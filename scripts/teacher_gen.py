#!/usr/bin/env python3
"""Generate S3 speech tokens with the fine-tuned Cantonese CosyVoice2 LLM for a pool of sentences (tokens only: no flow decoder, no vocoder, no audio).

One process per shard; run several per GPU. Resumable (finished ids are skipped), one JSON line per sentence:
    {"id", "mode", "prompt", "n_text_tokens", "tokens": [...], "gen_s"}

    python scripts/teacher_gen.py --cosyvoice-dir ~/teacher/CosyVoice --base-model-dir ~/teacher/base --llm-ckpt <model_acc_0.1813.pt> \\
        --qwen-dir ~/teacher/ft/qwen_pretrained/blank --sentences pool.jsonl --prompts ~/teacher/prompts/meta.json \\
        --prompt-mode alternate --shard 0 --num-shards 8 --out out/tokens_0.jsonl

--prompt-mode: `none` (no LLM prompt; speaker only through the flow x-vector), `zero_shot` (a speaker prompt's text and tokens go in front of the text: the delivery
follows the prompt's pace) or `alternate` (half of the ids each, chosen by a hash of the id; the prompt index comes from the same hash). Reuses the model patches
of teacher_bench.py (transformers 4.57 KV-cache mask, config-only Qwen encoder).
"""
import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "teacher_bench"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cosyvoice-dir", required=True)
    ap.add_argument("--base-model-dir", required=True)
    ap.add_argument("--llm-ckpt", required=True)
    ap.add_argument("--qwen-dir", required=True)
    ap.add_argument("--yaml", default=None)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--prompt-mode", default="alternate", choices=["none", "zero_shot", "alternate"])
    ap.add_argument("--prompt-index", type=int, default=None, help="use this prompt (index into the prompts file) for every sentence instead of the hash-chosen one")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-dir", required=True, help="scratch for the model-dir symlinks")
    ap.add_argument("--max-ratio", type=float, default=8.0, help="cap on speech tokens per text token (the diagnostics found 4-6 typical)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    import teacher_bench as tb

    tb.setup_imports(a.cosyvoice_dir)
    # cosyvoice imports a few packages that only its training / download paths use (modelscope: snapshot_download of a missing model dir; pyworld: F0 extraction in
    # the training data pipeline). Generation touches neither, so empty stand-ins avoid installing them.
    import importlib.util
    import types

    for name in ("modelscope", "pyworld"):
        if importlib.util.find_spec(name) is None:
            stub = types.ModuleType(name)

            def _unavailable(*args, _name=name, **kwargs):
                raise RuntimeError(f"{_name} is stubbed out in teacher_gen.py")

            stub.snapshot_download = stub.harvest = stub.stonemask = stub.cheaptrick = stub.d4c = _unavailable
            sys.modules[name] = stub

    yaml_src = tb.find_yaml(a.base_model_dir, a.cosyvoice_dir, a.yaml)
    work = Path(a.work_dir) / f"shard{a.shard}"
    model_dir = tb.build_model_dir(a.base_model_dir, a.llm_ckpt, work / "model_dir", yaml_src, a.qwen_dir)
    from cosyvoice.cli.cosyvoice import CosyVoice2

    tb.patch_qwen_encoder_from_config()
    tb.patch_qwen_encoder_cache()
    cosy = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=True)

    import torchaudio

    meta_dir = Path(a.prompts).parent  # the wav paths in meta.json are relative to another host: take the file names and look next to the meta file
    prompts = []
    for m in json.load(open(a.prompts, encoding="utf-8"))["prompts"]:
        wav, sr = torchaudio.load(str(meta_dir / Path(m["wav"]).name))
        assert sr == 16000, "prompts must be 16 kHz"
        prompts.append({"meta": m, "wav": wav.mean(dim=0, keepdim=True)})

    sentences = [json.loads(line) for line in open(a.sentences, encoding="utf-8")]
    mine = [s for i, s in enumerate(sentences) if i % a.num_shards == a.shard]
    if a.limit:
        mine = mine[: a.limit]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["id"] for line in open(a.out, encoding="utf-8") if line.strip()}
    todo = [s for s in mine if s["id"] not in done]
    print(f"shard {a.shard}/{a.num_shards}: {len(mine)} sentences, {len(done)} already done, {len(todo)} to go", flush=True)

    cache = {}
    t0, n_tok, n_done = time.time(), 0, 0
    with open(a.out, "a", encoding="utf-8") as out:
        for s in todo:
            h = zlib.crc32(s["id"].encode())
            mode = a.prompt_mode if a.prompt_mode != "alternate" else ("zero_shot" if h % 2 == 0 else "none")
            p_idx = (h // 2) % len(prompts) if a.prompt_index is None else a.prompt_index
            key = (p_idx, mode)
            if key not in cache:
                cache[key] = tb.base_inputs(cosy, prompts[p_idx], mode)
            mi = dict(cache[key])
            text_token, _ = cosy.frontend._extract_text_token(s["text"])
            t1 = time.time()
            toks = tb.llm_generate(cosy, mi, text_token, a.seed + h, f"gen-{s['id']}", a.max_ratio)
            gen_s = time.time() - t1
            out.write(json.dumps({"id": s["id"], "mode": mode, "prompt": p_idx, "n_text_tokens": int(text_token.shape[1]), "tokens": toks,
                                  "hit_max": len(toks) >= int(text_token.shape[1] * a.max_ratio), "gen_s": round(gen_s, 2)}) + "\n")
            out.flush()
            n_tok += len(toks)
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t0
                print(f"shard {a.shard}: {n_done}/{len(todo)} sentences, {n_tok / el:.0f} tok/s, {n_done / el * 3600:.0f} sentences/h", flush=True)
    print(f"shard {a.shard}: finished {n_done} sentences in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
