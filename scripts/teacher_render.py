#!/usr/bin/env python3
"""Render some of the teacher's generated token sequences to audio (frozen flow decoder + HiFT), to compare the teacher with the student on the SAME sentences.

    python scripts/teacher_render.py ... --tokens 'out/pilot/tokens_*.jsonl' --sentences pool_pilot.jsonl --n 60 --min-commas 2 --out-dir render_pilot

Writes <out-dir>/wavs/<id>.wav (24 kHz) and <out-dir>/manifest.jsonl ({wav, text, id, mode, prompt}), the format scripts/pause_stats.py reads.
"""
import argparse
import json
import os
import sys
import types
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "teacher_bench"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cosyvoice-dir", required=True)
    ap.add_argument("--base-model-dir", required=True)
    ap.add_argument("--llm-ckpt", required=True)
    ap.add_argument("--qwen-dir", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--min-commas", type=int, default=2)
    ap.add_argument("--min-chars", type=int, default=25)
    ap.add_argument("--max-chars", type=int, default=70)
    ap.add_argument("--keep-hit-max", action="store_true", help="also render sequences that hit the token cap (possibly cut off); the manifest says which")
    a = ap.parse_args()

    import glob

    import soundfile as sf
    import torch
    import torchaudio
    import teacher_bench as tb

    tb.setup_imports(a.cosyvoice_dir)
    for name in ("modelscope", "pyworld"):
        import importlib.util

        if importlib.util.find_spec(name) is None:
            stub = types.ModuleType(name)
            stub.snapshot_download = stub.harvest = stub.stonemask = stub.cheaptrick = stub.d4c = lambda *x, **k: (_ for _ in ()).throw(RuntimeError("stubbed"))
            sys.modules[name] = stub
    model_dir = tb.build_model_dir(a.base_model_dir, a.llm_ckpt, Path(a.work_dir) / "render_model_dir", tb.find_yaml(a.base_model_dir, a.cosyvoice_dir), a.qwen_dir)
    from cosyvoice.cli.cosyvoice import CosyVoice2

    tb.patch_qwen_encoder_from_config()
    tb.patch_qwen_encoder_cache()
    cosy = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=True)

    meta_dir = Path(a.prompts).parent
    prompts = []
    for m in json.load(open(a.prompts, encoding="utf-8"))["prompts"]:
        wav, sr = torchaudio.load(str(meta_dir / Path(m["wav"]).name))
        prompts.append({"meta": m, "wav": wav.mean(dim=0, keepdim=True)})

    pool = {}
    for line in open(a.sentences, encoding="utf-8"):
        r = json.loads(line)
        pool[r["id"]] = r
    gens = []
    for fn in sorted(glob.glob(a.tokens)):
        for line in open(fn, encoding="utf-8"):
            if line.strip():
                g = json.loads(line)
                t = pool[g["id"]]["text"]
                if (a.keep_hit_max or not g["hit_max"]) and a.min_chars <= len(t) <= a.max_chars and t[:-1].count("，") >= a.min_commas:
                    gens.append(g)
    gens = sorted(gens, key=lambda g: g["id"])[: a.n]
    os.makedirs(os.path.join(a.out_dir, "wavs"), exist_ok=True)
    cache, out = {}, open(os.path.join(a.out_dir, "manifest.jsonl"), "w", encoding="utf-8")
    for g in gens:
        key = (g["prompt"], g["mode"])
        if key not in cache:
            cache[key] = tb.base_inputs(cosy, prompts[g["prompt"]], g["mode"])
        wav = tb.render(cosy, dict(cache[key]), g["tokens"], f"render-{g['id']}")
        path = os.path.join("wavs", f"{g['id']}.wav")
        sf.write(os.path.join(a.out_dir, path), wav.squeeze(0).numpy(), cosy.sample_rate)
        out.write(json.dumps({"wav": path, "text": pool[g["id"]]["text"], "id": g["id"], "mode": g["mode"], "prompt": g["prompt"], "hit_max": g["hit_max"]}, ensure_ascii=False) + "\n")
        out.flush()
    print(f"rendered {len(gens)} sentences -> {a.out_dir}")


if __name__ == "__main__":
    main()
