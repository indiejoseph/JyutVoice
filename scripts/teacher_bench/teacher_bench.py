#!/usr/bin/env python3
"""Benchmark the Cantonese CosyVoice2 LLM as a distillation teacher on one (MIG) GPU.

Phases (--modes):
  gen     autoregressive generation on long Cantonese sentences: tokens/s, RTF, memory, EOS behaviour, and
          rendered wavs (frozen flow + HiFT) to listen to. Run for each --prompt_modes entry:
            zero_shot  speaker prompt text+tokens go into the LLM prefix (CosyVoice2 zero-shot)
            none       no LLM prompt (cross-lingual style); speaker only via the flow x-vector
  hidden  teacher-forced hidden-state extraction on real (text, speech-token) pairs from the LLM's own
          training CSV: throughput and peak memory per batch size, and what caching would cost on disk.

Nothing outside --work_dir is written. The model dir is a scratch dir of symlinks (the fine-tuned checkpoint
is linked as llm.pt; the yaml is copied to cosyvoice2.yaml, which the stock CosyVoice2 class insists on).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosyvoice_dir", required=True, help="stock CosyVoice checkout (provides the cosyvoice package)")
    ap.add_argument("--base_model_dir", required=True, help="CosyVoice2-0.5B dir with flow.pt, hift.pt, campplus, tokenizer, yaml")
    ap.add_argument("--llm_ckpt", required=True, help="fine-tuned Qwen2LM state dict (e.g. best_model/model_acc_0.1813.pt)")
    ap.add_argument("--qwen_dir", required=True, help="the fine-tune's Qwen config+tokenizer dir (qwen_pretrained/blank)")
    ap.add_argument("--yaml", default=None, help="cosyvoice2.yaml to use (default: base dir's, else the upstream libritts example)")
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--sentences", help="jsonl from make_text_pool.py")
    ap.add_argument("--prompts", help="meta.json from make_prompts.py")
    ap.add_argument("--train_csv", help="dataset20250907.csv (key,text,codes) for the hidden-state phase")
    ap.add_argument("--modes", default="gen,hidden")
    ap.add_argument("--prompt_modes", default="zero_shot,none")
    ap.add_argument("--n_sentences", type=int, default=100)
    ap.add_argument("--n_hidden", type=int, default=512, help="(text, tokens) pairs for the hidden-state phase")
    ap.add_argument("--hidden_batch_sizes", default="8,16,32,64")
    ap.add_argument("--no_render", action="store_true", help="skip flow+HiFT (tokens only)")
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_ratio", type=float, default=20, help="cap on speech tokens per text token (Qwen2LM default 20); lower it to cut runaway generations short")
    return ap.parse_args()


def setup_imports(cosy_dir):
    # The venv's flash_attn (built for torch 2.7) and xformers fail to load against its torch 2.8 on GPU nodes
    # (undefined symbol), and diffusers -> xformers -> flash_attn pulls them in while cosyvoice imports. Neither is
    # needed here, so make them look uninstalled for this process only (find_spec() returns None for these).
    for name in ("flash_attn", "xformers"):
        sys.modules[name] = None
    for p in (cosy_dir, os.path.join(cosy_dir, "third_party", "Matcha-TTS")):
        if p not in sys.path:
            sys.path.insert(0, p)
    # The CosyVoice frontend imports a text-normaliser at import time. We normalise nothing here, so a no-op
    # stand-in is enough when neither ttsfrd nor wetext is installed.
    try:
        import ttsfrd  # noqa: F401
    except ImportError:
        try:
            import wetext  # noqa: F401
        except ImportError:
            stub = types.ModuleType("wetext")

            class Normalizer:
                def __init__(self, *a, **k):
                    pass

                def normalize(self, text):
                    return text

            stub.Normalizer = Normalizer
            sys.modules["wetext"] = stub


def find_yaml(base, cosy_dir, override=None):
    """A yaml matching the current cosyvoice code. The old `cosyvoice.yaml` in the 0.5B dir builds the flow
    estimator as ConditionalDecoder(causal=True), which the current code no longer accepts, so it is not a fallback."""
    cands = [override] if override else [
        Path(base) / "cosyvoice2.yaml",
        Path(cosy_dir) / "examples" / "libritts" / "cosyvoice2" / "conf" / "cosyvoice2.yaml",
    ]
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(f"no cosyvoice2.yaml found, tried {[str(c) for c in cands]}; pass --yaml")


def patch_qwen_encoder_from_config():
    """The fine-tune's Qwen dir (config + tokenizer, no weights) is not loadable by Qwen2ForCausalLM.from_pretrained,
    which the stock Qwen2Encoder uses. Build the encoder from the config alone, as the fine-tune script did;
    every weight then comes from llm.pt."""
    import cosyvoice.llm.llm as cosy_llm
    from transformers import AutoConfig, Qwen2ForCausalLM

    def init(self, pretrain_path):
        torch.nn.Module.__init__(self)
        self.model = Qwen2ForCausalLM(AutoConfig.from_pretrained(pretrain_path))

    cosy_llm.Qwen2Encoder.__init__ = init


def patch_qwen_encoder_cache():
    """CosyVoice pins transformers==4.51.3 and decodes step by step with attention_mask=masks[:, -1, :], which is length 1
    after the first step while the KV cache keeps growing. Newer transformers (4.57 here) then attend wrongly: on a tiny
    random Qwen2, cached decoding deviated from the full forward by 0.12 (vs 2e-7 on 4.51.3), and the real model never
    produced end-of-speech, even the upstream control. Mask the whole sequence (cache + current chunk) instead; with
    batch size 1 and no padding that is exactly what the original mask means, and it matches the full forward to 2e-7."""
    import cosyvoice.llm.llm as cosy_llm

    def forward_one_step(self, xs, masks, cache=None):
        if cache is None:
            past = 0
        elif hasattr(cache, "get_seq_length"):
            past = cache.get_seq_length()
        else:  # legacy tuple cache
            past = cache[0][0].shape[2]
        input_masks = torch.ones(xs.shape[0], past + xs.shape[1], dtype=torch.bool, device=xs.device)
        outs = self.model(inputs_embeds=xs, attention_mask=input_masks, output_hidden_states=True, return_dict=True,
                          use_cache=True, past_key_values=cache)
        return outs.hidden_states[-1], outs.past_key_values

    cosy_llm.Qwen2Encoder.forward_one_step = forward_one_step


def build_model_dir(base, llm_ckpt, dst, yaml_src, qwen_dir):
    base, dst = Path(base), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    def link(name, target):
        p = dst / name
        if p.is_symlink() or p.exists():
            p.unlink()
        p.symlink_to(Path(target).resolve())

    for name in ["campplus.onnx", "speech_tokenizer_v2.onnx", "flow.pt", "hift.pt", "spk2info.pt"]:
        if (base / name).exists():
            link(name, base / name)
    # The fine-tune's Qwen config + tokenizer (vocab_size 152064 with 367 added tokens, mostly Cantonese characters),
    # NOT the stock CosyVoice-BlankEN (151936): the checkpoint's embedding shape and the token ids depend on it.
    link("CosyVoice-BlankEN", qwen_dir)
    link("llm.pt", llm_ckpt)
    (dst / "cosyvoice2.yaml").write_bytes(Path(yaml_src).read_bytes())
    print(f"model dir {dst} (yaml: {yaml_src})", flush=True)
    return str(dst)


def gpu_info():
    info = {"cuda": torch.cuda.is_available()}
    if info["cuda"]:
        props = torch.cuda.get_device_properties(0)
        info.update(name=props.name, total_gb=round(props.total_memory / 2**30, 1), sm_count=props.multi_processor_count)
        try:
            info["nvidia_smi_L"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            pass
    return info


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_gb():
    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1)))] if xs else None


# ----------------------------------------------------------------------------------------------- generation
def load_prompts(meta_path):
    import torchaudio

    metas = json.load(open(meta_path, encoding="utf-8"))["prompts"]
    out = []
    for m in metas:
        wav, sr = torchaudio.load(m["wav"])
        assert sr == 16000, "prompts must be 16 kHz"
        out.append({"meta": m, "wav": wav.mean(dim=0, keepdim=True)})
    return out


def base_inputs(cosy, prompt, prompt_mode):
    """Prompt-dependent model inputs, computed once per (prompt, mode); text is filled in per sentence."""
    fe = cosy.frontend
    if prompt_mode == "zero_shot":
        mi = fe.frontend_zero_shot("x", prompt["meta"]["text"], prompt["wav"], cosy.sample_rate, "")
    else:
        mi = fe.frontend_cross_lingual("x", prompt["wav"], cosy.sample_rate, "")
    return {k: v for k, v in mi.items() if k not in ("text", "text_len")}


def llm_generate(cosy, mi, text_token, seed, uid, max_ratio=20):
    m, dev = cosy.model, cosy.model.device
    zeros_i = torch.zeros(1, 0, dtype=torch.int32)
    prompt_text = mi.get("prompt_text", zeros_i)
    prompt_tok = mi.get("llm_prompt_speech_token", zeros_i)
    emb = mi.get("llm_embedding", torch.zeros(0, 192))
    torch.manual_seed(seed)
    toks = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=bool(m.fp16) and torch.cuda.is_available()):
        for t in m.llm.inference(
            text=text_token.to(dev),
            text_len=torch.tensor([text_token.shape[1]], dtype=torch.int32).to(dev),
            prompt_text=prompt_text.to(dev),
            prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(dev),
            prompt_speech_token=prompt_tok.to(dev),
            prompt_speech_token_len=torch.tensor([prompt_tok.shape[1]], dtype=torch.int32).to(dev),
            embedding=emb.to(dev),
            max_token_text_ratio=max_ratio,
            uuid=uid,
        ):
            toks.append(int(t))
    return toks


def render(cosy, mi, toks, uid):
    m = cosy.model
    m.hift_cache_dict[uid] = None
    with torch.inference_mode():
        wav = m.token2wav(
            token=torch.tensor(toks).unsqueeze(0),
            prompt_token=mi["flow_prompt_speech_token"],
            prompt_feat=mi["prompt_speech_feat"],
            embedding=mi["flow_embedding"],
            token_offset=0,
            uuid=uid,
            stream=False,
            finalize=True,
        )
    m.hift_cache_dict.pop(uid, None)
    return wav.float().cpu()


def run_gen(cosy, args, work, sentences, prompts, prompt_mode, results):
    import soundfile as sf

    wav_dir = work / "wavs" / prompt_mode
    wav_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    rows = []
    token_rows = []
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    for i, s in enumerate(sentences):
        p_idx = i % len(prompts)
        key = (p_idx, prompt_mode)
        if key not in cache:
            cache[key] = base_inputs(cosy, prompts[p_idx], prompt_mode)
        mi = dict(cache[key])
        text_token, _ = cosy.frontend._extract_text_token(s["text"])
        max_len = int(text_token.shape[1] * args.max_ratio)  # Qwen2LM.inference stops generating here
        uid = f"bench-{prompt_mode}-{i}"
        sync()
        t0 = time.time()
        toks = llm_generate(cosy, mi, text_token, args.seed + i, uid, args.max_ratio)
        sync()
        gen_s = time.time() - t0
        audio_s, wav_name = None, ""
        if toks and not args.no_render:
            wav = render(cosy, mi, toks, uid)
            audio_s = wav.shape[1] / cosy.sample_rate
            wav_name = f"{s['id']}_p{p_idx}.wav"
            sf.write(str(wav_dir / wav_name), wav.squeeze(0).numpy(), cosy.sample_rate)
        elif toks:
            audio_s = len(toks) / 25.0
        row = {
            "id": s["id"], "mode": prompt_mode, "prompt": p_idx, "bin": s["bin"], "kind": s["kind"],
            "n_chars": s["n_chars"], "n_text_tokens": int(text_token.shape[1]), "n_speech_tokens": len(toks),
            "gen_s": round(gen_s, 3), "tokens_per_s": round(len(toks) / gen_s, 1) if gen_s > 0 else None,
            "audio_s": round(audio_s, 2) if audio_s else None,
            "chars_per_s": round(s["n_chars"] / audio_s, 2) if audio_s else None,
            "rtf": round(gen_s / audio_s, 3) if audio_s else None,
            "hit_max_len": len(toks) >= max_len,
            "distinct_last200": round(len(set(toks[-200:])) / max(1, min(200, len(toks))), 2),
            "wav": wav_name, "text": s["text"],
        }
        rows.append(row)
        token_rows.append({"id": s["id"], "mode": prompt_mode, "prompt": p_idx, "n_text_tokens": int(text_token.shape[1]), "tokens": toks})
        print(f"[{prompt_mode} {i + 1}/{len(sentences)}] {row['n_chars']}ch -> {row['n_speech_tokens']} tok, "
              f"{row['tokens_per_s']} tok/s, {row['audio_s']}s audio, {row['chars_per_s']} ch/s, max_len_hit={row['hit_max_len']}", flush=True)
    with open(work / f"tokens_{prompt_mode}.jsonl", "w") as f:
        for t in token_rows:
            f.write(json.dumps(t) + "\n")
    tps = [r["tokens_per_s"] for r in rows if r["tokens_per_s"]]
    cps = [r["chars_per_s"] for r in rows if r["chars_per_s"]]
    tok_per_char = [r["n_speech_tokens"] / r["n_chars"] for r in rows if r["n_chars"]]
    results["gen"][prompt_mode] = {
        "n": len(rows),
        "tokens_per_s": {"mean": round(statistics.mean(tps), 1), "p50": pct(tps, .5), "p10": pct(tps, .1)} if tps else None,
        "rtf_median": pct([r["rtf"] for r in rows if r["rtf"]], .5),
        "hit_max_len": sum(r["hit_max_len"] for r in rows),
        "chars_per_s": {"p10": pct(cps, .1), "p50": pct(cps, .5), "p90": pct(cps, .9)} if cps else None,
        "speech_tokens_per_char": {"p10": round(pct(tok_per_char, .1), 2), "p50": round(pct(tok_per_char, .5), 2), "p90": round(pct(tok_per_char, .9), 2)},
        "peak_gpu_gb": peak_gb(),
        "by_bin": {
            b: {
                "n": sum(1 for r in rows if r["bin"] == b),
                "eos_rate": round(1 - sum(r["hit_max_len"] for r in rows if r["bin"] == b) / max(1, sum(1 for r in rows if r["bin"] == b)), 2),
                "speech_tokens_per_char_p50": pct([r["n_speech_tokens"] / r["n_chars"] for r in rows if r["bin"] == b], .5),
            }
            for b in sorted({r["bin"] for r in rows}, key=lambda x: int(x.split("-")[0]))
        },
        "by_bin_chars_per_s_p50": {
            b: pct([r["chars_per_s"] for r in rows if r["bin"] == b and r["chars_per_s"]], .5) for b in sorted({r["bin"] for r in rows})
        },
    }
    return rows


# ------------------------------------------------------------------------------------------------- hidden
def run_hidden(cosy, args, work, results):
    import pandas as pd
    from torch.nn.utils.rnn import pad_sequence

    llm, dev = cosy.model.llm, cosy.model.device
    tok = cosy.frontend.tokenizer
    df = pd.read_csv(args.train_csv, nrows=max(20000, args.n_hidden * 8))
    df = df.sample(n=args.n_hidden, random_state=args.seed)
    samples = []
    for text, codes in zip(df["text"], df["codes"]):
        tt = tok.encode(str(text), allowed_special=cosy.frontend.allowed_special)
        st = [int(x) for x in str(codes).split()]
        if tt and st:
            samples.append((tt, st))
    samples.sort(key=lambda x: len(x[0]) + len(x[1]))  # length-sorted batches, like a bucketed loader
    total_speech = sum(len(s) for _, s in samples)
    print(f"hidden phase: {len(samples)} pairs, {total_speech} speech tokens (~{total_speech / 25:.0f}s of speech)", flush=True)

    sos = llm.llm_embedding.weight[llm.sos_eos].reshape(1, -1)
    task = llm.llm_embedding.weight[llm.task_id].reshape(1, -1)
    embed_text = llm.llm.model.model.embed_tokens
    out = {"hidden_size": int(sos.shape[-1]), "n_pairs": len(samples), "speech_tokens": total_speech, "batches": {}}
    for bs in [int(x) for x in args.hidden_batch_sizes.split(",")]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            sync()
            t0 = time.time()
            n_tok = 0
            with torch.inference_mode():
                for b in range(0, len(samples), bs):
                    chunk = samples[b:b + bs]
                    seqs, lens = [], []
                    for tt, st in chunk:
                        t = torch.tensor(tt, device=dev)
                        s = torch.tensor(st, device=dev)
                        seqs.append(torch.cat([sos, embed_text(t), task, llm.speech_embedding(s)], dim=0))
                        lens.append(seqs[-1].shape[0])
                    x = pad_sequence(seqs, batch_first=True)
                    hidden, _ = llm.llm(x, torch.tensor(lens, dtype=torch.int32, device=dev))
                    # state at position 1+Ti (the task-id slot) .. 1+Ti+Si-1 predicts speech token 0..Si-1
                    for i, (tt, st) in enumerate(chunk):
                        h = hidden[i, 1 + len(tt):1 + len(tt) + len(st)]
                        assert h.shape[0] == len(st)
                        n_tok += h.shape[0]
            sync()
            dt = time.time() - t0
            out["batches"][str(bs)] = {
                "seconds": round(dt, 2), "speech_tokens_per_s": round(n_tok / dt, 0),
                "audio_s_per_s": round(n_tok / 25 / dt, 1), "peak_gpu_gb": peak_gb(),
            }
            print(f"  batch {bs}: {n_tok / dt:.0f} speech tok/s, peak {peak_gb()} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            out["batches"][str(bs)] = {"oom": True}
            print(f"  batch {bs}: OOM", flush=True)
            break
    best = max((v["speech_tokens_per_s"] for v in out["batches"].values() if "speech_tokens_per_s" in v), default=None)
    if best:
        hrs_484 = 484 * 3600 * 25
        out["extrapolation"] = {
            "best_speech_tokens_per_s": best,
            "hours_for_484h_corpus": round(hrs_484 / best / 3600, 1),
            "cache_gb_fp16_last_layer_484h": round(hrs_484 * out["hidden_size"] * 2 / 1e9, 1),
        }
    results["hidden"] = out


# ---------------------------------------------------------------------------------------------------- main
def main():
    args = parse_args()
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    setup_imports(args.cosyvoice_dir)
    modes = set(args.modes.split(","))
    results = {"gpu": gpu_info(), "args": vars(args), "gen": {}}
    print("GPU:", json.dumps(results["gpu"], ensure_ascii=False), flush=True)

    yaml_src = find_yaml(args.base_model_dir, args.cosyvoice_dir, args.yaml)
    model_dir = build_model_dir(args.base_model_dir, args.llm_ckpt, work / "model_dir", yaml_src, args.qwen_dir)
    from cosyvoice.cli.cosyvoice import CosyVoice2

    patch_qwen_encoder_from_config()
    patch_qwen_encoder_cache()

    t0 = time.time()
    cosy = CosyVoice2(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=not args.no_fp16)
    results["load_s"] = round(time.time() - t0, 1)
    results["after_load_gpu_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0
    print(f"model loaded in {results['load_s']}s, {results['after_load_gpu_gb']} GB allocated", flush=True)

    all_rows = []
    if "gen" in modes:
        sentences = [json.loads(l) for l in open(args.sentences, encoding="utf-8")][: args.n_sentences]
        prompts = load_prompts(args.prompts)
        for pm in args.prompt_modes.split(","):
            all_rows += run_gen(cosy, args, work, sentences, prompts, pm, results)
        cols = ["id", "mode", "prompt", "bin", "kind", "n_chars", "n_text_tokens", "n_speech_tokens", "gen_s",
                "tokens_per_s", "audio_s", "chars_per_s", "rtf", "hit_max_len", "distinct_last200", "wav", "text"]
        with open(work / "manifest.tsv", "w", encoding="utf-8") as f:
            f.write("\t".join(cols) + "\n")
            for r in all_rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
    if "hidden" in modes:
        run_hidden(cosy, args, work, results)

    with open(work / "report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print("\n=== report ===")
    print(json.dumps({k: v for k, v in results.items() if k != "args"}, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
