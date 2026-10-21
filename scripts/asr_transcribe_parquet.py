"""Transcribe the clips of HF parquet shards (an `audio` column of {bytes, path} WAV, a text column) with a Qwen3-ASR vLLM endpoint, no forced language.

    python scripts/asr_transcribe_parquet.py --parquet '~/data/mce_raw/data/*.parquet' --out ~/data/mce_asr.jsonl --id-prefix mce \\
        --base-url http://127.0.0.1:18000 --model Qwen/Qwen3-ASR-1.7B --workers 2

One JSON line per clip: {"id": "<prefix>_<shard>_<row>", "asr": "...", "dur": seconds}. Resumable (finished ids are skipped). The request is the one of
winrex's eval_asr.py: chat completions with `input_audio`, temperature 0. With a low --workers it is a light load for a service that other people use.
"""
import argparse
import base64
import concurrent.futures as cf
import glob
import io
import json
import os
import time

import pyarrow.parquet as pq
import requests
import soundfile as sf


def transcribe(base_url, model, wav_bytes, max_tokens):
    payload = {
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": base64.b64encode(wav_bytes).decode(), "format": "wav"}}]}],
    }  # fmt: skip
    for attempt in range(5):
        try:
            r = requests.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=payload, timeout=300)
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"] or ""
                if "<asr_text>" in text:  # `language X<asr_text>...` when the endpoint keeps the tags
                    text = text.split("<asr_text>", 1)[1]
                return text.strip()
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id-prefix", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:18000")
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out_path = os.path.expanduser(a.out)
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(line)["id"] for line in open(out_path, encoding="utf-8") if line.strip()}
    files = sorted(glob.glob(os.path.expanduser(a.parquet)))
    t0, n_new, n_fail = time.time(), 0, 0
    with open(out_path, "a", encoding="utf-8") as out, cf.ThreadPoolExecutor(a.workers) as ex:
        for shard, f in enumerate(files):
            pending = []
            row = 0
            for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["audio"]):
                for audio in batch.column("audio").to_pylist():
                    cid = f"{a.id_prefix}_{shard:02d}_{row:05d}"
                    row += 1
                    if cid in done or (a.limit and n_new + len(pending) >= a.limit):
                        continue
                    pending.append((cid, audio["bytes"]))
                if len(pending) >= 256:
                    n_new, n_fail = flush(ex, a, pending, out, n_new, n_fail, t0)
                    pending = []
            n_new, n_fail = flush(ex, a, pending, out, n_new, n_fail, t0)
            if a.limit and n_new >= a.limit:
                break
    print(f"finished: {n_new} transcribed, {n_fail} failed, {time.time() - t0:.0f} s", flush=True)


def flush(ex, a, pending, out, n_new, n_fail, t0):
    futs = {ex.submit(transcribe, a.base_url, a.model, wav, a.max_tokens): (cid, wav) for cid, wav in pending}
    for fut in cf.as_completed(futs):
        cid, wav = futs[fut]
        text = fut.result()
        if text is None:
            n_fail += 1
            continue
        info = sf.info(io.BytesIO(wav))
        out.write(json.dumps({"id": cid, "asr": text, "dur": round(info.duration, 2)}, ensure_ascii=False) + "\n")
        n_new += 1
    out.flush()
    print(f"{n_new} done ({n_new / (time.time() - t0):.1f}/s), {n_fail} failed", flush=True)
    return n_new, n_fail


if __name__ == "__main__":
    main()
