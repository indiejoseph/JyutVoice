#!/usr/bin/env python3
"""Pick a few diverse Cantonese speaker prompts (16 kHz wav + transcript) from the JyutVoice dataset (CPU only).

The teacher needs a prompt clip to choose a voice. Candidates are random `yue` rows of 3.5-8 s with a
transcript, taken from a few random arrow shards; the final K are chosen by farthest-point sampling on the
stored x-vectors so the voices differ. Shards are memory-mapped one at a time (mapping all 138 GB at once, as
datasets.load_from_disk does, fails on the login node). Read-only on the dataset; writes wavs + meta.json.

    python make_prompts.py --dataset .../tmp/dataset_processed --out_dir .../prompts --k 4
"""
import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n_shards", type=int, default=8)
    ap.add_argument("--per_shard", type=int, default=60, help="random yue rows examined per shard")
    ap.add_argument("--min_dur", type=float, default=3.5)
    ap.add_argument("--max_dur", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.dataset)
    files = [f["filename"] for f in json.load(open(root / "state.json"))["_data_files"]]
    rng = random.Random(args.seed)
    rng.shuffle(files)

    kept = []
    for fn in files[: args.n_shards]:
        with pa.memory_map(str(root / fn)) as mm:
            t = pa.ipc.open_stream(mm).read_all()
            lang = t.column("lang").to_pylist()
            yue = [i for i, l in enumerate(lang) if l == "yue"]
            for i in rng.sample(yue, min(args.per_shard, len(yue))):
                if "text_valid" in t.column_names and not t.column("text_valid")[i].as_py():
                    continue
                text = t.column("text")[i].as_py().replace(" ", "")
                if not (10 <= len(text) <= 60):
                    continue
                audio = t.column("audio")[i].as_py()
                raw = audio.get("bytes") if isinstance(audio, dict) else None
                if not raw:
                    continue
                info = sf.info(io.BytesIO(raw))
                if not (args.min_dur <= info.duration <= args.max_dur):
                    continue
                arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
                if arr.ndim > 1:
                    arr = arr.mean(axis=1)
                spk = np.asarray(t.column("spk_emb")[i].as_py(), dtype=np.float32)
                spk = spk / (np.linalg.norm(spk) + 1e-9)
                kept.append({"shard": fn, "row": i, "id": t.column("id")[i].as_py(), "text": text,
                             "arr": arr, "sr": int(sr), "dur": float(info.duration), "spk": spk})
        print(f"{fn}: {len(kept)} usable candidates so far", flush=True)
    if len(kept) < args.k:
        raise SystemExit("not enough candidates; raise --n_shards/--per_shard or relax --min_dur/--max_dur")

    # farthest-point sampling on cosine distance
    chosen = [rng.randrange(len(kept))]
    while len(chosen) < args.k:
        best, best_d = None, -1.0
        for j, c in enumerate(kept):
            if j in chosen:
                continue
            d = min(1.0 - float(c["spk"] @ kept[k]["spk"]) for k in chosen)
            if d > best_d:
                best, best_d = j, d
        chosen.append(best)

    import librosa

    prompts = []
    for n, j in enumerate(chosen):
        c = kept[j]
        arr = c["arr"]
        if c["sr"] != 16000:
            arr = librosa.resample(arr, orig_sr=c["sr"], target_sr=16000)
        peak = float(np.abs(arr).max()) or 1.0
        if peak > 0.99:
            arr = arr * (0.99 / peak)
        wav = out / f"prompt{n}.wav"
        sf.write(str(wav), arr, 16000)
        prompts.append({"wav": str(wav), "text": c["text"], "dur": round(c["dur"], 2), "id": c["id"],
                        "shard": c["shard"], "row": c["row"]})
        print(f"prompt{n}: {c['dur']:.1f}s  {c['text'][:40]}")
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"prompts": prompts}, f, ensure_ascii=False, indent=2)
    print("wrote", out / "meta.json")


if __name__ == "__main__":
    main()
