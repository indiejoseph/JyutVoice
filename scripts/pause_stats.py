"""How many pauses does a set of utterances contain, compared with the commas in its text?

    python scripts/pause_stats.py --name teacher --manifest m.jsonl --audio-root <dir>        # jsonl lines {wav, text}
    python scripts/pause_stats.py --name tts_corpus --dataset-dir ~/jyutvoice_data/dataset_yue --n 150

Internal pauses = silences of at least `--min-pause` s (default 0.15) between speech segments (librosa.effects.split, top_db 35 below the clip's peak),
leading and trailing silence ignored. Internal punctuation = punctuation marks in the text except the last character. Reported: mean clip length,
pauses per clip and per minute, internal punctuation per clip, pauses >= 0.3 s per clip, and the median / 90th percentile pause length.
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import soundfile as sf

PUNCT = set("，。！？、；：,.!?;:…")


def pauses_of(y, sr, min_pause, top_db=35):
    import librosa

    iv = librosa.effects.split(y, top_db=top_db, frame_length=int(sr * 0.032), hop_length=int(sr * 0.008))
    return [(iv[i + 1][0] - iv[i][1]) / sr for i in range(len(iv) - 1) if (iv[i + 1][0] - iv[i][1]) / sr >= min_pause]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--audio-root", default="")
    ap.add_argument("--dataset-dir")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--min-pause", type=float, default=0.15)
    a = ap.parse_args()
    items = []
    if a.manifest:
        for line in open(a.manifest, encoding="utf-8"):
            r = json.loads(line)
            y, sr = sf.read(os.path.join(a.audio_root, r["wav"]), dtype="float32")
            items.append((y if y.ndim == 1 else y.mean(1), sr, r["text"]))
            if len(items) >= a.n:
                break
    else:
        import glob

        import pyarrow as pa
        import pyarrow.ipc as ipc

        files = sorted(glob.glob(os.path.join(a.dataset_dir, "data-*.arrow")))[:: max(1, 162 // 30)]
        for fn in files:
            b = next(iter(ipc.open_stream(pa.memory_map(fn)))).select(["audio", "text"])
            for r in b.slice(0, 6).to_pylist():
                y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                items.append((y if y.ndim == 1 else y.mean(1), sr, r["text"]))
            if len(items) >= a.n:
                break
    dur, n_p, n_p3, n_punct, allp = [], [], [], [], []
    for y, sr, text in items:
        p = pauses_of(y, sr, a.min_pause)
        dur.append(len(y) / sr)
        n_p.append(len(p))
        n_p3.append(sum(x >= 0.3 for x in p))
        allp += p
        n_punct.append(max(sum(c in PUNCT for c in text) - (1 if text.strip() and text.strip()[-1] in PUNCT else 0), 0))
    tot = sum(dur)
    q = (np.median(allp), np.percentile(allp, 90)) if allp else (float("nan"),) * 2
    print(f"{a.name:12s} n={len(items):3d} clip {np.mean(dur):5.1f}s | pauses>={a.min_pause:.2f}s: {np.mean(n_p):4.2f}/clip {60 * sum(n_p) / tot:5.1f}/min | "
          f"internal punctuation {np.mean(n_punct):4.2f}/clip | pauses>=0.3s {np.mean(n_p3):4.2f}/clip | pause length median {q[0]:.2f}s p90 {q[1]:.2f}s")


if __name__ == "__main__":
    main()
