"""Pitch statistics per model of a synth_compare.py output directory (plus any reference added with add_reference_to_synth.py).

    python scripts/f0_stats.py --synth-dir tmp/synth_dp [--jobs 8]

F0 is tracked with pyin (70-500 Hz, 10 ms hop) and turned into semitones around the clip's own median, so voices of different pitch compare. Per clip:
    spread      std of the semitone track
    range       90th - 10th percentile
    slope       least-squares slope over the clip, semitones per second (global declination)
    phrase decl mean over phrases (voiced stretches separated by >= 0.15 s without voicing, at least 0.8 s long) of the first-third mean minus the last-third
                mean: how much a phrase falls from its start to its end
    liveliness  median absolute change between consecutive voiced frames, in semitones per 10 ms
    reset       mean over phrase boundaries of (mean of the next phrase's first third) - (mean of the previous phrase's last third): how far the pitch jumps
                back up after a pause
Reported as the mean over clips per model. These are descriptive; a difference to the reference says "different", not "worse".
"""
import argparse
import collections
import json
import os
from multiprocessing import Pool

import numpy as np

SR = 16000
HOP = 160


def track(path):
    import librosa

    y, _ = librosa.load(path, sr=SR)
    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=500, sr=SR, frame_length=1024, hop_length=HOP)
    ok = voiced & ~np.isnan(f0)
    st = np.full(len(f0), np.nan)
    if ok.sum() >= 20:
        st[ok] = 12 * np.log2(f0[ok] / np.median(f0[ok]))
    return st


def phrases(st, min_gap=15, min_len=80):
    """[(start, stop)] of the voiced stretches: gaps of fewer than min_gap unvoiced frames are bridged; stretches shorter than min_len frames are dropped."""
    v = ~np.isnan(st)
    idx = np.where(v)[0]
    if len(idx) == 0:
        return []
    out, s, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > min_gap:
            out.append((s, prev + 1))
            s = i
        prev = i
    out.append((s, prev + 1))
    return [(a, b) for a, b in out if b - a >= min_len]


def stats(path):
    st = track(path)
    v = ~np.isnan(st)
    if v.sum() < 50:
        return None
    t = np.arange(len(st))[v] * HOP / SR
    vals = st[v]
    d = np.abs(np.diff(st))
    d = d[~np.isnan(d)]
    decl, firsts, lasts = [], [], []
    for a, b in phrases(st):
        seg = st[a:b]
        n = len(seg) // 3
        f, l = np.nanmean(seg[:n]), np.nanmean(seg[-n:])
        decl.append(f - l)
        firsts.append(f)
        lasts.append(l)
    reset = [firsts[i + 1] - lasts[i] for i in range(len(firsts) - 1)]
    return {
        "spread": float(vals.std()), "range": float(np.percentile(vals, 90) - np.percentile(vals, 10)), "slope": float(np.polyfit(t, vals, 1)[0]),
        "phrase decl": float(np.mean(decl)) if decl else np.nan, "liveliness": float(np.median(d)) if len(d) else np.nan,
        "reset": float(np.mean(reset)) if reset else np.nan, "phrases": len(decl),
    }  # fmt: skip


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(os.path.join(a.synth_dir, "results.jsonl"), encoding="utf-8")]
    with Pool(a.jobs) as p:
        res = p.map(stats, [os.path.join(a.synth_dir, r["wav"]) for r in rows])
    by = collections.defaultdict(list)
    for r, s in zip(rows, res):
        if s is not None:
            by[r["model"]].append(s)
    keys = ["spread", "range", "slope", "phrase decl", "liveliness", "reset", "phrases"]
    print(f"{'model':12s} {'n':>3s} " + " ".join(f"{k:>12s}" for k in keys))
    for m, ss in by.items():
        cells = []
        for k in keys:
            v = np.array([s[k] for s in ss], dtype=float)
            cells.append(f"{np.nanmean(v):6.2f}±{np.nanstd(v) / np.sqrt(max(1, np.sum(~np.isnan(v)))):4.2f}")
        print(f"{m:12s} {len(ss):3d} " + " ".join(f"{c:>12s}" for c in cells))


if __name__ == "__main__":
    main()
