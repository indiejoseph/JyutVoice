#!/usr/bin/env python3
"""Can pauses be read off the flow-encoder states? Train a silence probe on real corpus frames, apply it to the teacher's states.

    python scripts/silence_probe.py --corpus ~/jyutvoice_data/dataset_yue --teacher ~/teacher/rows_pilot --n-corpus 3000

S3 tokens have no single silence code, so a pause is not a run of one token. The probe is a logistic regression on the 80-d `decoder_h` frame, labelled from the AUDIO of
real corpus clips (a frame is silent when it lies in a gap of librosa.effects.split with top_db 35, edges included). Applied to a set of `decoder_h` sequences it
gives silent runs: reported are leading/trailing silence and the INTERNAL runs (pauses) per sentence, per comma and per minute, for the corpus (held-out clips) and for the
teacher rows. Frame t of decoder_h covers samples [480 t, 480 (t + 1)) of the 24 kHz audio (50 Hz).
"""
import argparse
import glob
import io
import json
import os

import sys

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from jyutvoice.utils.silence_probe import SilenceProbe  # noqa: E402

PUNCT = set("，。！？、；：,.!?;:…")


def read_rows(directory, columns, n, stride_files=1, per_file=40, with_audio=False):
    files = sorted(glob.glob(os.path.join(directory, "data-*.arrow")))[::stride_files]
    out = []
    for fn in files:
        for b in ipc.open_stream(pa.memory_map(fn)):
            out += b.select(columns).slice(0, per_file).to_pylist()
            break
        if len(out) >= n:
            break
    return out[:n]


def silent_frames(wav, n_frames):
    import librosa

    iv = librosa.effects.split(wav, top_db=35, frame_length=960, hop_length=240)
    speech = np.zeros(n_frames, dtype=bool)
    for a, b in iv:
        speech[a // 480: (b + 479) // 480] = True
    return ~speech


def runs_of(mask):
    out, n, start = [], 0, 0
    for i, m in enumerate(list(mask) + [False]):
        if m:
            if n == 0:
                start = i
            n += 1
        elif n:
            out.append((start, n))
            n = 0
    return out


def internal_pauses(sil):
    r = runs_of(sil)
    return [n for s, n in r if s > 0 and s + n < len(sil)], (r[0][1] if r and r[0][0] == 0 else 0), (r[-1][1] if r and r[-1][0] + r[-1][1] == len(sil) else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--n-corpus", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", type=float, default=0.85, help="held-out precision the decision threshold is chosen for")
    ap.add_argument("--save", help="write the probe (npz) here")
    a = ap.parse_args()

    rows = read_rows(a.corpus, ["audio", "decoder_h", "text"], a.n_corpus, stride_files=3, per_file=45)
    X, Y, clips = [], [], []
    for r in rows:
        h = np.asarray(r["decoder_h"], dtype=np.float32)
        y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        y = y if y.ndim == 1 else y.mean(1)
        assert sr == 24000
        sil = silent_frames(y, len(h))
        X.append(h)
        Y.append(sil)
        clips.append((h, sil, r["text"]))
    n_train = int(len(clips) * 0.7)  # fit on whole clips, so that the report below is on clips the probe has not seen
    probe, rep = SilenceProbe.fit(np.concatenate(X[:n_train]), np.concatenate(Y[:n_train]), holdout=0.3, precision_target=a.precision)
    print("probe fit:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep.items()}, flush=True)
    if a.save:
        probe.save(a.save)
        print("saved", a.save)
    mu, sd, w, b = torch.from_numpy(probe.mean), torch.from_numpy(probe.std), torch.from_numpy(probe.w), probe.b

    def stats(name, seqs, texts):
        P, lead, trail, nclip, dur, commas = [], [], [], 0, 0, 0
        for h, t in zip(seqs, texts):
            sil = probe.mask(h)
            p, l, e = internal_pauses(sil)
            P += p; lead.append(l); trail.append(e); nclip += 1; dur += len(h) / 50
            commas += sum(c in PUNCT for c in t.replace(" ", "")[:-1])
        P = np.array(P) if P else np.zeros(1)
        print(f"{name:18s} n={nclip:4d} {dur / nclip:5.1f}s/clip | leading {np.mean(lead) * 20:4.0f} ms trailing {np.mean(trail) * 20:4.0f} ms | internal silent runs: "
              f"{len(P) / nclip:4.2f}/clip, >=100 ms {np.sum(P >= 5) / nclip:4.2f}/clip ({np.sum(P >= 5) / dur * 60:4.1f}/min), >=300 ms {np.sum(P >= 15) / nclip:4.2f}/clip "
              f"({np.sum(P >= 15) / dur * 60:4.1f}/min) | commas {commas / nclip:4.2f}/clip | median run {np.median(P) * 20:.0f} ms p90 {np.percentile(P, 90) * 20:.0f} ms")

    stats("corpus (held out)", [c[0] for c in clips[n_train:]], [c[2] for c in clips[n_train:]])
    trows = read_rows(a.teacher, ["decoder_h", "text", "mode"], 10**9, per_file=10**9) if glob.glob(os.path.join(a.teacher, "data-*.arrow")) else []
    for mode in ("zero_shot", "none"):
        sel = [r for r in trows if r["mode"] == mode]
        stats(f"teacher {mode}", [np.asarray(r["decoder_h"], dtype=np.float32) for r in sel], [r["text"] for r in sel])
    stats("teacher all", [np.asarray(r["decoder_h"], dtype=np.float32) for r in trows], [r["text"] for r in trows])


if __name__ == "__main__":
    main()
