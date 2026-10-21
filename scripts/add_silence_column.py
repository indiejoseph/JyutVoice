"""Add the per-frame `silence` column (uint8, the SilenceProbe of the row's `decoder_h`) to dataset shards that lack it.

    python scripts/add_silence_column.py '<dir>/shards/shard_*' --probe ~/teacher/silence_probe.npz --out-dir <dir>/shards_sil

Why: the pause-aware alignment (`pause_bonus`) needs `silence`, and `TextMelDataModule` keeps only the columns common to all the datasets of a mix, so one set without
it (the MCE clips next to the teacher rows) would switch the pause-aware alignment off for every row. The probe is the one `teacher_rows.py` used for the teacher rows
(fitted on real corpus frames, so it applies to recordings as it is). Each shard goes to `<out-dir>/<shard name>`; the input is not modified.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch  # noqa: F401  (imported before datasets/pyarrow: the libstdc++ CXXABI clash with scipy)
from datasets import Features, Sequence, Value, load_from_disk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jyutvoice.utils.silence_probe import SilenceProbe  # noqa: E402


def add_silence(ds, probe):
    def fn(row):
        return {"silence": probe.mask(np.asarray(row["decoder_h"], dtype=np.float32)).astype(np.uint8)}

    feats = ds.features.copy()
    feats["silence"] = Sequence(Value("uint8"))
    return ds.map(fn, features=Features(feats))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", help="glob of shard directories (quote it)")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    probe = SilenceProbe.load(os.path.expanduser(a.probe))
    paths = sorted(glob.glob(os.path.expanduser(a.shards)))
    os.makedirs(a.out_dir, exist_ok=True)
    n_frames = n_silent = 0
    for p in paths:
        ds = load_from_disk(p)
        if "silence" in ds.column_names:
            raise SystemExit(f"{p} already has a silence column")
        out = add_silence(ds, probe)
        out.save_to_disk(os.path.join(a.out_dir, os.path.basename(p)))
        for s in out["silence"]:
            n_frames += len(s)
            n_silent += int(np.sum(s))
        print(os.path.basename(p), len(out), "rows", flush=True)
    print(f"{len(paths)} shards, {n_silent / max(1, n_frames):.1%} of {n_frames} frames silent -> {a.out_dir}")


if __name__ == "__main__":
    main()
