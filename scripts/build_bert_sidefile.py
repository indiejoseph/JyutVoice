"""Write ModernCantoneseBert-large features for every row of a Hugging Face dataset as a side file (see jyutvoice/data/bert_sidefile.py).

    python scripts/build_bert_sidefile.py --dataset ~/jyutvoice_data/dataset_yue --out ~/jyutvoice_data/bert_yue \\
        --model ~/sd3_build/models/ModernCantoneseBert-large-ext --device cuda --threads 8

For every row: the cleaned text (the characters g2p sees), one BERT feature per character plus the two pad units, standardised per dimension with
statistics measured on `--calibrate-n` random rows. When the row has a `phone` label, its token count must equal the number of characters
(`g2p(jyutping=...)` requires it); rows where it does not are written to bad_ids.json so that the training can leave them out.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

BERT_DIM = 1024


def read_columns(directory):
    """id / lang / text / phone of every row, read straight from the arrow files (columns only: `ds["text"]` through `datasets` took over ten
    minutes on 162 shards of 270 MB with audio and decoder_h in them)."""
    import glob

    import pyarrow as pa
    import pyarrow.ipc as ipc

    cols = {"id": [], "lang": [], "text": [], "phone": []}
    for fn in sorted(glob.glob(os.path.join(directory, "data-*.arrow"))):
        for b in ipc.open_stream(pa.memory_map(fn)):
            for name in cols:
                cols[name] += b.column(name).to_pylist() if name in b.schema.names else [None] * b.num_rows
    return cols["id"], cols["lang"], cols["text"], cols["phone"]


def main():
    import torch

    from jyutvoice.text.bert_align import BertPhoneFeatures, unit_chars
    from jyutvoice.text.cantonese.readings import cleaned

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", default="yue", help="only rows of this language (the Cantonese BERT covers yue / zh text, not English)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--calibrate-n", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="testing: first N rows only")
    a = ap.parse_args()

    t0 = time.time()
    ids, langs, texts, phones = read_columns(a.dataset)
    print(f"read {len(ids)} rows of id/lang/text/phone in {time.time() - t0:.0f} s", flush=True)
    keep = [i for i in range(len(ids)) if langs[i] == a.lang]
    if a.limit:
        keep = keep[: a.limit]
    ids = [ids[i] for i in keep]
    t0 = time.time()
    norms = [cleaned(texts[i]) for i in keep]
    print(f"cleaned {len(norms)} texts in {time.time() - t0:.0f} s", flush=True)
    n_units = np.array([len(unit_chars(n)) for n in norms])
    bad = [ids[j] for j, i in enumerate(keep) if phones[i] is not None and len(phones[i].split()) != n_units[j]]
    print(f"{len(ids)} rows of lang {a.lang}; {len(bad)} with a label whose token count differs from the character count", flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(a.threads)
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    ext = BertPhoneFeatures.from_pretrained(a.model, device=device, dtype={"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype])
    rng = np.random.default_rng(0)
    sample = [norms[i] for i in rng.choice(len(norms), min(a.calibrate_n, len(norms)), replace=False)]
    t0 = time.time()
    ext.calibrate(sample, batch_size=a.batch_size)
    np.savez(os.path.join(a.out, "stats.npz"), mean=ext.mean.numpy(), std=ext.std.numpy())
    print(f"calibrated on {len(sample)} texts in {time.time() - t0:.0f} s on {device}", flush=True)

    rows = n_units + 2
    offsets = np.concatenate([[0], np.cumsum(rows)])
    out = np.lib.format.open_memmap(os.path.join(a.out, "bert.npy"), mode="w+", dtype=np.float16, shape=(int(offsets[-1]), BERT_DIM))
    order = np.argsort([len(n) for n in norms])
    t0 = time.time()
    for k in range(0, len(order), a.batch_size):
        idx = order[k : k + a.batch_size]
        for i, f in zip(idx, ext.unit_features([norms[i] for i in idx])):
            assert f.shape[0] == rows[i], (ids[i], f.shape, rows[i])
            out[offsets[i] : offsets[i + 1]] = f.numpy().astype(np.float16)
        if (k // a.batch_size) % 100 == 0:
            print(f"bert {k + len(idx)}/{len(order)} ({(k + len(idx)) / (time.time() - t0):.0f} texts/s)", flush=True)
    out.flush()
    np.save(os.path.join(a.out, "offsets.npy"), offsets)
    json.dump(ids, open(os.path.join(a.out, "ids.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(bad, open(os.path.join(a.out, "bad_ids.json"), "w", encoding="utf-8"), ensure_ascii=False)
    finite = bool(np.isfinite(out[:: max(1, len(out) // 100000)].astype(np.float32)).all())
    json.dump({"model": a.model, "layer": ext.layer, "dtype": a.dtype, "dataset": a.dataset, "lang": a.lang, "rows": int(offsets[-1]), "texts": len(ids),
               "bad": len(bad), "finite_sample": finite, "calibrate_n": len(sample)}, open(os.path.join(a.out, "meta.json"), "w"))
    print(f"done: {len(ids)} rows, {int(offsets[-1])} feature rows, {time.time() - t0:.0f} s, finite={finite}, bad={len(bad)}")


if __name__ == "__main__":
    main()
