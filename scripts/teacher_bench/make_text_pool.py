#!/usr/bin/env python3
"""Build a stratified pool of Cantonese test sentences from the hk_news summaries (CPU only).

Every piece goes through text_filter.CantoneseTextFilter first: leaked reasoning tags, code-mixed English, currency, odd
symbols, unsupported punctuation and characters the teacher never saw are rejected; numbers are converted to Chinese
numerals; the rest is mapped to the punctuation the teacher was trained on. Half of each length bin is plain text, half
had numbers (so number reading can be judged separately). Writes one JSON object per line (clean `text` plus the `raw`).

    python make_text_pool.py --parquet .../news_summarized_dedup.parquet --out sentences.jsonl --n 100
    python make_text_pool.py ... --bins 8-16,16-24,24-32,32-48,48-72,72-110 --clauses --n 48   # short pieces too
    python make_text_pool.py ... --survey 20000      # just report how much text survives each filter
"""
import argparse
import json
import os
import random
import re
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text_filter import DIGIT, CantoneseTextFilter, has_markup  # noqa: E402

SENT_SPLIT = re.compile(r"(?<=[。！？!?；;])")
CLAUSE_SPLIT = re.compile(r"(?<=[。！？!?；;，,、])")


def raw_pieces(summary, splitter):
    for para in re.split(r"[\r\n]+", summary):
        for x in splitter.split(para):
            x = x.strip()
            if x:
                yield x


def candidates(summaries, splitter, flt, pairs=True):
    """Yield (clean_text, raw, n_pieces) for single pieces and 2-piece chunks that pass the filter."""
    for s in summaries:
        if not s:
            continue
        if has_markup(s):  # a leaked tag may wrap reasoning text, so the whole summary goes
            flt.stats["summary_markup"] += 1
            continue
        pcs = list(raw_pieces(s, splitter))
        for i, a in enumerate(pcs):
            text, _ = flt(a)
            if text:
                yield text, a, 1
            if pairs and i + 1 < len(pcs):
                text, _ = flt(a + pcs[i + 1])
                if text:
                    yield text, a + pcs[i + 1], 2


def print_stats(flt, label):
    total = sum(v for k, v in flt.stats.items() if k not in ("kept_with_numbers", "summary_markup"))
    print(f"\n{label}: {total:,} candidate pieces")
    for k, v in flt.stats.most_common():
        if k in ("kept_with_numbers", "summary_markup"):
            continue
        print(f"  {k:12s} {v:8,d}  {100 * v / max(1, total):5.1f}%")
    print(f"  (of the kept, {flt.stats['kept_with_numbers']:,} had numbers converted; "
          f"{flt.stats['summary_markup']:,} whole summaries dropped for leaked markup)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=100, help="total sentences, split evenly over length bins")
    ap.add_argument("--bins", default="20-40,40-60,60-90,90-140", help="comma list of lo-hi character ranges (hi exclusive)")
    ap.add_argument("--clauses", action="store_true", help="also split on commas, to get short pieces")
    ap.add_argument("--digits", default="normalize", choices=["normalize", "drop"])
    ap.add_argument("--char_counts", default=None, help="json of character counts in the teacher's training texts")
    ap.add_argument("--min_char_count", type=int, default=20)
    ap.add_argument("--survey", type=int, default=0, help="only report filter yield over this many summaries")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    BINS = [tuple(int(x) for x in b.split("-")) for b in args.bins.split(",")]
    splitter = CLAUSE_SPLIT if args.clauses else SENT_SPLIT
    flt = CantoneseTextFilter(digits=args.digits, char_counts=args.char_counts, min_char_count=args.min_char_count)
    summaries = pq.read_table(args.parquet, columns=["summary"]).column("summary").to_pylist()
    rng = random.Random(args.seed)
    rng.shuffle(summaries)

    if args.survey:
        for _ in candidates(summaries[: args.survey], splitter, flt, pairs=False):
            pass
        print_stats(flt, f"yield over {args.survey:,} summaries (single pieces)")
        return

    per_bin = args.n // len(BINS)
    need = max(per_bin * 20, 40)
    buckets = {(b, k): [] for b in BINS for k in ("plain", "numeric")}
    seen = set()
    lo, hi = min(b[0] for b in BINS), max(b[1] for b in BINS)
    for text, raw, n_piece in candidates(summaries, splitter, flt):
        n = len(text)
        b = next((b for b in BINS if b[0] <= n < b[1]), None)
        if text in seen or b is None or not (lo <= n < hi):
            continue
        kind = "numeric" if DIGIT.search(raw) else "plain"
        if len(buckets[(b, kind)]) < need:
            buckets[(b, kind)].append((text, raw, n_piece))
            seen.add(text)
        if all(len(v) >= need for v in buckets.values()):
            break

    rows = []
    for (b, kind), pool in buckets.items():
        for text, raw, n_piece in rng.sample(pool, min(len(pool), per_bin // 2)):
            rows.append({"text": text, "raw": raw, "n_chars": len(text), "bin": f"{b[0]}-{b[1]}", "kind": kind,
                         "n_sentences": n_piece})
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"s{i:04d}"
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} sentences -> {args.out}")
    for (b, kind), pool in buckets.items():
        used = sum(1 for r in rows if r["bin"] == f"{b[0]}-{b[1]}" and r["kind"] == kind)
        print(f"  {b[0]:>3}-{b[1]:<3} {kind:7s}: pool={len(pool):5d} used={used}")
    print_stats(flt, "filter outcome while filling the pool")


if __name__ == "__main__":
    main()
