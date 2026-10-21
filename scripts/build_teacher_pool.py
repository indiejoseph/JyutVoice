#!/usr/bin/env python3
"""Build a pool of teacher-safe Cantonese sentences from the speech-dataset3 transcripts (the turns that went into the SD3 training shards).

    python scripts/build_teacher_pool.py --turns ~/sd3_build/out/plan/turns.jsonl --char-counts train_char_counts.json --out pool.jsonl --n 30000

The turns are real conversational / talk-show / news speech transcribed by R8 (word-segmented; the spaces are removed again here), so they carry the
phrasing and punctuation of spoken Cantonese, unlike the news summaries the first teacher benchmark used. Every piece goes through
text_filter.CantoneseTextFilter (the teacher was trained on `，。？！` only, no digits, no Latin, Traditional characters it has seen >= 20 times). The pool is
stratified over length bins and channels and de-duplicated; `id` is stable (t00000...) and `raw` keeps the unfiltered text.
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "teacher_bench"))
from text_filter import CantoneseTextFilter  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--turns", required=True)
    ap.add_argument("--char-counts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--bins", default="6-12,12-20,20-30,30-45,45-70,70-110", help="lo-hi character ranges (hi exclusive) and the relative weight after ':'")
    ap.add_argument("--weights", default="1,2,3,3,2,1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    bins = [tuple(int(x) for x in b.split("-")) for b in a.bins.split(",")]
    weights = [float(w) for w in a.weights.split(",")]
    flt = CantoneseTextFilter(digits="drop", char_counts=a.char_counts)
    rng = random.Random(a.seed)
    rows = [json.loads(line) for line in open(a.turns, encoding="utf-8")]
    rng.shuffle(rows)
    buckets = collections.defaultdict(list)
    seen = set()
    for r in rows:
        raw = r["text"].replace(" ", "")
        text, _ = flt(raw)
        if not text or text in seen:
            continue
        b = next((b for b in bins if b[0] <= len(text) < b[1]), None)
        if b is None:
            continue
        seen.add(text)
        buckets[b].append({"text": text, "raw": raw, "n_chars": len(text), "bin": f"{b[0]}-{b[1]}", "channel": r.get("channel"), "src_id": r["id"]})
    total_w = sum(weights)
    out = []
    for b, w in zip(bins, weights):
        take = int(a.n * w / total_w)
        pool = buckets[b]
        out += pool[:take]  # already shuffled
    rng.shuffle(out)
    for i, r in enumerate(out):
        r["id"] = f"t{i:05d}"
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} turns -> {sum(len(v) for v in buckets.values())} pass the filter -> pool of {len(out)} -> {a.out}")
    print("filter outcomes:", dict(flt.stats.most_common()))
    for b in bins:
        print(f"  {b[0]:>3}-{b[1]:<3} available {len(buckets[b]):6d} used {sum(1 for r in out if r['bin'] == f'{b[0]}-{b[1]}')}")
    ch = collections.Counter(r["channel"] for r in out)
    punct = sum(r["text"].count("，") for r in out) / max(1, len(out))
    print("channels:", dict(ch), f"| commas per sentence {punct:.2f}")


if __name__ == "__main__":
    main()
