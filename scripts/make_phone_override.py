"""Label overrides for dataset_processed's Cantonese rows from the jyutping project's corrected label file.

    python scripts/make_phone_override.py --dataset-dir ~/jyutvoice_data/dataset_yue \\
        --corrected ~/projects/wav2vec2bert-jyutping/tmp/label_mining/corrected_datasets/tts_train_corrected_v2.jsonl --out phone_override.json

How the two datasets line up (verified on 39,176 clips: the text equals in all of them, the original labels equal in all 373,343 syllables):
  * dataset_processed's `yue_<N>` is row N of `indiejoseph/tts20250516` as stored, and the jyutping project's train split is
    `shuffle(seed=42)` of that dataset without its first 1,000 rows (500 test + 500 validation): `idx = position[N] - 1000`, where
    `position[perm[p]] = p` for `perm = np.random.default_rng(42).permutation(261524)`. Rows with idx < 0 are the held-out clips and are left as they are.
  * dataset_processed's `phone` is the corpus label (syllables) plus the punctuation of the text as tokens ("." for 。), the corrected file has the
    syllables only. The override puts the corrected syllables back into that layout, so that `g2p(jyutping=...)` accepts it: one token per character
    of the cleaned text, every punctuation mark written as "." like the dataset does.

Output: {row id: phone string} for the clips whose label changed or whose punctuation tokens are misplaced; every other clip keeps its own `phone`. `TextMelDataModule(phone_override=...)` applies it.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import correct_readings as cr  # noqa: E402  (its clean() applies the cleaners' punctuation rules)

N_STORED = 261524  # rows of indiejoseph/tts20250516
HELD_OUT = 1000


def with_punctuation(text, syllables):
    """The syllables laid out over the characters of the cleaned text: a token per character. dataset_processed marks EVERY punctuation character
    with "." (a comma or a question mark too), so that is what the override does."""
    out, it = [], iter(syllables)
    for c in cr.clean(text):
        out.append("." if c in cr.PUNCTUATIONS else next(it))
    rest = list(it)
    if rest:
        raise ValueError(f"{len(rest)} syllables left over for {text!r}")
    return " ".join(out)


def main():
    import pyarrow as pa
    import pyarrow.ipc as ipc

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, help="directory with the data-*.arrow files (a partial copy is fine)")
    ap.add_argument("--corrected", required=True, help="tts_train_corrected_v2.jsonl (export_corrected_labels.py of the jyutping project)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus", default="tts")
    a = ap.parse_args()

    labels = {}
    for line in open(a.corrected, encoding="utf-8"):
        r = json.loads(line)
        if r["corpus"] == a.corpus:
            labels[r["idx"]] = r
    perm = np.random.default_rng(42).permutation(N_STORED)
    position = np.empty(N_STORED, dtype=np.int64)
    position[perm] = np.arange(N_STORED)

    overrides, stats = {}, {"rows": 0, "held_out": 0, "text_differs": 0, "layout_repaired": 0, "changed_clips": 0, "changed_syllables": 0}
    for fn in sorted(glob.glob(os.path.join(a.dataset_dir, "data-*.arrow"))):
        try:
            batches = list(ipc.open_stream(pa.memory_map(fn)))  # a shard still being copied raises: skipped, it is picked up on a later run
        except (OSError, pa.ArrowInvalid):
            print(f"skipping unreadable {os.path.basename(fn)}", file=sys.stderr)
            continue
        for b in batches:
            for r in b.select(["id", "lang", "text", "phone"]).to_pylist():
                if r["lang"] != "yue" or not r["id"].startswith("yue_"):
                    continue
                stats["rows"] += 1
                idx = int(position[int(r["id"].split("_")[1])]) - HELD_OUT
                if idx < 0:
                    stats["held_out"] += 1
                    continue
                lab = labels[idx]
                if lab["text"].replace(" ", "") != r["text"].replace(" ", ""):
                    stats["text_differs"] += 1
                    continue
                # the layout rebuilt from the ORIGINAL label reproduces the dataset's own phone string for all but a handful of clips; in those the
                # dataset's "." sits at a different place than the punctuation of the text (the syllables after it are shifted against their
                # characters), so they are overridden with the canonical layout too
                canonical = with_punctuation(r["text"], lab["phone_corrected"].split())
                misplaced = with_punctuation(r["text"], lab["phone_original"].split()) != " ".join((r["phone"] or "").split())
                if misplaced:
                    stats["layout_repaired"] += 1
                if lab["changed"] or misplaced:
                    overrides[r["id"]] = canonical
                if lab["changed"]:
                    stats["changed_clips"] += 1
                    stats["changed_syllables"] += len(lab["changed"])
    json.dump(overrides, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps(stats))
    print(f"{len(overrides)} clips get a corrected label -> {a.out}")


if __name__ == "__main__":
    main()
