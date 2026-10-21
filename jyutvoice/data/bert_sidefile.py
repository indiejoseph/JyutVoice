"""Per-character BERT features kept next to a dataset instead of inside it, looked up by row id.

Rewriting a 46-138 GB dataset to add a `bert` column is not worth it, so `scripts/build_bert_sidefile.py` writes the features to a directory

    ids.json       row ids, in the order of the feature blocks
    offsets.npy    int64, block i is bert.npy[offsets[i]:offsets[i + 1]]  (n_chars + 2 rows: the two pad units included)
    bert.npy       float16 memory map (rows, bert_dim), standardised per dimension with stats.npz
    stats.npz      mean / std used for the standardisation (needed again at inference time)
    bad_ids.json   ids whose label has a different number of tokens than the text has characters (g2p would fail on them)

and `TextMelDataModule(bert_dir=...)` reads them through `BertSideFile`. Rows that are not in the file get zeros (no BERT input).
"""
import json
import os

import numpy as np


class BertSideFile:
    def __init__(self, directory: str):
        self.directory = directory
        ids = json.load(open(os.path.join(directory, "ids.json"), encoding="utf-8"))
        self.index = {row_id: i for i, row_id in enumerate(ids)}
        self.offsets = np.load(os.path.join(directory, "offsets.npy"))
        self.bert = np.load(os.path.join(directory, "bert.npy"), mmap_mode="r")
        self.dim = int(self.bert.shape[1])
        bad = os.path.join(directory, "bad_ids.json")
        self.bad_ids = set(json.load(open(bad))) if os.path.exists(bad) else set()

    def __contains__(self, row_id):
        return row_id in self.index

    def get(self, row_id):
        """(n_chars + 2, dim) float16 features of a row, or None when the row is not in the file."""
        i = self.index.get(row_id)
        if i is None:
            return None
        return np.asarray(self.bert[self.offsets[i] : self.offsets[i + 1]])
