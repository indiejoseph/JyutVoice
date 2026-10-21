"""Make a directory with only some of a Hugging Face dataset's arrow files loadable by `load_from_disk`.

    python scripts/subset_hf_dataset_state.py <dir>

A partial copy of a `save_to_disk` dataset still has the original state.json, which lists every shard. This rewrites it to the data-*.arrow files that
are present (the original is kept as state.json.orig), and reports the row count.
"""
import glob
import json
import os
import shutil
import sys


def main():
    d = sys.argv[1]
    state = json.load(open(os.path.join(d, "state.json")))
    present = sorted(os.path.basename(f) for f in glob.glob(os.path.join(d, "data-*.arrow")))
    if not os.path.exists(os.path.join(d, "state.json.orig")):
        shutil.copy(os.path.join(d, "state.json"), os.path.join(d, "state.json.orig"))
    state["_data_files"] = [{"filename": f} for f in present]
    json.dump(state, open(os.path.join(d, "state.json"), "w"), indent=1)
    from datasets import load_from_disk

    ds = load_from_disk(d)
    print(f"{len(present)} shards listed, {len(ds)} rows, columns {ds.column_names}")


if __name__ == "__main__":
    main()
