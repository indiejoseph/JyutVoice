"""Make Hugging Face dataset shards written by `datasets` >= 4 readable by `datasets` 3.x (winrex has 4.8.5, the local box 3.6.0).

    python scripts/downgrade_hf_features.py '<dir>/shards/shard_*' [--out-dir OUT]

`datasets` 4 saves `Sequence` features as `List`, which 3.x does not know ("Feature type 'List' not found"). The type name is in `dataset_info.json` and in the
schema metadata of the Arrow file; the data is the same, so each shard's metadata is rewritten (`List` -> `Sequence`). Without --out-dir the shards are
rewritten in place (to a temporary file, renamed after it is complete); shards that need no change are left alone.
"""
import argparse
import glob
import json
import os
import shutil

import pyarrow as pa


def downgrade(node):
    """Feature dict with every `"_type": "List"` renamed to `"Sequence"` (same fields: feature, length)."""
    if isinstance(node, dict):
        return {k: ("Sequence" if k == "_type" and v == "List" else downgrade(v)) for k, v in node.items()}
    if isinstance(node, list):
        return [downgrade(v) for v in node]
    return node


def convert_shard(src, dst):
    """Rewrite shard directory `src` into `dst` (may be the same directory). Returns False when the shard did not need it."""
    info_path = os.path.join(src, "dataset_info.json")
    if '"_type": "List"' not in open(info_path, encoding="utf-8").read():
        if dst != src:
            shutil.copytree(src, dst, dirs_exist_ok=True)
        return False
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        path = os.path.join(src, name)
        if name.endswith(".arrow"):
            with pa.memory_map(path) as f:
                table = pa.ipc.open_stream(f).read_all()
            meta = dict(table.schema.metadata or {})
            if b"huggingface" in meta:
                meta[b"huggingface"] = json.dumps(downgrade(json.loads(meta[b"huggingface"]))).encode()
            table = table.replace_schema_metadata(meta)
            tmp = os.path.join(dst, name + ".tmp")
            with pa.OSFile(tmp, "wb") as sink, pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            del table
            os.replace(tmp, os.path.join(dst, name))
        elif name == "dataset_info.json":
            info = downgrade(json.load(open(path, encoding="utf-8")))
            with open(os.path.join(dst, name) + ".tmp", "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
            os.replace(os.path.join(dst, name) + ".tmp", os.path.join(dst, name))
        elif dst != src:
            shutil.copy2(path, os.path.join(dst, name))
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", help="glob of shard directories (each written by Dataset.save_to_disk)")
    ap.add_argument("--out-dir", help="write the converted shards here (same directory names) instead of in place")
    a = ap.parse_args()
    paths = [p for p in sorted(glob.glob(os.path.expanduser(a.shards))) if os.path.isdir(p) and not p.endswith(".tmp")]
    changed = 0
    for p in paths:
        dst = os.path.join(a.out_dir, os.path.basename(p)) if a.out_dir else p
        changed += convert_shard(p, dst)
    print(f"{len(paths)} shards, {changed} converted")


if __name__ == "__main__":
    main()
