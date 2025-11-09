#!/usr/bin/env python3
"""
Merge distributed dataset shards back into a single dataset.

Usage:
    python merge_dataset_shards.py --base_output tmp/output --worlds 32 --final_output tmp/final_dataset
"""

import argparse
import os
from datasets import load_from_disk, concatenate_datasets


def merge_dataset_shards(base_output: str, worlds: int, final_output: str):
    """
    Merge all dataset shards from distributed processing into a single dataset.

    Args:
        base_output: Base output path used in the distributed processing (without shard suffix)
        worlds: Number of worlds (shards) used in distributed processing
        final_output: Path to save the merged dataset
    """
    print(f"🔀 Merging {worlds} dataset shards...")

    # Collect all shard datasets
    shard_datasets = []
    for rank in range(worlds):
        shard_path = f"{base_output}_{rank + 1}_{worlds}"
        if os.path.exists(shard_path):
            print(f"📁 Loading shard {rank + 1}/{worlds}: {shard_path}")
            shard_ds = load_from_disk(shard_path)
            shard_datasets.append(shard_ds)
            print(f"   ✅ Loaded {len(shard_ds)} samples")
        else:
            print(f"❌ Missing shard: {shard_path}")
            return False

    # Concatenate all shards
    print("🔗 Concatenating datasets...")
    merged_dataset = concatenate_datasets(shard_datasets)

    # Save the merged dataset
    print(f"💾 Saving merged dataset to {final_output}...")
    os.makedirs(final_output, exist_ok=True)
    merged_dataset.save_to_disk(final_output)

    print("✅ Dataset merging complete!")
    print(f"📊 Final dataset info:")
    print(f"   Total samples: {len(merged_dataset)}")
    print(f"   Columns: {merged_dataset.column_names}")

    # Show sample info
    if len(merged_dataset) > 0:
        sample = merged_dataset[0]
        print(f"   Sample keys: {list(sample.keys())}")
        if sample.get("spk_emb") is not None:
            print(f"   Speaker embedding shape: {len(sample['spk_emb'])}")
        if sample.get("decoder_h") is not None:
            import numpy as np

            print(
                f"   Decoder hidden state shape: {np.array(sample['decoder_h']).shape}"
            )

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge distributed dataset shards into a single dataset"
    )
    parser.add_argument(
        "--base_output",
        type=str,
        required=True,
        help="Base output path used in distributed processing (e.g., tmp/output)",
    )
    parser.add_argument(
        "--worlds",
        type=int,
        required=True,
        help="Number of worlds (shards) used in distributed processing",
    )
    parser.add_argument(
        "--final_output",
        type=str,
        required=True,
        help="Path to save the merged dataset",
    )

    args = parser.parse_args()

    success = merge_dataset_shards(args.base_output, args.worlds, args.final_output)
    if not success:
        print("❌ Failed to merge datasets")
        exit(1)
