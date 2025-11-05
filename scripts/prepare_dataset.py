"""
Dataset preparation script for JyutVoice TTS.

Performs word segmentation and text validation on your dataset.

Usage:
    # From JSON file
    python prepare_dataset.py --dataset data.json --output processed_data/

    # From CSV file
    python prepare_dataset.py --dataset data.csv --output processed_data/

    # From HuggingFace dataset directory
    python prepare_dataset.py --dataset ./dataset_dir --output processed_data/

    # From HuggingFace Hub
    python prepare_dataset.py --dataset mozilla-foundation/common_voice_17_0 --output processed_data/

Expected dataset format (JSON/CSV):
    {
        "text": "你好世界",
        "lang": "zh",      # Language code: "zh" (Chinese), "en" (English), etc.
        "audio": "path/to/audio.wav",
        "speaker_id": 1
    }

Output:
    - Segmented text with word boundaries
    - Filtered samples (valid Chinese/English only)
    - Saved to HuggingFace dataset format
"""

import os
from pydips import BertModel
from datasets import load_dataset, load_from_disk
import argparse

ws_model = BertModel()


def is_chinese(char: str) -> bool:
    # Check CJK Unified Ideographs (Most Common)
    if "\u4e00" <= char <= "\u9fff":
        return True
    # Check CJK Extension A (Where '䨇' is found)
    if "\u3400" <= char <= "\u4dbf":
        return True
    # More extensions (B, C, D, E, F) exist but are far less common
    return False


def word_seg(row):
    """
    Perform word segmentation and validate text.

    Returns:
        dict: With keys 'text', 'valid'
              'text': Segmented text with spaces between words
              'valid': Boolean indicating if text passed validation
    """
    lang = row.get("lang")
    text = row.get("text", "")

    # Validate inputs exist
    if not lang or not text:
        return {
            "text": text,
            "valid": False,
        }

    try:
        if lang != "en":
            # Chinese text: apply word segmentation
            seg = ws_model.cut(text, mode="coarse")
            text = " ".join(seg)

            # Only accept Chinese characters, spaces and basic punctuation
            valid = all(
                is_chinese(c) or c.isspace() or c in "。，、！？；：、…—·「」『』（）"
                for c in text
            )

            return {
                "text": text,
                "valid": valid,
            }
        else:
            # English text: just validate
            valid = all(c.isalpha() or c.isspace() or c in ".,!?;:'-" for c in text)
            return {
                "text": text,
                "valid": valid,
            }
    except (ValueError, KeyError, TypeError, Exception) as e:
        # If any error occurs during processing, mark as invalid
        return {
            "text": text,
            "valid": False,
        }

    return {
        "text": text,
        "valid": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Word segmentation for dataset preparation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the input dataset file or HuggingFace dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output dataset file",
    )

    args = parser.parse_args()

    # Load dataset from various sources
    if os.path.isdir(args.dataset):
        try:
            # Try to load as HuggingFace dataset directory
            dataset = load_from_disk(args.dataset)
            print(f"✅ Loaded dataset from directory: {args.dataset}")
        except Exception as e:
            print(f"❌ Error loading dataset from directory: {e}")
            print(f"   Directory must contain a valid HuggingFace dataset structure")
            print(f"   Or use a JSON/CSV file path instead")
            exit(1)
    elif args.dataset.endswith(".json"):
        try:
            from datasets import load_dataset

            dataset = load_dataset("json", data_files=args.dataset, split="train")
            print(f"✅ Loaded dataset from JSON: {args.dataset}")
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            exit(1)
    elif args.dataset.endswith(".csv"):
        try:
            from datasets import load_dataset

            dataset = load_dataset("csv", data_files=args.dataset, split="train")
            print(f"✅ Loaded dataset from CSV: {args.dataset}")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            exit(1)
    else:
        # Try to load from HuggingFace Hub
        try:
            dataset = load_dataset(args.dataset, split="train")
            print(f"✅ Loaded dataset from HuggingFace Hub: {args.dataset}")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print(f"   Supported formats:")
            print(f"   - HuggingFace dataset directory (with dataset_dict.json)")
            print(f"   - JSON file (--dataset data.json)")
            print(f"   - CSV file (--dataset data.csv)")
            print(f"   - HuggingFace Hub dataset (--dataset dataset_name)")
            exit(1)

    print(f"Processing dataset with {len(dataset)} samples...")
    initial_count = len(dataset)

    # Apply word segmentation
    dataset = dataset.map(word_seg, num_proc=4)
    print(f"✅ Word segmentation complete")

    # Filter valid samples
    print(f"🔍 Filtering invalid samples...")
    dataset = dataset.filter(lambda x: x["valid"])
    filtered_count = len(dataset)
    removed_count = initial_count - filtered_count

    print(f"✅ Filtering complete:")
    print(f"   Original samples: {initial_count}")
    print(f"   Valid samples: {filtered_count}")
    print(f"   Removed invalid: {removed_count}")
    print(f"   Retention rate: {100 * filtered_count / initial_count:.1f}%")

    # Remove the valid column before saving
    dataset = dataset.remove_columns(["valid"])
    print(f"✅ Removed 'valid' column")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # export the dataset
    dataset.save_to_disk(args.output)
    print(f"✅ Dataset saved to: {args.output}")
