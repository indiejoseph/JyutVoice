import os
from pydips import BertModel
from datasets import load_dataset, load_from_disk
import argparse

ws_model = BertModel()


def word_seg(row):
    lang = row["lang"]
    text = row["text"]

    if lang != "en":
        seg = ws_model.cut(text, mode="coarse")
        text = " ".join(seg)

        return {"text": text}

    return {"text": text}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Word segmentation for dataset preparation"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the input dataset file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output dataset file",
    )

    args = parser.parse_args()

    if os.path.isdir(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path, split="train")

    dataset = dataset.map(word_seg, num_proc=4)

    # export the dataset
    dataset.save_to_disk(args.output_path)
