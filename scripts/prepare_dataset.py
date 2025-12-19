"""
Dataset preparation script for JyutVoice TTS.

Performs word segmentation, text validation, speaker embedding extraction,
and decoder hidden state extraction on your dataset.

Usage:
    # From JSON file
    python prepare_dataset.py --dataset data.json --output processed_data/ \
        --speaker_model pretrained_models/campplus.onnx \
        --flow_encoder pretrained_models/flow_encoder.pt \
        --speech_tokenizer pretrained_models/speech_tokenizer_v2.onnx

    # From CSV file
    python prepare_dataset.py --dataset data.csv --output processed_data/ \
        --speaker_model pretrained_models/campplus.onnx \
        --flow_encoder pretrained_models/flow_encoder.pt \
        --speech_tokenizer pretrained_models/speech_tokenizer_v2.onnx

    # From HuggingFace dataset directory
    python prepare_dataset.py --dataset ./dataset_dir --output processed_data/ \
        --speaker_model pretrained_models/campplus.onnx \
        --flow_encoder pretrained_models/flow_encoder.pt \
        --speech_tokenizer pretrained_models/speech_tokenizer_v2.onnx

Expected dataset format (JSON/CSV):
    {
        "text": "你好世界",
        "lang": "zh",      # Language code: "zh" (Chinese), "en" (English), etc.
        "audio": "path/to/audio.wav" or {"array": [...], "sampling_rate": 16000}
    }

Output:
    - Segmented text with word boundaries
    - Filtered samples (valid Chinese/English only)
    - Extracted speaker embeddings
    - Extracted decoder hidden states
    - Saved to HuggingFace dataset format
"""

import os
import numpy as np
import librosa
from datasets import load_dataset, load_from_disk
import argparse


def load_audio_from_path(audio_path):
    """Load audio from file path and convert to required formats."""
    try:
        # Load audio using librosa
        audio, sr = librosa.load(audio_path, sr=None)

        # Convert to required sample rates
        audio16k = (
            librosa.resample(audio, orig_sr=sr, target_sr=16000)
            if sr != 16000
            else audio
        )

        return audio16k
    except Exception as e:
        print(f"Error loading audio from {audio_path}: {e}")
        return None


def process_audio_features(row, spk_model, flow_encoder, speech_tokenizer):
    """
    Process audio to extract speaker embedding and decoder hidden state.

    Args:
        row: Dataset row containing audio information
        spk_model: Speaker embedding ONNX session
        flow_encoder: Flow encoder model for hidden state extraction
        speech_tokenizer: Speech tokenizer ONNX session

    Returns:
        dict: Updated row with spk_emb and decoder_h fields
    """
    try:
        # Get audio data
        audio_data = row.get("audio")
        if audio_data is None:
            return {**row, "spk_emb": None, "decoder_h": None, "audio_processed": False}

        # Handle different audio formats
        if isinstance(audio_data, dict):
            # HuggingFace datasets format
            if "array" in audio_data:
                audio = np.array(audio_data["array"], dtype=np.float32)
                sr = audio_data.get("sampling_rate", 16000)
            elif "path" in audio_data:
                audio = load_audio_from_path(audio_data["path"])
                sr = 16000  # After loading with librosa
            else:
                return {
                    **row,
                    "spk_emb": None,
                    "decoder_h": None,
                    "audio_processed": False,
                }
        elif isinstance(audio_data, str):
            # File path
            audio = load_audio_from_path(audio_data)
            sr = 16000  # After loading with librosa
        else:
            return {**row, "spk_emb": None, "decoder_h": None, "audio_processed": False}

        if audio is None:
            return {**row, "spk_emb": None, "decoder_h": None, "audio_processed": False}

        # Ensure audio is 16kHz for processing
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # Extract speaker embedding
        spk_emb = None
        if spk_model is not None:
            try:
                spk_emb = get_spk_embedding(audio, spk_model)
            except Exception as e:
                print(f"Error extracting speaker embedding: {e}")

        # Extract decoder hidden state
        decoder_h = None
        if flow_encoder is not None and speech_tokenizer is not None:
            try:
                # Extract speech tokens
                speech_token, speech_token_len = extract_speech_token(
                    audio, speech_tokenizer
                )
            except Exception as e:
                print(f"Error extracting decoder hidden state: {e}")

        return {
            **row,
            "spk_emb": spk_emb,
            "audio_processed": True,
        }

    except Exception as e:
        print(f"Error processing audio features: {e}")
        return {**row, "spk_emb": None, "decoder_h": None, "audio_processed": False}
    # Check CJK Unified Ideographs (Most Common)
    if "\u4e00" <= char <= "\u9fff":
        return True
    # Check CJK Extension A (Where '䨇' is found)
    if "\u3400" <= char <= "\u4dbf":
        return True
        # More extensions (B, C, D, E, F) exist but are far less common
        return {**row, "spk_emb": None, "decoder_h": None, "audio_processed": False}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Word segmentation and feature extraction for dataset preparation"
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
    parser.add_argument(
        "--speaker_model",
        type=str,
        default=None,
        help="Path to speaker embedding ONNX model (e.g., pretrained_models/campplus.onnx)",
    )
    parser.add_argument(
        "--flow_encoder",
        type=str,
        default=None,
        help="Path to flow encoder weights (e.g., pretrained_models/flow_encoder.pt)",
    )
    parser.add_argument(
        "--speech_tokenizer",
        type=str,
        default=None,
        help="Path to speech tokenizer ONNX model (e.g., pretrained_models/speech_tokenizer_v2.onnx)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for inference (cpu or cuda)",
    )

    args = parser.parse_args()

    # Load models if paths provided
    spk_model = None
    flow_encoder = None
    speech_tokenizer = None

    if args.speaker_model:
        print(f"📦 Loading speaker embedding model from {args.speaker_model}...")
        spk_model = load_spk_embedding_model(args.speaker_model)
        print("✅ Speaker embedding model loaded")

    if args.flow_encoder:
        print(f"📦 Loading flow encoder from {args.flow_encoder}...")
        flow_encoder = load_flow_encoder(args.flow_encoder, device=args.device)
        print("✅ Flow encoder loaded")

    if args.speech_tokenizer:
        print(f"📦 Loading speech tokenizer from {args.speech_tokenizer}...")
        speech_tokenizer = load_speech_tokenizer(args.speech_tokenizer)
        print("✅ Speech tokenizer loaded")

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
    print("🔤 Applying word segmentation...")
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

    # Remove the valid column before audio processing
    dataset = dataset.remove_columns(["valid"])

    # Extract audio features if models are provided
    if spk_model is not None or (
        flow_encoder is not None and speech_tokenizer is not None
    ):
        print("🎵 Extracting audio features...")

        def extract_features(row):
            return process_audio_features(
                row, spk_model, flow_encoder, speech_tokenizer
            )

        dataset = dataset.map(
            extract_features, num_proc=1
        )  # Use single process for model inference

        # Report processing results
        processed_count = sum(1 for x in dataset if x.get("audio_processed", False))
        print(f"✅ Audio feature extraction complete:")
        print(f"   Successfully processed: {processed_count}/{len(dataset)} samples")

        if spk_model is not None:
            spk_emb_count = sum(1 for x in dataset if x.get("spk_emb") is not None)
            print(f"   Speaker embeddings extracted: {spk_emb_count}/{len(dataset)}")

        if flow_encoder is not None and speech_tokenizer is not None:
            decoder_h_count = sum(1 for x in dataset if x.get("decoder_h") is not None)
            print(
                f"   Decoder hidden states extracted: {decoder_h_count}/{len(dataset)}"
            )

        # Remove the processing status column
        dataset = dataset.remove_columns(["audio_processed"])

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # export the dataset
    dataset.save_to_disk(args.output)
    print(f"✅ Dataset saved to: {args.output}")

    # Print final dataset info
    sample = dataset[0]
    print(f"\n📊 Final dataset info:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Columns: {dataset.column_names}")
    print(f"   Sample keys: {list(sample.keys())}")
    if sample.get("spk_emb") is not None:
        print(f"   Speaker embedding shape: {len(sample['spk_emb'])}")
    if sample.get("decoder_h") is not None:
        print(f"   Decoder hidden state shape: {np.array(sample['decoder_h']).shape}")
