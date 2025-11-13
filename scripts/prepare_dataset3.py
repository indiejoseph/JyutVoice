#!/usr/bin/env python3
"""
Fast dataset preparation script for adding speech tokens to JyutVoice TTS dataset.

This script extracts speech tokens from audio using the speech tokenizer ONNX model
and adds them to the dataset for training.

Usage:
    python prepare_dataset3.py --dataset tmp/dataset --output tmp/dataset_with_tokens \
        --speech_tokenizer pretrained_models/speech_tokenizer_v2.onnx

Expected dataset format:
    HuggingFace dataset directory with columns:
    - audio: Audio data (path or array format)

Output:
    - Extracted speech tokens added to each sample
    - Saved to HuggingFace dataset format
"""

import os
import numpy as np
import torch
import librosa
import onnxruntime
import argparse
from datasets import load_from_disk
from tqdm import tqdm
import torchaudio


def load_speech_tokenizer(speech_tokenizer_path: str):
    """Load speech tokenizer ONNX model."""
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        speech_tokenizer_path,
        sess_options=option,
        providers=["CPUExecutionProvider"],
    )
    return session


def _log_mel_spectrogram(audio, n_mels=128, n_fft=400, hop_length=160):
    """Compute log mel spectrogram similar to whisper."""
    import torchaudio

    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=8000,
        norm="slaney",
        mel_scale="slaney",
    )

    # Compute mel spectrogram
    mel_spec = mel_transform(audio)

    # Convert to log scale (add small epsilon to avoid log(0))
    log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))

    # Normalize (mean=0, std=1) like whisper does
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-10)

    return log_mel_spec


def extract_speech_token(audio, speech_tokenizer_session):
    """
    Extract speech tokens from audio using speech tokenizer.

    Args:
        audio: audio signal (torch.Tensor or numpy.ndarray), shape (T,) at 16kHz
        speech_tokenizer_session: ONNX speech tokenizer session

    Returns:
        speech_token: list of token IDs
        speech_token_len: length of token sequence
    """
    # Ensure audio is numpy array
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    elif isinstance(audio, np.ndarray):
        pass
    else:
        raise ValueError("Audio must be torch.Tensor or numpy.ndarray")

    # Convert to torch tensor for mel-spectrogram
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

    # Extract mel-spectrogram (whisper-style: 128 mel bins, 16kHz)
    feat = _log_mel_spectrogram(audio_tensor, n_mels=128)

    # Run speech tokenizer
    speech_token = (
        speech_tokenizer_session.run(
            None,
            {
                speech_tokenizer_session.get_inputs()[0]
                .name: feat.detach()
                .cpu()
                .numpy(),
                speech_tokenizer_session.get_inputs()[1].name: np.array(
                    [feat.shape[2]], dtype=np.int32
                ),
            },
        )[0]
        .flatten()
        .tolist()
    )

    return speech_token, len(speech_token)


def load_audio_from_path(audio_path):
    """Load audio from file path and convert to required formats."""
    try:
        # Load audio using librosa
        audio, sr = librosa.load(audio_path, sr=None)

        # Convert to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        return audio
    except Exception as e:
        print(f"Error loading audio from {audio_path}: {e}")
        return None


def process_sample(sample, speech_tokenizer_session):
    """
    Process a single sample to extract speech tokens.

    Args:
        sample: Dataset sample with 'audio' field
        speech_tokenizer_session: Loaded ONNX speech tokenizer session

    Returns:
        Processed sample with added speech tokens
    """
    # Get audio data and handle different formats
    audio_data = sample.get("audio")
    if audio_data is None:
        return sample

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
            return sample
    elif isinstance(audio_data, str):
        # File path
        audio = load_audio_from_path(audio_data)
        sr = 16000  # After loading with librosa
    else:
        return sample

    if audio is None:
        return sample

    # Ensure audio is 16kHz for processing
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    # Extract speech tokens
    try:
        speech_token, speech_token_len = extract_speech_token(
            audio, speech_tokenizer_session
        )

        # Add to sample
        sample["speech_token"] = speech_token
        sample["speech_token_len"] = speech_token_len

    except Exception as e:
        print(f"Error extracting speech tokens: {e}")
        return sample

    return sample


def main():
    parser = argparse.ArgumentParser(
        description="Add speech tokens to dataset using ONNX speech tokenizer"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the input HuggingFace dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output dataset directory",
    )
    parser.add_argument(
        "--speech_tokenizer",
        type=str,
        default="pretrained_models/speech_tokenizer_v2.onnx",
        help="Path to speech tokenizer ONNX model",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes to use for parallel processing (default: 1)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank of this process in distributed processing (default: 0)",
    )
    parser.add_argument(
        "--worlds",
        type=int,
        default=1,
        help="Total number of processes in distributed processing (default: 1)",
    )

    args = parser.parse_args()

    # Load dataset
    print(f"📁 Loading dataset from {args.dataset}...")
    ds = load_from_disk(args.dataset)
    print(f"✅ Loaded dataset with {len(ds)} samples")

    # Shard dataset for distributed processing
    if args.worlds > 1:
        print(f"🔀 Sharding dataset: rank {args.rank}/{args.worlds}")
        ds = ds.shard(num_shards=args.worlds, index=args.rank)
        print(
            f"✅ Processing shard {args.rank + 1}/{args.worlds} with {len(ds)} samples"
        )

    # Modify output path for distributed processing
    if args.worlds > 1:
        args.output = f"{args.output}_{args.rank + 1}_{args.worlds}"
        print(f"📁 Output path modified to: {args.output}")

    # Load speech tokenizer (shared across workers)
    print(f"🎯 Loading speech tokenizer from {args.speech_tokenizer}...")
    speech_tokenizer_session = load_speech_tokenizer(args.speech_tokenizer)
    print("✅ Speech tokenizer loaded!")

    # Process speech tokens using parallel map
    print("🎵 Extracting speech tokens...")

    def process_sample_wrapper(sample):
        return process_sample(sample, speech_tokenizer_session)

    # Use datasets.map with num_proc for parallel processing
    ds = ds.map(
        process_sample_wrapper,
        num_proc=args.num_proc,
        desc="Extracting speech tokens",
    )

    print("✅ Speech token extraction complete!")

    # Count successful extractions
    token_count = sum(1 for sample in ds if sample.get("speech_token") is not None)
    print(f"   Speech tokens extracted: {token_count}/{len(ds)}")

    # Save the processed dataset
    print(f"💾 Saving processed dataset to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    ds.save_to_disk(args.output)

    print("✅ Dataset saved successfully!")
    print(f"\n📊 Final dataset info:")
    print(f"   Total samples: {len(ds)}")
    print(f"   Columns: {ds.column_names}")

    # Show sample info
    if len(ds) > 0:
        sample = ds[0]
        print(f"   Sample keys: {list(sample.keys())}")
        if sample.get("speech_token") is not None:
            print(f"   Speech token length: {len(sample['speech_token'])}")


if __name__ == "__main__":
    main()
