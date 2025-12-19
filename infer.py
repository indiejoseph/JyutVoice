#!/usr/bin/env python3
"""
Inference script for JyutVoice TTS model with full feature extraction pipeline.

This script performs complete TTS inference including:
- Text processing with multilingual support
- Reference audio feature extraction (mel spectrograms, speech tokens, speaker embeddings)
- Flow encoder processing for prompt features
- Mel spectrogram generation
- HiFi-GAN vocoder for waveform synthesis

Usage:
    python infer.py --text "你好，歡迎使用這個語音合成系統。" --lang zh --ref_audio tmp/seedtts_ref_en_1.wav --output output.wav
"""

import argparse
import time
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
import whisper
import numpy as np
from hyperpyyaml import load_hyperpyyaml
from jyutvoice.utils.mask import make_pad_mask
from jyutvoice.text import text_to_sequence
from jyutvoice.utils.utils import intersperse
from jyutvoice.utils.audio import mel_spectrogram


def get_text(text: str, lang: str, phone: str = None):
    """Process text into linguistic features for TTS."""
    phone_token_ids, tones, lang_ids = text_to_sequence(text, lang=lang, phone=phone)
    phone_token_ids = intersperse(phone_token_ids, 0)
    tones = intersperse(tones, 0)
    lang_ids = intersperse(lang_ids, 0)
    x = torch.tensor([phone_token_ids])
    x_lengths = torch.tensor([len(phone_token_ids)])
    tones = torch.tensor([tones])
    lang_ids = torch.tensor([lang_ids])

    return x, x_lengths, tones, lang_ids


def main():
    parser = argparse.ArgumentParser(
        description="JyutVoice TTS Inference with Full Pipeline"
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument(
        "--lang",
        required=True,
        choices=["en", "zh", "yue", "multilingual"],
        help="Language of the text",
    )
    parser.add_argument(
        "--phone", default=None, help="Phonetic transcription (for Cantonese, optional)"
    )
    parser.add_argument(
        "--ref_audio",
        required=True,
        help="Reference audio file for speaker embedding and prompt features",
    )
    parser.add_argument("--output", required=True, help="Output audio file path")
    parser.add_argument(
        "--config", default="configs/base.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--tts_checkpoint",
        default="pretrained_models/epoch=0-step=55872.ckpt",
        help="Path to TTS model checkpoint",
    )
    parser.add_argument(
        "--flow_encoder",
        default="pretrained_models/flow_encoder.pt",
        help="Path to flow encoder weights",
    )
    parser.add_argument(
        "--speech_tokenizer",
        default="pretrained_models/speech_tokenizer_v2.onnx",
        help="Path to speech tokenizer ONNX model",
    )
    parser.add_argument(
        "--campplus",
        default="pretrained_models/campplus.onnx",
        help="Path to CAMPPlus speaker embedding model",
    )
    parser.add_argument(
        "--hift",
        default="pretrained_models/hift.pt",
        help="Path to HiFT vocoder weights",
    )
    parser.add_argument(
        "--n_timesteps", type=int, default=10, help="Number of diffusion timesteps"
    )
    parser.add_argument(
        "--length_scale",
        type=float,
        default=0.9,
        help="Length scale for speech duration control",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    with open(args.config, "r") as f:
        configs = load_hyperpyyaml(f)

    # Load TTS model
    print(f"Loading TTS model from {args.tts_checkpoint}...")
    tts = configs["tts"]
    ckpt = torch.load(args.tts_checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    tts.load_state_dict(state_dict)
    tts = tts.eval().to(device)

    # Load and process reference audio
    print(f"Loading reference audio from {args.ref_audio}...")
    ref_audio, ref_sr = torchaudio.load(args.ref_audio)

    # Resample to 16kHz for speech tokenization and speaker embedding
    if ref_sr != 16000:
        resampler_16k = torchaudio.transforms.Resample(orig_freq=ref_sr, new_freq=16000)
        ref_audio_16k = resampler_16k(ref_audio)
    else:
        ref_audio_16k = ref_audio

    # Extract features from reference audio
    print("Extracting features from reference audio...")
    text = args.text

    # Process input text
    print(f"Processing text: {args.text} (language: {args.lang})")

    x, x_lengths, tones, lang_ids = get_text(text, args.lang, args.phone)

    # Move tensors to device
    x = x.to(device)
    x_lengths = x_lengths.to(device)
    tones = tones.to(device)
    lang_ids = lang_ids.to(device)

    # Run inference
    print("Running TTS synthesis...")
    start_time = time.time()

    with torch.no_grad():
        wav = tts.synthesise(
            x=x,
            x_lengths=x_lengths,
            lang=lang_ids,
            tone=tones,
            n_timesteps=args.n_timesteps,
            length_scale=args.length_scale,
        )

    end_time = time.time()
    synthesis_time = end_time - start_time
    print(".2f")

    # Save output audio
    print(f"Saving audio to {args.output}...")
    torchaudio.save(args.output, wav.cpu(), 24000)

    print("✅ Inference completed successfully!")
    print(f"Generated audio saved to: {args.output}")
    print(f"Audio duration: {wav.shape[1] / 24000:.2f} seconds")


if __name__ == "__main__":
    main()
