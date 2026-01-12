#!/usr/bin/env python3
"""
Inference script for JyutVoice TTS model.

This script performs TTS inference including:
- Text processing with multilingual support
- Speaker embedding extraction from reference audio
- Mel spectrogram generation via text encoder and flow decoder
- HiFi-GAN vocoder for waveform synthesis

Note: Speaker embedding is normalized and projected through spk_embed_affine_layer
inside the TTS model's synthesise() method.

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
from typing import Optional
from pydips import BertModel
from hyperpyyaml import load_hyperpyyaml
from jyutvoice.text import text_to_sequence
from jyutvoice.utils.utils import intersperse

ws_model = BertModel()


def extract_spk_embedding(spk_model, speech):
    """Extract speaker embedding from audio."""
    feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
    spk_feat = feat - feat.mean(dim=0, keepdim=True)

    embedding = (
        spk_model.run(
            None,
            {spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()},
        )[0]
        .flatten()
        .tolist()
    )
    embedding = torch.tensor([embedding])

    return embedding


def get_text(text: str, lang: str, phone: Optional[str] = None):
    """Process text into linguistic features for TTS."""
    phone_token_ids, tones, word_pos, syllable_pos, lang_ids = text_to_sequence(
        text, lang=lang, phone=phone
    )
    phone_token_ids = intersperse(phone_token_ids, 0)
    tones = intersperse(tones, 0)
    word_pos = intersperse(word_pos, 0)
    syllable_pos = intersperse(syllable_pos, 0)
    lang_ids = intersperse(lang_ids, 0)
    x = torch.tensor([phone_token_ids])
    x_lengths = torch.tensor([len(phone_token_ids)])
    tones = torch.tensor([tones])
    word_pos = torch.tensor([word_pos])
    syllable_pos = torch.tensor([syllable_pos])
    lang_ids = torch.tensor([lang_ids])

    return x, x_lengths, tones, word_pos, syllable_pos, lang_ids


def word_seg(text: str):
    text = ws_model.cut(text, mode="coarse")
    text = " ".join(text)
    return text


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
        default="pretrained_models/20260102_encoder_only.ckpt",
        help="Path to TTS model checkpoint",
    )
    parser.add_argument(
        "--flow_decoder_checkpoint",
        default="pretrained_models/flow_decoder.pt",
        help="Path to flow decoder weights",
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
        default=1.0,
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

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    has_decoder_weights = any("decoder" in k for k in state_dict.keys())

    if not has_decoder_weights:
        print("Loading decoder weights from flow_decoder.pt")
        decoder_weights = torch.load(args.flow_decoder_checkpoint, map_location=device)
        state_dict.update(decoder_weights)

    tts.load_state_dict(state_dict)
    tts = tts.eval().to(device)

    # Load HiFT vocoder
    print(f"Loading HiFT vocoder from {args.hift}...")
    hift = configs["hift"].eval().to(device)
    hift.load_state_dict(torch.load(args.hift, map_location="cpu"))

    # Load speaker embedding model
    print("Loading speaker embedding model...")
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    spk_model = onnxruntime.InferenceSession(
        args.campplus, sess_options=option, providers=["CPUExecutionProvider"]
    )

    # Load and process reference audio
    print(f"Loading reference audio from {args.ref_audio}...")
    ref_audio, ref_sr = torchaudio.load(args.ref_audio)

    # Resample to 16kHz for speech tokenization and speaker embedding
    if ref_sr != 16000:
        resampler_16k = torchaudio.transforms.Resample(orig_freq=ref_sr, new_freq=16000)
        ref_audio_16k = resampler_16k(ref_audio)
    else:
        ref_audio_16k = ref_audio

    # Extract speaker embedding from reference audio
    print("Extracting speaker embedding from reference audio...")
    # Note: Speaker embedding will be normalized (L2) and projected through
    # spk_embed_affine_layer inside tts.synthesise() method
    spk_embed = extract_spk_embedding(spk_model, ref_audio_16k)
    text = args.text

    if args.lang in ["zh", "yue"]:
        text = word_seg(text)

    # Process input text
    print(f"Processing text: {args.text} (language: {args.lang})")

    x, x_lengths, tones, word_pos, syllable_pos, lang_ids = get_text(
        text, args.lang, args.phone
    )

    # Move tensors to device
    x = x.to(device)
    x_lengths = x_lengths.to(device)
    tones = tones.to(device)
    word_pos = word_pos.to(device)
    syllable_pos = syllable_pos.to(device)
    lang_ids = lang_ids.to(device)
    spk_embed = spk_embed.to(device)

    # Run inference
    print("Running TTS synthesis...")
    start_time = time.time()

    with torch.no_grad():
        result = tts.synthesise(
            x=x,
            x_lengths=x_lengths,
            lang=lang_ids,
            tone=tones,
            word_pos=word_pos,
            syllable_pos=syllable_pos,
            spk_embed=spk_embed,
            n_timesteps=args.n_timesteps,
            length_scale=args.length_scale,
        )
        print(
            f"Mel stats - Max: {result['mel'].abs().max().item():.4f}, Mean: {result['mel'].mean().item():.4f}, Std: {result['mel'].std().item():.4f}"
        )
        wav, _ = hift.inference(result["mel"])
        print(
            f"Wav stats - Max: {wav.abs().max().item():.4f}, Mean: {wav.mean().item():.4f}, Std: {wav.std().item():.4f}"
        )

    end_time = time.time()
    synthesis_time = end_time - start_time
    print(".2f")

    # Save output audio
    print(f"Saving audio to {args.output}...")
    torchaudio.save(args.output, wav.cpu(), 24000)

    print("✅ Inference completed successfully!")
    print(f"Generated audio saved to: {args.output}")
    print(f"Synthesis time: {synthesis_time:.2f} seconds")
    print(f"Audio duration: {wav.shape[1] / 24000:.2f} seconds")


if __name__ == "__main__":
    main()
