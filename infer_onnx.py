#!/usr/bin/env python3
"""
ONNX Inference script for JyutVoice TTS model.

This script performs TTS inference using ONNX Runtime for the text encoder and duration predictor,
with PyTorch for the flow decoder and vocoder.

Note: Speaker embedding is normalized (L2) and projected through spk_embed_affine_layer
inside the TTS model. For ONNX inference, this projection happens in the synthesise_onnx function.

Usage:
    python infer_onnx.py --text "你好，歡迎使用這個語音合成系統。" --lang zh --ref_audio tmp/seedtts_ref_en_1.wav --output output.wav
"""

import argparse
import time
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
import numpy as np
from typing import Optional
from pydips import BertModel
from hyperpyyaml import load_hyperpyyaml
from jyutvoice.text import text_to_sequence
from jyutvoice.utils.utils import intersperse
from jyutvoice.utils.model import sequence_mask, generate_path

ws_model = BertModel()


def load_onnx_session(onnx_path: str, use_gpu: bool = False):
    """Load ONNX model session."""
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1

    providers = ["CPUExecutionProvider"]
    if use_gpu and "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")

    session = onnxruntime.InferenceSession(
        onnx_path,
        sess_options=option,
        providers=providers,
    )
    return session


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


def run_text_encoder_onnx(
    encoder_session, x, x_lengths, lang, tone, word_pos, syllable_pos, spk_embed
):
    """Run text encoder using ONNX Runtime."""
    ort_inputs = {
        "x": x.cpu().numpy().astype(np.int64),
        "x_lengths": x_lengths.cpu().numpy().astype(np.int64),
        "lang": lang.cpu().numpy().astype(np.int64),
        "tone": tone.cpu().numpy().astype(np.int64),
        "word_pos": word_pos.cpu().numpy().astype(np.int64),
        "syllable_pos": syllable_pos.cpu().numpy().astype(np.int64),
        "spk_embed": spk_embed.cpu().numpy().astype(np.float32),
    }

    # Run ONNX inference
    ort_outputs = encoder_session.run(None, ort_inputs)

    # Convert outputs back to torch tensors
    encoder_output = torch.from_numpy(ort_outputs[0])  # encoder_output
    mu_x = torch.from_numpy(ort_outputs[1])  # mu_x
    x_mask = torch.from_numpy(ort_outputs[2])  # x_mask

    return encoder_output, mu_x, x_mask


def run_duration_predictor_onnx(dp_session, encoder_output, x_mask):
    """Run duration predictor using ONNX Runtime."""
    ort_inputs = {
        "encoder_output": encoder_output.cpu().numpy().astype(np.float32),
        "x_mask": x_mask.cpu().numpy().astype(np.float32),
    }

    # Run ONNX inference
    ort_outputs = dp_session.run(None, ort_inputs)

    # Convert output back to torch tensor
    log_durations = torch.from_numpy(ort_outputs[0])

    return log_durations


def synthesise_onnx(
    encoder_session,
    dp_session,
    decoder,
    spk_embed_affine_layer,
    x,
    x_lengths,
    lang,
    tone,
    word_pos,
    syllable_pos,
    spk_embed,
    n_timesteps=10,
    temperature=1.0,
    length_scale=1.0,
    device="cpu",
):
    """
    Synthesize speech using ONNX text encoder and duration predictor, PyTorch decoder.

    Args:
        encoder_session: ONNX Runtime session for text encoder
        dp_session: ONNX Runtime session for duration predictor
        decoder: PyTorch flow decoder model
        spk_embed_affine_layer: Speaker embedding projection layer
        x: Phoneme token IDs (B, T)
        x_lengths: Sequence lengths (B,)
        lang: Language IDs (B, T)
        tone: Tone IDs (B, T)
        word_pos: Word position IDs (B, T)
        syllable_pos: Syllable position IDs (B, T)
        spk_embed: Speaker embeddings (B, spk_dim)
        n_timesteps: Number of diffusion timesteps
        temperature: Temperature for diffusion sampling
        length_scale: Duration scaling factor
        device: Device for PyTorch operations

    Returns:
        dict with "mel" key containing generated mel-spectrogram
    """
    # Project speaker embedding for FM conditioning
    spk_embed = torch.nn.functional.normalize(spk_embed, dim=1)
    c = spk_embed_affine_layer(spk_embed)

    # Run text encoder (ONNX)
    encoder_output, mu_x, x_mask = run_text_encoder_onnx(
        encoder_session, x, x_lengths, lang, tone, word_pos, syllable_pos, spk_embed
    )

    # Move to device for subsequent operations
    encoder_output = encoder_output.to(device)
    mu_x = mu_x.to(device)
    x_mask = x_mask.to(device)

    # Run duration predictor (ONNX)
    log_durations = run_duration_predictor_onnx(dp_session, encoder_output, x_mask)
    log_durations = log_durations.to(device)

    # Generate durations
    durations = torch.exp(log_durations) * x_mask
    durations = torch.ceil(durations * length_scale).long()

    # Expand features according to durations (using generate_path for alignment)
    y_lengths = torch.clamp_min(torch.sum(durations, [1, 2]), 1).long()
    y_max_length = y_lengths.max()

    # Create alignment mask
    y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask.dtype)
    attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
    attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

    # Align encoded text to get mu_y
    mu_y = torch.matmul(
        attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)
    ).transpose(1, 2)
    print(
        f"mu_y stats - Max: {mu_y.abs().max().item():.4f}, Mean: {mu_y.mean().item():.4f}, Std: {mu_y.std().item():.4f}"
    )

    # Prepare conditioning (no prompt)
    conds = torch.zeros_like(mu_y).to(mu_y.dtype)
    from jyutvoice.utils.mask import make_pad_mask

    mask = (~make_pad_mask(y_lengths)).to(mu_y.dtype)

    # Run flow decoder (PyTorch)
    with torch.no_grad():
        mel, _ = decoder(
            mu=mu_y,
            mask=mask.unsqueeze(1),
            spks=c,
            cond=conds,
            n_timesteps=n_timesteps,
            temperature=temperature,
            streaming=False,
        )

    return {"mel": mel}


def main():
    parser = argparse.ArgumentParser(description="JyutVoice TTS ONNX Inference")
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
        help="Reference audio file for speaker embedding",
    )
    parser.add_argument("--output", required=True, help="Output audio file path")
    parser.add_argument(
        "--config",
        default="configs/onnx_infer.yaml",
        help="Configuration file path (for decoder and vocoder)",
    )
    parser.add_argument(
        "--encoder_onnx",
        default="pretrained_models/onnx_models/text_encoder.onnx",
        help="Path to text encoder ONNX model",
    )
    parser.add_argument(
        "--duration_predictor_onnx",
        default="pretrained_models/onnx_models/duration_predictor.onnx",
        help="Path to duration predictor ONNX model",
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
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU for ONNX inference if available"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    with open(args.config, "r") as f:
        configs = load_hyperpyyaml(f)

    # Load ONNX models
    print(f"Loading text encoder ONNX model from {args.encoder_onnx}...")
    encoder_session = load_onnx_session(args.encoder_onnx, use_gpu=args.use_gpu)

    print(
        f"Loading duration predictor ONNX model from {args.duration_predictor_onnx}..."
    )
    dp_session = load_onnx_session(args.duration_predictor_onnx, use_gpu=args.use_gpu)

    # Load flow decoder (PyTorch)
    print(f"Loading flow decoder from {args.flow_decoder_checkpoint}...")
    decoder = configs["decoder"].eval().to(device)
    checkpoint = torch.load(args.flow_decoder_checkpoint, map_location=device)

    # Extract decoder weights and spk_embed_affine_layer weights
    decoder_state_dict = {}
    spk_embed_affine_state_dict = {}

    for key, val in checkpoint.items():
        if key.startswith("decoder."):
            decoder_state_dict[key.replace("decoder.", "")] = val
        elif key.startswith("spk_embed_affine_layer."):
            spk_embed_affine_state_dict[key.replace("spk_embed_affine_layer.", "")] = (
                val
            )

    decoder.load_state_dict(decoder_state_dict)

    # Initialize and load spk_embed_affine_layer
    spk_embed_dim = configs.get("spk_embed_dim", 192)
    output_size = configs.get("n_feats", 80)
    spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size).to(device)
    spk_embed_affine_layer.load_state_dict(spk_embed_affine_state_dict)
    spk_embed_affine_layer.eval()

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

    # Resample to 16kHz for speaker embedding
    if ref_sr != 16000:
        resampler_16k = torchaudio.transforms.Resample(orig_freq=ref_sr, new_freq=16000)
        ref_audio_16k = resampler_16k(ref_audio)
    else:
        ref_audio_16k = ref_audio

    # Extract speaker embedding from reference audio
    print("Extracting speaker embedding from reference audio...")
    spk_embed = extract_spk_embedding(spk_model, ref_audio_16k)

    text = args.text
    if args.lang in ["zh", "yue"]:
        text = word_seg(text)

    # Process input text
    print(f"Processing text: {args.text} (language: {args.lang})")
    x, x_lengths, tones, word_pos, syllable_pos, lang_ids = get_text(
        text, args.lang, args.phone
    )

    # Move tensors to device (for decoder operations)
    # Note: ONNX inputs will be converted to numpy in the ONNX functions
    x = x.to(device)
    x_lengths = x_lengths.to(device)
    tones = tones.to(device)
    word_pos = word_pos.to(device)
    syllable_pos = syllable_pos.to(device)
    lang_ids = lang_ids.to(device)
    spk_embed = spk_embed.to(device)

    # Run inference
    print("Running TTS synthesis with ONNX encoder and duration predictor...")
    start_time = time.time()

    with torch.no_grad():
        result = synthesise_onnx(
            encoder_session=encoder_session,
            dp_session=dp_session,
            decoder=decoder,
            spk_embed_affine_layer=spk_embed_affine_layer,
            x=x,
            x_lengths=x_lengths,
            lang=lang_ids,
            tone=tones,
            word_pos=word_pos,
            syllable_pos=syllable_pos,
            spk_embed=spk_embed,
            n_timesteps=args.n_timesteps,
            temperature=1.0,
            length_scale=args.length_scale,
            device=device,
        )

        # Decoder outputs mel in log-scale, HiFT vocoder expects log-mel
        mel_for_vocoder = result["mel"]
        print(
            f"Mel stats - Max: {mel_for_vocoder.abs().max().item():.4f}, Mean: {mel_for_vocoder.mean().item():.4f}, Std: {mel_for_vocoder.std().item():.4f}"
        )

        wav, _ = hift.inference(mel_for_vocoder)
        print(
            f"Wav stats - Max: {wav.abs().max().item():.4f}, Mean: {wav.mean().item():.4f}, Std: {wav.std().item():.4f}"
        )

    end_time = time.time()
    synthesis_time = end_time - start_time

    # Save output audio
    print(f"Saving audio to {args.output}...")
    torchaudio.save(args.output, wav.cpu(), 24000)

    print("✅ ONNX Inference completed successfully!")
    print(f"Generated audio saved to: {args.output}")
    print(f"Synthesis time: {synthesis_time:.2f} seconds")
    print(f"Audio duration: {wav.shape[1] / 24000:.2f} seconds")


if __name__ == "__main__":
    main()
