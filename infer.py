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
from pydips import BertModel
from hyperpyyaml import load_hyperpyyaml
from jyutvoice.utils.mask import make_pad_mask
from jyutvoice.transformer.upsample_encoder import UpsampleConformerEncoder
from jyutvoice.text import text_to_sequence
from jyutvoice.utils.utils import intersperse
from jyutvoice.utils.audio import mel_spectrogram

ws_model = BertModel()


class FlowEncoder(torch.nn.Module):
    """Flow encoder for processing speech tokens into hidden representations."""

    def __init__(self, vocab_size=6561, input_size=512, output_size=80):
        super().__init__()
        self.input_embedding = torch.nn.Embedding(vocab_size, input_size)
        # Instantiate encoder with CosyVoice2 architecture
        self.encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer="linear",
            pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn",
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
            static_chunk_size=25,
        )
        # Project encoder output from 512 to output_size (80)
        self.encoder_proj = torch.nn.Linear(512, output_size)

    def forward(self, token, token_len):
        """
        Process speech tokens through the encoder.

        Args:
            token: speech tokens, shape (batch, seq_len)
            token_len: token sequence lengths, shape (batch,)

        Returns:
            h: encoder output, shape (batch, seq_len, 80)
            h_lengths: output lengths, shape (batch,)
        """
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # Encode
        h, h_lengths = self.encoder(token, token_len, streaming=False)
        # Project to output size (80)
        h = self.encoder_proj(h)

        return h, h_lengths


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


def extract_speech_token(audio, speech_tokenizer_session):
    """
    Extract speech tokens from audio using speech tokenizer.

    Args:
        audio: audio signal (torch.Tensor or numpy.ndarray), shape (T,) at 16kHz
        speech_tokenizer_session: ONNX speech tokenizer session

    Returns:
        speech_token: tensor of shape (1, num_tokens)
        speech_token_len: tensor of shape (1,) with token sequence length
    """
    # Ensure audio is on CPU for processing
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    elif isinstance(audio, np.ndarray):
        pass
    else:
        raise ValueError("Audio must be torch.Tensor or numpy.ndarray")

    # Convert to torch tensor for mel-spectrogram
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

    # Extract mel-spectrogram (whisper format)
    feat = whisper.log_mel_spectrogram(audio_tensor, n_mels=128)

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

    speech_token = torch.tensor([speech_token], dtype=torch.int32)
    speech_token_len = torch.tensor([len(speech_token[0])], dtype=torch.int32)

    return speech_token, speech_token_len


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


def extract_speech_feat(speech, device):
    """Extract mel spectrogram features from audio."""
    speech_feat = (
        mel_spectrogram(
            speech,
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=8000,
            center=False,
        )
        .squeeze(dim=0)
        .transpose(0, 1)
        .to(device)
    )
    speech_feat = speech_feat.unsqueeze(dim=0)
    speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(device)
    return speech_feat, speech_feat_len


def get_text(text: str, lang: str, phone: str = None):
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


def load_flow_encoder(flow_encoder_path, device="cpu"):
    """
    Load the UpsampleConformerEncoder from a pretrained model checkpoint.

    Args:
        flow_encoder_path (str): Path to the pretrained flow model weights
        device (str or torch.device): Device to load model on

    Returns:
        torch.nn.Module: Loaded encoder module ready for inference
    """
    if flow_encoder_path is None:
        return None

    flow_encoder = FlowEncoder().to(device)

    # Load pretrained weights
    state_dict = torch.load(flow_encoder_path, map_location=device, weights_only=True)
    flow_encoder.load_state_dict(state_dict)
    flow_encoder.eval()

    return flow_encoder


def word_seg(text: str):
    text = ws_model.cut(text, mode="coarse")
    text = " ".join(text)
    return text


def get_decoder_hidden_state(speech_token, speech_token_len, flow_encoder, device):
    """
    Extract hidden state from the flow encoder (CosyVoice2's encoder).

    This function processes speech tokens through the encoder to get
    the hidden representation used for prior loss computation during training.

    Args:
        speech_token (torch.Tensor): Speech tokens, shape (batch, token_len)
        speech_token_len (torch.Tensor): Lengths of speech token sequences
        flow_encoder (torch.nn.Module): The flow encoder
        device (torch.device): Device to run inference on

    Returns:
        torch.Tensor: Hidden state from encoder, shape (batch, token_len, 512)
    """
    if flow_encoder is None:
        raise ValueError(
            "flow_encoder must be provided to extract decoder hidden state"
        )

    speech_token = speech_token.to(device)
    speech_token_len = speech_token_len.to(device)

    with torch.no_grad():
        # Get encoder output and lengths
        h, h_lengths = flow_encoder(speech_token, speech_token_len)
        # h shape: (batch, token_len, 512)

    return h


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

    # Load HiFT vocoder
    print(f"Loading HiFT vocoder from {args.hift}...")
    hift = configs["hift"].eval().to(device)
    hift.load_state_dict(torch.load(args.hift, map_location="cpu"))

    # Load auxiliary models
    print("Loading auxiliary models...")
    speech_tokenizer_session = load_speech_tokenizer(args.speech_tokenizer)

    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    spk_model = onnxruntime.InferenceSession(
        args.campplus, sess_options=option, providers=["CPUExecutionProvider"]
    )

    flow_encoder = load_flow_encoder(args.flow_encoder, device)

    # Load and process reference audio
    print(f"Loading reference audio from {args.ref_audio}...")
    ref_audio, ref_sr = torchaudio.load(args.ref_audio)

    # Resample to 16kHz for speech tokenization and speaker embedding
    if ref_sr != 16000:
        resampler_16k = torchaudio.transforms.Resample(orig_freq=ref_sr, new_freq=16000)
        ref_audio_16k = resampler_16k(ref_audio)
    else:
        ref_audio_16k = ref_audio

    # Resample to 24kHz for mel spectrogram extraction
    if ref_sr != 24000:
        resampler_24k = torchaudio.transforms.Resample(orig_freq=ref_sr, new_freq=24000)
        ref_audio_24k = resampler_24k(ref_audio)
    else:
        ref_audio_24k = ref_audio

    # Extract features from reference audio
    print("Extracting features from reference audio...")
    prompt_feat, prompt_feat_len = extract_speech_feat(ref_audio_24k, device)
    prompt_token, prompt_tokens_len = extract_speech_token(
        ref_audio_16k.squeeze(0), speech_tokenizer_session
    )
    prompt_h = get_decoder_hidden_state(
        prompt_token, prompt_tokens_len, flow_encoder, device
    )
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
            prompt_feat=prompt_feat,
            prompt_h=prompt_h,
            spk_embed=spk_embed,
            n_timesteps=args.n_timesteps,
            length_scale=args.length_scale,
        )
        wav, _ = hift.inference(result["mel"])

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
