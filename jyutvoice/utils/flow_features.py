"""CosyVoice2 front-end features for training data: speaker embedding, S3 speech tokens and the flow encoder's hidden state `decoder_h`.

The functions are the ones in scripts/prepare_dataset2.py (same fbank settings, tokenizer call and encoder), moved into the package so
other builders can import them without that script's module-level side effects (it loads a word-segmentation model on import).
"""
import numpy as np
import onnxruntime
import torch
import torchaudio.compliance.kaldi as kaldi
import whisper

from jyutvoice.transformer.upsample_encoder import UpsampleConformerEncoder
from jyutvoice.utils.mask import make_pad_mask


def load_spk_embedding_model(onnx_path: str):
    """CampPlus speaker embedding (ONNX, CPU, one thread: run one session per worker process)."""
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    return onnxruntime.InferenceSession(onnx_path, sess_options=option, providers=["CPUExecutionProvider"])


def get_spk_embedding(audio, onnx_session):
    """192-d speaker embedding of 16 kHz mono audio (numpy array or tensor), as a list of floats."""
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0) if isinstance(audio, np.ndarray) else audio.float().reshape(1, -1)
    feat = kaldi.fbank(audio_tensor, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    return (
        onnx_session.run(None, {onnx_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
    )


def load_speech_tokenizer(speech_tokenizer_path: str):
    """CosyVoice2's S3 speech tokenizer v2 (ONNX, CPU, one thread)."""
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    return onnxruntime.InferenceSession(speech_tokenizer_path, sess_options=option, providers=["CPUExecutionProvider"])


def extract_speech_token(audio, speech_tokenizer_session):
    """S3 tokens (25 Hz) of 16 kHz mono audio of at most 30 s: ((1, N) int32 tensor, (1,) lengths)."""
    audio = audio.cpu().numpy() if isinstance(audio, torch.Tensor) else audio
    feat = whisper.log_mel_spectrogram(torch.from_numpy(audio).float().unsqueeze(0), n_mels=128)
    token = speech_tokenizer_session.run(
        None,
        {
            speech_tokenizer_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
            speech_tokenizer_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32),
        },
    )[0].flatten().tolist()
    return torch.tensor([token], dtype=torch.int32), torch.tensor([len(token)], dtype=torch.int32)


class FlowEncoder(torch.nn.Module):
    """CosyVoice2's flow encoder (token embedding + upsampling conformer + projection to the 80-d mel space): tokens at 25 Hz -> 50 Hz."""

    def __init__(self, vocab_size=6561, input_size=512, output_size=80, device="cpu"):
        super().__init__()
        self.device = device
        self.input_embedding = torch.nn.Embedding(vocab_size, input_size)
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
        ).to(device)
        self.encoder_proj = torch.nn.Linear(512, output_size)

    def forward(self, token, token_len):
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(self.device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask
        h, h_lengths = self.encoder(token, token_len, streaming=False)
        return self.encoder_proj(h), h_lengths


def load_flow_encoder(flow_encoder_path, device="cpu"):
    flow_encoder = FlowEncoder(device=device)
    flow_encoder.load_state_dict(torch.load(flow_encoder_path, map_location=device, weights_only=True))
    return flow_encoder.to(device).eval()


@torch.no_grad()
def get_decoder_hidden_state(speech_token, speech_token_len, flow_encoder, device):
    """(1, 2 * N, 80) hidden state of the flow encoder for S3 tokens."""
    h, _ = flow_encoder(speech_token.to(device), speech_token_len.to(device))
    return h
