"""
Fast dataset preparation script for JyutVoice TTS.


and decoder hidden state extraction on your dataset using direct iteration
for faster processing compared to HuggingFace datasets.map().

Usage:
    # From HuggingFace dataset directory (fast processing)
    python prepare_dataset2.py --dataset tmp/dataset --output tmp/dataset_processed \
        --speaker_model pretrained_models/campplus.onnx \
        --flow_encoder pretrained_models/flow_encoder.pt \
        --speech_tokenizer pretrained_models/speech_tokenizer_v2.onnx

    # Text processing only (no audio features)
    python prepare_dataset2.py --dataset tmp/dataset --output tmp/dataset_processed

    # With specific device for model inference
    python prepare_dataset2.py --dataset tmp/dataset --output tmp/dataset_processed \
        --speaker_model pretrained_models/campplus.onnx \
        --flow_encoder pretrained_models/flow_encoder.pt \
        --speech_tokenizer pretrained_models/speech_tokenizer_v2.onnx \
        --device cuda

Expected dataset format:
    HuggingFace dataset directory with columns:
    - text: Raw text content
    - lang: Language code ("yue", "zh", "en", etc.)
    - audio: Audio data (path or array format)

Output:
    - Extracted speaker embeddings (if models provided)
    - Extracted decoder hidden states (if models provided)
    - Saved to HuggingFace dataset format
"""

import os
import numpy as np
import torch
import whisper
import librosa
import torchaudio.compliance.kaldi as kaldi
import onnxruntime
import argparse
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from pydips import BertModel
from jyutvoice.transformer.upsample_encoder import UpsampleConformerEncoder
from jyutvoice.utils.mask import make_pad_mask

ws_model = BertModel()


def load_spk_embedding_model(onnx_path: str):
    """Load speaker embedding ONNX model."""
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(
        onnx_path, sess_options=option, providers=["CPUExecutionProvider"]
    )
    return ort_session


def get_spk_embedding(audio, onnx_session):
    """Extract speaker embedding from audio using CampPlus model."""
    audio_tensor = None

    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(dim=0)
    elif isinstance(audio, torch.Tensor):
        if audio.dim() == 1:
            audio_tensor = audio.float().unsqueeze(dim=0)
        elif audio.dim() == 2:
            audio_tensor = audio.float()
        else:
            raise ValueError("Audio tensor must be 1D or 2D.")
    if audio_tensor is None:
        raise ValueError("Audio must be a numpy array or a torch tensor.")

    feat = kaldi.fbank(audio_tensor, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = (
        onnx_session.run(
            None,
            {onnx_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()},
        )[0]
        .flatten()
        .tolist()
    )

    return embedding


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


class FlowEncoder(torch.nn.Module):
    def __init__(self, vocab_size=6561, input_size=512, output_size=80, device="cpu"):
        super().__init__()
        self.device = device
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
        ).to(device)
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
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(self.device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # Encode
        h, h_lengths = self.encoder(token, token_len, streaming=False)
        # Project to output size (80)
        h = self.encoder_proj(h)

        return h, h_lengths


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

    flow_encoder = FlowEncoder(device=device)

    # Load pretrained weights
    state_dict = torch.load(flow_encoder_path, map_location=device, weights_only=True)
    flow_encoder.load_state_dict(state_dict)
    flow_encoder.eval()

    return flow_encoder


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


def process_sample(
    sample,
    spk_model_path=None,
    flow_encoder_path=None,
    speech_tokenizer_path=None,
    device="cpu",
):
    """
    Process a single sample to extract audio features.

    Args:
        sample: Dataset sample with 'audio' field
        spk_model_path: Path to speaker embedding model
        flow_encoder_path: Path to flow encoder
        speech_tokenizer_path: Path to speech tokenizer
        device: Device for inference

    Returns:
        Processed sample with added features
    """
    # Load models if paths provided (loaded per worker to avoid multiprocessing issues)
    spk_model = None
    flow_encoder = None
    speech_tokenizer = None

    if spk_model_path:
        spk_model = load_spk_embedding_model(spk_model_path)

    if flow_encoder_path:
        flow_encoder = load_flow_encoder(flow_encoder_path, device=device)

    if speech_tokenizer_path:
        speech_tokenizer = load_speech_tokenizer(speech_tokenizer_path)

    # Process audio features
    spk_emb = None
    decoder_h = None

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

    if spk_model is not None:
        try:
            spk_emb = get_spk_embedding(audio, spk_model)
        except Exception as e:
            print(f"Error extracting speaker embedding: {e}")

    if flow_encoder is not None and speech_tokenizer is not None:
        try:
            # Extract speech tokens
            speech_token, speech_token_len = extract_speech_token(
                audio, speech_tokenizer
            )

            # Get hidden state
            h = get_decoder_hidden_state(
                speech_token, speech_token_len, flow_encoder, device
            )
            decoder_h = h.squeeze(
                0
            ).tolist()  # Remove batch dim and convert to list for serialization
        except Exception as e:
            print(f"Error extracting decoder hidden state: {e}")

    # Add features to sample
    if spk_emb is not None:
        sample["spk_emb"] = spk_emb
    if decoder_h is not None:
        sample["decoder_h"] = decoder_h
    if speech_token is not None:
        sample["speech_token"] = speech_token.squeeze(0).tolist()

    return sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fast word segmentation and feature extraction for dataset preparation"
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
        "--speaker_model",
        type=str,
        default="pretrained_models/campplus.onnx",
        help="Path to speaker embedding ONNX model (e.g., pretrained_models/campplus.onnx)",
    )
    parser.add_argument(
        "--flow_encoder",
        type=str,
        default="pretrained_models/flow_encoder.pt",
        help="Path to flow encoder weights (e.g., pretrained_models/flow_encoder.pt)",
    )
    parser.add_argument(
        "--speech_tokenizer",
        type=str,
        default="pretrained_models/speech_tokenizer_v2.onnx",
        help="Path to speech tokenizer ONNX model (e.g., pretrained_models/speech_tokenizer_v2.onnx)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for inference (cpu or cuda)",
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

    # Check if models are provided (will be loaded in workers)
    has_spk_model = args.speaker_model is not None
    has_flow_encoder = args.flow_encoder is not None
    has_speech_tokenizer = args.speech_tokenizer is not None

    if has_spk_model:
        print(
            f"📦 Will load speaker embedding model from {args.speaker_model} in workers"
        )

    if has_flow_encoder:
        print(f"📦 Will load flow encoder from {args.flow_encoder} in workers")

    if has_speech_tokenizer:
        print(f"📦 Will load speech tokenizer from {args.speech_tokenizer} in workers")

    # Process audio features using parallel map if models are provided
    if has_spk_model or (has_flow_encoder and has_speech_tokenizer):
        print("🎵 Processing audio features using parallel processing...")

        # Define the processing function with model paths
        def process_sample_wrapper(sample):
            return process_sample(
                sample,
                spk_model_path=args.speaker_model,
                flow_encoder_path=args.flow_encoder,
                speech_tokenizer_path=args.speech_tokenizer,
                device=args.device,
            )

        # Use datasets.map with num_proc for parallel processing
        ds = ds.map(
            process_sample_wrapper,
            num_proc=args.num_proc,  # Use configurable number of processes
            desc="Processing audio features",
        )

        print("✅ Audio feature extraction complete!")

        # Count successful extractions
        spk_emb_count = sum(1 for sample in ds if sample.get("spk_emb") is not None)
        decoder_h_count = sum(1 for sample in ds if sample.get("decoder_h") is not None)

        if has_spk_model:
            print(f"   Speaker embeddings extracted: {spk_emb_count}")
        if has_flow_encoder and has_speech_tokenizer:
            print(f"   Decoder hidden states extracted: {decoder_h_count}")

    # Save the processed dataset
    print(f"💾 Saving processed dataset to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    ds.save_to_disk(args.output)

    print(f"✅ Dataset saved successfully!")
    print(f"\n📊 Final dataset info:")
    print(f"   Total samples: {len(ds)}")
    print(f"   Columns: {ds.column_names}")

    # Show sample info
    if len(ds) > 0:
        sample = ds[0]
        print(f"   Sample keys: {list(sample.keys())}")
        if sample.get("spk_emb") is not None:
            print(f"   Speaker embedding shape: {len(sample['spk_emb'])}")
        if sample.get("decoder_h") is not None:
            print(
                f"   Decoder hidden state shape: {np.array(sample['decoder_h']).shape}"
            )
