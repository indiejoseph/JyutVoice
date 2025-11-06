#!/usr/bin/env python3
"""
Inference script for JyutVoice TTS model.

Usage:
    python infer.py --model pretrained_models/pretrain.pt --text "Hello world" --output output.wav
"""

import argparse
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from hyperpyyaml import load_hyperpyyaml
from jyutvoice.text.cantonese.g2p import g2p
from jyutvoice.text.symbols import symbol_to_id
import onnxruntime


def extract_spk_embedding(spk_model, speech, device):
    """Extract speaker embedding from audio."""
    feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)
    spk_feat = feat - feat.mean(dim=0, keepdim=True)

    if isinstance(spk_model, onnxruntime.InferenceSession):
        embedding = (
            spk_model.run(
                None,
                {
                    spk_model.get_inputs()[0]
                    .name: spk_feat.unsqueeze(dim=0)
                    .cpu()
                    .numpy()
                },
            )[0]
            .flatten()
            .tolist()
        )
        embedding = torch.tensor([embedding]).to(device)
    else:
        [spk_model, stream], trt_engine = spk_model.acquire_estimator()
        with torch.cuda.device(device):
            torch.cuda.current_stream().synchronize()
            spk_feat = spk_feat.unsqueeze(dim=0).to(device)
            batch_size = spk_feat.size(0)

            with stream:
                spk_model.set_input_shape("input", (batch_size, spk_feat.size(1), 80))
                embedding = torch.empty((batch_size, 192), device=spk_feat.device)

                data_ptrs = [
                    spk_feat.contiguous().data_ptr(),
                    embedding.contiguous().data_ptr(),
                ]
                for i, j in enumerate(data_ptrs):
                    spk_model.set_tensor_address(trt_engine.get_tensor_name(i), j)
                assert (
                    spk_model.execute_async_v3(torch.cuda.current_stream().cuda_stream)
                    is True
                )
                torch.cuda.current_stream().synchronize()
            spk_model.release_estimator(spk_model, stream)

    return embedding


def load_reference_audio(ref_audio_path, device):
    """Load and process reference audio for speaker embedding."""
    # Load ONNX models
    campplus_model = "pretrained_models/campplus.onnx"
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1

    spk_model = onnxruntime.InferenceSession(
        campplus_model, sess_options=option, providers=["CPUExecutionProvider"]
    )

    # Load audio
    audio, sr = torchaudio.load(ref_audio_path)

    if sr == 16000:
        audio_16k = audio
    elif sr == 24000:
        audio_16k = torchaudio.transforms.Resample(orig_freq=24000, new_freq=16000)(
            audio
        )
    else:
        audio_16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio)

    # Extract speaker embedding
    speaker_embedding = extract_spk_embedding(spk_model, audio_16k, device)

    return speaker_embedding


def process_text(text):
    """Process text to get phonemes, tones, and other linguistic features."""
    # Get phonemes and tones from Cantonese text
    phones, tones, _, word_pos, syllable_pos = g2p(text)

    # Convert phone symbols to indices
    phone_indices = [symbol_to_id.get(p, symbol_to_id.get("UNK", 0)) for p in phones]

    # Convert to tensors
    phone_tensor = torch.tensor([phone_indices], dtype=torch.int32)
    tone_tensor = torch.tensor([tones], dtype=torch.int32)
    lang_tensor = torch.tensor(
        [[0] * len(phone_indices)], dtype=torch.int32
    )  # 0 = Cantonese
    word_pos_tensor = torch.tensor([word_pos], dtype=torch.int32)
    syllable_pos_tensor = torch.tensor([syllable_pos], dtype=torch.int32)
    phone_len_tensor = torch.tensor([len(phone_indices)], dtype=torch.int32)

    return {
        "phone": phone_tensor,
        "tone": tone_tensor,
        "lang": lang_tensor,
        "word_pos": word_pos_tensor,
        "syllable_pos": syllable_pos_tensor,
        "phone_len": phone_len_tensor,
    }


def main():
    parser = argparse.ArgumentParser(description="JyutVoice TTS Inference")
    parser.add_argument(
        "--model", required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", required=True, help="Output audio file path")
    parser.add_argument(
        "--ref_audio",
        default="tmp/ref123.wav",
        help="Reference audio for speaker embedding",
    )
    parser.add_argument(
        "--config", default="configs/base.yaml", help="Configuration file path"
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    with open(args.config, encoding="utf-8") as f:
        cfg = load_hyperpyyaml(f)

    # Load model
    print(f"Loading model from {args.model}...")
    model = cfg["tts"]
    model.load_state_dict(
        torch.load(args.model, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    # Load HiFT generator
    print("Loading HiFT generator...")
    hift = cfg["hift"]
    hift.load_state_dict(
        torch.load(
            "pretrained_models/hift.pt",
            map_location=device,
            weights_only=True,
        )
    )
    hift.to(device)
    hift.eval()

    # Process text
    print(f"Processing text: {args.text}")
    text_features = process_text(args.text)

    # Load reference audio
    print(f"Loading reference audio from {args.ref_audio}...")
    speaker_embedding = load_reference_audio(args.ref_audio, device)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        # Move text features to device
        for key in text_features:
            text_features[key] = text_features[key].to(device)

        outputs = model.synthesise(
            x=text_features["phone"],
            x_lengths=text_features["phone_len"],
            lang=text_features["lang"],
            tone=text_features["tone"],
            word_pos=text_features["word_pos"],
            syllable_pos=text_features["syllable_pos"],
            spk_embed=speaker_embedding,
            n_timesteps=10,
            temperature=1.0,
            length_scale=1.0,
        )

        # Extract mel spectrogram
        mel = outputs["mel"]  # Shape: [1, 80, T]

        # Convert mel to wav
        speech, _ = hift.inference(mel)

    # Save audio
    print(f"Saving audio to {args.output}...")
    torchaudio.save(args.output, speech.cpu(), 24000)

    print("✅ Inference completed successfully!")
    print(f"Generated audio saved to: {args.output}")


if __name__ == "__main__":
    main()
