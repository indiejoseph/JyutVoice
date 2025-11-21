#!/usr/bin/env python3
"""
Download and Extract Pretrained Weights for JyutVoice

This script downloads the CosyVoice2 flow and HiFT model weights from HuggingFace,
extracts the encoder and decoder components from the flow model, and saves them as
separate checkpoint files for use in JyutVoice.

Usage:
    python scripts/download_pretrain_weights.py
"""

import os
import torch
import requests
from hyperpyyaml import load_hyperpyyaml


def download_file(url: str, destination: str, chunk_size: int = 8192) -> None:
    """
    Download a file from URL with progress bar.

    Args:
        url: URL to download from
        destination: Local path to save the file
        chunk_size: Size of chunks to download
    """
    print(f"📥 Downloading from {url}...")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    downloaded = 0
    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(
                        f"\r📥 Downloading... {progress:.1f}% ({downloaded}/{total_size} bytes)",
                        end="",
                        flush=True,
                    )

    print(f"\n✅ Downloaded to {destination}")


def prepare_pretrain_weights():
    print("📁 Loading configuration from configs/base.yaml...")
    with open("configs/base.yaml") as f:
        cfg = load_hyperpyyaml(f)

    print("✅ Configuration loaded!")

    # Instantiate the model
    print("\n🧠 Instantiating JyutVoiceTTS model...")
    model = cfg["tts"]
    model.eval()

    print(f"✅ Model instantiated successfully!")
    print(f"Model architecture:")
    print(f"  Text Encoder: {type(model.encoder).__name__}")
    print(f"  Decoder (frozen): {type(model.decoder).__name__}")
    print(f"  Speaker Embed Affine: {type(model.spk_embed_affine_layer).__name__}")

    import torch

    # Load CosyVoice2 flow decoder pretrained weights and transfer to JyutVoiceTTS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        "\n📦 Loading CosyVoice2 flow decoder weights from: pretrained_models/flow_decoder.pt"
    )
    decoder_weights = torch.load(
        "pretrained_models/flow_decoder.pt", map_location=device
    )

    print(f"✅ Loaded {len(decoder_weights)} weight entries from flow_decoder.pt")
    print(f"Decoder weight keys (first 5): {list(decoder_weights.keys())[:5]}")

    # Load the weights into the model
    print("\n⚙️  Loading decoder weights into JyutVoiceTTS model...")
    model.load_state_dict(decoder_weights, strict=False)

    print("✅ Decoder weights loaded successfully!")

    # Save the model as pretrain checkpoint
    print("\n💾 Saving pretrained model to: pretrained_models/pretrain.pt")
    torch.save(model.state_dict(), "pretrained_models/pretrain.pt")

    print("✅ Pretrained model saved successfully!")
    print(f"\n📊 Model checkpoint summary:")
    print(f"  Total parameters saved: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(
        f"  Frozen parameters: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}"
    )
    print(f"\n✨ Pretrained checkpoint ready for transfer learning training!")
    print(f"   Use: model.load_state_dict(torch.load('pretrained_models/pretrain.pt'))")

    # Double check: Compare pretrained.pt with flow_decoder.pt
    print("🔍 Verifying pretrained checkpoint against flow_decoder.pt...\n")

    # Load both checkpoints
    pretrain_weights = torch.load("pretrained_models/pretrain.pt", map_location="cpu")
    decoder_weights_original = torch.load(
        "pretrained_models/flow_decoder.pt", map_location="cpu"
    )

    print(f"📊 Pretrain checkpoint:")
    print(f"  Total keys: {len(pretrain_weights)}")
    print(f"  Keys (first 10): {list(pretrain_weights.keys())[:10]}\n")

    print(f"📊 Flow decoder checkpoint:")
    print(f"  Total keys: {len(decoder_weights_original)}")
    print(f"  Keys (first 10): {list(decoder_weights_original.keys())[:10]}\n")

    # Check if decoder weights are in pretrain
    decoder_keys_in_pretrain = 0
    for key in decoder_weights_original.keys():
        if key in pretrain_weights:
            decoder_keys_in_pretrain += 1
            # Verify the weights are identical
            if torch.allclose(
                pretrain_weights[key], decoder_weights_original[key], atol=1e-6
            ):
                status = "✅ MATCH"
            else:
                status = "⚠️  DIFFERENT"
            # Print sample keys
            if decoder_keys_in_pretrain <= 3:
                shape_pretrain = pretrain_weights[key].shape
                shape_decoder = decoder_weights_original[key].shape
                print(f"{status} | {key}")
                print(
                    f"       Pretrain shape: {shape_pretrain}, Decoder shape: {shape_decoder}"
                )

    print(
        f"\n✅ Decoder keys found in pretrain: {decoder_keys_in_pretrain}/{len(decoder_weights_original)}"
    )

    # Check for additional keys in pretrain (should be text encoder and other components)
    additional_keys = set(pretrain_weights.keys()) - set(
        decoder_weights_original.keys()
    )
    print(
        f"📝 Additional keys in pretrain (text encoder + others): {len(additional_keys)}"
    )
    if len(additional_keys) > 0:
        print(f"   Examples: {list(additional_keys)[:5]}")

    print(f"\n✨ Verification complete!")
    if decoder_keys_in_pretrain == len(decoder_weights_original):
        print(f"   ✅ All decoder weights correctly transferred to pretrain.pt")
    else:
        print(
            f"   ⚠️  WARNING: Only {decoder_keys_in_pretrain}/{len(decoder_weights_original)} decoder weights found!"
        )


def extract_flow_weights(flow_checkpoint_path: str, output_dir: str) -> None:
    """
    Extract encoder and decoder weights from flow checkpoint.

    Args:
        flow_checkpoint_path: Path to the flow.pt checkpoint
        output_dir: Directory to save extracted weights
    """
    print(f"🔍 Loading flow checkpoint from {flow_checkpoint_path}...")

    # Load the state dict
    state_dict = torch.load(flow_checkpoint_path, map_location="cpu", weights_only=True)
    print(f"✅ Loaded checkpoint with {len(state_dict)} weight entries")

    # Extract encoder weights
    print("🔧 Extracting encoder weights...")
    encoder_state_dict = {
        k: v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
        or k.startswith("input_embedding.")
        or k.startswith("encoder_proj.")
    }
    print(f"   Found {len(encoder_state_dict)} encoder weights")

    # Extract decoder weights
    print("🔧 Extracting decoder weights...")
    decoder_state_dict = {
        k: v
        for k, v in state_dict.items()
        if k.startswith("decoder.") or k.startswith("spk_embed_affine_layer.")
    }
    print(f"   Found {len(decoder_state_dict)} decoder weights")

    # Save extracted weights
    os.makedirs(output_dir, exist_ok=True)

    encoder_path = os.path.join(output_dir, "flow_encoder.pt")
    decoder_path = os.path.join(output_dir, "flow_decoder.pt")

    print(f"💾 Saving encoder weights to {encoder_path}...")
    torch.save(encoder_state_dict, encoder_path)

    print(f"💾 Saving decoder weights to {decoder_path}...")
    torch.save(decoder_state_dict, decoder_path)

    print("✅ Flow weight extraction complete!")


def main():
    """Main function to download and extract weights."""
    # Configuration
    FLOW_URL = "https://huggingface.co/lucyknada/CosyVoice2-0.5B/resolve/main/flow.pt"
    HIFT_URL = "https://huggingface.co/lucyknada/CosyVoice2-0.5B/resolve/main/hift.pt"
    CAMPPLUS_URL = (
        "https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/resolve/main/campplus.onnx"
    )
    SPEECH_TOKENIZER_URL = "https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/resolve/main/speech_tokenizer_v2.onnx"
    FLOW_CHECKPOINT = "flow.pt"
    HIFT_CHECKPOINT = "hift.pt"
    CAMPPLUS_CHECKPOINT = "campplus.onnx"
    SPEECH_TOKENIZER_CHECKPOINT = "speech_tokenizer_v2.onnx"
    OUTPUT_DIR = "pretrained_models"

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download the flow checkpoint if it doesn't exist
    flow_path = os.path.join(OUTPUT_DIR, FLOW_CHECKPOINT)
    if not os.path.exists(flow_path):
        try:
            download_file(FLOW_URL, flow_path)
        except (requests.RequestException, OSError) as e:
            print(f"❌ Failed to download flow checkpoint: {e}")
            return
    else:
        print(f"📁 Flow checkpoint already exists at {flow_path}")

    # Download the HiFT checkpoint if it doesn't exist
    hift_path = os.path.join(OUTPUT_DIR, HIFT_CHECKPOINT)
    if not os.path.exists(hift_path):
        try:
            download_file(HIFT_URL, hift_path)
        except (requests.RequestException, OSError) as e:
            print(f"❌ Failed to download HiFT checkpoint: {e}")
            return
    else:
        print(f"📁 HiFT checkpoint already exists at {hift_path}")

    # Download the campplus checkpoint if it doesn't exist
    campplus_path = os.path.join(OUTPUT_DIR, CAMPPLUS_CHECKPOINT)
    if not os.path.exists(campplus_path):
        try:
            download_file(CAMPPLUS_URL, campplus_path)
        except (requests.RequestException, OSError) as e:
            print(f"❌ Failed to download campplus checkpoint: {e}")
            return
    else:
        print(f"📁 Campplus checkpoint already exists at {campplus_path}")

    # Download the speech tokenizer checkpoint if it doesn't exist
    speech_tokenizer_path = os.path.join(OUTPUT_DIR, SPEECH_TOKENIZER_CHECKPOINT)
    if not os.path.exists(speech_tokenizer_path):
        try:
            download_file(SPEECH_TOKENIZER_URL, speech_tokenizer_path)
        except (requests.RequestException, OSError) as e:
            print(f"❌ Failed to download speech tokenizer checkpoint: {e}")
            return
    else:
        print(
            f"📁 Speech tokenizer checkpoint already exists at {speech_tokenizer_path}"
        )

    # Extract flow weights
    try:
        extract_flow_weights(flow_path, OUTPUT_DIR)
        print("\n🎉 All operations completed successfully!")
        print(f"   Flow encoder weights: {os.path.join(OUTPUT_DIR, 'flow_encoder.pt')}")
        print(f"   Flow decoder weights: {os.path.join(OUTPUT_DIR, 'flow_decoder.pt')}")
        print(f"   HiFT weights: {os.path.join(OUTPUT_DIR, 'hift.pt')}")
        print(f"   Campplus weights: {os.path.join(OUTPUT_DIR, 'campplus.onnx')}")
        print(
            f"   Speech tokenizer: {os.path.join(OUTPUT_DIR, 'speech_tokenizer_v2.onnx')}"
        )
    except (OSError, torch.serialization.pickle.UnpicklingError, KeyError) as e:
        print(f"❌ Failed to extract weights: {e}")
        import traceback

        traceback.print_exc()

    prepare_pretrain_weights()


if __name__ == "__main__":
    main()
