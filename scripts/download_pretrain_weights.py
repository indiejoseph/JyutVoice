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
    FLOW_CHECKPOINT = "flow.pt"
    HIFT_CHECKPOINT = "hift.pt"
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

    # Extract flow weights
    try:
        extract_flow_weights(flow_path, OUTPUT_DIR)
        print("\n🎉 All operations completed successfully!")
        print(f"   Flow encoder weights: {os.path.join(OUTPUT_DIR, 'flow_encoder.pt')}")
        print(f"   Flow decoder weights: {os.path.join(OUTPUT_DIR, 'flow_decoder.pt')}")
        print(f"   HiFT weights: {os.path.join(OUTPUT_DIR, 'hift.pt')}")
    except (OSError, torch.serialization.pickle.UnpicklingError, KeyError) as e:
        print(f"❌ Failed to extract weights: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
