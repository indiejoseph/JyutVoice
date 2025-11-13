#!/usr/bin/env python3
"""
Download Pretrained Weights for JyutVoice

This script downloads the CosyVoice2 flow encoder, decoder, and HiFT model weights from HuggingFace
and saves them as separate files for use in JyutVoice.

Usage:
    python scripts/download_pretrain_weights.py
"""

import os
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


def main():
    """Main function to download and extract weights."""
    # Configuration
    HIFT_URL = "https://huggingface.co/lucyknada/CosyVoice2-0.5B/resolve/main/hift.pt"
    FLOW_ENCODER_URL = "https://huggingface.co/lucyknada/CosyVoice2-0.5B/resolve/main/flow.encoder.fp32.zip"
    FLOW_DECODER_URL = "https://huggingface.co/lucyknada/CosyVoice2-0.5B/resolve/main/flow.decoder.estimator.fp32.onnx"
    HIFT_CHECKPOINT = "hift.pt"
    ENCODER_ZIP = "flow_encoder.zip"
    OUTPUT_DIR = "pretrained_models"

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    # Download the flow encoder zip if it doesn't exist
    encoder_zip_path = os.path.join(OUTPUT_DIR, ENCODER_ZIP)
    if not os.path.exists(encoder_zip_path):
        try:
            download_file(FLOW_ENCODER_URL, encoder_zip_path)
        except (requests.RequestException, OSError) as e:
            print(f"❌ Failed to download flow encoder zip: {e}")
            return
    else:
        print(f"📁 Flow encoder zip already exists at {encoder_zip_path}")

    # Download the flow decoder if it doesn't exist
    decoder_path = os.path.join(OUTPUT_DIR, "flow_decoder.onnx")
    if not os.path.exists(decoder_path):
        try:
            download_file(FLOW_DECODER_URL, decoder_path)
        except (requests.RequestException, OSError) as e:
            print(f"❌ Failed to download flow decoder: {e}")
            return
    else:
        print(f"📁 Flow decoder already exists at {decoder_path}")

    print("\n🎉 All operations completed successfully!")
    print(f"   Flow encoder weights: {encoder_zip_path}")
    print(f"   Flow decoder weights: {decoder_path}")
    print(f"   HiFT weights: {os.path.join(OUTPUT_DIR, 'hift.pt')}")


if __name__ == "__main__":
    main()
