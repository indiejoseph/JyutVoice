#!/usr/bin/env python3
"""
ONNX Export Script for JyutVoice TTS Components

Exports the following components to ONNX format:
- Text Encoder: Converts phoneme sequences to mel-spectrogram features
- Duration Predictor: Predicts phoneme durations

Usage:
    python scripts/export_onnx.py --config configs/onnx.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort
from hyperpyyaml import load_hyperpyyaml
from jyutvoice.utils.common import convert_onnx_to_trt


class TextEncoderWrapper(torch.nn.Module):
    """Wrapper for text encoder to make it ONNX-exportable"""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, x_lengths, lang, tone, word_pos, syllable_pos, spk_embed):
        """
        Forward pass for text encoder.

        Args:
            x: Phoneme token IDs (B, T)
            x_lengths: Sequence lengths (B,)
            lang: Language IDs (B, T)
            tone: Tone IDs (B, T)
            word_pos: Word position IDs (B, T)
            syllable_pos: Syllable position IDs (B, T)
            spk_embed: Speaker embeddings (B, spk_dim)

        Returns:
            encoder_output: Encoded features (B, n_feats, T)
            mu_x: Mean features (B, n_feats, T)
            x_mask: Attention mask (B, 1, T)
        """
        return self.encoder(x, x_lengths, lang, tone, word_pos, syllable_pos, spk_embed)


class DurationPredictorWrapper(torch.nn.Module):
    """Wrapper for duration predictor to make it ONNX-exportable"""

    def __init__(self, dp):
        super().__init__()
        self.dp = dp

    def forward(self, x, x_mask):
        """
        Forward pass for duration predictor.

        Args:
            x: Encoder outputs (B, n_feats, T)
            x_mask: Attention mask (B, 1, T)

        Returns:
            log_durations: Log-scaled durations (B, 1, T)
        """
        return self.dp(x, x_mask)


def load_model(checkpoint_path, base_config_path):
    """Load TTS model from checkpoint"""
    print(f"Loading base config from {base_config_path}...")
    with open(base_config_path, "r") as f:
        configs = load_hyperpyyaml(f)

    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    tts = configs["tts"]
    tts.load_state_dict(state_dict, strict=False)
    tts.eval()

    return tts, configs


def create_example_inputs(config, device="cpu"):
    """Create example inputs for ONNX export"""
    example_cfg = config["example_inputs"]

    batch_size = example_cfg["batch_size"]
    seq_len = example_cfg["seq_len"]
    n_vocab = example_cfg["n_vocab"]
    spk_embed_dim = example_cfg["spk_embed_dim"]

    # Create dummy inputs
    x = torch.randint(
        0, n_vocab, (batch_size, seq_len), dtype=torch.long, device=device
    )
    x_lengths = torch.tensor([seq_len] * batch_size, dtype=torch.long, device=device)
    lang = torch.randint(0, 4, (batch_size, seq_len), dtype=torch.long, device=device)
    tone = torch.randint(0, 7, (batch_size, seq_len), dtype=torch.long, device=device)
    word_pos = torch.randint(
        0, 3, (batch_size, seq_len), dtype=torch.long, device=device
    )
    syllable_pos = torch.randint(
        0, 3, (batch_size, seq_len), dtype=torch.long, device=device
    )
    spk_embed = torch.randn(batch_size, spk_embed_dim, device=device)

    return {
        "x": x,
        "x_lengths": x_lengths,
        "lang": lang,
        "tone": tone,
        "word_pos": word_pos,
        "syllable_pos": syllable_pos,
        "spk_embed": spk_embed,
    }


def export_text_encoder(tts, config, output_path, device="cpu"):
    """Export text encoder to ONNX"""
    print("\n" + "=" * 70)
    print("Exporting Text Encoder to ONNX")
    print("=" * 70)

    # Wrap encoder
    encoder_wrapper = TextEncoderWrapper(tts.encoder).to(device)
    encoder_wrapper.eval()

    # Create example inputs
    example_inputs = create_example_inputs(config, device)

    # Input names
    input_names = [
        "x",
        "x_lengths",
        "lang",
        "tone",
        "word_pos",
        "syllable_pos",
        "spk_embed",
    ]
    output_names = ["encoder_output", "mu_x", "x_mask"]

    # Export to ONNX
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        encoder_wrapper,
        (
            example_inputs["x"],
            example_inputs["x_lengths"],
            example_inputs["lang"],
            example_inputs["tone"],
            example_inputs["word_pos"],
            example_inputs["syllable_pos"],
            example_inputs["spk_embed"],
        ),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=config.get("dynamic_axes", {}),
        opset_version=config.get("opset_version", 17),
        do_constant_folding=True,
        export_params=True,
    )

    print(f"✓ Text encoder exported to {output_path}")
    return output_path


def export_duration_predictor(tts, config, output_path, device="cpu"):
    """Export duration predictor to ONNX"""
    print("\n" + "=" * 70)
    print("Exporting Duration Predictor to ONNX")
    print("=" * 70)

    # Wrap DP
    dp_wrapper = DurationPredictorWrapper(tts.dp).to(device)
    dp_wrapper.eval()

    # Create example inputs (need encoder outputs)
    example_inputs = create_example_inputs(config, device)

    with torch.no_grad():
        encoder_output, _, x_mask = tts.encoder(
            example_inputs["x"],
            example_inputs["x_lengths"],
            example_inputs["lang"],
            example_inputs["tone"],
            example_inputs["word_pos"],
            example_inputs["syllable_pos"],
            example_inputs["spk_embed"],
        )

    # Input names
    input_names = ["encoder_output", "x_mask"]
    output_names = ["log_durations"]

    # Export to ONNX
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        dp_wrapper,
        (encoder_output, x_mask),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=config.get("dynamic_axes", {}),
        opset_version=config.get("opset_version", 17),
        do_constant_folding=True,
        export_params=True,
    )

    print(f"✓ Duration predictor exported to {output_path}")
    return output_path


def validate_onnx_model(
    onnx_path, pytorch_model, example_inputs, model_type="text_encoder"
):
    """Validate ONNX model by comparing outputs with PyTorch model"""
    print(f"\nValidating ONNX model: {onnx_path}")

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model structure is valid")

    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Prepare inputs
    ort_inputs = {}
    if model_type == "text_encoder":
        ort_inputs = {
            "x": example_inputs["x"].cpu().numpy(),
            "x_lengths": example_inputs["x_lengths"].cpu().numpy(),
            "lang": example_inputs["lang"].cpu().numpy(),
            "tone": example_inputs["tone"].cpu().numpy(),
            "word_pos": example_inputs["word_pos"].cpu().numpy(),
            "syllable_pos": example_inputs["syllable_pos"].cpu().numpy(),
            "spk_embed": example_inputs["spk_embed"].cpu().numpy(),
        }

        # PyTorch inference
        with torch.no_grad():
            pt_outputs = pytorch_model(
                example_inputs["x"],
                example_inputs["x_lengths"],
                example_inputs["lang"],
                example_inputs["tone"],
                example_inputs["word_pos"],
                example_inputs["syllable_pos"],
                example_inputs["spk_embed"],
            )

    # ONNX inference
    ort_outputs = session.run(None, ort_inputs)

    # Compare outputs
    print("Comparing PyTorch vs ONNX outputs:")
    for i, (pt_out, ort_out) in enumerate(zip(pt_outputs, ort_outputs)):
        pt_np = pt_out.cpu().numpy()
        max_diff = np.abs(pt_np - ort_out).max()
        mean_diff = np.abs(pt_np - ort_out).mean()
        print(f"  Output {i}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        if max_diff > 1e-3:
            print(f"    ⚠️  Warning: Large difference detected!")
        else:
            print(f"    ✓ Outputs match within tolerance")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export JyutVoice TTS components to ONNX"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/export_onnx.yaml",
        help="Path to ONNX export config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for export (cpu or cuda)",
    )
    args = parser.parse_args()

    # Load config
    print(f"Loading export config from {args.config}...")
    with open(args.config, "r") as f:
        config = load_hyperpyyaml(f)

    # Create output directory
    output_dir = Path(config["export"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    tts, base_configs = load_model(config["checkpoint_path"], config["base_config"])

    device = torch.device(args.device)
    tts = tts.to(device)

    # Export text encoder
    if config["export"]["text_encoder"]["enabled"]:
        encoder_path = output_dir / config["export"]["text_encoder"]["filename"]
        encoder_config = config["export"]["text_encoder"]

        export_text_encoder(tts, encoder_config, str(encoder_path), device)

        # Validate
        if config["validation"]["test_inference"]:
            example_inputs = create_example_inputs(encoder_config, device)
            encoder_wrapper = TextEncoderWrapper(tts.encoder).to(device)
            validate_onnx_model(
                str(encoder_path), encoder_wrapper, example_inputs, "text_encoder"
            )

    # Export duration predictor
    if config["export"]["duration_predictor"]["enabled"]:
        dp_path = output_dir / config["export"]["duration_predictor"]["filename"]
        dp_config = config["export"]["duration_predictor"]

        dp_onnx_path = export_duration_predictor(tts, dp_config, str(dp_path), device)

    # Export flow decoder estimator to TensorRT
    min_shape = [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)]
    opt_shape = [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)]
    max_shape = [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)]
    input_names = ["x", "mask", "mu", "cond"]
    trt_kwargs = {
        "min_shape": min_shape,
        "opt_shape": opt_shape,
        "max_shape": max_shape,
        "input_names": input_names,
    }
    fp16 = True
    flow_decoder_estimator_model = (
        "pretrained_models/flow.decoder.estimator.fp16.mygpu.plan"
    )
    flow_decoder_onnx_model = "pretrained_models/flow.decoder.estimator.fp32.onnx"

    convert_onnx_to_trt(
        flow_decoder_estimator_model, trt_kwargs(), flow_decoder_onnx_model, fp16
    )

    print("\n" + "=" * 70)
    print("ONNX Export Complete!")
    print("=" * 70)
    print(f"\nExported models saved to: {output_dir}")
    print("\nYou can now use these ONNX models for inference with ONNXRuntime:")
    print(
        f"  - Text Encoder: {output_dir / config['export']['text_encoder']['filename']}"
    )
    if config["export"]["duration_predictor"]["enabled"]:
        print(
            f"  - Duration Predictor: {output_dir / config['export']['duration_predictor']['filename']}"
        )


if __name__ == "__main__":
    main()
