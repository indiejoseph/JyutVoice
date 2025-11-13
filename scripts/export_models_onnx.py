#!/usr/bin/env python3
"""
Export JyutVoice models to ONNX format

This script loads the JyutVoice models and exports them to ONNX format for inference.

Usage:
    python scripts/export_models_onnx.py
"""

import torch
from torch.nn import functional as F
import torch.onnx
from jyutvoice.transformer.upsample_encoder import UpsampleConformerEncoder
from hyperpyyaml import load_hyperpyyaml


class FlowEncoder(torch.nn.Module):
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
        # Create mask manually for ONNX compatibility
        max_len = token.size(1)
        mask = torch.arange(max_len, device=token.device).unsqueeze(
            0
        ) < token_len.unsqueeze(1)
        mask = mask.float().unsqueeze(-1)

        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # Encode
        h, h_lengths = self.encoder(token, token_len, streaming=False)
        # Project to output size (80)
        h = self.encoder_proj(h)

        return h, h_lengths


class TextEncoderStyleEncoder(torch.nn.Module):
    def __init__(self, encoder, style_encoder, spk_embed_affine_layer, style_proj):
        super().__init__()
        self.encoder = encoder
        self.style_encoder = style_encoder
        self.spk_embed_affine_layer = spk_embed_affine_layer
        self.style_proj = style_proj

    def forward(
        self, x, x_lengths, lang, tone, word_pos, syllable_pos, spk_embed, prompt_feat
    ):
        """
        Encode text and style.

        Args:
            x: text tokens (batch, text_len)
            x_lengths: text lengths (batch,)
            lang, tone, word_pos, syllable_pos: linguistic features (batch, text_len)
            spk_embed: speaker embedding (batch, spk_dim)
            prompt_feat: prompt mel (batch, n_mel, prompt_len)

        Returns:
            mu_x: encoded text (batch, n_feats, text_len)
            logw: log durations (batch, 1, text_len)
            x_mask: text mask (batch, 1, text_len)
            style_cond: style conditioning (batch, n_feats)
            spk_embed_proj: projected speaker embedding (batch, n_feats)
        """
        # Project speaker embedding
        spk_embed = F.normalize(spk_embed, dim=1)
        spk_embed_proj = self.spk_embed_affine_layer(spk_embed)

        # Style encoding
        style_emb = self.style_encoder(prompt_feat)  # (B, gst_token_dim)
        style_cond = self.style_proj(style_emb)  # (B, n_feats)

        # Text encoding
        mu_x, logw, x_mask = self.encoder(
            x, x_lengths, lang, tone, word_pos, syllable_pos, spk_embed_proj
        )
        mu_x = mu_x + style_cond.unsqueeze(2)  # (B, n_feats, T_text)

        return mu_x, logw, x_mask, style_cond, spk_embed_proj


class FlowDecoder(torch.nn.Module):
    def __init__(self, decoder, output_size=80):
        super().__init__()
        self.decoder = decoder
        self.output_size = output_size

    def forward(self, mu, mask, spks, cond, n_timesteps, temperature):
        """
        Flow decoder forward.

        Args:
            mu: mean (batch, n_feats, mel_len)
            mask: mask (batch, 1, mel_len)
            spks: speaker embedding (batch, n_feats)
            cond: conditioning (batch, n_feats, mel_len)
            n_timesteps: int
            temperature: float

        Returns:
            decoder_outputs: (batch, n_feats, mel_len)
        """
        decoder_outputs, _ = self.decoder(
            mu=mu,
            mask=mask,
            spks=spks,
            cond=cond,
            n_timesteps=n_timesteps,
            temperature=temperature,
            streaming=False,
        )
        return decoder_outputs


def main():
    """Export JyutVoice models to ONNX."""
    device = "cpu"  # Use CPU for ONNX export

    # Load TTS config
    hyper_yaml_path = "configs/base.yaml"
    with open(hyper_yaml_path, "r") as f:
        configs = load_hyperpyyaml(f)

    # Load TTS model
    tts = configs["tts"].to(device)
    ckpt_path = "pretrained_models/epoch=0-step=55872.ckpt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    tts.load_state_dict(state_dict)
    tts.eval()

    # Export FlowEncoder
    print("Exporting FlowEncoder...")
    flow_encoder = FlowEncoder().to(device)
    flow_encoder_path = "pretrained_models/flow_encoder.pt"
    state_dict = torch.load(flow_encoder_path, map_location=device, weights_only=True)
    flow_encoder.load_state_dict(state_dict)
    flow_encoder.eval()

    batch_size = 1
    seq_len = 100
    vocab_size = 6561
    dummy_token = torch.randint(
        0, vocab_size, (batch_size, seq_len), dtype=torch.long
    ).to(device)
    dummy_token_len = torch.tensor([seq_len], dtype=torch.long).to(device)

    torch.onnx.export(
        flow_encoder,
        (dummy_token, dummy_token_len),
        "pretrained_models/flow_encoder.onnx",
        input_names=["token", "token_len"],
        output_names=["h", "h_lengths"],
        dynamic_axes={
            "token": {0: "batch_size", 1: "seq_len"},
            "token_len": {0: "batch_size"},
            "h": {0: "batch_size", 1: "seq_len"},
            "h_lengths": {0: "batch_size"},
        },
        opset_version=14,
        verbose=False,
    )
    print("✅ FlowEncoder exported to pretrained_models/flow_encoder.onnx")

    # Export TextEncoderStyleEncoder
    print("Exporting TextEncoderStyleEncoder...")
    text_style_encoder = TextEncoderStyleEncoder(
        tts.encoder, tts.style_encoder, tts.spk_embed_affine_layer, tts.style_proj
    ).to(device)
    text_style_encoder.eval()

    # Dummy inputs
    max_text_len = 50
    spk_dim = 192
    n_mel = 80
    prompt_len = 50
    dummy_x = torch.randint(0, 97, (batch_size, max_text_len), dtype=torch.long).to(
        device
    )
    dummy_x_lengths = torch.tensor([max_text_len], dtype=torch.long).to(device)
    dummy_lang = torch.randint(0, 3, (batch_size, max_text_len), dtype=torch.long).to(
        device
    )
    dummy_tone = torch.randint(0, 7, (batch_size, max_text_len), dtype=torch.long).to(
        device
    )
    dummy_word_pos = torch.randint(
        0, 3, (batch_size, max_text_len), dtype=torch.long
    ).to(device)
    dummy_syllable_pos = torch.randint(
        0, 3, (batch_size, max_text_len), dtype=torch.long
    ).to(device)
    dummy_spk_embed = torch.randn(batch_size, spk_dim).to(device)
    dummy_prompt_feat = torch.randn(batch_size, prompt_len, n_mel).to(device)

    torch.onnx.export(
        text_style_encoder,
        (
            dummy_x,
            dummy_x_lengths,
            dummy_lang,
            dummy_tone,
            dummy_word_pos,
            dummy_syllable_pos,
            dummy_spk_embed,
            dummy_prompt_feat,
        ),
        "pretrained_models/text_style_encoder.onnx",
        input_names=[
            "x",
            "x_lengths",
            "lang",
            "tone",
            "word_pos",
            "syllable_pos",
            "spk_embed",
            "prompt_feat",
        ],
        output_names=["mu_x", "logw", "x_mask", "style_cond", "spk_embed_proj"],
        dynamic_axes={
            "x": {0: "batch_size", 1: "text_len"},
            "x_lengths": {0: "batch_size"},
            "lang": {0: "batch_size", 1: "text_len"},
            "tone": {0: "batch_size", 1: "text_len"},
            "word_pos": {0: "batch_size", 1: "text_len"},
            "syllable_pos": {0: "batch_size", 1: "text_len"},
            "spk_embed": {0: "batch_size"},
            "prompt_feat": {0: "batch_size", 1: "prompt_len"},
            "mu_x": {0: "batch_size", 2: "text_len"},
            "logw": {0: "batch_size", 2: "text_len"},
            "x_mask": {0: "batch_size", 2: "text_len"},
            "style_cond": {0: "batch_size"},
            "spk_embed_proj": {0: "batch_size"},
        },
        opset_version=14,
        verbose=False,
    )
    print(
        "✅ TextEncoderStyleEncoder exported to pretrained_models/text_style_encoder.onnx"
    )

    # Export FlowDecoder
    print("Exporting FlowDecoder...")
    flow_decoder = FlowDecoder(tts.decoder, tts.output_size).to(device)
    flow_decoder.eval()

    # Dummy inputs for decoder
    mel_len = 100
    dummy_mu = torch.randn(batch_size, tts.n_feats, mel_len).to(device)
    dummy_mask = torch.ones(batch_size, 1, mel_len, dtype=torch.float).to(device)
    dummy_spks = torch.randn(batch_size, tts.output_size).to(device)
    dummy_cond = torch.randn(batch_size, tts.output_size, mel_len).to(device)
    dummy_n_timesteps = torch.tensor(10, dtype=torch.long).to(device)
    dummy_temperature = torch.tensor(1.0, dtype=torch.float).to(device)

    torch.onnx.export(
        flow_decoder,
        (
            dummy_mu,
            dummy_mask,
            dummy_spks,
            dummy_cond,
            dummy_n_timesteps,
            dummy_temperature,
        ),
        "pretrained_models/flow_decoder.onnx",
        input_names=["mu", "mask", "spks", "cond", "n_timesteps", "temperature"],
        output_names=["decoder_outputs"],
        dynamic_axes={
            "mu": {0: "batch_size", 2: "mel_len"},
            "mask": {0: "batch_size", 2: "mel_len"},
            "spks": {0: "batch_size"},
            "cond": {0: "batch_size", 2: "mel_len"},
            "decoder_outputs": {0: "batch_size", 2: "mel_len"},
        },
        opset_version=14,
        verbose=False,
    )
    print("✅ FlowDecoder exported to pretrained_models/flow_decoder.onnx")

    print("🎉 All models exported successfully!")


if __name__ == "__main__":
    main()
