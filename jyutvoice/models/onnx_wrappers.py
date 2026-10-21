"""The parts of JyutVoice that run in ONNX Runtime, as modules `torch.onnx.export` can trace (scripts/export_onnx.py), and the numpy side that drives them
(jyutvoice/utils/onnx_synth.py).

Four graphs, every one with batch and sequence length dynamic:

    TextEncoderWrapper          token ids + positions + speaker -> encoder_output, mu_x, x_mask
    DurationPredictorWrapper    encoder_output, x_mask -> log durations (the regression predictor: deterministic)
    FlowDurationSamplerWrapper  encoder_output, x_mask, speaker, noise -> log durations (the flow-matching predictor: its Euler loop is unrolled in the graph)
    ProsodyCFMSamplerWrapper    mu_y, h_text_y, tau, dur, y_mask, speaker, noise -> the decoder's input features (ProsodyCFM's Euler loop, unrolled)

Random numbers are inputs of the sampler graphs, not operators inside them: the caller draws the noise (already scaled by the temperature, which may differ
per token) so a render can be seeded and the graphs compared with PyTorch on identical noise. The expansion of token features to frames (the alignment
`generate_path` builds) is the one step left to the caller: its output length depends on the predicted durations.
"""
import copy

import torch
import torch.nn as nn


def _ramp(x_like: torch.Tensor, value: float) -> torch.Tensor:
    """(B,) tensor filled with `value`, the batch size taken from `x_like` so that the traced graph keeps a dynamic batch."""
    return torch.full_like(x_like[:, 0, 0], value)


@torch.no_grad()
def warm_rotary_caches(encoder, max_tokens: int):
    """Build the encoder's rotary position tables up to `max_tokens` tokens. They are built lazily to the longest sequence seen so far and `torch.onnx.export` bakes them
    in as they are: a graph traced on a 93-token sentence fails on a 97-token one (the tables are sliced to the input length, never extended)."""
    was_training = encoder.training
    encoder.eval()
    ids = torch.zeros(1, max_tokens, dtype=torch.long)
    encoder(ids, torch.tensor([max_tokens]), ids, ids, ids, ids, torch.zeros(1, encoder.gin_channels))
    encoder.train(was_training)


class TextEncoderWrapper(nn.Module):
    def __init__(self, encoder, max_tokens: int = 1024):
        """`max_tokens`: the longest token sequence (blanks included) the exported graph accepts."""
        super().__init__()
        if getattr(encoder, "bert_dim", 0) > 0:
            raise ValueError("an encoder with BERT features (bert_dim > 0) needs the BERT model next to it: not exportable by this wrapper")
        self.encoder = encoder
        self.max_tokens = max_tokens
        warm_rotary_caches(encoder, max_tokens)

    def forward(self, x, x_lengths, lang, tone, word_pos, syllable_pos, spk_embed):
        """x, lang, tone, word_pos, syllable_pos: (B, T) int64; x_lengths: (B,) int64; spk_embed: (B, spk_dim), L2-normalised.
        Returns encoder_output (B, n_channels, T), mu_x (B, n_feats, T), x_mask (B, 1, T)."""
        return self.encoder(x, x_lengths, lang, tone, word_pos, syllable_pos, spk_embed)


class DurationPredictorWrapper(nn.Module):
    """The regression duration predictor (`DurationPredictor`)."""

    def __init__(self, dp):
        super().__init__()
        self.dp = dp

    def forward(self, encoder_output, x_mask):
        """-> log durations (B, 1, T)."""
        return self.dp(encoder_output, x_mask)


class FlowDurationSamplerWrapper(nn.Module):
    """`FlowDurationPredictor.forward` without guidance (cfg_scale 0, the setting of every model trained so far) and with the noise as an input."""

    def __init__(self, dp, n_steps: int = 10):
        super().__init__()
        if dp.infer_cfg_scale != 0.0:
            raise ValueError(f"the sampler graph has no guidance branch, but this predictor's infer_cfg_scale is {dp.infer_cfg_scale}")
        if dp.bert_proj is not None:
            raise ValueError("a duration predictor with BERT features is not exportable by this wrapper")
        self.dp = copy.deepcopy(dp)
        self.dp.grad_scale = 1.0  # the training-time gradient scaling is the identity in the forward pass: keep it out of the graph
        self.n_steps = n_steps

    def forward(self, encoder_output, x_mask, spk_embed, noise):
        """encoder_output (B, C, T), x_mask (B, 1, T), spk_embed (B, spk_dim) L2-normalised, noise (B, 1, T) = randn * temperature (per token) -> log durations (B, 1, T)."""
        cond = self.dp.extract_cond(encoder_output, x_mask, g=spk_embed)
        w = noise * x_mask
        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            w = w + dt * self.dp.forward_velocity(w, _ramp(x_mask, i * dt), cond, x_mask)
        return w * x_mask


class ProsodyCFMSamplerWrapper(nn.Module):
    """`ProsodyCFM.forward` without guidance, with the noise as an input and the within-phone position (`ProsodyCFM.compute_fractional_pos`) computed by the caller."""

    def __init__(self, prosody_cfm, n_steps: int = 10):
        super().__init__()
        if prosody_cfm.use_mark:
            raise ValueError("a ProsodyCFM with punctuation features (use_mark=True) is not exportable by this wrapper")
        self.pcfm = copy.deepcopy(prosody_cfm)
        self.n_steps = n_steps

    def forward(self, mu_y, h_text_y, tau, dur, y_mask, spk_embed, noise):
        """mu_y (B, 80, T_mel) the encoder's mean per frame, h_text_y (B, C, T_mel) encoder_output per frame, tau / dur (B, 1, T_mel) position in the phone (0..1) and its
        frames, y_mask (B, 1, T_mel), spk_embed (B, spk_dim) L2-normalised, noise (B, 80, T_mel) = randn * temperature -> the decoder's input features (B, 80, T_mel)."""
        cond = self.pcfm.extract_cond(mu_y, h_text_y, tau, dur, y_mask, g=spk_embed)
        h = noise * y_mask
        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            h = h + dt * self.pcfm.forward_velocity(h, _ramp(y_mask, i * dt), cond, y_mask)
        if self.pcfm.residual:  # the flow generated what the encoder's mean misses
            h = h + mu_y
        return h * y_mask
