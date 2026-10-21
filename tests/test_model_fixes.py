import os
import sys

import pytest
import torch
from omegaconf import DictConfig

# Make sure the repository root is on sys.path so tests can import the package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jyutvoice.models.jyutvoice_tts as tts_module
from jyutvoice.models.duration_predictor import DurationPredictor
from jyutvoice.models.jyutvoice_tts import JyutVoiceTTS
from jyutvoice.models.slp import (
    SpeechLengthPredictor,
    ce_logits_to_length,
    distribute_frames,
)
from jyutvoice.models.text_encoder import TextEncoder

N_CH = 32
SPK = 16


def make_encoder(gin_channels=SPK):
    return TextEncoder(
        encoder_type="RoPE Encoder",
        encoder_params=DictConfig(
            {
                "n_feats": 80,
                "n_channels": N_CH,
                "filter_channels": 64,
                "filter_channels_dp": 16,
                "n_heads": 2,
                "n_layers": 3,
                "kernel_size": 3,
                "p_dropout": 0.0,
                "prenet": False,
                "gin_channels": gin_channels,
            }
        ),
        n_vocab=97,
        n_lang=4,
        n_tone=7,
    )


def make_dp():
    return DurationPredictor(
        in_channels=N_CH, filter_channels=16, kernel_size=3, p_dropout=0.0
    )


def make_tts(dp=None, slp=None):
    return JyutVoiceTTS(
        encoder=make_encoder(),
        decoder=None,
        dp=dp if dp is not None else make_dp(),
        slp=slp,
        output_size=80,
        spk_embed_dim=SPK,
    )


def make_ids(B, T):
    ids = torch.randint(0, 4, (B, T))
    return {
        "x": torch.randint(1, 97, (B, T)),
        "x_lengths": torch.full((B,), T),
        "lang": ids,
        "tone": ids.clone(),
        "word_pos": ids.clone(),
        "syllable_pos": ids.clone(),
        "spk_embed": torch.randn(B, SPK),
    }


class ConstantLength(torch.nn.Module):
    """SLP stand-in that always predicts the same total length (L1-style scalar output)."""

    def __init__(self, frames):
        super().__init__()
        self.frames = frames

    def forward(self, x, x_mask, **kwargs):
        return torch.full((x.size(0), 1), float(self.frames))


class RampDurations(torch.nn.Module):
    """Duration predictor stand-in: token i gets relative duration i + 1 (clearly non-uniform)."""

    def forward(self, x, x_mask):
        T = x.size(-1)
        logw = torch.log(torch.arange(1, T + 1, dtype=torch.float32)).view(1, 1, T)
        return logw.expand(x.size(0), 1, T) * x_mask


def test_speaker_conditioning_when_gin_channels_differs_from_hidden():
    B, T = 2, 10
    enc = make_encoder(gin_channels=48)  # 48 != N_CH
    ids = make_ids(B, T)
    x, mu, x_mask = enc(
        ids["x"],
        ids["x_lengths"],
        ids["lang"],
        ids["tone"],
        ids["word_pos"],
        ids["syllable_pos"],
        torch.randn(B, 48),
    )
    assert x.shape == (B, N_CH, T)
    assert mu.shape == (B, 80, T)


class FixedMu(torch.nn.Module):
    """Encoder stand-in with a fixed fp32 mu, so only the alignment maths can differ between runs."""

    n_feats = 80

    def __init__(self, mu):
        super().__init__()
        self.mu = mu

    def forward(self, x, x_lengths, lang, tone, word_pos, syllable_pos, g):
        B, T = x.shape
        x_mask = (torch.arange(T)[None] < x_lengths[:, None]).unsqueeze(1).float()
        return torch.zeros(B, N_CH, T), self.mu * x_mask, x_mask


def run_forward(model, ids, Ty, decoder_h):
    B = ids["x"].size(0)
    return model(
        x=ids["x"],
        x_lengths=ids["x_lengths"],
        y=torch.zeros(B, 80, Ty),
        y_lengths=torch.full((B,), Ty),
        lang=ids["lang"],
        tone=ids["tone"],
        word_pos=ids["word_pos"],
        syllable_pos=ids["syllable_pos"],
        spk_embed=ids["spk_embed"],
        decoder_h=decoder_h,
    )


def test_mas_log_prior_is_accurate_under_bf16_autocast(monkeypatch):
    # bf16 matmuls over 80 feature dims put ~0.1 of error on log_prior values of magnitude ~40, which is
    # as large as the score differences MAS chooses between; the alignment must be computed in fp32.
    captured = []
    real = tts_module.monotonic_align.maximum_path

    def spy(value, mask):
        captured.append(value.detach().double().clone())
        return real(value, mask)

    monkeypatch.setattr(tts_module.monotonic_align, "maximum_path", spy)

    B, Tx, Ty = 2, 12, 40
    torch.manual_seed(0)
    mu, h = torch.randn(B, 80, Tx), torch.randn(B, Ty, 80)
    model = make_tts()
    model.encoder = FixedMu(mu)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        run_forward(model, make_ids(B, Tx), Ty, h)

    # log N(h_s | mu_t, I) summed over features, in float64
    diff = h.transpose(1, 2).double().unsqueeze(1) - mu.transpose(1, 2).double().unsqueeze(-1)
    expected = -0.5 * (diff**2).sum(2) - 0.5 * 80 * torch.log(torch.tensor(2 * torch.pi)).double()
    assert (captured[0] - expected).abs().max() < 1e-2


def test_forward_under_bf16_autocast_gives_a_valid_alignment_and_gradients():
    B, Tx, Ty = 2, 12, 40
    model = make_tts().train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        dur_loss, prior_loss, attn = run_forward(
            model, make_ids(B, Tx), Ty, torch.randn(B, Ty, 80)
        )
    assert torch.isfinite(dur_loss) and torch.isfinite(prior_loss)
    # every frame is assigned to exactly one token, and every token gets at least one frame
    assert torch.equal(attn.sum(dim=1).squeeze(1), torch.ones(B, Ty))
    assert (attn.sum(dim=-1) >= 1).all()
    (dur_loss + prior_loss).backward()
    assert model.encoder.proj.weight.grad is not None


def test_ce_logits_to_length_is_median_with_in_bin_interpolation():
    n = 10
    sharp = torch.zeros(1, 1, 301)
    sharp[..., 7] = 30.0  # essentially one-hot on class 7 -> bin [70, 80)
    assert ce_logits_to_length(sharp, n).item() == pytest.approx(75.0, abs=0.5)

    split = torch.full((1, 1, 301), -30.0)
    split[..., 7] = 0.0
    split[..., 8] = 0.0  # 50/50 between classes 7 and 8 -> median on the bin edge
    assert ce_logits_to_length(split, n).item() == pytest.approx(80.0, abs=0.5)

    shifted = torch.zeros(1, 1, 301)
    shifted[..., 9] = 30.0
    assert ce_logits_to_length(shifted, n) > ce_logits_to_length(sharp, n)
    assert ce_logits_to_length(sharp, n).shape == (1, 1)


def test_distribute_frames_hits_total_and_respects_padding():
    w = torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]], [[2.0, 2.0, 2.0, 9.0, 9.0]]])
    mask = torch.tensor([[[1.0] * 5], [[1.0, 1.0, 1.0, 0.0, 0.0]]])
    total = torch.tensor([[30.0], [12.0]])
    dur = distribute_frames(w, mask, total)
    assert dur.shape == w.shape
    assert dur.sum(dim=(1, 2)).tolist() == [30.0, 12.0]
    assert dur[1, 0, 3:].tolist() == [0.0, 0.0]  # padding gets nothing
    assert torch.equal(dur, dur.round())  # whole frames
    assert (dur[0, 0] >= 1).all()
    assert dur[0, 0].tolist() == sorted(dur[0, 0].tolist())  # ramp stays a ramp


def test_synthesise_rescales_dp_durations_to_slp_total():
    T, total = 10, 90
    model = make_tts(dp=RampDurations(), slp=ConstantLength(total)).eval()
    ids = make_ids(1, T)
    out = model.synthesise(
        x=ids["x"],
        x_lengths=ids["x_lengths"],
        lang=ids["lang"],
        tone=ids["tone"],
        word_pos=ids["word_pos"],
        syllable_pos=ids["syllable_pos"],
        spk_embed=ids["spk_embed"],
    )
    assert out["mel_lengths"].item() == total
    dur = out["attn"].squeeze(1).sum(-1).squeeze(0)  # frames per token
    assert dur.sum().item() == total
    ideal = total * torch.arange(1, T + 1) / torch.arange(1, T + 1).sum()
    assert (dur - ideal).abs().max() <= 1.0  # follows the DP's relative rhythm
    assert dur[-1] > dur[0]  # not uniform


def test_legacy_slp_in_dp_slot_still_runs_but_warns():
    slp = SpeechLengthPredictor(
        n_vocab=97,
        hidden_dim=N_CH,
        n_text_layer=1,
        n_cross_layer=1,
        n_head=2,
        output_dim=1,
        spk_embed_dim=SPK,
    )
    model = make_tts(dp=slp).eval()
    ids = make_ids(1, 8)
    with pytest.warns(UserWarning, match="slp"):
        out = model.synthesise(
            x=ids["x"],
            x_lengths=ids["x_lengths"],
            lang=ids["lang"],
            tone=ids["tone"],
            word_pos=ids["word_pos"],
            syllable_pos=ids["syllable_pos"],
            spk_embed=ids["spk_embed"],
        )
    assert out["attn"].shape[-1] == out["mel_lengths"].max().item()


class ConstantDurations(torch.nn.Module):
    """Duration predictor stand-in: every token lasts `frames` frames (fractional)."""

    def __init__(self, frames):
        super().__init__()
        self.frames = frames

    def forward(self, x, x_mask):
        return torch.full((x.size(0), 1, x.size(2)), float(torch.log(torch.tensor(self.frames)))) * x_mask


def synth_frames(rounding, frames=1.4, T=10):
    model = make_tts(dp=ConstantDurations(frames)).eval()
    ids = make_ids(1, T)
    kw = dict(x=ids["x"], x_lengths=ids["x_lengths"], lang=ids["lang"], tone=ids["tone"], word_pos=ids["word_pos"], syllable_pos=ids["syllable_pos"],
              spk_embed=ids["spk_embed"])  # fmt: skip
    out = model.synthesise(**kw) if rounding is None else model.synthesise(**kw, duration_rounding=rounding)
    return out["mel_lengths"].item(), out["attn"].squeeze().sum(-1)


def test_duration_rounding_default_is_still_ceil_and_the_other_modes_remove_its_bias():
    ceil_total, _ = synth_frames(None)
    assert ceil_total == synth_frames("ceil")[0] == 20  # 10 tokens x ceil(1.4) = 20: 43% longer than the 14 frames predicted
    round_total, dur_round = synth_frames("round")
    assert round_total == 10 and (dur_round == 1).all()
    cum_total, dur_cum = synth_frames("cumulative")
    assert cum_total == 14  # round(10 x 1.4)
    assert (dur_cum >= 1).all() and set(dur_cum.tolist()) <= {1.0, 2.0}  # whole frames, a mix of 1 and 2
    with pytest.raises(ValueError, match="duration_rounding"):
        synth_frames("floor")
