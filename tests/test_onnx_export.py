"""ONNX export (scripts/export_onnx.py, jyutvoice/models/onnx_wrappers.py) and the numpy side of ONNX inference (jyutvoice/utils/onnx_synth.py).

Models are tiny and random (the layers that start at zero are randomised so the samplers do something); what is checked is that the exported graphs and the numpy
steps between them reproduce PyTorch: `JyutVoiceTTS.synthesise` for the same random draws.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from jyutvoice.models.duration_predictor import DurationPredictor
from jyutvoice.models.flow_duration_predictor import FlowDurationPredictor
from jyutvoice.models.jyutvoice_tts import JyutVoiceTTS
from jyutvoice.models.onnx_wrappers import FlowDurationSamplerWrapper, ProsodyCFMSamplerWrapper, TextEncoderWrapper
from jyutvoice.models.prosody_cfm import ProsodyCFM
from jyutvoice.models.slp import distribute_frames
from jyutvoice.models.text_encoder import TextEncoder
from jyutvoice.utils import onnx_synth as osy
from jyutvoice.utils.model import generate_path, sequence_mask

spec = importlib.util.spec_from_file_location("export_onnx", os.path.join(ROOT, "scripts", "export_onnx.py"))
export_onnx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_onnx)

SPK = 16
C = 32  # encoder width


def randomise_zero_init(module):
    """Output layers start at zero (the sampler would predict nothing): give them weights."""
    g = torch.Generator().manual_seed(7)
    for p in module.parameters():
        if p.abs().max() == 0:
            p.data = torch.randn(p.shape, generator=g) * 0.1


def tiny_model(kind="flow", prosody=True, residual=False):
    torch.manual_seed(0)
    enc = TextEncoder(
        encoder_type="RoPE Encoder",
        encoder_params=DictConfig(
            {"n_feats": 80, "n_channels": C, "filter_channels": 64, "filter_channels_dp": 32, "n_heads": 2, "n_layers": 3, "kernel_size": 3, "p_dropout": 0.0,
             "prenet": False, "gin_channels": SPK}
        ),  # fmt: skip
        n_vocab=97,
        n_lang=4,
        n_tone=11,
    )
    if kind == "flow":
        dp = FlowDurationPredictor(in_channels=C, filter_channels=32, n_layers=2, kernel_size=5, bert_dim=0, spk_embed_dim=SPK, infer_cfg_scale=0.0)
    else:
        dp = DurationPredictor(C, 32, 3, 0.0)
    pm = ProsodyCFM(in_channels=80, text_channels=C, hidden_channels=32, n_blocks=2, kernel_size=7, dilations=(1, 2), spk_embed_dim=SPK, residual=residual) if prosody else None
    tts = JyutVoiceTTS(encoder=enc, decoder=None, dp=dp, output_size=80, spk_embed_dim=SPK)
    for m in (tts, pm):
        if m is not None:
            randomise_zero_init(m)
    return tts.eval(), (pm.eval() if pm is not None else None)


COMBOS = {"flow": ("flow", True, False), "flow_residual": ("flow", True, True), "regression": ("regression", False, False)}


@pytest.fixture(scope="module", params=list(COMBOS))
def exported(request, tmp_path_factory):
    tts, pm = tiny_model(*COMBOS[request.param])
    out = str(tmp_path_factory.mktemp(f"onnx_{request.param}"))
    ok = export_onnx.export_all(tts, pm, out, add_blank=True, n_steps=4, opset=17, max_tokens=512, atol=1e-4, source={"checkpoint": "random", "base_config": "tiny"})
    assert ok
    return tts, pm, out


# ---- numpy steps between the graphs against PyTorch -----------------------------------------------------------------------------------------------------


def torch_counts(logw, rounding, scale):
    """The lines of JyutVoiceTTS.synthesise between the duration model and the alignment."""
    x_mask = torch.ones_like(logw)
    w = torch.exp(logw) * x_mask
    if rounding == "cumulative":
        w = distribute_frames(w, x_mask, w.sum(dim=(1, 2)).round().clamp_min(1.0).unsqueeze(-1))
    elif rounding == "round":
        w = torch.clamp_min(torch.round(w), 1.0) * x_mask
    w_ceil = torch.ceil(w) * scale
    y_len = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = sequence_mask(y_len, y_len.max()).unsqueeze(1).to(x_mask.dtype)
    attn = generate_path(w_ceil.squeeze(1), (x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)).squeeze(1))
    return attn[0], int(y_len)


@pytest.mark.parametrize("rounding", ["ceil", "round", "cumulative"])
@pytest.mark.parametrize("scale", [0.5, 1.0, 1.25, 1.5, 0.8, 1.1, 1.37])
def test_frame_counts_match_torch(rounding, scale):
    exact = scale in (0.5, 1.0, 1.25, 1.5)  # representable in binary: every sum is exact
    g = torch.Generator().manual_seed(1)
    for T in (5, 35, 120, 400):
        logw = torch.randn(1, 1, T, generator=g) * 0.7 + 0.9  # a few frames per token, as predicted
        attn, y_len = torch_counts(logw, rounding, scale)
        counts, n = osy.frame_counts(osy.round_durations(logw.numpy(), rounding), scale)
        want = attn.sum(-1).numpy().astype(np.int64)
        if exact:
            assert n == y_len and np.array_equal(counts, want)
        else:
            # torch sums in float32, whose last bit can put a sum that is an integer in exact arithmetic (400 * 0.8 ...) just below it: the mel is then one frame
            # shorter and only the last tokens differ
            assert abs(n - y_len) <= 1 and np.array_equal(counts[:-2], want[:-2])


def test_expand_and_phone_position_match_torch():
    g = torch.Generator().manual_seed(2)
    logw = torch.randn(1, 1, 60, generator=g) * 0.8 + 0.9
    for scale in (0.5, 1.0, 1.25):  # below 1 some tokens get no frame at all
        attn, y_len = torch_counts(logw, "round", scale)
        counts, _ = osy.frame_counts(osy.round_durations(logw.numpy(), "round"), scale)
        feats = torch.randn(1, 8, 60, generator=g)
        assert torch.equal(torch.from_numpy(osy.expand_frames(feats.numpy(), counts)), torch.matmul(attn.T.unsqueeze(0), feats.transpose(1, 2)).transpose(1, 2))
        tau, dur = ProsodyCFM.compute_fractional_pos(attn.unsqueeze(0))
        tau_np, dur_np = osy.phone_position(counts)
        assert np.allclose(tau_np, tau.numpy()) and np.allclose(dur_np, dur.numpy())


def test_pause_mask_and_temperatures():
    x = np.array([0, 5, 0, 7, 0, 9, 0, 3, 0, 5, 0, 4, 0, 6, 0, 8, 0])
    pm = osy.pause_token_mask(x, [7, 3], add_blank=True)
    assert pm.tolist() == [True, True, True, True, False, False, False, True, False, False, False, False, False, False, True, True, True]
    assert osy.pause_token_mask(x, [7], add_blank=False)[[0, -1, 3]].all()
    t = osy.token_temperatures(pm, 0.3, 1.0)
    assert t.shape == (1, 1, len(x)) and t.dtype == np.float32 and t[0, 0, 0] == 1.0 and np.isclose(t[0, 0, 4], 0.3)


# ---- the exported graphs ----------------------------------------------------------------------------------------------------------------------------------


def test_export_matches_torch_and_manifest(exported):
    tts, pm, out = exported  # export_all already compared every graph and the whole chain with synthesise at three lengths, the longest far beyond the traced one
    model = osy.OnnxAcousticModel(out)
    assert model.dp_kind == ("flow" if isinstance(tts.dp, FlowDurationPredictor) else "regression")
    assert model.has_prosody == (pm is not None)
    assert model.max_tokens == 512 and model.add_blank


def test_dynamic_lengths_and_seed(exported):
    tts, pm, out = exported
    model = osy.OnnxAcousticModel(out)
    spk = torch.nn.functional.normalize(torch.randn(1, SPK), dim=1).numpy()
    for text in ("我 喺 Zoom 開會 ，", "你 好 。", "佢 話 個 interview 好 順利 ， 希望 可以 攞到 offer 。"):
        tok = osy.text_tokens(text)
        args = (tok["x"], tok["lang"], tok["tone"], tok["word_pos"], tok["syllable_pos"], spk)
        a = model(*args, seed=1)
        b = model(*args, seed=1)
        assert a.mu_y.shape == (1, 80, int(a.counts.sum())) and a.counts.shape == (tok["x"].shape[1],)
        assert np.array_equal(a.mu_y, b.mu_y)
        if model.dp_kind == "flow" or model.has_prosody:
            assert not np.array_equal(a.mu_y, model(*args, seed=2).mu_y)
        assert model(*args, seed=1, length_scale=1.5).counts.sum() > a.counts.sum()
    with pytest.raises(ValueError, match="at most"):
        model.max_tokens = 10
        model(*args)


def test_without_prosody_gives_the_encoder_mean(exported):
    tts, pm, out = exported
    if pm is None:
        pytest.skip("exported without ProsodyCFM")
    model = osy.OnnxAcousticModel(out)
    tok = osy.text_tokens("我 喺 Zoom 開會 ，")
    r = model(tok["x"], tok["lang"], tok["tone"], tok["word_pos"], tok["syllable_pos"], np.ones((1, SPK), np.float32) / 4, use_prosody=False, seed=3)
    assert r.mu_y_mean is None and r.mu_y.shape[1] == 80


# ---- what is refused ----------------------------------------------------------------------------------------------------------------------------------------


def test_wrappers_refuse_what_they_cannot_export():
    tts, pm = tiny_model("flow", True)
    tts.dp.infer_cfg_scale = 1.5
    with pytest.raises(ValueError, match="guidance"):
        FlowDurationSamplerWrapper(tts.dp)
    with pytest.raises(ValueError, match="punctuation"):
        ProsodyCFMSamplerWrapper(ProsodyCFM(in_channels=80, text_channels=C, hidden_channels=32, n_blocks=1, spk_embed_dim=SPK, use_mark=True))
    bert_enc = TextEncoder(
        encoder_type="RoPE Encoder",
        encoder_params=DictConfig({"n_feats": 80, "n_channels": C, "filter_channels": 64, "filter_channels_dp": 32, "n_heads": 2, "n_layers": 3, "kernel_size": 3, "p_dropout": 0.0, "prenet": False, "gin_channels": SPK}),
        n_vocab=97,
        n_lang=4,
        bert_dim=8,
    )  # fmt: skip
    with pytest.raises(ValueError, match="BERT"):
        TextEncoderWrapper(bert_enc)


def test_old_manifest_format_is_refused(tmp_path):
    import json

    (tmp_path / "manifest.json").write_text(json.dumps({"format": 99}))
    with pytest.raises(ValueError, match="format"):
        osy.OnnxAcousticModel(str(tmp_path))
