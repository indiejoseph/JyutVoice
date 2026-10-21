"""Tests for the pieces that let teacher-generated rows (text + flow-encoder states, no audio) train the duration predictor: encoder freezing,
per-row weight in the prior loss, rows without audio in the datamodule.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_teacher_rows.py
"""
import os
import sys

import pytest
import torch
from omegaconf import DictConfig

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from jyutvoice.models.duration_predictor import DurationPredictor
from jyutvoice.models.jyutvoice_tts import JyutVoiceTTS
from jyutvoice.models.text_encoder import TextEncoder

N_CH, SPK = 32, 16


def make_tts(freeze_encoder=False):
    enc = TextEncoder(
        encoder_type="RoPE Encoder",
        encoder_params=DictConfig({"n_feats": 80, "n_channels": N_CH, "filter_channels": 64, "filter_channels_dp": 16, "n_heads": 2, "n_layers": 3,
                                   "kernel_size": 3, "p_dropout": 0.1, "prenet": False, "gin_channels": SPK}),
        n_vocab=97, n_lang=4, n_tone=7,
    )  # fmt: skip
    dp = DurationPredictor(in_channels=N_CH, filter_channels=16, kernel_size=3, p_dropout=0.1)
    return JyutVoiceTTS(encoder=enc, decoder=None, dp=dp, output_size=80, spk_embed_dim=SPK, freeze_encoder=freeze_encoder)


def batch(B=2, T=10, Ty=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 4, (B, T), generator=g)
    return dict(x=torch.randint(1, 97, (B, T), generator=g), x_lengths=torch.full((B,), T), y=torch.zeros(B, 80, Ty), y_lengths=torch.full((B,), Ty),
                lang=ids, tone=ids.clone(), word_pos=ids.clone(), syllable_pos=ids.clone(), spk_embed=torch.randn(B, SPK, generator=g),
                decoder_h=torch.randn(B, Ty, 80, generator=g))  # fmt: skip


def test_frozen_encoder_stays_put_and_the_duration_predictor_learns():
    model = make_tts(freeze_encoder=True).train()
    assert not any(p.requires_grad for p in model.encoder.parameters()) and all(p.requires_grad for p in model.dp.parameters())
    assert not model.encoder.training and model.dp.training  # train() keeps the frozen encoder in eval mode
    enc0 = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    dp0 = {k: v.clone() for k, v in model.dp.state_dict().items()}
    opt = torch.optim.AdamW([p for p in model.parameters()], lr=1e-2, weight_decay=0.0)
    for _ in range(3):
        dur, prior, _ = model(**batch())
        (dur + prior).backward()
        opt.step()
        opt.zero_grad()
    assert all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in enc0.items())
    assert any(not torch.equal(v, model.dp.state_dict()[k]) for k, v in dp0.items())


def test_prior_weight_zero_rows_only_train_the_durations():
    model = make_tts().eval()
    b = batch(B=2)
    with torch.no_grad():
        d_all, p_all, _ = model(**b)
        d_w, p_w, _ = model(**b, prior_weight=torch.tensor([1.0, 0.0]))
        p_ones = model(**b, prior_weight=torch.ones(2))[1]
        b0 = {k: v[:1] for k, v in b.items()}
        d_0, p_0, _ = model(**b0)
        p_zero = model(**b, prior_weight=torch.zeros(2))[1]
    assert torch.allclose(p_ones, p_all) and torch.allclose(d_w, d_all)  # weights do not touch the duration loss
    assert torch.allclose(p_w, p_0, atol=1e-5)  # the prior loss is that of row 0 alone
    assert p_zero.item() == 0.0


def test_row_without_audio_gets_a_zero_mel_of_the_right_length_and_a_prior_weight():
    pytest.importorskip("pycantonese")
    try:
        import jyutvoice.data.text_mel_datamodule as dm
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"datamodule imports unavailable: {e!r}")
    ds = dm.TextMelDataset.__new__(dm.TextMelDataset)
    ds.add_blank, ds.bert_dim, ds.phone_override, ds.load_durations, ds.n_mels, ds.bert_side = True, 0, {}, False, 80, None
    ds.teacher_prior_weight = 0.0
    T = 37
    row = {"id": "t00001", "text": "我 哋 去 食 飯 。", "lang": "yue", "phone": None, "decoder_h": torch.randn(T, 80).tolist(), "spk_emb": [0.1] * 192,
           "source": "teacher"}  # fmt: skip
    item = ds.get_datapoint(row)
    assert item["y"].shape == (80, T) and item["y"].abs().sum() == 0 and item["prior_weight"] == 0.0
    assert item["decoder_h"].shape == (T, 80) and item["durations"] is None
    other = ds.get_datapoint({**row, "source": "tts"})
    assert other["prior_weight"] == 1.0
    out = dm.TextMelBatchCollate()([item, other])
    assert out["prior_weight"].tolist() == [0.0, 1.0]
    assert "prior_weight" not in dm.TextMelBatchCollate()([other, other])  # untouched batches look exactly as before


def test_init_weights_with_prefixes_load_only_the_encoder_and_leave_a_different_predictor_alone(tmp_path):
    from jyutvoice.models.flow_duration_predictor import FlowDurationPredictor
    from jyutvoice.train import load_init_weights

    src = make_tts()
    torch.save({"state_dict": src.state_dict()}, tmp_path / "src.ckpt")
    dst = make_tts()
    dst.dp = FlowDurationPredictor(in_channels=N_CH, filter_channels=16, n_layers=2, bert_dim=0, spk_embed_dim=SPK)  # keys differ from the checkpoint's dp
    flow0 = {k: v.clone() for k, v in dst.dp.state_dict().items()}
    with pytest.raises(RuntimeError):
        load_init_weights(dst, str(tmp_path / "src.ckpt"))  # a full strict load cannot work
    load_init_weights(dst, str(tmp_path / "src.ckpt"), ["encoder.", "spk_embed_affine_layer."])
    assert all(torch.equal(v, dst.encoder.state_dict()[k]) for k, v in src.encoder.state_dict().items())
    assert all(torch.equal(v, dst.dp.state_dict()[k]) for k, v in flow0.items())
    dur, prior, attn = dst.train()(**batch())  # and it trains end to end
    assert torch.isfinite(dur) and attn.shape == (2, 10, 40)
