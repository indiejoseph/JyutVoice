"""Per-token duration-loss weights: characters the teacher barely saw are dropped from the flow-matching duration loss.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_dur_weight.py
"""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from jyutvoice.models.flow_duration_predictor import FlowDurationPredictor


def make_dp():
    dp = FlowDurationPredictor(in_channels=16, filter_channels=16, n_layers=2, bert_dim=0, spk_embed_dim=8, cfg_dropout=0.0)
    for prm in dp.out_proj.parameters():
        torch.nn.init.normal_(prm, std=0.3)
    return dp


def loss(dp, weight, seed=0, T=6):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 16, T, generator=g)
    mask = torch.ones(2, 1, T)
    dur = torch.randint(1, 9, (2, 1, T), generator=g).float()
    torch.manual_seed(seed)
    return dp.compute_loss(x, mask, dur, torch.full((2,), T), g=torch.randn(2, 8, generator=g), token_weight=weight)


def test_flow_duration_loss_with_weights():
    dp = make_dp().eval()
    base = loss(dp, None)
    assert torch.allclose(loss(dp, torch.ones(2, 6)), base)  # all ones = the plain loss
    assert loss(dp, torch.zeros(2, 6)).item() == 0.0  # nothing to learn from
    w = torch.ones(2, 6)
    w[:, 2] = 0.0
    part = loss(dp, w)
    assert part.item() != base.item() and torch.isfinite(part)  # a dropped token changes the loss; the others are still averaged by their own count


def test_dur_weight_marks_the_tokens_of_rare_characters(tmp_path):
    pytest.importorskip("pycantonese")
    try:
        import jyutvoice.data.text_mel_datamodule as dm
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"datamodule imports unavailable: {e!r}")
    text = "之前 ， 我 已經 提醒 過 佢哋 㗎 喇 。"
    counts = {c: 10**6 for c in text if not c.isspace() and c not in "，。"}  # punctuation has no entry: it must still not be masked
    counts["㗎"], counts["喇"] = 1014, 96
    path = tmp_path / "counts.json"
    path.write_text(json.dumps(counts, ensure_ascii=False))
    ds = dm.TextMelDataset.__new__(dm.TextMelDataset)
    ds.add_blank, ds.dur_weight_min_count, ds.char_counts = True, 2000, counts
    _, _, phone_ids, *_, word2ph = ds.get_text(text, "yue", add_blank=True)
    w = ds.get_dur_weight(text, "yue", word2ph, phone_ids.shape[0])
    assert w.shape == phone_ids.shape and w[0] == 1 and w[-1] == 1  # pads untouched
    assert int((w == 0).sum()) == 9  # 4 phones of 㗎喇 + the 5 blanks around and between them
    ds.dur_weight_min_count = 50  # nothing is rarer than that
    assert (ds.get_dur_weight(text, "yue", word2ph, phone_ids.shape[0]) == 1).all()
    assert (ds.get_dur_weight(text, "en", word2ph, phone_ids.shape[0]) == 1).all()  # other languages are never masked
