"""The frame-level prosody model's training pieces: alignment from durations, batch building, frame statistics, and that a few steps lower the loss.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_prosody_train.py
"""
import os
import random
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import train_prosody_cfm as tp  # noqa: E402
from jyutvoice.models.prosody_cfm import ProsodyCFM  # noqa: E402
from jyutvoice.utils.model import generate_path  # noqa: E402


def fake_rows(n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(n):
        tt = int(torch.randint(5, 12, (1,), generator=g))
        dur = torch.randint(1, 5, (tt,), generator=g)
        ty = int(dur.sum())
        mu = torch.randn(tt, 80, generator=g)
        h = mu.repeat_interleave(dur, dim=0) + 0.3 * torch.randn(ty, 80, generator=g)  # real features = the token average plus detail
        rows.append({"x": torch.randn(tt, 192, generator=g).half(), "mu": mu.half(), "dur": dur.to(torch.int16), "h": h.half(), "spk": torch.randn(192, generator=g)})
    return rows


def test_alignment_from_durations_matches_generate_path():
    dur = torch.tensor([[2, 3, 1, 0], [1, 1, 4, 2]])
    ty = 8
    mask = (torch.arange(ty)[None, None, :] < dur.sum(-1)[:, None, None]).float().expand(-1, dur.shape[1], -1)
    ref = generate_path(dur.float(), mask)
    got = tp.attn_from_durations(dur, ty)
    assert torch.equal(got, ref[:, :, :ty] * (got.sum(1, keepdim=True) > 0)) or torch.equal(got, ref)
    assert got.sum(-1).long().tolist() == dur.tolist()  # frames per token are the durations
    assert (got.sum(1) <= 1).all()  # a frame belongs to at most one token


def test_batch_has_each_token_vector_repeated_over_its_frames():
    rows = fake_rows(3)
    b = tp.make_batch(rows, "cpu")
    r = rows[0]
    ty = r["h"].shape[0]
    assert b["h"].shape[1] == 80 and b["mu_y"].shape == b["h"].shape and b["h_text_y"].shape[1] == 192
    assert torch.allclose(b["mu_y"][0, :, :ty].T, r["mu"].float().repeat_interleave(r["dur"].long(), dim=0))
    assert b["y_mask"][0, 0].sum() == ty and b["y_lengths"][0] == ty
    assert torch.allclose(b["spk"].norm(dim=1), torch.ones(3))


def test_frame_stats_of_the_real_features_and_of_a_smoothed_copy():
    g = torch.Generator().manual_seed(1)
    h = torch.cumsum(torch.randn(200, 80, generator=g), 0) * 0.1
    assert all(abs(v - 1.0) < 1e-6 for v in tp.frame_stats(h, h))
    smooth = torch.nn.functional.avg_pool1d(h.T[None], 9, 1, 4)[0].T
    r2, std, delta = tp.frame_stats(h, smooth)
    assert 0.5 < r2 < 1.0 and std < 1.0 and delta < 1.0  # smoothing lowers the movement ratios


def test_a_few_steps_lower_the_loss_and_sampling_runs():
    rows = fake_rows(16)
    model = ProsodyCFM(in_channels=80, text_channels=192, hidden_channels=64, n_blocks=2, kernel_size=5, spk_embed_dim=192, cfg_dropout=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    b = tp.make_batch(rows, "cpu")

    def loss():
        torch.manual_seed(0)
        return float(model.compute_loss(decoder_h=b["h"], mu_y=b["mu_y"], h_text_y=b["h_text_y"], attn=b["attn"], y_mask=b["y_mask"], y_lengths=b["y_lengths"], g=b["spk"]))

    model.eval()
    before = loss()
    model.train()
    for _ in range(40):
        l = model.compute_loss(decoder_h=b["h"], mu_y=b["mu_y"], h_text_y=b["h_text_y"], attn=b["attn"], y_mask=b["y_mask"], y_lengths=b["y_lengths"], g=b["spk"])
        opt.zero_grad()
        l.backward()
        opt.step()
    model.eval()
    assert loss() < before
    st = tp.sample_stats(model, rows, "cpu", steps=3, cfg_scale=0.0, temperature=1.0, k_mean=2, batch_size=8)
    assert set(st) == {"mu", "sample", "mean_k"} and all(len(v) == 3 for v in st.values())
    assert 0 < tp.val_loss(model, rows, "cpu") < 10
    batches = tp.length_batches(rows, 4, True, random.Random(0))
    assert sorted(i for g in batches for i in g) == list(range(16))


def test_synthesise_can_switch_the_prosody_model_off_and_pass_its_sampling_settings():
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_teacher_rows import N_CH, SPK, make_tts

    tts = make_tts().eval()
    z = torch.zeros(1, 9, dtype=torch.long)
    kw = dict(x=torch.randint(1, 97, (1, 9)), x_lengths=torch.tensor([9]), lang=z, tone=z, word_pos=z, syllable_pos=z, spk_embed=torch.randn(1, SPK))
    with torch.no_grad():
        mean_only = tts.synthesise(**kw)["encoder_outputs"]
        tts.prosody_cfm = ProsodyCFM(in_channels=80, text_channels=N_CH, hidden_channels=32, n_blocks=2, kernel_size=5, spk_embed_dim=SPK)
        for prm in tts.prosody_cfm.out_proj.parameters():
            torch.nn.init.normal_(prm, std=0.3)  # the zero-initialised output layer would make every sample the noise scaled by nothing useful
        tts.prosody_cfm.eval()  # attached after tts.eval(): without this its dropout would still be on
        off = tts.synthesise(**kw, use_prosody=False)["encoder_outputs"]
        torch.manual_seed(0)
        on = tts.synthesise(**kw, prosody_kwargs=dict(n_steps=4, cfg_scale=0.0, temperature=0.0))["encoder_outputs"]
        torch.manual_seed(1)
        on2 = tts.synthesise(**kw, prosody_kwargs=dict(n_steps=4, cfg_scale=0.0, temperature=0.0))["encoder_outputs"]
        torch.manual_seed(1)
        noisy = tts.synthesise(**kw, prosody_kwargs=dict(n_steps=4, cfg_scale=0.0, temperature=1.0))["encoder_outputs"]
    assert torch.equal(mean_only, off)  # switched off = the encoder's mean, exactly as without the model
    assert on.shape == off.shape and not torch.equal(on, off)  # on: the sampled features replace it
    assert torch.equal(on, on2)  # temperature 0: no randomness left, whatever the seed
    assert not torch.equal(on, noisy)  # temperature 1 changes the result


def test_residual_model_adds_the_encoder_average_back_and_trains():
    rows = fake_rows(16)
    b = tp.make_batch(rows, "cpu")
    kw = dict(in_channels=80, text_channels=192, hidden_channels=64, n_blocks=2, kernel_size=5, spk_embed_dim=192, cfg_dropout=0.1)
    res, plain = ProsodyCFM(**kw, residual=True).eval(), ProsodyCFM(**kw).eval()
    sample = lambda m: m(b["mu_y"], b["h_text_y"], b["attn"], b["y_mask"], g=b["spk"], n_steps=3, cfg_scale=0.0, temperature=0.0)  # noqa: E731
    assert torch.equal(sample(res), b["mu_y"] * b["y_mask"])  # untrained (zero velocity) and without noise: exactly the encoder's average
    assert float(sample(plain).abs().max()) == 0.0  # the plain model starts from nothing
    res.train()
    opt = torch.optim.AdamW(res.parameters(), lr=2e-3)

    def loss():
        torch.manual_seed(0)
        return float(res.compute_loss(decoder_h=b["h"], mu_y=b["mu_y"], h_text_y=b["h_text_y"], attn=b["attn"], y_mask=b["y_mask"], y_lengths=b["y_lengths"], g=b["spk"]))

    before = loss()
    for _ in range(40):
        l = res.compute_loss(decoder_h=b["h"], mu_y=b["mu_y"], h_text_y=b["h_text_y"], attn=b["attn"], y_mask=b["y_mask"], y_lengths=b["y_lengths"], g=b["spk"])
        opt.zero_grad()
        l.backward()
        opt.step()
    assert loss() < before


def test_mark_features_give_the_next_mark_and_the_distance_to_it():
    from jyutvoice.models.prosody_cfm import MARK_MAX_DIST, mark_features

    table = torch.zeros(12, dtype=torch.long)
    table[7], table[9] = 1, 3  # symbol 7 is a comma, 9 a question mark
    x = torch.tensor([[0, 5, 0, 6, 0, 7, 0, 5, 0, 9, 0, 0]])  # blank, w, blank, w, blank, COMMA, blank, w, blank, QUESTION, blank, pad
    dur = torch.tensor([[1, 2, 1, 3, 1, 4, 1, 2, 1, 5, 1, 0]])  # frames 0-11 up to and including the comma (frames 8-11), 12-20 up to the question mark (16-20), 21 after it
    mark, dist = mark_features(x, attn=tp.attn_from_durations(dur, int(dur.sum())), class_of_id=table)
    assert mark[0].tolist() == [1] * 12 + [3] * 9 + [0]  # the next mark at or after the frame's own token; nothing ahead after the last mark
    assert dist[0, 0].tolist() == [8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0] + [4, 3, 2, 1, 0, 0, 0, 0, 0] + [float(MARK_MAX_DIST)]
    none, far = mark_features(torch.tensor([[0, 5, 0, 6, 0]]), tp.attn_from_durations(torch.tensor([[1, 2, 1, 3, 1]]), 8), table)
    assert none.sum() == 0 and bool((far == MARK_MAX_DIST).all())  # no mark anywhere: class 0 and the cap


def test_punctuation_aware_model_needs_the_mark_features_and_trains_with_them():
    from jyutvoice.models.prosody_cfm import mark_class_table

    rows = fake_rows(12)
    for i, r in enumerate(rows):
        ids = torch.randint(1, 8, (r["x"].shape[0],))
        ids[-2] = 4  # a "question mark" id for the test
        r["ids"] = ids.to(torch.int16)
    table = mark_class_table(["_", "a", "b", "c", "?", "d", "e", "f"])
    assert table.tolist() == [0, 0, 0, 0, 3, 0, 0, 0]
    b = tp.make_batch(rows, "cpu", table)
    assert b["mark_y"].shape == (12, b["h"].shape[2]) and b["dist_y"].shape == (12, 1, b["h"].shape[2])
    kw = dict(in_channels=80, text_channels=192, hidden_channels=64, n_blocks=2, kernel_size=5, spk_embed_dim=192, cfg_dropout=0.0, use_mark=True)
    m = ProsodyCFM(**kw)
    m.train()
    args = dict(decoder_h=b["h"], mu_y=b["mu_y"], h_text_y=b["h_text_y"], attn=b["attn"], y_mask=b["y_mask"], y_lengths=b["y_lengths"], g=b["spk"])
    try:
        m.compute_loss(**args)
        raise AssertionError("a use_mark model must refuse to run without the mark features")
    except ValueError:
        pass
    assert float(m.compute_loss(**args, mark_y=b["mark_y"], dist_y=b["dist_y"])) > 0
    m.eval()
    out = m(b["mu_y"], b["h_text_y"], b["attn"], b["y_mask"], g=b["spk"], n_steps=2, mark_y=b["mark_y"], dist_y=b["dist_y"])
    assert out.shape == b["mu_y"].shape
    plain = ProsodyCFM(**{**kw, "use_mark": False}).eval()
    assert not hasattr(plain, "mark_emb") and plain(b["mu_y"], b["h_text_y"], b["attn"], b["y_mask"], g=b["spk"], n_steps=2).shape == b["mu_y"].shape  # the old models are untouched
