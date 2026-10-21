"""Tests for the pause-aware monotonic alignment: silent frames go to pause tokens (punctuation, edge pads), and nothing changes when the option is off
or the batch has no silence flags.

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_pause_alignment.py
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from test_teacher_rows import batch, make_tts  # noqa: E402


def silent_block(B, Ty, start, stop):
    sil = torch.zeros(B, Ty)
    sil[:, start:stop] = 1.0
    return sil


def pause_at(B, T, idx):
    m = torch.zeros(B, T, dtype=torch.bool)
    m[:, idx] = True
    return m


def test_a_large_bonus_gives_the_silent_block_to_the_pause_token():
    T, Ty, k = 10, 40, 4
    b = batch(B=2, T=T, Ty=Ty)
    model = make_tts().eval()
    with torch.no_grad():
        _, _, attn = model(**b, silence=silent_block(2, Ty, 15, 25), pause_mask=pause_at(2, T, k))
        model.pause_bonus = 1e3
        _, _, attn_b = model(**b, silence=silent_block(2, Ty, 15, 25), pause_mask=pause_at(2, T, k))
    assert attn.shape == (2, T, Ty) and attn_b.shape == (2, T, Ty)
    assert torch.equal(attn_b[:, k, 15:25], torch.ones(2, 10))  # every silent frame is the pause token's ...
    assert attn_b[:, k].sum(-1).tolist() == [10.0, 10.0]  # ... and it takes nothing else
    assert (attn_b.sum(1) == 1).all() and (attn_b.sum(2) >= 1).all()  # still a monotonic path: one token per frame, every token present


def test_without_bonus_flags_or_mask_the_alignment_is_the_plain_one():
    T, Ty = 10, 40
    b = batch(B=2, T=T, Ty=Ty)
    sil, pm = silent_block(2, Ty, 15, 25), pause_at(2, T, 4)
    model = make_tts().eval()
    with torch.no_grad():
        plain = model(**b)[2]
        assert torch.equal(model(**b, silence=sil, pause_mask=pm)[2], plain)  # bonus 0: flags are ignored
        model.pause_bonus = 50.0
        assert torch.equal(model(**b)[2], plain)  # no flags
        assert torch.equal(model(**b, silence=sil)[2], plain)  # no mask
        assert not torch.equal(model(**b, silence=sil, pause_mask=pm)[2], plain)


def test_an_empty_pause_mask_changes_nothing_even_with_a_huge_bonus():
    T, Ty = 10, 40
    b = batch(B=1, T=T, Ty=Ty)
    model = make_tts().eval()
    with torch.no_grad():
        plain = model(**b)[2]
        model.pause_bonus = 1e3
        attn = model(**b, silence=silent_block(1, Ty, 15, 25), pause_mask=torch.zeros(1, T, dtype=torch.bool))[2]
    assert torch.equal(attn, plain)


def test_the_pause_mask_marks_punctuation_and_the_edges():
    pytest.importorskip("pycantonese")
    try:
        import jyutvoice.data.text_mel_datamodule as dm
        from jyutvoice.text import text_to_sequence
        from jyutvoice.utils.utils import intersperse
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"datamodule imports unavailable: {e!r}")
    ds = dm.TextMelDataset.__new__(dm.TextMelDataset)
    ds.add_blank = True
    ids = text_to_sequence("我 哋 去 食 飯 ， 好 呀 。", "yue")[0]
    ids = torch.tensor(intersperse(ids, 0))
    mask = ds.get_pause_mask(ids)
    assert mask.shape == ids.shape and mask[:3].all() and mask[-3:].all()
    punct = torch.isin(ids, dm.PUNCT_IDS)
    assert punct.sum() >= 2 and mask[punct].all()
    assert mask.sum() < len(ids) / 2  # the speech tokens in the middle are not pause tokens
    ds.add_blank = False
    m = ds.get_pause_mask(torch.tensor(text_to_sequence("我 哋 去 食 飯 ， 好 呀 。", "yue")[0]))
    assert m[0] and m[-1] and not m[1]


def test_collate_pads_silence_and_the_pause_mask_only_when_present():
    pytest.importorskip("pycantonese")
    try:
        import jyutvoice.data.text_mel_datamodule as dm
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"datamodule imports unavailable: {e!r}")
    ds = dm.TextMelDataset.__new__(dm.TextMelDataset)
    ds.add_blank, ds.bert_dim, ds.phone_override, ds.load_durations, ds.n_mels, ds.bert_side = True, 0, {}, False, 80, None
    ds.teacher_prior_weight = 0.0
    T1, T2 = 37, 52
    row = {"id": "t1", "text": "我 哋 去 食 飯 ， 好 呀 。", "lang": "yue", "phone": None, "decoder_h": torch.randn(T1, 80).tolist(), "spk_emb": [0.1] * 192,
           "source": "teacher", "silence": [0] * 10 + [1] * 7 + [0] * 20}  # fmt: skip
    a = ds.get_datapoint(row)
    b = ds.get_datapoint({**row, "id": "t2", "decoder_h": torch.randn(T2, 80).tolist(), "silence": [1] * 5 + [0] * (T2 - 5)})
    c = ds.get_datapoint({k: v for k, v in row.items() if k != "silence"})
    assert c["silence"] is None and c["pause_mask"] is None
    out = dm.TextMelBatchCollate()([a, b, c])
    assert out["silence"].shape == (3, T2) and out["pause_mask"].shape == (3, out["x"].shape[1])
    assert out["silence"][0, :T1].sum() == 7 and out["silence"][0, T1:].sum() == 0 and out["silence"][1, :5].sum() == 5
    assert out["silence"][2].sum() == 0 and not out["pause_mask"][2].any()  # a row without flags never gets a pause bonus
    assert out["pause_mask"][0, : a["pause_mask"].shape[0]].equal(a["pause_mask"])
    assert "silence" not in dm.TextMelBatchCollate()([c, c])
    bad = {**row, "silence": [0] * (T1 - 1)}
    with pytest.raises(AssertionError):
        ds.get_datapoint(bad)


def test_flow_duration_sampler_noise_can_be_set_per_token():
    from omegaconf import DictConfig

    from jyutvoice.models.flow_duration_predictor import FlowDurationPredictor
    from jyutvoice.models.jyutvoice_tts import JyutVoiceTTS
    from jyutvoice.models.text_encoder import TextEncoder

    C, SPK, T = 32, 16, 10
    enc = TextEncoder(
        encoder_type="RoPE Encoder",
        encoder_params=DictConfig({"n_feats": 80, "n_channels": C, "filter_channels": 64, "filter_channels_dp": 16, "n_heads": 2, "n_layers": 3,
                                   "kernel_size": 3, "p_dropout": 0.0, "prenet": False, "gin_channels": SPK}),
        n_vocab=97, n_lang=4, n_tone=7,
    )  # fmt: skip
    dp = FlowDurationPredictor(in_channels=C, filter_channels=32, n_layers=2, bert_dim=0, spk_embed_dim=SPK, infer_cfg_scale=0.0)
    for prm in dp.out_proj.parameters():  # the zero-initialised output layer would make every sample equal the noise; give the velocity net something to do
        torch.nn.init.normal_(prm, std=0.5)
    tts = JyutVoiceTTS(encoder=enc, decoder=None, dp=dp, output_size=80, spk_embed_dim=SPK).eval()
    z = torch.zeros(1, T, dtype=torch.long)
    kw = dict(x=torch.randint(1, 97, (1, T)), x_lengths=torch.tensor([T]), lang=z, tone=z, word_pos=z, syllable_pos=z, spk_embed=torch.randn(1, SPK),
              duration_rounding="cumulative")  # fmt: skip

    def durations(seed, **extra):
        torch.manual_seed(seed)
        with torch.no_grad():
            out = tts.synthesise(**kw, **extra)
        return out["attn"].squeeze().sum(-1)

    mask = pause_at(1, T, 4)
    quiet = [durations(s, dp_temperature=0.0, dp_pause_temperature=0.0, dp_pause_mask=mask) for s in range(3)]
    assert all(torch.equal(quiet[0], q) for q in quiet)  # no noise: the same durations whatever the seed
    mixed = [durations(s, dp_temperature=0.0, dp_pause_temperature=2.0, dp_pause_mask=mask) for s in range(6)]
    torch.manual_seed(0)
    g = torch.nn.functional.normalize(kw["spk_embed"], dim=1)
    with torch.no_grad():
        x_enc, _, x_mask = tts.encoder(kw["x"], kw["x_lengths"], z, z, z, z, g)
        temp = torch.zeros_like(x_mask)
        temp[:, :, 4] = 2.0  # noise on the flagged token only
        runs = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            runs.append(dp(x_enc, x_mask, g=g, temperature=temp))
        silent = [dp(x_enc, x_mask, g=g, temperature=torch.zeros_like(x_mask)) for _ in range(2)]
    assert torch.equal(silent[0], silent[1])  # temperature 0 everywhere: the sampler is deterministic
    assert not torch.equal(runs[0][:, :, 4], runs[1][:, :, 4])  # the flagged token is random
    # (the other tokens move a little too: the output GroupNorm pools statistics over the whole sequence, so noise is not strictly per token)


def test_decoder_prompt_goes_in_front_and_is_cut_from_the_output():
    from omegaconf import DictConfig

    from jyutvoice.models.duration_predictor import DurationPredictor
    from jyutvoice.models.jyutvoice_tts import JyutVoiceTTS
    from jyutvoice.models.text_encoder import TextEncoder

    C, SPK, T = 32, 16, 8

    class Recorder(torch.nn.Module):
        """Stands in for the flow decoder: returns its `mu` and remembers what it was given."""

        def forward(self, mu, mask, spks, cond, n_timesteps, temperature, streaming=False):
            self.seen = dict(mu=mu.clone(), mask=mask.clone(), cond=cond.clone(), spks=spks.clone())
            return mu.clone(), None

    enc = TextEncoder(
        encoder_type="RoPE Encoder",
        encoder_params=DictConfig({"n_feats": 80, "n_channels": C, "filter_channels": 64, "filter_channels_dp": 16, "n_heads": 2, "n_layers": 3,
                                   "kernel_size": 3, "p_dropout": 0.0, "prenet": False, "gin_channels": SPK}),
        n_vocab=97, n_lang=4, n_tone=7,
    )  # fmt: skip
    dec = Recorder()
    tts = JyutVoiceTTS(encoder=enc, decoder=dec, dp=DurationPredictor(in_channels=C, filter_channels=16, kernel_size=3, p_dropout=0.0), output_size=80,
                       spk_embed_dim=SPK).eval()  # fmt: skip
    z = torch.zeros(1, T, dtype=torch.long)
    kw = dict(x=torch.randint(1, 97, (1, T)), x_lengths=torch.tensor([T]), lang=z, tone=z, word_pos=z, syllable_pos=z, spk_embed=torch.randn(1, SPK))
    with torch.no_grad():
        plain = tts.synthesise(**kw)
        n_new = plain["mel"].shape[-1]
        torch.manual_seed(0)
        prompt_feat, prompt_h = torch.randn(1, 30, 80), torch.randn(1, 30, 80)
        out = tts.synthesise(**kw, prompt_feat=prompt_feat, prompt_h=prompt_h)
    seen = dec.seen
    assert seen["mu"].shape == (1, 80, 30 + n_new) and seen["cond"].shape == (1, 80, 30 + n_new) and seen["mask"].shape[-1] == 30 + n_new
    assert torch.equal(seen["mu"][:, :, :30], prompt_h.transpose(1, 2))  # the prompt's states come first
    assert torch.equal(seen["cond"][:, :, :30], prompt_feat.transpose(1, 2)) and seen["cond"][:, :, 30:].abs().sum() == 0  # its mel is the cond, zeros after
    assert out["mel"].shape[-1] == n_new  # only the generated frames come back
    assert torch.allclose(out["mel"], seen["mu"][:, :, 30:])
