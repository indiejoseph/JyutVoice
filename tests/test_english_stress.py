"""Tests for English stress as encoder input: jyutvoice/text/english/stress.py (pure), the tone table of TextEncoder (old 7-row checkpoints still load), and,
where the g2p dependencies are installed, the real English g2p.

Run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_english_stress.py`. The real-g2p tests are skipped without pycantonese / nltk data.
"""
import importlib.util
import os
import sys

import pytest
import torch
from omegaconf import DictConfig
from torch import nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# stress.py has no dependencies; importing it as jyutvoice.text.english.stress would pull in the whole g2p stack.
_spec = importlib.util.spec_from_file_location("stress", os.path.join(ROOT, "jyutvoice", "text", "english", "stress.py"))
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)


def test_stress_has_tone_ids_of_its_own_not_the_cantonese_ones():
    assert st.arpa_tone("UW1") == ("uw", 9)
    assert st.arpa_tone("AH0") == ("ah", 8)
    assert st.arpa_tone("AE2") == ("ae", 10)
    assert st.arpa_tone("G") == ("g", 7)
    assert st.arpa_tone("NG") == ("ng", 7)
    used = {st.arpa_tone(p)[1] for p in ["AH0", "AH1", "AH2", "G", "K"]}
    assert used.isdisjoint(range(0, 7))  # 0 pad / punctuation, 1-6 Cantonese tones
    assert max(used) < st.N_TONE


def test_is_name_like():
    assert all(st.is_name_like(w) for w in ["Google", "Tokyo", "iPhone", "YouTube", "WhatsApp"])
    assert not any(st.is_name_like(w) for w in ["google", "hiking", "NASA", "OK"])


def test_initial_stress_puts_the_primary_stress_on_the_first_vowel():
    assert st.initial_stress(["W", "AH2", "T", "S", "AE1", "P"]) == ["W", "AH1", "T", "S", "AE2", "P"]  # WhatsApp: guessed stress on `app`
    assert st.initial_stress(["IH2", "N", "S", "T", "AH0", "G", "R", "AE2", "M"]) == ["IH1", "N", "S", "T", "AH0", "G", "R", "AE2", "M"]  # Instagram: no primary
    assert st.initial_stress(["G", "UW1", "G", "AH0", "L"]) == ["G", "UW1", "G", "AH0", "L"]  # already right
    assert st.initial_stress(["F", "EY1", "S", "B", "UH1", "K"]) == ["F", "EY1", "S", "B", "UH2", "K"]  # one primary stress per name
    assert st.initial_stress(["K", "AH0", "L"]) == ["K", "AH1", "L"]
    assert st.initial_stress(["S", "T"]) == ["S", "T"]  # no vowel: unchanged
    pron = ["W", "AH2", "T"]
    st.initial_stress(pron)
    assert pron == ["W", "AH2", "T"]  # the input is not modified


# ---- the tone table of the encoder ----------------------------------------------------------------------------------------------------------------------

from jyutvoice.models.text_encoder import TextEncoder  # noqa: E402

N_CH = 32


def make_encoder(n_tone):
    return TextEncoder(
        encoder_type="RoPE Encoder",
        encoder_params=DictConfig(
            {"n_feats": 80, "n_channels": N_CH, "filter_channels": 64, "filter_channels_dp": 16, "n_heads": 2, "n_layers": 3, "kernel_size": 3,
             "p_dropout": 0.0, "prenet": False, "gin_channels": 16}
        ),
        n_vocab=97,
        n_lang=4,
        n_tone=n_tone,
    )  # fmt: skip


class Holder(nn.Module):
    """The encoder under a prefix, as in JyutVoiceTTS (`encoder.tone_emb.weight`)."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder


def test_default_tone_table_has_the_english_ids():
    assert TextEncoder.__init__.__defaults__[0] == st.N_TONE


def test_a_seven_row_checkpoint_loads_into_the_larger_tone_table():
    torch.manual_seed(0)
    old = Holder(make_encoder(7)).eval()
    new = Holder(make_encoder(st.N_TONE)).eval()
    fresh = new.encoder.tone_emb.weight.detach().clone()
    result = new.load_state_dict(old.state_dict())  # strict: no missing or unexpected keys
    assert not result.missing_keys and not result.unexpected_keys
    w = new.encoder.tone_emb.weight
    assert torch.equal(w[:7], old.encoder.tone_emb.weight)  # the trained rows are kept
    assert torch.equal(w[7:], fresh[7:])  # the new ones keep their initialisation

    # Cantonese input gives the same output as before
    B, T = 2, 9
    ids = torch.randint(0, 4, (B, T))
    args = dict(x=torch.randint(1, 97, (B, T)), x_lengths=torch.full((B,), T), lang=ids, tone=torch.randint(0, 7, (B, T)), word_pos=ids.clone(),
                syllable_pos=ids.clone(), g=torch.randn(B, 16))
    with torch.no_grad():
        a = old.encoder(**args)
        b = new.encoder(**args)
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(a[:2], b[:2]))
    args["tone"] = torch.randint(7, st.N_TONE, (B, T))  # English ids run through the new table
    with torch.no_grad():
        assert torch.isfinite(new.encoder(**args)[1]).all()


def test_a_full_size_checkpoint_loads_unchanged_and_a_larger_one_does_not_shrink():
    a = Holder(make_encoder(st.N_TONE))
    b = Holder(make_encoder(st.N_TONE))
    b.load_state_dict(a.state_dict())
    assert torch.equal(a.encoder.tone_emb.weight, b.encoder.tone_emb.weight)
    with pytest.raises(RuntimeError, match="size mismatch"):
        Holder(make_encoder(7)).load_state_dict(a.state_dict())


# ---- the real English g2p -------------------------------------------------------------------------------------------------------------------------------


def _real_g2p():
    pytest.importorskip("pycantonese")
    nltk = pytest.importorskip("nltk")
    try:
        nltk.pos_tag(["a"])
    except LookupError as e:
        pytest.skip(f"NLTK data missing: {e}")
    from jyutvoice.text.english import g2p as g

    return g


VOWELS = {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"}


def _stress(g, word):
    phones, tones, *_ = g.g2p(word, padding=False)
    return "".join(str(t - 8) for p, t in zip(phones, tones) if p in VOWELS), phones, tones


def test_real_english_tones_never_use_the_cantonese_ids():
    g = _real_g2p()
    phones, tones, *_ = g.g2p("Google is my favourite search engine, really!", padding=False)
    assert set(tones) <= {0} | set(range(7, st.N_TONE))
    assert all(t == 0 for p, t in zip(phones, tones) if p in {",", "!", "_"})


def test_real_brand_names_are_stressed_on_the_first_syllable_once():
    g = _real_g2p()
    # Google, YouTube: dictionary entries; WhatsApp, Instagram: corrected in engdict-hot.rep (the dictionary had the stress on `app` / none);
    # Netflix, TikTok, WeChat, Klook, Xiaomi: not in any dictionary (the guess gave nearly every syllable a primary stress)
    for name in ["Google", "YouTube", "WhatsApp", "Instagram", "Netflix", "TikTok", "WeChat", "Klook", "Xiaomi"]:
        stress, _, _ = _stress(g, name)
        assert stress[0] == "1", (name, stress)
        assert "1" not in stress[1:], (name, stress)


def test_real_dictionary_names_keep_a_later_stress():
    g = _real_g2p()
    # no first-syllable rule for the words the dictionary knows: these are stressed on a later syllable
    for name in ["Japan", "Macau", "Berlin", "Chanel", "Toyota", "Nintendo", "IKEA", "Decathlon"]:
        stress, _, _ = _stress(g, name)
        assert stress.index("1") >= 1, (name, stress)


def test_real_hot_dictionary_exceptions_to_the_name_rule():
    g = _real_g2p()
    for name in ["Deliveroo", "Lazada", "Uniqlo", "Zalando"]:  # brand names stressed on a later syllable, listed in engdict-hot.rep
        stress, _, _ = _stress(g, name)
        assert stress.index("1") >= 1, (name, stress)
