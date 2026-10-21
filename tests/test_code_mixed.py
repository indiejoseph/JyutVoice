"""Tests for code-mixed (Chinese + English) text: jyutvoice/text/script_split.py (pure) and, where pycantonese / nltk are installed, the real front end
(jyutvoice.text.cleaners.clean_text).

Run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_code_mixed.py`. The real-g2p tests are skipped without pycantonese, ToJyutping and the NLTK
tagger data (python -m nltk.downloader averaged_perceptron_tagger_eng).
"""
import importlib.util
import os
import re
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# script_split has no package dependencies; importing it as jyutvoice.text.script_split would pull in the whole g2p stack (pycantonese, ...).
_spec = importlib.util.spec_from_file_location("script_split", os.path.join(ROOT, "jyutvoice", "text", "script_split.py"))
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)

PUNCT = ["!", "?", "…", ",", ".", "'", "-"]


def fake_zh(text, padding=False):
    """One phone per character (the character itself), tone 3, marks as marks; like the Cantonese g2p, one word per whitespace word."""
    assert padding is False
    phones, tones, w2p, wp, sp = [], [], [], [], []
    for w in text.split():
        for i, c in enumerate(w):
            phones.append(c)
            tones.append(0 if c in PUNCT else 3)
            w2p.append(1)
            wp.append(1 if i == 0 else 3)
            sp.append(0 if c in PUNCT else 1)
    return phones, tones, w2p, wp, sp, [0] * len(phones)


def fake_en(text, padding=False):
    """One upper-case phone per letter, marks as tokens of their own, `_` between all tokens (like the English g2p), language id 2 everywhere."""
    assert padding is False
    assert not re.search(r"\s[!?…,.'\-]", text), "a mark after a space: the real English g2p would drop it"
    phones, tones, w2p, wp, sp = [], [], [], [], []
    tokens = re.findall(r"[A-Za-z]+|[!?…,.'\-]", text)
    for i, tok in enumerate(tokens):
        for c in tok:
            phones.append(c if c in PUNCT else c.upper())
            tones.append(0)
            w2p.append(1)
            wp.append(1)
            sp.append(0 if c in PUNCT else 1)
        if i != len(tokens) - 1:
            phones.append("_")
            tones.append(0)
            w2p.append(1)
            wp.append(1)
            sp.append(0)
    return phones, tones, w2p, wp, sp, [2] * len(phones)


G2PS = {"zh": fake_zh, "en": fake_en}


def mixed(text, **kw):
    return ss.mixed_g2p(text, G2PS, punct=PUNCT, **kw)


def test_split_latin_keeps_numbers_and_contractions_whole():
    assert ss.split_latin("$999.") == [("$999", "latin"), (".", "other")]
    assert ss.split_latin("3.5%") == [("3.5%", "latin")]
    assert ss.split_latin("1,200") == [("1,200", "latin")]
    assert ss.split_latin("wi-fi") == [("wi-fi", "latin")]
    assert ss.split_latin("don’t") == [("don't", "latin")]  # typographic apostrophe


def test_split_latin_splits_at_the_script_boundary():
    assert ss.split_latin("order咗") == [("order", "latin"), ("咗", "other")]
    assert ss.split_latin("meeting，order") == [("meeting", "latin"), ("，", "other"), ("order", "latin")]


def test_a_number_touching_chinese_is_not_latin():
    assert ss.split_latin("有3個") == [("有", "other"), ("3", "number"), ("個", "other")]
    assert ss.split_latin("3G") == [("3G", "latin")]  # has a letter: English
    assert ss.split_latin("15") == [("15", "latin")]  # alone: English


def test_bare_numbers_are_english_only_after_an_english_word():
    f = ss.drop_chinese_context_numbers
    assert f("我 哋 用 iPhone 15 打 電話".split()) == "我 哋 用 iPhone 15 打 電話".split()
    assert f("我 有 3 個 meeting".split()) == "我 有 個 meeting".split()
    assert f("Windows 11 同 12 個".split()) == "Windows 11 同 個".split()
    assert f("iPhone 15 256 GB".split()) == "iPhone 15 256 GB".split()  # a run of numbers after an English word
    assert f("我 有 3.5% 同 $9".split()) == "我 有 同".split()


def test_has_latin_ignores_digits_and_cjk():
    assert ss.has_latin("我今日有個meeting")
    assert not ss.has_latin("我有3個，好嘢！")
    assert not ss.has_latin("１２３")


def test_tokenize_kinds():
    units = ss.tokenize("我 今日 有 個 meeting , order咗 杯 coffee .", PUNCT)
    assert units == [
        ("我", "zh"), ("今日", "zh"), ("有", "zh"), ("個", "zh"), ("meeting", "en"), (",", "punct"),
        ("order", "en"), ("咗", "zh"), ("杯", "zh"), ("coffee", "en"), (".", "punct"),
    ]  # fmt: skip


def test_tokenize_drops_stray_characters_and_touching_numbers():
    assert ss.tokenize("我 有3個 @ meeting", PUNCT) == [("我", "zh"), ("有", "zh"), ("個", "zh"), ("meeting", "en")]


def test_marks_inside_a_chinese_word_stay_in_it():
    assert ss.tokenize("咗杯, hello", PUNCT) == [("咗杯,", "zh"), ("hello", "en")]


def test_groups_keep_word_segmentation_and_marks_join_the_group_before():
    units = ss.tokenize("我 今日 meeting , order 咗 .", PUNCT)
    assert ss.group_units(units) == [("zh", ["我", "今日"]), ("en", ["meeting,", "order"]), ("zh", ["咗", "."])]  # comma glued in English, a word in Chinese


def test_leading_mark_joins_the_first_group():
    assert ss.group_units(ss.tokenize(", 你好", PUNCT)) == [("zh", [",", "你好"])]
    assert ss.group_units(ss.tokenize(", hello", PUNCT)) == [("en", [",hello"])]


def test_lists_stay_aligned_and_word2ph_covers_the_phones():
    for text in ["我 今日 有 個 meeting , order咗 杯 coffee .", "hello world .", "你好 .", ", hello"]:
        phones, tones, w2p, wp, sp, lang = mixed(text)
        assert len(phones) == len(tones) == len(wp) == len(sp) == len(lang) == sum(w2p), text
        assert phones[0] == phones[-1] == "_" and w2p[0] == w2p[-1] == 1 and tones[0] == tones[-1] == 0, text


def test_language_ids_marks_follow_the_matrix_language():
    phones, _, _, _, _, lang = mixed("我 今日 有 個 meeting , order 咗 杯 coffee .", zh_lang="yue", padding=False)
    by_ph = list(zip(phones, lang))
    assert ("M", 2) in by_ph and ("C", 2) in by_ph  # English letters: 2
    assert ("今", 0) in by_ph  # Cantonese: 0
    assert [l for p, l in by_ph if p in PUNCT] == [0, 0]  # the comma after meeting and the full stop after coffee are Cantonese-matrix marks
    assert lang[-1] == 0  # last token is the mark after English "coffee"


def test_pads_and_marks_follow_zh_lang_for_mandarin_text():
    _, _, _, _, _, lang = mixed("你 好 hello .", zh_lang="zh")
    assert lang[0] == lang[-1] == 1 and lang[-2] == 1


def test_pure_english_text_is_matrix_english():
    phones, _, _, _, _, lang = mixed("hello world .", zh_lang="yue")
    assert lang == [2] * len(phones)  # pads and the mark are English too


def test_no_padding_option():
    phones, tones, w2p, wp, sp, lang = mixed("你好 hello", padding=False)
    assert phones[0] != "_" and phones[-1] != "_"
    assert len(phones) == sum(w2p)


def test_english_word_boundary_token_only_between_english_words():
    phones, *_ = mixed("你好 foo bar 你好", padding=False)
    assert phones == ["你", "好", "F", "O", "O", "_", "B", "A", "R", "你", "好"]


# ---- the real front end -------------------------------------------------------------------------------------------------------------------------------

def _real():
    pytest.importorskip("pycantonese")
    pytest.importorskip("ToJyutping")
    nltk = pytest.importorskip("nltk")
    try:
        nltk.pos_tag(["a"])
    except LookupError as e:  # NLTK tagger data missing
        pytest.skip(f"NLTK data missing: {e}")
    from jyutvoice.text import cleaners

    return cleaners


def test_real_cantonese_text_without_latin_is_unchanged():
    cl = _real()
    from jyutvoice.text.cantonese.g2p import g2p as yue

    text = "佢 哋 今日 好 開心 ， 你 呢 ？"
    norm, phones, tones, word_pos, syllable_pos, lang, w2p = cl.clean_text(text, "yue", return_word2ph=True)
    ref = yue(" ".join(cl.text_normalize(w, "yue") for w in text.split()))
    assert (phones, tones, w2p, word_pos, syllable_pos, lang) == (ref[0], ref[1], ref[2], ref[3], ref[4], ref[5])
    assert set(lang) == {0}


def test_real_code_mixed_cantonese_keeps_the_english_words():
    cl = _real()
    norm, phones, tones, word_pos, syllable_pos, lang, w2p = cl.clean_text("我 今日 有 個 meeting ， order 咗 杯 coffee 。", "yue", return_word2ph=True)
    assert set(lang) == {0, 2}
    assert len(phones) == len(tones) == len(word_pos) == len(syllable_pos) == len(lang) == sum(w2p)
    english = [p for p, l in zip(phones, lang) if l == 2 and p not in ("_", ",", ".")]
    assert "m" in english and "iy" in english  # "meeting": m iy t ih ng
    assert [l for p, l in zip(phones, lang) if p in (",", ".")] == [0, 0]
    assert phones[0] == phones[-1] == "_"


def test_real_numbers_in_a_code_mixed_sentence():
    cl = _real()
    norm, *_ = cl.clean_text("我 哋 用 iPhone 15 打 電話 。", "yue")
    assert "fifteen" in norm  # right after an English word: English
    norm, *_ = cl.clean_text("我 有 3 個 meeting 。", "yue")
    assert "three" not in norm and "3" not in norm  # a Cantonese number: no reader, dropped (as in plain Chinese text)


def test_real_mandarin_with_english():
    cl = _real()
    _, phones, _, _, _, lang = cl.clean_text("我 今天 有 个 meeting 。", "zh")
    assert set(lang) == {1, 2}
    assert lang[-1] == 1  # the pad: matrix language


def test_real_english_numbers_and_symbols_are_read():
    cl = _real()
    norm, phones, *_ = cl.clean_text("The iPhone 15 costs $999.", "en")
    assert "$" not in norm and "15" not in norm
    assert "fifteen" in norm and "dollars" in norm
    assert "$" not in phones


def test_real_english_acronym_and_brand_words():
    cl = _real()
    norm, phones, tones, word_pos, syllable_pos, lang, w2p = cl.clean_text("OK DSPGAN iPhone", "en", return_word2ph=True)
    assert "iPhone" in norm and "O K" not in norm  # no camel-case / all-caps splitting in the normaliser
    assert len(phones) == sum(w2p)


def test_split_phrases_keeps_english_phrases_whole_for_the_segmenter():
    assert ss.split_phrases("最近real estate嘅價格，我哋用iPhone 15打電話。") == [
        ("最近", False), ("real estate", True), ("嘅價格，我哋用", False), ("iPhone 15", True), ("打電話。", False)]
    assert ss.split_phrases("我好enjoy, but 唔得！") == [("我好", False), ("enjoy, but", True), (" 唔得！", False)]
    assert ss.split_phrases("好似fishing!") == [("好似", False), ("fishing", True), ("!", False)]  # ! ? are not part of a phrase (group_units glues them back)
    assert ss.split_phrases("今日天氣好。") == [("今日天氣好。", False)]
    assert ss.split_phrases("user-friendly同$20") == [("user-friendly", True), ("同", False), ("$20", True)]
