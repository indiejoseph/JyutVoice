"""Tests for jyutvoice/text/text_processor.py (the WeTextProcessing stage before segmentation and g2p).

Run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_text_processor.py`. The routing, space handling, cache fingerprint and fallback tests are pure; the
real-normaliser tests are skipped without the indiejoseph/WeTextProcessing fork (and pynini, opencc-python-reimplemented); the first English graph build takes about
a minute, then it is cached in $JYUTVOICE_TN_CACHE.
"""
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# text_processor has no package dependencies; importing it as jyutvoice.text.text_processor would pull in the whole g2p stack (pycantonese, ...).
_spec = importlib.util.spec_from_file_location("text_processor", os.path.join(ROOT, "jyutvoice", "text", "text_processor.py"))
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)


@pytest.fixture(autouse=True)
def fresh_state(request, monkeypatch, tmp_path):
    monkeypatch.setattr(tp, "_normalizers", {})
    monkeypatch.setattr(tp, "_warned", False)
    monkeypatch.delenv("JYUTVOICE_TN", raising=False)
    if not request.node.name.startswith("test_real"):  # the real normalisers keep their compiled graphs in the usual cache; the pure tests get an empty one
        monkeypatch.setenv("JYUTVOICE_TN_CACHE", str(tmp_path / "tn"))


class FakeNormalizer:
    def __init__(self, tag):
        self.tag = tag

    def normalize(self, text):
        return f"[{self.tag}]{text}"


def test_routing_chinese_text_goes_to_the_language_normaliser_and_other_text_to_english():
    assert tp.pick_kind("我有3個meeting", "yue") == "yue"
    assert tp.pick_kind("我有3個meeting", "zh") == "zh"
    assert tp.pick_kind("我有3個meeting", "multilingual") == "yue"
    assert tp.pick_kind("The iPhone 15 costs $999.", "en") == "en"
    assert tp.pick_kind("我有3個meeting", "en") == "en"  # lang en: English normaliser whatever the text is
    assert tp.pick_kind("meeting at 3pm", "yue") == "en"  # no Chinese character: nothing for the Cantonese rules to read around


def test_squeeze_spaces_keeps_only_the_spaces_inside_english_phrases():
    assert tp.squeeze_spaces("我 今日 有 個 meeting , order 咗 iced coffee 。") == "我今日有個meeting,order咗iced coffee。"
    assert tp.squeeze_spaces("a  b   c") == "a b c"  # runs collapse to one space
    assert tp.squeeze_spaces("  你 好  ") == "你好"


def test_prepare_text_normalises_before_segmenting(monkeypatch):
    monkeypatch.setitem(tp._normalizers, ("yue", False), FakeNormalizer("粵"))  # a tag without Latin letters: those would be taken for English
    seen = []

    def segmenter(t):
        seen.append(t)
        return list(t)

    out = tp.prepare_text("我 有 3 個", "yue", segmenter)
    assert seen == ["[粵]我有3個"]  # the normaliser saw the raw text, the segmenter the normalised one without segmentation spaces
    assert out == " ".join("[粵]我有3個")


def test_prepare_text_without_segmenter_or_chinese_only_normalises(monkeypatch):
    monkeypatch.setitem(tp._normalizers, ("yue", False), FakeNormalizer("yue"))
    monkeypatch.setitem(tp._normalizers, ("en", False), FakeNormalizer("en"))
    assert tp.prepare_text("我有3個", "yue") == "[yue]我有3個"
    assert tp.prepare_text("meeting at 3pm", "yue", segmenter=lambda t: ["never called"]) == "[en]meeting at 3pm"


def test_joined_english_tokens_are_hidden_from_the_chinese_rules_and_restored(monkeypatch):
    seen = []

    class Spy:
        def normalize(self, text):
            seen.append(text)
            return text

    monkeypatch.setitem(tp._normalizers, ("yue", False), Spy())
    text = "呢個app好user-friendly，COVID-19之後80s嘅Wi-Fi，AT&T同google.com都得，170cm"
    assert tp.normalize_text(text, "yue") == text  # restored exactly
    assert "-" not in seen[0] and "COVID" not in seen[0] and "80s" not in seen[0] and "&" not in seen[0]  # the Chinese rules never saw them
    assert "170cm" in seen[0]  # units the Chinese rules read well stay visible to them
    assert "ZQAA" in seen[0] and "ZQAF" in seen[0]


def test_numbers_with_a_joiner_are_not_protected(monkeypatch):
    seen = []

    class Spy:
        def normalize(self, text):
            seen.append(text)
            return text

    monkeypatch.setitem(tp._normalizers, ("yue", False), Spy())
    tp.normalize_text("2024/3/8同3.5同10-20同6:30am", "yue")
    assert seen == ["2024/3/8同3.5同10-20同6:30am"]


def test_text_with_the_placeholder_alphabet_is_left_alone(monkeypatch):
    seen = []

    class Spy:
        def normalize(self, text):
            seen.append(text)
            return text

    monkeypatch.setitem(tp._normalizers, ("yue", False), Spy())
    tp.normalize_text("我ZQAA同user-friendly", "yue")
    assert seen == ["我ZQAA同user-friendly"]


def test_lone_d_is_the_cantonese_particle_in_cantonese_text_only(monkeypatch):
    seen = {}

    class Spy:
        def __init__(self, k):
            self.k = k

        def normalize(self, text):
            seen[self.k] = text
            return text

    monkeypatch.setitem(tp._normalizers, ("yue", False), Spy("yue"))
    monkeypatch.setitem(tp._normalizers, ("zh", False), Spy("zh"))
    monkeypatch.setitem(tp._normalizers, ("en", False), Spy("en"))
    tp.normalize_text("著厚d嘅clothes，d 80s vibes，Vitamin D，4d，don'd，red", "yue")
    assert seen["yue"] == "著厚啲嘅clothes，啲 ZQAA vibes，Vitamin D，4d，ZQAB，red"  # 4d, D and red are not the particle; don'd is an English token (hidden)
    tp.normalize_text("the d key", "en")
    assert seen["en"] == "the d key"
    tp.normalize_text("好d", "zh")
    assert seen["zh"] == "好d"


def test_fullwidth_letters_digits_and_signs_are_folded_to_ascii_but_marks_stay(monkeypatch):
    seen = []

    class Spy:
        def normalize(self, text):
            seen.append(text)
            return text

    monkeypatch.setitem(tp._normalizers, ("yue", False), Spy())
    tp.normalize_text("ＷｉＦｉ密碼係＄２０，３．５％！", "yue")
    assert seen == ["WiFi密碼係$20，3.5%！"]


def test_empty_text_is_returned_as_is():
    assert tp.normalize_text("", "yue") == ""
    assert tp.normalize_text("  ", "yue") == "  "


def test_without_the_package_text_comes_back_unchanged_with_one_warning(monkeypatch, caplog):
    def no_package(*a, **k):
        raise ImportError("No module named 'tn'")

    monkeypatch.setattr(tp, "_make", no_package)
    with caplog.at_level("WARNING"):
        assert tp.normalize_text("我有3個meeting", "yue") == "我有3個meeting"
        assert tp.normalize_text("我有4個meeting", "yue") == "我有4個meeting"
    assert sum("WeTextProcessing is not available" in r.message for r in caplog.records) == 1


def test_required_mode_raises_without_the_package(monkeypatch):
    def no_package(*a, **k):
        raise ImportError("No module named 'tn'")

    monkeypatch.setattr(tp, "_make", no_package)
    monkeypatch.setenv("JYUTVOICE_TN", "required")
    with pytest.raises(ImportError, match="indiejoseph/WeTextProcessing"):
        tp.normalize_text("我有3個meeting", "yue")


def test_cache_is_rebuilt_unless_it_was_built_with_the_current_options(monkeypatch, tmp_path):
    calls = []

    def fake_make(kind, simple_to_traditional, overwrite):
        calls.append((kind, overwrite))
        return FakeNormalizer(kind)

    monkeypatch.setattr(tp, "_make", fake_make)
    stamp = tp.cache_dir() / "yue_tn.options.json"
    tp.get_normalizer("yue")
    assert calls == [("yue", True)] and json.loads(stamp.read_text()) == tp.OPTIONS["yue"]  # no stamp: compile and write it
    tp._normalizers.clear()
    tp.get_normalizer("yue")
    assert calls[-1] == ("yue", False)  # stamp matches: load the cached graph
    tp._normalizers.clear()
    stamp.write_text(json.dumps({**tp.OPTIONS["yue"], "tag_oov": True}))  # a graph built with the fork's shipped options (oov tags on)
    tp.get_normalizer("yue")
    assert calls[-1] == ("yue", True)
    assert json.loads(stamp.read_text()) == tp.OPTIONS["yue"]


def test_options_keep_real_syllables():
    for kind in ("yue", "zh"):
        assert tp.OPTIONS[kind]["remove_interjections"] is False  # 啊 / 喇 are spoken
        assert tp.OPTIONS[kind]["remove_erhua"] is False
        assert tp.OPTIONS[kind]["tag_oov"] is False  # 係 / 咗 / 哋 must not be wrapped in <oov>
        assert tp.OPTIONS[kind]["traditional_to_simple"] is False


# ---- the real normalisers -----------------------------------------------------------------------------------------------------------------------------

def _real(kind):
    pytest.importorskip("pynini")
    pytest.importorskip("opencc")
    try:
        import tn.cantonese.normalizer  # noqa: F401  (upstream WeTextProcessing has no Cantonese)
    except ImportError:
        pytest.skip("the indiejoseph/WeTextProcessing fork is not installed")
    return lambda text, lang="yue": tp.normalize_text(text, lang)


def test_real_cantonese_numbers_money_dates_and_times():
    norm = _real("yue")
    assert norm("我有3個meeting，$20") == "我有三個meeting，二十蚊"
    assert norm("2024年10月21日") == "二零二四年十月二十一日"
    assert norm("下午3:30開會") == "下午三點半開會"
    assert norm("電話號碼係98765432") == "電話號碼係九八七六五四三二"
    assert norm("50%折扣") == "百分之五十折扣"


def test_real_cantonese_keeps_colloquial_characters_and_particles():
    norm = _real("yue")
    assert norm("佢哋係咪食咗飯呀？") == "佢哋係咪食咗飯呀？"  # untouched, marks included: the segmenter must see what it saw in the corpus
    assert norm("好啊，喇！") == "好啊，喇！"
    assert "<oov>" not in norm("嘅咗啲喺")


def test_real_english_words_in_chinese_text_are_left_alone():
    norm = _real("yue")
    assert norm("我今日有個meeting，order咗杯coffee。") == "我今日有個meeting，order咗杯coffee。"
    assert norm("我哋用iPhone 15打電話") == "我哋用iPhone 十五打電話"  # the number is read in Cantonese, the brand stays


def test_real_joined_english_tokens_and_lone_d():
    norm = _real("yue")
    assert norm("呢個app好user-friendly，COVID-19之後") == "呢個app好user-friendly，COVID-19之後"  # not user減friendly / COVID負十九
    assert norm("喜歡80s嘅歌") == "喜歡80s嘅歌"  # not 八十秒
    assert norm("著厚d嘅clothes") == "著厚啲嘅clothes"
    assert norm("身高170cm，6:30am") == "身高一百七十釐米，上午六點半"


def test_real_fullwidth_text():
    norm = _real("yue")
    assert norm("２０２４年１０月２１日，價錢係＄２０。") == "二零二四年十月二十一日，價錢係二十蚊。"
    assert norm("ＷｉＦｉ好快") == "WiFi好快"


def test_real_english_numbers_and_money():
    norm = _real("en")
    assert norm("The iPhone 15 costs $999.", "en") == "The iPhone fifteen costs nine hundred and ninety nine dollars."
    assert norm("I got 100% on the test.", "en") == "I got one hundred percent on the test."


def test_segment_mixed_keeps_english_phrases_whole_and_cuts_only_the_chinese(monkeypatch):
    pytest.importorskip("pycantonese")  # the package import of script_split pulls in the g2p stack
    cut = []

    def segmenter(t):
        cut.append(t)
        return list(t)  # one word per character; would cut English words apart

    words = tp.segment_mixed("最近 real  estate嘅價格，好 fair 嘅 市場 。", segmenter)
    assert words == ["最", "近", "real estate", "嘅", "價", "格", "，", "好", "fair", "嘅", "市", "場", "。"]
    assert cut == ["最近", "嘅價格，好", "嘅市場。"]  # the segmenter never saw an English word


def test_segment_mixed_without_english_is_the_plain_segmenter_call():
    seen = []

    def segmenter(t):
        seen.append(t)
        return [t[:2], t[2:]]

    assert tp.segment_mixed("今日 天氣 好 。", segmenter) == ["今日", "天氣好。"]
    assert seen == ["今日天氣好。"]  # spaces removed, one call (no package import needed for Chinese-only text)
