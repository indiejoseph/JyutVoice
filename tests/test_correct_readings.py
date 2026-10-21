"""Tests for scripts/correct_readings.py (voting logic, needs ToJyutping) and jyutvoice/text/cantonese/readings.py (needs the g2p stack).

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_correct_readings.py
"""
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for path in (ROOT, os.path.join(ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

pytest.importorskip("ToJyutping")
import correct_readings as cr  # noqa: E402


# --------------------------------------------------------------------------------------------------------------------- voting


def test_documented_mergers_are_recognised_and_other_differences_are_not():
    assert cr.merger_only("wai6", "wai4") and cr.merger_only("wui5", "wui2") and cr.merger_only("gam3", "gam6")  # tone 4/6, 2/5, 3/6
    assert cr.merger_only("lei5", "nei5") and cr.merger_only("ok3", "ngok3") and cr.merger_only("gwok3", "gok3")  # onset n/l, ng/-, gw/g
    assert not cr.merger_only("fan1", "fan6")  # tone 1/6 is no merger
    assert not cr.merger_only("haang4", "hang4")  # rime
    assert not cr.merger_only("lei5", "nei6")  # onset merger but the tone differs too


def test_decide_needs_enough_agreeing_models_and_ignores_ties_and_agreement_with_g2p():
    assert cr.decide("fan1", {"a": "fan6", "b": "fan6", "c": "fan1"}) == ("fan6", 2)
    assert cr.decide("fan1", {"a": "fan6", "b": "fan1", "c": "fan1"}) is None  # one model is not enough
    assert cr.decide("fan1", {"a": "fan1", "b": "fan1", "c": "fan6"}) is None
    assert cr.decide("fan1", {"a": "fan6", "b": "fan3"}) is None  # tie between two different answers
    assert cr.decide("fan1", {"a": "fan6", "b": "fan6"}, min_agree=3) is None
    assert cr.decide("fan1", {}) is None


def test_merger_pair_corrections_need_every_model():
    assert cr.decide("wai6", {"a": "wai4", "b": "wai4", "c": "wai6"}) is None  # 4/6: 2 of 3 is not enough
    assert cr.decide("wai6", {"a": "wai4", "b": "wai4", "c": "wai4"}) == ("wai4", 3)
    assert cr.decide("wai6", {"a": "wai4", "b": "wai4"}) == ("wai4", 2)  # only two models aligned this character: both agree
    assert cr.decide("wai6", {"a": "wai4", "b": "wai4", "c": "wai6"}, merger_agree=2) == ("wai4", 2)


def test_clean_matches_the_cleaners_rules():
    assert cr.clean("好嘅，咁就得！ 你好…") == "好嘅,咁就得!你好..."  # full-width marks become ASCII, '…' becomes three dots, spaces go
    assert cr.clean("OK 好 abc 123") == "好"  # everything that is not Chinese or punctuation is dropped, as in replace_punctuation
    assert cr.clean("（好）「嘅」") == "'好''嘅'"


def decode(*syllables):
    return " ".join(syllables)


def test_clip_overrides_corrects_only_the_agreed_position():
    # 百分之: g2p reads 分 as fan1; two of three models heard fan6
    text = "百分之五十"
    units = cr.rp.text_to_syllables(cr.clean(text), True, all_units=True)
    default = [r for _, r in units]
    fixed = list(default)
    fixed[1] = "fan6"
    wrong = list(default)
    wrong[1] = "fan3"
    norm, overrides, changes, han, reverted = cr.clip_overrides(text, {"a": decode(*fixed), "b": decode(*fixed), "c": decode(*default)})
    assert norm == "百分之五十" and han == list("百分之五十")
    assert overrides == {1: "fan6"} and changes[0]["char"] == "分" and changes[0]["prev"] == "百" and changes[0]["next"] == "之"
    assert changes[0]["default"] == default[1] and changes[0]["agree"] == 2 and not changes[0]["merger"]
    # no agreement -> no change
    assert cr.clip_overrides(text, {"a": decode(*fixed), "b": decode(*wrong), "c": decode(*default)})[1] == {}


def test_clip_overrides_ignores_unrelated_decodes_and_keeps_unit_indices_across_punctuation():
    text = "好嘅，百分之五十。"
    units = cr.rp.text_to_syllables(cr.clean(text), True, all_units=True)
    assert [c for c, _ in units][3] == "百"  # the comma is unit 2, so 分 is unit 4
    han_syllables = [r for c, r in units if cr.rp.HAN.match(c)]
    fixed = list(han_syllables)
    fixed[3] = "fan6"  # 分 is the 4th Chinese syllable
    _, overrides, _, _, _ = cr.clip_overrides(text, {"a": decode(*fixed), "b": decode(*fixed)})
    assert overrides == {4: "fan6"}
    garbage = list(han_syllables)
    garbage[3] = "zyut6"  # shares nothing with fan1: an alignment slip, not a reading
    assert cr.clip_overrides(text, {"a": decode(*garbage), "b": decode(*garbage)})[1] == {}


# ---------------------------------------------------------------------------------------- readings.py with the real g2p


@pytest.fixture(scope="module")
def g2p_stack():
    pytest.importorskip("pycantonese")
    from jyutvoice.text.cantonese.g2p import g2p
    from jyutvoice.text.cantonese.readings import cleaned, jyutping_with_overrides

    return g2p, cleaned, jyutping_with_overrides


TEXTS = ["佢 哋 去 搵 嘢 ， 好 唔 好 ？", "百分之 五十 嘅 銀行 局長 … 即係 返 屋企", "我 喐 咗 少少 。", "專業 講解員 唔 單止 帶領 遊客 ！"]


@pytest.mark.parametrize("text", TEXTS)
def test_overrides_reproduce_g2p_defaults_when_there_are_none(g2p_stack, text):
    g2p, cleaned, jyutping_with_overrides = g2p_stack
    norm = cleaned(text)
    assert g2p(norm, jyutping=jyutping_with_overrides(text)) == g2p(norm)
    assert g2p(norm, jyutping=jyutping_with_overrides(text, {})) == g2p(norm)


def test_an_override_changes_only_that_syllables_phones(g2p_stack):
    from jyutvoice.text.cantonese.readings import default_jyutping

    g2p, cleaned, jyutping_with_overrides = g2p_stack
    text = "百分之 五十"
    norm = cleaned(text)
    assert norm.replace(" ", "")[1] == "分"
    default = default_jyutping(norm)[1]
    other = default[:-1] + ("1" if default[-1] != "1" else "6")  # same syllable, a different tone
    base = g2p(norm)
    fixed = g2p(norm, jyutping=jyutping_with_overrides(text, {1: other}))
    assert base[2] == fixed[2] and len(base[0]) == len(fixed[0])  # same shape: only the tone of one syllable changed
    differing = [i for i in range(len(base[0])) if (base[0][i], base[1][i]) != (fixed[0][i], fixed[1][i])]
    start = base[2][0] + base[2][1]  # the leading pad and 百 come first
    assert differing and all(start <= i < start + base[2][2] for i in differing)
    assert {fixed[1][i] for i in differing} == {int(other[-1])}


def test_bad_overrides_are_refused(g2p_stack):
    _, _, jyutping_with_overrides = g2p_stack
    with pytest.raises(ValueError, match="outside"):
        jyutping_with_overrides("百分之 五十", {99: "fan6"})
    with pytest.raises(ValueError, match="punctuation"):
        jyutping_with_overrides("好 嘅 ， 唔 好", {2: "fan6"})
