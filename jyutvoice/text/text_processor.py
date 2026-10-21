"""Written text -> spoken form (numbers, dates, times, money, units, symbols), the stage BEFORE word segmentation and g2p.

Wraps WeTextProcessing, the indiejoseph fork that adds Cantonese to the Mandarin and English normalisers:

    pip install git+https://github.com/indiejoseph/WeTextProcessing.git opencc-python-reimplemented   (needs pynini)

It must see the raw sentence: on word-segmented text the spaces break its patterns (`2024 年 10 月` is read as a cardinal, `3 : 30` as a ratio). So the order is
normalise -> segment (pydips) -> `clean_text`; `prepare_text` does the first two. The compiled graphs are cached in `$JYUTVOICE_TN_CACHE` (default
`~/.cache/jyutvoice/tn`; the first Cantonese build takes seconds, Mandarin about half a minute, English about a minute) together with the options they were
built with: the fork ships Cantonese graphs compiled with `tag_oov` and interjection removal on, which would wrap 係/咗/哋 in <oov> tags and delete 啊/喇, so a
cache that does not match OPTIONS is rebuilt, never loaded.

Language routing: text with a Chinese character goes to the Chinese normaliser of `lang` (Cantonese unless lang is "zh"): numbers inside a Chinese sentence are
read in Chinese, English words are left alone. Text without one (lang "en", or English text under another lang) goes to the English normaliser.
Without WeTextProcessing installed the text comes back unchanged (one warning), or an ImportError with JYUTVOICE_TN=required.

Two things the Chinese rules get wrong around English words, handled here (found on 14k code-mixed transcripts):
- English tokens joined by - . & ' / (user-friendly, COVID-19, Wi-Fi, AT&T) and decades (80s) are hidden from the Chinese rules and put back afterwards: left to them
  they became `user減friendly` (the hyphen read as minus), `COVID負十九` and `八十秒`. They are English words for the English normaliser in `clean_text`.
  Numbers and units that the Chinese rules read well (`170cm`, `6:30am`, `$20`) are not touched.
- A lone lowercase `d` in a Cantonese sentence is the written form of 啲 (`著厚d嘅clothes`): it is rewritten, or it would be spelled as the letter D.
"""
import json
import logging
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Every option that changes the compiled graph. interjections / erhua / oov tags stay as they are in the text: they are real syllables of the speech.
# full_to_half is off: ，。！？ stay full-width, as the word segmenter saw them in the corpus (the normaliser still reads full-width digits); the letters, digits and signs
# of English words and amounts are folded to ASCII here instead (FULLWIDTH_ASCII), and clean_text maps the marks.
OPTIONS: Dict[str, dict] = {
    "yue": dict(remove_interjections=False, remove_erhua=False, tag_oov=False, traditional_to_simple=False, remove_puncts=False, full_to_half=False),
    "zh": dict(remove_interjections=False, remove_erhua=False, tag_oov=False, traditional_to_simple=False, remove_puncts=False, full_to_half=False),
    "en": dict(),
}
FULLWIDTH_ASCII = {**{0xFF21 + i: 0x41 + i for i in range(26)}, **{0xFF41 + i: 0x61 + i for i in range(26)}, **{0xFF10 + i: 0x30 + i for i in range(10)},
                   **{ord(f): ord(a) for f, a in zip("＄％＆＠＋＝／－．", "$%&@+=/-.")}}
_GRAPH_NAME = {"yue": "yue_tn", "zh": "zh_tn", "en": "en_tn"}
_CHINESE = re.compile("[㐀-䶿一-鿿]")
# an English token with an internal joiner (user-friendly, COVID-19, AT&T, google.com) and a decade (80s, 1990s)
_JOINED_LATIN = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]+[0-9]*|[0-9]+[A-Za-z]+)(?:[-'.&/][A-Za-z0-9]+)+(?![A-Za-z0-9])|(?<![A-Za-z0-9])[0-9]{2,4}s(?![A-Za-z0-9])")
_LONE_D = re.compile(r"(?<![A-Za-z0-9'\-])d(?![A-Za-z0-9'\-])")
_PLACEHOLDER = re.compile(r"ZQ[A-Z]{2}")
_normalizers: dict = {}
_warned = False


def cache_dir() -> Path:
    return Path(os.environ.get("JYUTVOICE_TN_CACHE", os.path.join("~", ".cache", "jyutvoice", "tn"))).expanduser()


def has_chinese(text: str) -> bool:
    return _CHINESE.search(text) is not None


def pick_kind(text: str, lang: str = "yue") -> str:
    """'yue' | 'zh' | 'en': the normaliser for this text (see the module docstring)."""
    if lang == "en" or not has_chinese(text):
        return "en"
    return "zh" if lang == "zh" else "yue"


def _unavailable(err: Exception) -> None:
    global _warned
    if os.environ.get("JYUTVOICE_TN") == "required":
        raise ImportError("WeTextProcessing (the indiejoseph fork) is required: pip install git+https://github.com/indiejoseph/WeTextProcessing.git opencc-python-reimplemented") from err
    if not _warned:
        _warned = True
        logger.warning("WeTextProcessing is not available (%s: %s): numbers and symbols are not normalised", type(err).__name__, err)


def _make(kind: str, simple_to_traditional: bool, overwrite: bool):
    d = cache_dir()
    if kind == "yue":
        from tn.cantonese.normalizer import Normalizer
    elif kind == "zh":
        from tn.chinese.normalizer import Normalizer
    else:
        from tn.english.normalizer import Normalizer
    extra = {"simple_to_traditional": simple_to_traditional} if kind == "yue" else {}
    return Normalizer(cache_dir=str(d), overwrite_cache=overwrite, **OPTIONS[kind], **extra)


def get_normalizer(kind: str, simple_to_traditional: bool = False):
    """The cached normaliser of `kind`, or None when WeTextProcessing is not installed. `simple_to_traditional` (OpenCC s2hk, Cantonese only) is for
    Simplified input; it is applied as a text conversion before the graph, so it does not enter the cache fingerprint."""
    key = (kind, simple_to_traditional and kind == "yue")
    if key in _normalizers:
        return _normalizers[key]
    cache_dir().mkdir(parents=True, exist_ok=True)
    stamp = cache_dir() / f"{_GRAPH_NAME[kind]}.options.json"
    try:
        have = json.loads(stamp.read_text()) if stamp.exists() else None
        n = _make(kind, key[1], overwrite=have != OPTIONS[kind])  # no stamp, or built with other options: compile again
        if have != OPTIONS[kind]:
            stamp.write_text(json.dumps(OPTIONS[kind], sort_keys=True))
    except ImportError as e:
        _unavailable(e)
        n = None
    _normalizers[key] = n
    return n


def _protect(text: str):
    """text -> (text with the joined English tokens replaced by letter-only placeholders the Chinese rules pass through, the tokens)."""
    if "ZQ" in text:  # the placeholder alphabet is taken: leave the text alone
        return text, []
    tokens: List[str] = []

    def hide(m):
        if len(tokens) >= 26 * 26:
            return m.group()
        tokens.append(m.group())
        i = len(tokens) - 1
        return "ZQ" + chr(65 + i // 26) + chr(65 + i % 26)

    return _JOINED_LATIN.sub(hide, text), tokens


def _restore(text: str, tokens: List[str]) -> str:
    return _PLACEHOLDER.sub(lambda m: tokens[(ord(m.group()[2]) - 65) * 26 + ord(m.group()[3]) - 65], text) if tokens else text


def normalize_text(text: str, lang: str = "yue", simple_to_traditional: bool = False) -> str:
    """Raw sentence -> its spoken form, e.g. `我有3個meeting，$20` -> `我有三個meeting，二十蚊`. The marks are left as they are; `clean_text` maps them.
    `simple_to_traditional`: OpenCC s2hk for Simplified input (Cantonese only)."""
    if not text or not text.strip():
        return text
    text = text.translate(FULLWIDTH_ASCII)
    kind = pick_kind(text, lang)
    n = get_normalizer(kind, simple_to_traditional)
    if n is None:
        return text
    if kind == "en":
        return n.normalize(text)
    if kind == "yue":
        text = _LONE_D.sub("啲", text)
    text, tokens = _protect(text)
    return _restore(n.normalize(text), tokens)


def squeeze_spaces(text: str) -> str:
    """Remove the spaces the segmenter must not see: all of them except those between two ASCII letters / digits (the words of English phrases)."""
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"(?<![A-Za-z0-9]) | (?![A-Za-z0-9])", "", text)


def segment_mixed(text: str, segmenter: Callable[[str], List[str]]) -> List[str]:
    """Words of a sentence: the Chinese parts are cut by `segmenter` (str -> list of words, e.g. `lambda t: pydips_model.cut(t, mode="coarse")`), each English
    phrase stays one piece: the Chinese segmenters cut inside English words (`Bad minton`) and the spaces between English words are lost if they are stripped."""
    text = squeeze_spaces(text)
    if re.search("[A-Za-z]", text) is None:  # Chinese only (the common case): no need for the splitter
        return [w for w in segmenter(text) if w.strip()]
    from jyutvoice.text.script_split import split_phrases

    words: List[str] = []
    for piece, is_latin in split_phrases(text):
        if is_latin:
            words.append(piece)
        elif piece.strip():
            words += segmenter(piece.strip())
    return [w for w in words if w.strip()]


def prepare_text(text: str, lang: str = "yue", segmenter: Optional[Callable[[str], List[str]]] = None) -> str:
    """Raw sentence -> the word-segmented text `clean_text` takes: normalised, then cut into words by `segmenter` (see `segment_mixed`). English text (and no
    segmenter) is only normalised."""
    text = normalize_text(text, lang)
    if segmenter is None or not has_chinese(text):
        return text
    return " ".join(segment_mixed(text, segmenter))
