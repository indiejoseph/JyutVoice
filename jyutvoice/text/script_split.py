"""Code-mixed text: split a sentence into Chinese and Latin (English) units and assemble one g2p result.

No g2p dependency: the per-language g2p functions are passed in, so the splitting, the language ids and the punctuation rules can be tested without
pycantonese / nltk (jyutvoice.text.multilingual wires the real ones; jyutvoice.text.cleaners does the English number normalisation first).

Rules:
- a Latin unit is a word or number: ASCII letters and digits, the currency signs and %, with . , ' - / between alphanumerics (3.5%, 1,200, don't, $999);
- a numeric-only unit that touches a Chinese character, or a bare number that does not follow an English word (drop_chinese_context_numbers), is a Cantonese or
  Mandarin number; there is no reader for those, it is dropped as before;
- consecutive Chinese words stay one group (the segmentation spaces are kept: they give the word positions), Latin words likewise;
- a punctuation-only unit joins the group before it (the first group when the text starts with one), glued to the previous word in English;
- marks (and the edge pads) carry the language id of the matrix language, `zh_lang` when the text has any Chinese, else English, so ，。！？ mean the same
  token whatever word precedes them.
"""
import re
from typing import Callable, Dict, List, Optional, Tuple

LANG_IDS = {"yue": 0, "zh": 1, "en": 2}

LATIN_RUN = re.compile(r"[A-Za-z0-9$£€%]+(?:[.,'\-/][A-Za-z0-9%]+)*")
_APOSTROPHE = re.compile(r"(?<=[A-Za-z])[’‘](?=[A-Za-z])")


def is_chinese(char: str) -> bool:
    return "一" <= char <= "鿿" or "㐀" <= char <= "䶿"


def has_latin(text: str) -> bool:
    """True when the text has an ASCII letter (digits alone do not make a text code-mixed)."""
    return any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in text)


def split_latin(word: str) -> List[Tuple[str, str]]:
    """One whitespace-free word -> [(piece, kind)], kind 'latin' | 'other' | 'number' (a numeric-only run that touches a Chinese character)."""
    word = _APOSTROPHE.sub("'", word)
    pieces, pos = [], 0
    for m in LATIN_RUN.finditer(word):
        if m.start() > pos:
            pieces.append((word[pos : m.start()], "other"))
        run = m.group()
        touches_zh = (m.start() > 0 and is_chinese(word[m.start() - 1])) or (m.end() < len(word) and is_chinese(word[m.end()]))
        pieces.append((run, "number" if not has_latin(run) and touches_zh else "latin"))
        pos = m.end()
    if pos < len(word):
        pieces.append((word[pos:], "other"))
    return pieces


_LATIN_WORD = r"[A-Za-z0-9$£€%][A-Za-z0-9$£€%'’&./,\-]*"
LATIN_PHRASE = re.compile(f"{_LATIN_WORD}(?: {_LATIN_WORD})*")


def split_phrases(text: str) -> List[Tuple[str, bool]]:
    """Text -> [(piece, is_latin)] in order: an English phrase (words with single spaces between them; . , ' & - / inside or after a word stay in it, ! and ? do not) and the rest
    (Chinese text and marks) alternate. For word segmentation: only the pieces that are not Latin go to the Chinese segmenter, which cuts English words apart
    (`Bad minton`, `per formance`)."""
    pieces, pos = [], 0
    for m in LATIN_PHRASE.finditer(text):
        if m.start() > pos:
            pieces.append((text[pos : m.start()], False))
        pieces.append((m.group(), True))
        pos = m.end()
    if pos < len(text):
        pieces.append((text[pos:], False))
    return pieces


def drop_chinese_context_numbers(words: List[str]) -> List[str]:
    """Words of a Chinese sentence with English in it -> the words without the bare numbers (15, 3.5%, $9) that are not right after an English word. After one
    (iPhone 15, Windows 11) a number is English; anywhere else (有 3 個) it is a Cantonese / Mandarin number, which nothing here reads: dropped, as in plain
    Chinese text. The corpus text is word-segmented, so a number never touches the characters around it and `touches a Chinese character` cannot tell."""
    kept, prev_latin = [], False
    for w in words:
        numeric = any(c.isdigit() for c in w) and not has_latin(w) and not any(is_chinese(c) for c in w)
        if numeric and not prev_latin:
            continue
        kept.append(w)
        prev_latin = has_latin(w) or numeric
    return kept


def tokenize(text: str, punct) -> List[Tuple[str, str]]:
    """Whitespace-separated words of an already normalised text -> [(unit, kind)], kind 'zh' (has a Chinese character; marks inside stay in it), 'en' or
    'punct' (marks only). Characters that are neither Chinese, Latin nor a mark are dropped."""
    units = []
    for word in text.split():
        for piece, kind in split_latin(word):
            if kind == "latin":
                units.append((piece, "en"))
            elif kind == "number":
                continue
            else:
                kept = "".join(c for c in piece if is_chinese(c) or c in punct)
                if not kept:
                    continue
                units.append((kept, "zh" if any(is_chinese(c) for c in kept) else "punct"))
    return units


def group_units(units: List[Tuple[str, str]]) -> List[Tuple[str, List[str]]]:
    """[(unit, kind)] -> [(lang, [word, ...])]: runs of the same script. A punct unit joins the group before it (or the first one): as a word of its own after
    Chinese words (like the segmented corpus text), glued to the word before it in English (the English g2p drops a mark that follows a space)."""
    groups: List[Tuple[str, List[str]]] = []
    lead = ""
    for unit, kind in units:
        if kind == "punct":
            if not groups:
                lead += unit
            elif groups[-1][0] == "en":
                groups[-1][1][-1] += unit
            else:
                groups[-1][1].append(unit)
            continue
        if groups and groups[-1][0] == kind:
            groups[-1][1].append(unit)
            continue
        if kind == "en":
            groups.append((kind, [lead + unit]))
        else:
            groups.append((kind, ([lead] if lead else []) + [unit]))
        lead = ""
    return groups


def mixed_g2p(
    text: str,
    g2ps: Dict[str, Callable],
    zh_lang: str = "yue",
    padding: bool = True,
    punct: Optional[List[str]] = None,
):
    """text -> (phones, tones, word2ph, word_pos, syllable_pos, lang_ids), like the per-language g2ps. `g2ps` maps 'zh' (the Chinese language, whichever `zh_lang`
    is) and 'en' to a callable (text, padding=False) -> that same 6-tuple."""
    if punct is None:
        from jyutvoice.text.symbols import punctuations as punct
    groups = group_units(tokenize(text, punct))
    matrix = zh_lang if any(k == "zh" for k, _ in groups) else "en"
    phones, tones, word2ph, word_pos, syllable_pos, lang_ids = [], [], [], [], [], []
    for kind, words in groups:
        p, t, w2p, wp, sp, _ = g2ps[kind](" ".join(words), padding=False)
        group_lang = LANG_IDS[zh_lang if kind == "zh" else "en"]
        phones += p
        tones += t
        word2ph += w2p
        word_pos += wp
        syllable_pos += sp
        lang_ids += [LANG_IDS[matrix] if ph in punct else group_lang for ph in p]
    if padding:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
        word_pos = [0] + word_pos + [0]
        syllable_pos = [0] + syllable_pos + [0]
        lang_ids = [LANG_IDS[matrix]] + lang_ids + [LANG_IDS[matrix]]
    return phones, tones, word2ph, word_pos, syllable_pos, lang_ids
