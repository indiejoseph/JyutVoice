import re
from functools import partial
from jyutvoice.text.mandarin.g2p import g2p as mandarin_g2p
from jyutvoice.text.cantonese.g2p import g2p as cantonese_g2p
from jyutvoice.text.english.g2p import g2p as english_g2p
from jyutvoice.text.english.normalization import normalize as english_normalize
from jyutvoice.text.multilingual import g2p as multilingual_g2p
from jyutvoice.text.script_split import drop_chinese_context_numbers, has_latin, split_latin
from jyutvoice.text.symbols import punctuations

rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "…": "...",
    "⋯": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}


def is_chinese(char: str) -> bool:
    # Check CJK Unified Ideographs (Most Common)
    if "\u4e00" <= char <= "\u9fff":
        return True
    # Check CJK Extension A (Where '䨇' is found)
    if "\u3400" <= char <= "\u4dbf":
        return True
    # More extensions (B, C, D, E, F) exist but are far less common
    return False


def replace_punctuation(text: str, lang="yue") -> str:
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    # Keep only Chinese characters and punctuation
    if lang == "en":
        replaced_text = "".join(
            [
                text
                for text in replaced_text
                if (text.isalpha() or text in punctuations) and not text.isspace()
            ]
        )
    elif lang == "multilingual":
        # Keep Chinese characters, English letters, and punctuation
        replaced_text = "".join(
            [
                text
                for text in replaced_text
                if (is_chinese(text) or text.isalpha() or text in punctuations)
                and not text.isspace()
            ]
        )
    elif lang in ["yue", "zh"]:
        replaced_text = "".join(
            [
                text
                for text in replaced_text
                if (is_chinese(text) or text in punctuations) and not text.isspace()
            ]
        )
    else:
        raise ValueError(f"Language {lang} not supported for punctuation replacement.")

    return replaced_text


def normalize_latin_aware(text: str, lang: str) -> str:
    """Normalise a text with English in it. The Latin words and numbers are spelled out the English way (digits, $, % ... become words) BEFORE the cleaner
    strips them; the rest (Chinese, marks) goes through replace_punctuation as usual. For Chinese + English text the pieces of a word become separate words,
    so `order咗` is `order 咗`; for English text they are joined as they were (`$999.` is `nine hundred ninety-nine dollars.`: the English g2p drops a mark
    that follows a space)."""
    zh_lang = "en" if lang == "en" else ("zh" if lang == "zh" else "yue")
    out = []
    for word in text.split():
        for piece, kind in split_latin(word):
            if kind == "latin":
                piece = english_normalize(piece).strip()
            elif kind == "other":
                piece = replace_punctuation(piece, lang=zh_lang)
            else:  # a number that touches a Chinese character: dropped, as in plain Chinese text
                piece = ""
            if piece:
                out.append(piece)
    return ("" if lang == "en" else " ").join(out)


def is_code_mixed(text: str, lang: str) -> bool:
    """True when `clean_text` takes the code-mixed path: lang "multilingual", or Chinese ("yue"/"zh") text that has English words. Chinese text without
    Latin letters keeps the plain Cantonese / Mandarin path, unchanged."""
    return lang == "multilingual" or (lang in ("yue", "zh") and has_latin(text))


def text_normalize(text: str, lang="yue", mixed=None) -> str:
    """`mixed`: whether the sentence this text belongs to is code-mixed (clean_text decides it once for the whole sentence, so a bare number between
    English words is read in English); default: decide from this text alone."""
    text = text.strip()
    if mixed is None:
        mixed = is_code_mixed(text, lang)
    if lang == "en" or mixed:
        return normalize_latin_aware(text, lang)
    text = replace_punctuation(text, lang=lang)
    return text


def clean_text(text: str, lang: str = "yue", phoneme=None, padding=True, return_word2ph: bool = False):
    mixed = is_code_mixed(text, lang)
    words = text.split()
    if mixed and lang != "en":
        words = drop_chinese_context_numbers(words)
    norm_text = " ".join([text_normalize(w, lang=lang, mixed=mixed) for w in words])
    g2p = None

    if mixed:  # Chinese + English: the Chinese part is Cantonese unless lang is "zh"
        g2p = partial(multilingual_g2p, lang="zh" if lang == "zh" else "yue")
    elif lang == "yue":
        g2p = cantonese_g2p
    elif lang == "zh":
        g2p = mandarin_g2p
    elif lang == "en":
        g2p = english_g2p
    else:
        raise ValueError(f"Language {lang} not supported for text cleaning.")

    phones, tones, word2ph, word_pos, syllable_pos, lang_ids = g2p(
        norm_text, phoneme, padding=padding
    )

    if return_word2ph:  # phones per text unit, for jyutvoice.text.bert_align
        return norm_text, phones, tones, word_pos, syllable_pos, lang_ids, word2ph

    return norm_text, phones, tones, word_pos, syllable_pos, lang_ids


if __name__ == "__main__":
    text = "佢 邊係 想 辭工 吖 ， 跳下 草裙舞 想 加 人工 之嘛 。"
    jyutping = "keoi5 bin1 hai6 soeng2 ci4 gung1 aa1 , tiu3 haa6 cou2 kwan4 mou5 soeng2 gaa1 jan4 gung1 zi1 maa3 ."
    norm_text, phones, tones, word_pos, syllable_pos = clean_text(
        text, lang="yue", phoneme=jyutping
    )
    print("Original:", text)
    print("Normalized:", norm_text)
    print("Phones:", phones)
