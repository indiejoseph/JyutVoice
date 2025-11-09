import re
from jyutvoice.text.mandarin.g2p import g2p as mandarin_g2p
from jyutvoice.text.cantonese.g2p import g2p as cantonese_g2p
from jyutvoice.text.english.g2p import g2p as english_g2p
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
    if lang != "en":
        replaced_text = "".join(
            [
                text
                for text in replaced_text
                if (is_chinese(text) or text in punctuations) and not text.isspace()
            ]
        )
    elif lang == "en":
        replaced_text = "".join(
            [
                text
                for text in replaced_text
                if (text.isalpha() or text in punctuations) and not text.isspace()
            ]
        )
    else:
        raise ValueError(f"Language {lang} not supported for punctuation replacement.")

    return replaced_text


def text_normalize(text: str, lang="yue") -> str:
    text = text.strip()
    text = replace_punctuation(text, lang=lang)
    return text


def clean_text(text: str, lang: str = "yue", phoneme=None, padding=True):
    norm_text = " ".join([text_normalize(w, lang=lang) for w in text.split()])
    g2p = None

    if lang == "yue":
        g2p = cantonese_g2p
    elif lang == "zh":
        g2p = mandarin_g2p
    elif lang == "en":
        g2p = english_g2p
    else:
        raise ValueError(f"Language {lang} not supported for text cleaning.")

    phones, tones, word2ph, word_pos, syllable_pos = g2p(
        norm_text, phoneme, padding=padding
    )
    return norm_text, phones, tones, word_pos, syllable_pos


if __name__ == "__main__":
    text = "佢 邊係 想 辭工 吖 ， 跳下 草裙舞 想 加 人工 之嘛 。"
    jyutping = "keoi5 bin1 hai6 soeng2 ci4 gung1 aa1 , tiu3 haa6 cou2 kwan4 mou5 soeng2 gaa1 jan4 gung1 zi1 maa3 ."
    norm_text, phones, tones, word_pos, syllable_pos = clean_text(
        text, lang="yue", phoneme=jyutping
    )
    print("Original:", text)
    print("Normalized:", norm_text)
    print("Phones:", phones)
