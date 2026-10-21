from typing import List, Tuple

from jyutvoice.text.english.g2p import g2p as g2p_en
from jyutvoice.text.cantonese.g2p import g2p as g2p_yue
from jyutvoice.text.mandarin.g2p import g2p as g2p_zh
from jyutvoice.text.script_split import is_chinese, mixed_g2p, split_latin  # noqa: F401  (is_chinese, split_latin: kept importable from here)


def g2p(
    text: str, phoneme=None, padding: bool = True, lang: str = "yue"
) -> Tuple[List[str], List[int], List[int], List[int], List[int], List[int]]:
    """
    Grapheme to phoneme conversion for code-mixed text (Chinese and English).
    Splits the text into Chinese and English groups, runs each through its own G2P and joins the results (see jyutvoice.text.script_split for the rules).

    Args:
        text: Input text containing mixed English and Chinese, already normalised (jyutvoice.text.cleaners.clean_text does that).
        lang: Language for Chinese chunks, "yue" for Cantonese or "zh" for Mandarin.

    Returns:
        phones, tones, word2ph, word_pos, syllable_pos, lang
    """
    if phoneme is not None:
        raise NotImplementedError("Phoneme input not supported for multilingual G2P.")
    if lang not in ("yue", "zh"):
        raise ValueError(f"Invalid lang '{lang}' for Chinese. Use 'yue' or 'zh'.")
    return mixed_g2p(text, {"zh": g2p_yue if lang == "yue" else g2p_zh, "en": g2p_en}, zh_lang=lang, padding=padding)


if __name__ == "__main__":
    text = "Hello 世界"
    phones, tones, word2ph, word_pos, syllable_pos, lang_ids = g2p(text, lang="zh")
    print("Text:", text)
    print("Phones:", phones)
    print("Tones:", tones)
    print("Word2ph:", word2ph)
    print("Word pos:", word_pos)
    print("Syllable pos:", syllable_pos)
    print("Lang:", lang_ids)
