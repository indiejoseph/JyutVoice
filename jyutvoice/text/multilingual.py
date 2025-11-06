import re
from typing import List, Tuple
from jyutvoice.text.english.g2p import g2p as g2p_en
from jyutvoice.text.cantonese.g2p import g2p as g2p_yue
from jyutvoice.text.mandarin.g2p import g2p as g2p_zh


def is_chinese(char: str) -> bool:
    # Check CJK Unified Ideographs (Most Common)
    if "\u4e00" <= char <= "\u9fff":
        return True
    # Check CJK Extension A (Where '䨇' is found)
    if "\u3400" <= char <= "\u4dbf":
        return True
    # More extensions (B, C, D, E, F) exist but are far less common
    return False


def split_text(text: str) -> List[Tuple[str, bool]]:
    """Split text into segments of consecutive Chinese or non-Chinese characters."""
    segments = []
    current = ""
    last_is_chinese = None
    for char in text:
        curr_is_chinese = is_chinese(char)
        if last_is_chinese is None or curr_is_chinese == last_is_chinese:
            current += char
            if last_is_chinese is None:
                last_is_chinese = curr_is_chinese
        else:
            if current:  # Only append if not empty
                segments.append((current, last_is_chinese))
            current = char
            last_is_chinese = curr_is_chinese
    if current:
        segments.append((current, last_is_chinese))
    return segments


def g2p(
    text: str, lang: str = "zh", padding: bool = True
) -> Tuple[List[str], List[int], List[int], List[int], List[int]]:
    """
    Grapheme to phoneme conversion for multilingual text.
    Splits text into English and Chinese chunks, processes each with appropriate G2P,
    and combines the results.

    Args:
        text: Input text containing mixed English and Chinese.
        lang: Language for Chinese chunks, "yue" for Cantonese or "zh" for Mandarin.

    Returns:
        phones, tones, word2ph, word_pos, syllable_pos, lang
    """
    segments = split_text(text)
    all_phones = []
    all_tones = []
    all_word2ph = []
    all_word_pos = []
    all_syllable_pos = []
    all_lang = []
    total_words = 0

    for chunk, is_chinese in segments:
        if not chunk:
            continue
        if is_chinese:
            if lang == "yue":
                phones, tones, word2ph, word_pos, syllable_pos = g2p_yue(
                    chunk, padding=False
                )
                all_lang.extend([0] * len(phones))
            elif lang == "zh":
                phones, tones, word2ph, word_pos, syllable_pos = g2p_zh(
                    chunk, padding=False
                )
                all_lang.extend([1] * len(phones))
            else:
                raise ValueError(
                    f"Invalid lang '{lang}' for Chinese. Use 'yue' or 'zh'."
                )
        else:
            phones, tones, word2ph, word_pos, syllable_pos = g2p_en(
                chunk, padding=False
            )
            all_lang.extend([2] * len(phones))

        # Append to global lists
        all_phones.extend(phones)
        all_tones.extend(tones)
        all_word2ph.extend(word2ph)
        all_word_pos.extend(word_pos)
        all_syllable_pos.extend(syllable_pos)

        # Update total words
        total_words += len(word2ph)

    if padding:
        # Add padding between chunks
        all_phones = ["-"] + all_phones + ["_"]
        all_tones = [0] + all_tones + [0]
        all_word2ph = [1] + all_word2ph + [1]
        all_word_pos = [0] + all_word_pos + [0]
        all_syllable_pos = [0] + all_syllable_pos + [0]

    return all_phones, all_tones, all_word2ph, all_word_pos, all_syllable_pos


if __name__ == "__main__":
    # Test
    text = "Hello 世界"
    phones, tones, word2ph, word_pos, syllable_pos = g2p(text, lang="zh")
    print("Text:", text)
    print("Phones:", phones)
    print("Tones:", tones)
    print("Word2ph:", word2ph)
    print("Word pos:", word_pos)
    print("Syllable pos:", syllable_pos)
