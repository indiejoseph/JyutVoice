import re
from typing import Optional, List
import pypinyin
from pypinyin import Style
from pypinyin.style.finals import FinalsConverter
from pypinyin.style.initials import convert as initials_convert
from jyutvoice.text.symbols import punctuations


finals_converter = FinalsConverter()


def text_to_pinyin(word: str) -> List[tuple]:
    """Convert Chinese text to pinyin initials and finals."""
    initials_list = pypinyin.pinyin(word, style=Style.INITIALS, strict=False)
    finals_list = pypinyin.pinyin(word, style=Style.FINALS_TONE3, strict=False)
    initials_flat = [item[0] for item in initials_list]
    finals_flat = [item[0] for item in finals_list]
    return list(zip(initials_flat, finals_flat))


def split_pinyin_syllable(syllable: str) -> tuple:
    """Split a pinyin syllable into its initial and final components."""
    if re.match(r"^[a-zA-Z]+[0-9]$", syllable) is None:
        return ("", syllable)  # Treat as punctuation or invalid syllable
    initial = initials_convert(syllable, strict=True)
    final = finals_converter.to_finals_tone3(syllable, strict=True)
    return initial, final


def pinyin_to_phonemes(pinyin_syllables: List[tuple]) -> tuple:
    """Convert pinyin syllables to phonemes."""
    phonemes = []
    tones = []
    word2ph = []
    syllable_pos = []

    try:
        for initial, final in pinyin_syllables:
            if initial in punctuations or (
                initial == final and not re.match(r"[a-zA-Z]", initial)
            ):
                # Assuming final is also punctuation
                phonemes.append(initial)
                tones.append(0)
                word2ph.append(1)
                syllable_pos.append(0)
            else:
                tone = 0
                num_phones = 0
                syllable_pos_index = 1

                if final and final[-1].isdigit():
                    tone = int(final[-1])
                    final = final[:-1]

                if initial:
                    phonemes.append(initial)
                    tones.append(tone)
                    syllable_pos.append(syllable_pos_index)  # Initial
                    syllable_pos_index += 1
                    num_phones += 1
                if final:
                    phonemes.append(final)
                    tones.append(tone)
                    syllable_pos.append(syllable_pos_index)  # Final
                    syllable_pos_index += 1
                    num_phones += 1

                word2ph.append(num_phones)
    except Exception as e:
        raise ValueError(
            f"Failed to convert pinyin syllables: {pinyin_syllables}"
        ) from e

    return phonemes, tones, word2ph, syllable_pos


def g2p(
    text: str,
    pinyin: Optional[str] = None,
    padding=True,
):
    """Grapheme to phoneme conversion for Mandarin."""
    words = text.split()
    phones = []
    tones = []
    word2ph = []
    ws_labels = []
    word_pos = []
    syllable_pos = []
    word_pinyin = []

    if pinyin is None:
        word_pinyin = [(word, text_to_pinyin(word)) for word in words]
    elif isinstance(pinyin, str):
        pinyin_list = [split_pinyin_syllable(s) for s in pinyin.split(" ")]

        if len(pinyin_list) != len([c for w in words for c in w]):
            raise ValueError(
                "The number of pinyin syllables does not match the number of characters in the text."
            )

        index = 0
        for word in words:
            start_index = index
            end_index = start_index + len(word)
            word_pinyin.append((word, pinyin_list[start_index:end_index]))
            index = end_index

    for word, pinyin in word_pinyin:
        temp_phones, temp_tones, temp_word2ph, temp_syllable_pos = pinyin_to_phonemes(
            pinyin
        )
        phones += temp_phones
        tones += temp_tones
        word2ph += temp_word2ph
        syllable_pos += temp_syllable_pos

        if len(word) == 0:
            continue
        elif len(word) == 1:
            ws_labels.append(1)  # Begin
        elif len(word) == 2:
            ws_labels.extend([1, 3])  # End
        elif len(word) > 2:
            ws_labels.extend([1] + [2] * (len(word) - 2) + [3])  # Begin, Middle, End

    for i, ws_label in enumerate(ws_labels):
        num_phones = word2ph[i]
        word_pos.extend([ws_label] * num_phones)

    # Add padding
    if padding:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word_pos = [0] + word_pos + [0]
        syllable_pos = [0] + syllable_pos + [0]

    assert (
        len(phones) == len(tones) == len(word_pos) == len(syllable_pos)
    ), "Phones, tones, word positions, and syllable positions must have the same length."

    lang_ids = [2] * len(phones)  # 2 for Mandarin

    return phones, tones, word2ph, word_pos, syllable_pos, lang_ids


if __name__ == "__main__":
    text = "你好 世界 ！"
    phones, tones, word2ph, word_pos, syllable_pos = g2p(text)
    print("Text:", text)
    print("Phones:", phones)
    print("Tones:", tones)
    print("Word2ph:", word2ph)
    print("Word pos:", word_pos)
    print("Syllable pos:", syllable_pos)

    phones, tones, word2ph, word_pos, syllable_pos = g2p(
        text, pinyin="ni3 hao3 shi4 jie4 !"
    )
    print("Text with provided pinyin:", text)
    print("Phones:", phones)
    print("Tones:", tones)
    print("Word2ph:", word2ph)
    print("Word pos:", word_pos)
    print("Syllable pos:", syllable_pos)
