import re
import unicodedata
from typing import Optional
import pycantonese
import ToJyutping
from jyutvoice.text.symbols import punctuations


def word2jyutping(word):
    jyutpings = [
        pycantonese.characters_to_jyutping(w)[0][1]
        for w in word
        if unicodedata.name(w, "").startswith("CJK UNIFIED IDEOGRAPH")
    ]

    if None in jyutpings:
        raise ValueError(f"Failed to convert {word} to jyutping: {jyutpings}")

    return " ".join(jyutpings)


def jyutping_to_onsets_nucleuses_codas_tones(jyutping_syllables):
    onsets_nucleuses_codas = []
    tones = []
    word2ph = []

    try:
        for syllable in jyutping_syllables:
            if syllable in punctuations:
                onsets_nucleuses_codas.append(syllable)
                tones.append(0)
                word2ph.append(1)  # Add 1 for punctuation
            else:
                onset, nucleus, coda, tone = parse_jyutping(syllable)
                num_phones = 0

                if onset != "":
                    onsets_nucleuses_codas.append(onset)
                    tones.append(int(tone))
                    num_phones += 1
                if nucleus != "":
                    onsets_nucleuses_codas.append(nucleus)
                    tones.append(int(tone))
                    num_phones += 1
                if coda != "":
                    onsets_nucleuses_codas.append(coda)
                    tones.append(int(tone))
                    num_phones += 1
                word2ph.append(num_phones)
    except Exception as e:
        raise ValueError(f"Failed to parse jyutping: {jyutping_syllables}") from e

    assert len(onsets_nucleuses_codas) == len(tones)
    return onsets_nucleuses_codas, tones, word2ph


def get_jyutping(text):
    jyutping_array = []
    punct_pattern = re.compile(r"^[{}]+$".format(re.escape("".join(punctuations))))
    syllables = ToJyutping.get_jyutping_list(text)

    for word, syllable in syllables:
        if punct_pattern.match(word):
            puncts = re.split(r"([{}])".format(re.escape("".join(punctuations))), word)
            for punct in puncts:
                if len(punct) > 0:
                    jyutping_array.append(punct)
        else:
            # match multiple jyutping eg: liu4 ge3, or single jyutping eg: liu4
            if not re.search(r"^([a-z]+[1-6]+[ ]?)+$", syllable):
                raise ValueError(f"Failed to convert {word} to jyutping: {syllable}")

            jyutping_array.append(syllable)

    return jyutping_array


def parse_jyutping(jyutping: str):
    x = pycantonese.parse_jyutping(jyutping)

    if not x or len(x) == 0:
        raise ValueError(f"Failed to parse jyutping: {jyutping}")
    x = x[0]  # Take the first parsed result

    return x.onset, x.nucleus, x.coda, x.tone


def g2p(
    text: str,
    jyutping: Optional[str] = None,
    padding=True,
):
    """Grapheme to phoneme conversion for Cantonese."""
    words = text.split()
    phones = []
    tones = []
    word2ph = []
    ws_labels = []
    word_jyutping = []

    if jyutping is None:
        word_jyutping = [(word, get_jyutping(word)) for word in words]
    elif isinstance(jyutping, str):
        jyutping_list = jyutping.split(" ")

        if len(jyutping_list) != len([c for w in words for c in w]):
            raise ValueError(
                "The number of jyutping syllables does not match the number of characters in the text."
            )

        index = 0
        for word in words:
            start_index = index
            end_index = start_index + len(word)
            word_jyutping.append((word, jyutping_list[start_index:end_index]))
            index = end_index

    for word, jyutping in word_jyutping:
        temp_phones, temp_tones, temp_word2ph = (
            jyutping_to_onsets_nucleuses_codas_tones(jyutping)
        )
        phones += temp_phones
        tones += temp_tones
        word2ph += temp_word2ph

        if len(word) == 0:
            continue
        elif len(word) == 1:
            ws_labels.append(1)  # Begin
        elif len(word) == 2:
            ws_labels.extend([1, 3])  # End
        elif len(word) > 2:
            ws_labels.extend([1] + [2] * (len(word) - 2) + [3])  # Begin, Middle, End

    # Add padding
    if padding:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]

    assert len(phones) == len(tones), "Phones and tones must have the same length."

    lang_ids = [1] * len(phones)  # 1 for Cantonese

    return phones, tones, word2ph, lang_ids


if __name__ == "__main__":
    text = "佢邊係想辭工吖,跳下草裙舞想加人工之嘛."
    jyutping = "keoi5 bin1 hai6 soeng2 ci4 gung1 aa1 , tiu3 haa6 cou2 kwan4 mou5 soeng2 gaa1 jan4 gung1 zi1 maa3 ."

    phones, tones, word2ph, lang_ids = g2p(text)

    print(phones, tones, word2ph, lang_ids)

    phones, tones, word2ph, lang_ids = g2p(text, jyutping)

    print(phones, tones, word2ph, lang_ids)
