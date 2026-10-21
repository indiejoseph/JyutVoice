"""English stress as encoder input, and the default stress of names the dictionary does not know. No dependencies (the g2p imports nltk, g2p_en, ...).

Tone ids 1-6 are the Cantonese (jyutping) tones: 1 high level, 2 mid rising, 3 mid level, ... The English g2p used to write stress there (unstressed 1, primary 2,
secondary 3, consonants 3), so a model trained on Cantonese gave an unstressed vowel the high level tone: in `Google` (UW1 AH0) the second syllable came out
high, as if it were the stressed one. English stress has ids of its own, so the two systems cannot be confused (and `N_TONE` is the size of the tone table:
0 pad / punctuation, 1-6 Cantonese, 7-10 English).

Names: the dictionary decides the stress of the words it knows (Japan, Macau, Berlin and Chanel are stressed on the second syllable, so there is no general
rule). For a name it does not know (Google, WhatsApp, Instagram) the neural guess is unreliable (WhatsApp came out stressed on `app`, Instagram with no primary
stress at all), and brand names are almost always stressed on the first syllable: `initial_stress` puts the primary stress there.
"""
import re
from typing import List

TONE_CONSONANT = 7  # a consonant carries no stress mark
TONE_BY_STRESS = {0: 8, 1: 9, 2: 10}  # unstressed, primary, secondary vowel
N_TONE = 11

_STRESSED = re.compile(r"^([A-Za-z]+)(\d)$")


def arpa_tone(phn: str):
    """ARPAbet phone, stress digit on vowels -> (lower-case phone, tone id): 'UW1' -> ('uw', 9), 'AH0' -> ('ah', 8), 'G' -> ('g', 7)."""
    m = _STRESSED.match(phn)
    if m is None:
        return phn.lower(), TONE_CONSONANT
    return m.group(1).lower(), TONE_BY_STRESS.get(int(m.group(2)), TONE_BY_STRESS[0])


def is_name_like(word: str) -> bool:
    """A word written with a capital letter that is not all capitals: Google, Tokyo, iPhone, YouTube (not google, not NASA)."""
    return word != word.lower() and not word.isupper()


def initial_stress(pron: List[str]) -> List[str]:
    """ARPAbet phones -> the same with primary stress on the first vowel; any other primary stress becomes secondary."""
    out = list(pron)
    vowels = [i for i, p in enumerate(out) if p[-1:].isdigit()]
    for k, i in enumerate(vowels):
        if k == 0:
            out[i] = out[i][:-1] + "1"
        elif out[i][-1] == "1":
            out[i] = out[i][:-1] + "2"
    return out
