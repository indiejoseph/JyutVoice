"""Per-character reading overrides on top of g2p's defaults, for the `phone` column (what `g2p(text, jyutping=...)` consumes).

`scripts/correct_readings.py` finds characters whose spoken syllable differs from g2p's (recogniser agreement on audio) and writes them as
`{unit index: syllable}`; unit i is the i-th character of the cleaned text without spaces (`cleaners.replace_punctuation`, then spaces removed).
`jyutping_with_overrides` rebuilds g2p's own readings for a (word-segmented) text and applies those overrides, so every position without an
override reads exactly as `g2p(text)` would.
"""
from typing import Dict, List, Optional

from jyutvoice.text.cantonese import colloquial
from jyutvoice.text.cantonese.g2p import get_jyutping
from jyutvoice.text.cleaners import replace_punctuation
from jyutvoice.text.symbols import punctuations


def cleaned(text: str) -> str:
    """The text as `clean_text(text, 'yue')` hands the Chinese words to g2p (per space-separated word, spaces kept): Chinese characters and marks only, so the
    English words of a code-mixed sentence are not in it (they have no Cantonese reading to compare or align)."""
    return " ".join(replace_punctuation(w.strip(), lang="yue") for w in text.split())


def default_jyutping(norm_text: str) -> List[str]:
    """One syllable per character of cleaned `norm_text` (punctuation marks stand for themselves), built as `g2p()` builds them."""
    words = norm_text.split()
    word_jyutping = [(word, get_jyutping(word)) for word in words]
    if colloquial.COLLOQUIAL_READINGS:
        word_jyutping = list(zip(words, colloquial.apply(words, [j for _, j in word_jyutping])))
    return [syllable for _, js in word_jyutping for syllable in js]


def jyutping_with_overrides(text: str, overrides: Optional[Dict[int, str]] = None) -> str:
    """The `jyutping` argument for g2p(): g2p's readings of `text` with `overrides` ({unit index: syllable}) applied, space-separated.

    Raises when an index is out of range or points at punctuation, or when g2p's readings do not line up with the characters (a sign that
    `text` was cleaned differently from the text the indices were computed on).
    """
    norm = cleaned(text)
    syllables = default_jyutping(norm)
    n_chars = len("".join(norm.split()))
    if len(syllables) != n_chars:
        raise ValueError(f"g2p gives {len(syllables)} syllables for {n_chars} characters: {norm!r}")
    for index, syllable in (overrides or {}).items():
        index = int(index)
        if not 0 <= index < n_chars:
            raise ValueError(f"override index {index} is outside the {n_chars} characters of {norm!r}")
        if syllables[index] in punctuations:
            raise ValueError(f"override index {index} is punctuation ({syllables[index]!r}) in {norm!r}")
        syllables[index] = syllable
    return " ".join(syllables)
