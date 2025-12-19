"""from https://github.com/keithito/tacotron"""

from jyutvoice.text.symbols import symbols
from jyutvoice.text.cleaners import clean_text

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {
    i: s for i, s in enumerate(symbols)
}  # pylint: disable=unnecessary-comprehension

# Mapping for language codes
LANGUAGE_CODES = {
    "yue": 0,
    "zh": 1,
    "en": 2,
}


def text_to_sequence(text, lang: str, phone=None):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      lang: language of the text ('yue', 'zh', 'en')
      phone: optional phoneme input
    Returns:
      List of integers corresponding to the symbols in the text
    """

    _, phones, tones, lang_ids = clean_text(
        text, lang=lang, phoneme=phone, padding=True
    )
    phone_token_ids = cleaned_text_to_sequence(phones)

    return phone_token_ids, tones, lang_ids


def cleaned_text_to_sequence(cleaned_text):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = [_symbol_to_id[symbol] for symbol in cleaned_text]
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string"""
    result = ""
    for symbol_id in sequence:
        s = _id_to_symbol[symbol_id]
        result += s
    return result
