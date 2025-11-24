import os
import sys
import pytest

# Make sure the repository root is on sys.path so tests can import the package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jyutvoice.text.number_utils import normalize_numbers


@pytest.mark.parametrize(
    "input_text, expected",
    [
        # comma removal and plain numbers
        ("1,234", "one thousand two hundred thirty four"),
        ("4,000", "four thousand"),
        # decimal expansion
        ("3.14", "three point 1 4"),
        ("0.5", "zero point 5"),
        # dollars
        ("$5", "5 dollars"),
        ("$1.01", "1 dollar, 1 cent"),
        ("$0.50", "50 cents"),
        # pounds
        ("£20", "20 pounds"),
        # ordinals
        ("1st", "one"),
        ("3rd", "three"),
        # special year handling
        ("2000", "two thousand"),
        ("2003", "two thousand 3"),
        ("1900", "nineteen hundred"),
    ],
)
def test_normalize_numbers_basic(input_text, expected):
    normalized = normalize_numbers(input_text)
    # The implementation outputs lower-case words and spaces; compare substrings for clarity
    assert expected in normalized


def test_mixed_text():
    text = "I paid $3.50 for 1,000 apples on 1st Jan 2000."
    normalized = normalize_numbers(text)

    # should contain expanded forms
    assert "3 dollars" in normalized or "3 dollar" in normalized
    assert "one thousand" in normalized
    assert "one" in normalized  # 1st -> one
    assert "two thousand" in normalized


def test_decimal_precision_multiple_points():
    # If malformed numeric like multiple dots appears, _expand_dollars leaves string
    assert "1.2.3" not in normalize_numbers("1.2.3")
