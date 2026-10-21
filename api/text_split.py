"""Cutting the input of /v1/audio/speech into the chunks that are synthesised one after the other (and streamed as they finish).

Sentences end at 。！？!? ； ; … and line breaks (a full stop only before white space or the end, and not after a digit: `3.5`, `1. item`); closing quotes stay with their
sentence. Sentences are packed into chunks of at most `max_chars` characters, a longer sentence is cut at commas, then at spaces, then anywhere. Pure functions.
"""
from typing import List

SENTENCE_END = "。！？!?；;…\n"
CLOSERS = "」』”’\"')）"
CLAUSE_END = "，、：,:"


def _period_ends_sentence(text: str, i: int) -> bool:
    nxt = text[i + 1] if i + 1 < len(text) else ""
    prev = text[i - 1] if i > 0 else ""
    return (nxt == "" or nxt.isspace()) and not prev.isdigit()


def sentences(text: str) -> List[str]:
    """The sentences of `text`, terminators and closing quotes attached, white space between them kept at the end of the earlier one."""
    out, start, i, n = [], 0, 0, len(text)
    while i < n:
        ch = text[i]
        if ch in SENTENCE_END or (ch == "." and _period_ends_sentence(text, i)):
            while i + 1 < n and (text[i + 1] in SENTENCE_END or text[i + 1] in CLOSERS or text[i + 1] == "."):
                i += 1
            while i + 1 < n and text[i + 1].isspace():
                i += 1
            out.append(text[start : i + 1])
            start = i + 1
        i += 1
    if start < n:
        out.append(text[start:])
    return [s for s in out if s.strip()]


def _cut(sentence: str, max_chars: int) -> List[str]:
    """A sentence longer than `max_chars`: pieces of at most that many characters, cut after a comma if possible, else at a space, else anywhere."""
    pieces, rest = [], sentence
    while len(rest) > max_chars:
        window = rest[:max_chars]
        cut = max((window.rfind(c) for c in CLAUSE_END), default=-1)
        if cut < max_chars // 3:  # no comma, or one so early that the piece would be tiny
            cut = max(cut, window.rfind(" "))
        cut = cut + 1 if cut >= 0 else max_chars
        pieces.append(rest[:cut])
        rest = rest[cut:]
    if rest.strip():
        pieces.append(rest)
    return pieces


def split_text(text: str, max_chars: int = 100) -> List[str]:
    """`text` as a list of chunks (each stripped, at most `max_chars` long unless it is a single unbreakable run of that length)."""
    chunks: List[str] = []
    cur = ""
    for sentence in sentences(text):
        for piece in _cut(sentence, max_chars):
            if cur and len(cur.strip()) + len(piece.strip()) > max_chars:
                chunks.append(cur.strip())
                cur = ""
            cur += piece
    if cur.strip():
        chunks.append(cur.strip())
    return chunks
