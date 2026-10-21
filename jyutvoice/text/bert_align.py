"""Align sub-word BERT features with the g2p phone sequence, the bert-vits2 way: BERT feature per word -> repeated for its phones.

g2p returns `word2ph`: the number of phones of every text unit (one unit per Chinese character or punctuation mark, plus one entry for
each of the two pad phones when `padding=True`). BERT works on sub-word tokens of a different granularity, so a token can cover several
characters (佢哋, 好唔好) and a character can be split over several byte-level tokens (喐 -> two tokens). The mapping is therefore done in
character offsets, in three steps:

    BERT tokens  --offsets-->  units (one per character)  --word2ph-->  phones  [--add_blank-->  phones with blanks]

    1. `bert_text_and_spans`   the string BERT reads (punctuation put back to the full-width form BERT was trained on) and the
                              [start, end) span of every unit inside it
    2. `pool_token_features`  unit feature = overlap-weighted mean of the tokens that cover the unit; a token spanning k characters
                              therefore gives its vector to all k of them, a character split over several tokens gets their average;
                              [CLS] / [SEP] go to the leading / trailing pad unit, as bert-vits2 does
    3. `expand_to_phones`     repeat every unit feature `word2ph` times (initial, nucleus and coda share the unit's vector)

With `add_blank=True` the result matches `intersperse(phones, 0)` in `TextMelDataset.get_text` (length 2P+1) using bert-vits2's rule
(`word2ph = [2 * n for n in word2ph]; word2ph[0] += 1`): every blank takes the vector of the phone before it, and the leading blank takes
the leading pad's.

Only Cantonese and Mandarin text is covered (`lang` yue / zh): their units are characters. English units are words with a "_"
separator entry between them, which needs the English g2p's word list; `BertPhoneFeatures.phone_features` rejects such text because
the number of units no longer matches `word2ph`.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# g2p works on the ASCII punctuation set (cleaners.rep_map); BERT was trained on natural Cantonese text, so give it the full-width marks back.
BERT_PUNCT: Dict[str, str] = {",": "，", ".": "。", "?": "？", "!": "！"}


def unit_chars(norm_text: str) -> List[str]:
    """One unit per character of the cleaned text (`clean_text()`'s `norm_text`, spaces ignored), in `word2ph` order."""
    return [c for c in norm_text if not c.isspace()]


def bert_text_and_spans(units: Sequence[str], punct_map: Optional[Dict[str, str]] = BERT_PUNCT) -> Tuple[str, List[Tuple[int, int]]]:
    """The string to feed BERT and the [start, end) character span of every unit inside it.

    `rep_map` turns "…" into three "." units; they are handed to BERT as one "…" and the three units share its span, so they receive the
    same feature. `punct_map=None` keeps the g2p characters unchanged.
    """
    pieces: List[str] = []
    spans: List[Tuple[int, int]] = []
    pos, i = 0, 0
    while i < len(units):
        if punct_map is not None and units[i : i + 3] == [".", ".", "."]:
            piece, n = "…", 3
        else:
            piece, n = (punct_map or {}).get(units[i], units[i]), 1
        pieces.append(piece)
        spans.extend([(pos, pos + len(piece))] * n)
        pos += len(piece)
        i += n
    return "".join(pieces), spans


def pool_token_features(
    hidden: torch.Tensor,
    offsets: torch.Tensor,
    special_mask: torch.Tensor,
    spans: Sequence[Tuple[int, int]],
    pad: bool = True,
    strict: bool = True,
) -> torch.Tensor:
    """Unit-level features from token-level BERT output.

    Args:
        hidden:       (T, d) hidden states of one sequence, padding tokens already removed
        offsets:      (T, 2) character offsets of every token in the BERT string (special tokens have (0, 0))
        special_mask: (T,) 1 for [CLS] / [SEP]
        spans:        [start, end) of every unit in the BERT string (`bert_text_and_spans`)
        pad:          add the leading / trailing pad unit, taken from the first / last special token (zeros if there is none)
        strict:       raise when a unit is covered by no token; otherwise its feature is zeros

    Returns (len(spans) + 2 * pad, d). A token's weight for a unit is the fraction of the token's characters that lie inside the unit, so
    a token over k characters gives its whole vector to each of them (duplicated, not divided), and when several tokens meet in one unit
    (a character split into bytes, or a token that straddles a boundary) the unit gets their weighted mean.
    """
    hidden = hidden.float()
    offsets = offsets.long()
    special = special_mask.bool()
    real = ~special & (offsets[:, 1] > offsets[:, 0])
    tok_len = (offsets[:, 1] - offsets[:, 0]).clamp(min=1).float()
    sp = torch.as_tensor(spans, dtype=torch.long, device=offsets.device).reshape(-1, 2)
    overlap = (torch.minimum(offsets[None, :, 1], sp[:, None, 1]) - torch.maximum(offsets[None, :, 0], sp[:, None, 0])).clamp(min=0)
    weight = overlap.float() / tok_len[None] * real[None].float()  # (units, tokens)
    total = weight.sum(1, keepdim=True)
    missing = (total[:, 0] == 0).nonzero().flatten().tolist()
    if missing and strict:
        raise ValueError(f"no BERT token covers unit(s) {missing} (spans {[tuple(spans[i]) for i in missing]})")
    feats = (weight @ hidden) / total.clamp(min=1e-8)
    if not pad:
        return feats
    zero = hidden.new_zeros(1, hidden.shape[-1])
    lead = hidden[:1] if len(special) and special[0] else zero
    trail = hidden[-1:] if len(special) and special[-1] and len(special) > 1 else zero
    return torch.cat([lead, feats, trail], 0)


def phone_counts(word2ph: Sequence[int], add_blank: bool = False) -> List[int]:
    """How many positions each unit owns in the phone sequence (`add_blank`: in `intersperse(phones, 0)`, bert-vits2's rule)."""
    if not add_blank:
        return list(word2ph)
    counts = [2 * n for n in word2ph]
    counts[0] += 1
    return counts


def expand_to_phones(unit_feats: torch.Tensor, word2ph: Sequence[int], add_blank: bool = False) -> torch.Tensor:
    """Repeat every unit feature for its phones: (len(word2ph), d) -> (sum(word2ph), d), or (2 * sum(word2ph) + 1, d) with `add_blank`."""
    if unit_feats.shape[0] != len(word2ph):
        raise ValueError(f"{unit_feats.shape[0]} unit features for {len(word2ph)} word2ph entries")
    counts = torch.as_tensor(phone_counts(word2ph, add_blank), dtype=torch.long, device=unit_feats.device)
    return torch.repeat_interleave(unit_feats, counts, dim=0)


def expand_for_sequence(unit_feats: torch.Tensor, word2ph: Sequence[int], n_positions: int) -> torch.Tensor:
    """`expand_to_phones` for a phone sequence whose layout is only known by its length: P phones (no blanks) or 2P + 1 (interspersed
    blanks). Raises when `n_positions` fits neither, which means the features and the phones came from different texts."""
    n_phones = sum(word2ph)
    if n_positions not in (n_phones, 2 * n_phones + 1):
        raise ValueError(f"{n_positions} phone positions fit neither {n_phones} phones nor {2 * n_phones + 1} with blanks (word2ph sum)")
    return expand_to_phones(unit_feats, word2ph, add_blank=n_positions != n_phones)


class BertPhoneFeatures:
    """Runs a frozen BERT over the cleaned text and returns features at unit or phone resolution.

    `tokenizer` must be a Hugging Face fast tokenizer (offsets are needed); `model` any encoder returning hidden states. bert-vits2 uses
    the third layer from the top (`layer=-3`).

    Raw ModernCantoneseBert-large states are badly scaled: every layer has one massive dimension (dim 379, mean |x| about 1,000 against
    a median of 7, so the feature std is ~92 and LayerNorm does not help, it only shrinks the other dimensions). Call `calibrate()` on a
    sample of texts to standardise every dimension (or pass `mean` / `std` saved from an earlier run) before the features reach a
    projection layer; on held-out transcripts that gives std 1.0 and takes the mean cosine between characters from 0.76 to 0.03.
    """

    def __init__(
        self,
        tokenizer,
        model,
        layer: int = -3,
        punct_map: Optional[Dict[str, str]] = BERT_PUNCT,
        max_length: int = 512,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ):
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.layer = layer
        self.punct_map = punct_map
        self.max_length = max_length
        self.mean, self.std = mean, std

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu", dtype=torch.float32, attn_implementation: str = "sdpa", **kw) -> "BertPhoneFeatures":
        """`attn_implementation="sdpa"` keeps transformers from picking flash-attention 2 for ModernBERT on a GPU node whose flash_attn wheel was built
        for another torch (genkai's venv: ABI error on import)."""
        from transformers import AutoModel, AutoTokenizer  # imported late: only the feature extractor needs it

        tokenizer = AutoTokenizer.from_pretrained(path)
        if not tokenizer.is_fast:
            raise ValueError("a fast tokenizer is required for offset mapping")
        try:
            model = AutoModel.from_pretrained(path, dtype=dtype, attn_implementation=attn_implementation)
        except TypeError:  # transformers < 4.56 only knows torch_dtype
            model = AutoModel.from_pretrained(path, torch_dtype=dtype, attn_implementation=attn_implementation)
        return cls(tokenizer, model.to(device), **kw)

    @torch.no_grad()
    def _token_states(self, norm_texts: Sequence[str]):
        """(spans per text, tokenizer output, (B, T, d) float32 hidden states of `layer`, unstandardised)."""
        built = [bert_text_and_spans(unit_chars(t), self.punct_map) for t in norm_texts]
        enc = self.tokenizer(
            [b[0] for b in built],
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        out = self.model(
            input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device), output_hidden_states=True
        )
        return [b[1] for b in built], enc, out.hidden_states[self.layer].float().cpu()

    def _standardize(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden if self.mean is None else (hidden - self.mean) / self.std

    @torch.no_grad()
    def calibrate(self, norm_texts: Sequence[str], batch_size: int = 32, eps: float = 1e-5) -> None:
        """Per-dimension mean and std of the real (non-special) tokens' states over `norm_texts`; used from now on by `unit_features`."""
        self.mean = self.std = None
        n, s1, s2 = 0, 0.0, 0.0
        for k in range(0, len(norm_texts), batch_size):
            _, enc, hidden = self._token_states(norm_texts[k : k + batch_size])
            real = enc["attention_mask"].bool() & ~enc["special_tokens_mask"].bool()
            x = hidden[real].double()
            n, s1, s2 = n + len(x), s1 + x.sum(0), s2 + (x * x).sum(0)
        mean = s1 / n
        self.mean = mean.float()
        self.std = ((s2 / n - mean * mean).clamp(min=0).sqrt() + eps).float()

    @torch.no_grad()
    def unit_features(self, norm_texts: Sequence[str], strict: bool = True) -> List[torch.Tensor]:
        """Per text: (n_units + 2, d) float32 features, row 0 and the last row being the pad units."""
        spans_list, enc, hidden = self._token_states(norm_texts)
        res = []
        for i, spans in enumerate(spans_list):
            n = int(enc["attention_mask"][i].sum())
            if spans and spans[-1][1] > int(enc["offset_mapping"][i, :n, 1].max()):
                raise ValueError(f"text {i} was truncated at max_length={self.max_length}; BERT never saw all of it")
            res.append(
                pool_token_features(
                    self._standardize(hidden[i, :n]), enc["offset_mapping"][i, :n], enc["special_tokens_mask"][i, :n], spans, True, strict
                )
            )
        return res

    def phone_features(self, norm_texts: Sequence[str], word2ph_list: Sequence[Sequence[int]], add_blank: bool = False) -> List[torch.Tensor]:
        """Per text: (P, d), or (2P + 1, d) with `add_blank`, aligned with the g2p phones (and `intersperse`d phones)."""
        feats = self.unit_features(norm_texts)
        for t, f, w in zip(norm_texts, feats, word2ph_list):
            if f.shape[0] != len(w):
                raise ValueError(
                    f"g2p returned {len(w)} word2ph entries but the text has {f.shape[0] - 2} units + 2 pads: {t!r} "
                    "(was the text cleaned with the same lang and padding=True?)"
                )
        return [expand_to_phones(f, w, add_blank) for f, w in zip(feats, word2ph_list)]
