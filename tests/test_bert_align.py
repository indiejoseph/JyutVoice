"""Tests for jyutvoice/text/bert_align.py (BERT sub-word features -> g2p phone positions).

Run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/test_bert_align.py` if other pytest plugins fail to import. Optional extras:
    CANTONESE_BERT_TOKENIZER=<dir or tokenizer.json>   checks the real byte-level BPE tokenizer of CantoneseModernBert
    pycantonese + ToJyutping installed                 checks word2ph against the real Cantonese g2p
"""
import importlib.util
import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# bert_align has no package dependencies; importing it as jyutvoice.text.bert_align would pull in the whole g2p stack (pycantonese, ...).
_spec = importlib.util.spec_from_file_location("bert_align", os.path.join(ROOT, "jyutvoice", "text", "bert_align.py"))
ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ba)

try:
    from jyutvoice.utils.utils import intersperse  # what TextMelDataset.get_text uses
except Exception:  # noqa: BLE001  (utils needs gdown, matplotlib, omegaconf, ...)

    def intersperse(lst, item):
        result = [item] * (len(lst) * 2 + 1)
        result[1::2] = lst
        return result


def onehot_tokens(n_tokens):
    """hidden[t] = e_t, so a pooled unit row says exactly which tokens it was built from and with what weight."""
    return torch.eye(n_tokens)


# ----------------------------------------------------------------------------------------------------------- BERT string + spans


def test_bert_text_gives_full_width_punctuation_back():
    units = list("好嘅,咁就得!")
    text, spans = ba.bert_text_and_spans(units)
    assert text == "好嘅，咁就得！"
    assert [text[s:e] for s, e in spans] == units[:2] + ["，"] + units[3:6] + ["！"]


def test_ellipsis_units_share_one_span():
    units = list("好...啊")  # cleaners.rep_map turns "…" into "..."
    text, spans = ba.bert_text_and_spans(units)
    assert text == "好…啊"
    assert spans == [(0, 1), (1, 2), (1, 2), (1, 2), (2, 3)]


def test_punct_map_none_keeps_g2p_characters():
    text, spans = ba.bert_text_and_spans(list("好,啊..."), punct_map=None)
    assert text == "好,啊..."
    assert spans == [(i, i + 1) for i in range(len("好,啊..."))]


def test_unit_chars_ignores_segmentation_spaces():
    assert ba.unit_chars("佢 邊係 想 , 跳下 .") == list("佢邊係想,跳下.")


# ----------------------------------------------------------------------------------------------------------------- pooling


def test_multi_character_token_is_duplicated_not_divided():
    # tokens: [CLS] 佢哋(0,2) 去(2,3) [SEP]
    offsets = torch.tensor([[0, 0], [0, 2], [2, 3], [0, 0]])
    special = torch.tensor([1, 0, 0, 1])
    feats = ba.pool_token_features(onehot_tokens(4), offsets, special, [(0, 1), (1, 2), (2, 3)])
    assert feats.shape == (5, 4)  # 3 units + 2 pads
    assert torch.equal(feats[1], feats[2])  # 佢 and 哋 both get the whole 佢哋 vector
    assert torch.equal(feats[1], torch.tensor([0.0, 1.0, 0.0, 0.0]))
    assert torch.equal(feats[3], torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_character_split_over_byte_tokens_is_averaged_and_neighbours_do_not_leak():
    # 喐 is two byte-level tokens that both report (1, 2); 我 and 咗 are whole tokens
    offsets = torch.tensor([[0, 0], [0, 1], [1, 2], [1, 2], [2, 3], [0, 0]])
    special = torch.tensor([1, 0, 0, 0, 0, 1])
    feats = ba.pool_token_features(onehot_tokens(6), offsets, special, [(0, 1), (1, 2), (2, 3)])
    assert torch.allclose(feats[2], torch.tensor([0, 0, 0.5, 0.5, 0, 0]))
    assert torch.equal(feats[1], torch.tensor([0.0, 1.0, 0, 0, 0, 0]))
    assert torch.equal(feats[3], torch.tensor([0.0, 0, 0, 0, 1.0, 0]))


def test_token_straddling_a_boundary_is_weighted_by_the_share_inside_the_unit():
    # a byte token that carries the end of 喐 and the start of 咗 reports (1, 3) over both
    offsets = torch.tensor([[0, 0], [1, 2], [1, 3], [0, 0]])
    special = torch.tensor([1, 0, 0, 1])
    feats = ba.pool_token_features(onehot_tokens(4), offsets, special, [(1, 2), (2, 3)], pad=False)
    # unit 喐: token 1 weight 1 (all inside), token 2 weight 1/2; unit 咗: token 2 only
    assert torch.allclose(feats[0], torch.tensor([0, 1 / 1.5, 0.5 / 1.5, 0]))
    assert torch.allclose(feats[1], torch.tensor([0, 0, 1.0, 0]))


def test_cls_and_sep_land_on_the_pads_and_are_zero_without_specials():
    hidden = onehot_tokens(3)
    offsets = torch.tensor([[0, 0], [0, 1], [0, 0]])
    feats = ba.pool_token_features(hidden, offsets, torch.tensor([1, 0, 1]), [(0, 1)])
    assert torch.equal(feats[0], hidden[0]) and torch.equal(feats[2], hidden[2])
    bare = ba.pool_token_features(hidden[:1], torch.tensor([[0, 1]]), torch.tensor([0]), [(0, 1)])
    assert torch.equal(bare[0], torch.zeros(3)) and torch.equal(bare[2], torch.zeros(3))


def test_unit_without_token_raises_or_is_zero():
    offsets = torch.tensor([[0, 0], [0, 1], [0, 0]])
    special = torch.tensor([1, 0, 1])
    with pytest.raises(ValueError, match="no BERT token covers"):
        ba.pool_token_features(onehot_tokens(3), offsets, special, [(0, 1), (1, 2)])
    feats = ba.pool_token_features(onehot_tokens(3), offsets, special, [(0, 1), (1, 2)], strict=False)
    assert torch.equal(feats[2], torch.zeros(3))


# --------------------------------------------------------------------------------------------------------------- expansion


def test_expand_repeats_every_unit_word2ph_times():
    unit = torch.arange(4.0).reshape(4, 1)  # pad, 佢 (onset+nucleus), 哋 (onset+nucleus+coda), pad
    out = ba.expand_to_phones(unit, [1, 2, 3, 1])
    assert out.flatten().tolist() == [0, 1, 1, 2, 2, 2, 3]


def test_expand_with_blank_matches_intersperse_layout():
    word2ph = [1, 2, 3, 1]
    unit = torch.arange(4.0).reshape(4, 1)
    phone_owner = [u for u, n in enumerate(word2ph) for _ in range(n)]  # which unit every phone belongs to
    out = ba.expand_to_phones(unit, word2ph, add_blank=True).flatten().long().tolist()
    assert len(out) == len(intersperse(phone_owner, 0)) == 2 * sum(word2ph) + 1
    # every phone keeps its unit's vector, every blank takes the vector of the phone before it, the leading blank the leading pad's
    assert out[1::2] == phone_owner
    assert out[2::2] == phone_owner
    assert out[0] == 0


def test_expand_rejects_length_mismatch():
    with pytest.raises(ValueError, match="word2ph"):
        ba.expand_to_phones(torch.zeros(3, 2), [1, 2])


# ------------------------------------------------------------------------------- real tokenizer / extractor glue (optional)


def _real_tokenizer():
    from tokenizers import Tokenizer

    path = os.environ.get("CANTONESE_BERT_TOKENIZER")
    if not path:
        pytest.skip("set CANTONESE_BERT_TOKENIZER to the CantoneseModernBert tokenizer directory")
    return Tokenizer.from_file(os.path.join(path, "tokenizer.json") if os.path.isdir(path) else path)


SENTENCES = [
    "佢哋而家去搵嘢食喎,好唔好呀?",
    "我喐咗少少.",
    "噉但係,其實嗰啲唔係黑仔嚟嘅!",
    "專業講解員唔單止帶領遊客近距離觀賞仿生機械.",
    "好嘅...,咁就'得'喇-",
    "𠮟咗佢一句",
]


@pytest.mark.parametrize("sentence", SENTENCES)
def test_real_tokenizer_covers_every_unit_and_duplicates_shared_tokens(sentence):
    tok = _real_tokenizer()
    units = ba.unit_chars(sentence)
    text, spans = ba.bert_text_and_spans(units)
    enc = tok.encode(text)
    offsets = torch.tensor(enc.offsets)
    feats = ba.pool_token_features(onehot_tokens(len(enc.ids)), offsets, torch.tensor(enc.special_tokens_mask), spans)
    assert feats.shape[0] == len(units) + 2
    assert torch.allclose(feats[1:-1].sum(1), torch.ones(len(units)))  # every unit is a proper weighted mean
    for a in range(len(units) - 1):  # neighbouring units inside one token get identical features
        covering = lambda i: [t for t, (s, e) in enumerate(enc.offsets) if s < spans[i][1] and e > spans[i][0] and e > s]  # noqa: E731
        if len(covering(a)) == 1 and covering(a) == covering(a + 1) and spans[a] != spans[a + 1]:
            assert torch.equal(feats[a + 1], feats[a + 2])


def test_real_tokenizer_shows_a_multi_character_word_and_a_split_character():
    tok = _real_tokenizer()
    text, spans = ba.bert_text_and_spans(ba.unit_chars("佢哋去喐."))
    enc = tok.encode(text)
    feats = ba.pool_token_features(onehot_tokens(len(enc.ids)), torch.tensor(enc.offsets), torch.tensor(enc.special_tokens_mask), spans)
    assert torch.equal(feats[1], feats[2])  # 佢 / 哋 are one token in the extended vocabulary
    assert (feats[4] > 0).sum() >= 2  # 喐 is split into byte tokens and averaged


class _HFLikeTokenizer:
    """The slice of the Hugging Face fast-tokenizer call that BertPhoneFeatures uses, on top of a `tokenizers.Tokenizer`."""

    def __init__(self, tok):
        self.tok = tok
        tok.enable_padding(pad_id=0, pad_token="[PAD]")

    def __call__(self, texts, return_offsets_mapping, return_special_tokens_mask, padding, truncation, max_length, return_tensors):
        encs = self.tok.encode_batch(list(texts))
        return {
            "input_ids": torch.tensor([e.ids for e in encs]),
            "attention_mask": torch.tensor([e.attention_mask for e in encs]),
            "offset_mapping": torch.tensor([e.offsets for e in encs]),
            "special_tokens_mask": torch.tensor([e.special_tokens_mask for e in encs]),
        }


class _FakeBert(torch.nn.Module):
    """hidden_states = (embeddings, 2 * embeddings, 3 * embeddings, 4 * embeddings): layer -3 is 2 * embeddings."""

    def __init__(self, vocab, d=8):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(vocab, d)

    def forward(self, input_ids, attention_mask, output_hidden_states):
        e = self.emb(input_ids)
        return type("Out", (), {"hidden_states": (e, 2 * e, 3 * e, 4 * e)})()


def test_extractor_gives_phone_aligned_features_for_a_batch():
    tok = _real_tokenizer()
    texts = ["佢哋去搵嘢,好.", "我喐咗少少."]
    # word2ph: pad, then one entry per character / punctuation mark (2 to 3 phones per syllable, 1 per punctuation mark), pad
    word2ph = [[1] + [2, 3, 2, 3, 2, 1, 2, 1] + [1], [1] + [2, 2, 3, 2, 3, 1] + [1]]
    assert [len(w) - 2 for w in word2ph] == [len(ba.unit_chars(t)) for t in texts]
    model = _FakeBert(vocab=max(tok.get_vocab_size(), 1))
    extractor = ba.BertPhoneFeatures(_HFLikeTokenizer(tok), model, layer=-3)
    plain = extractor.phone_features(texts, word2ph)
    blank = extractor.phone_features(texts, word2ph, add_blank=True)
    for t, w, p, b in zip(texts, word2ph, plain, blank):
        assert p.shape == (sum(w), 8) and b.shape == (2 * sum(w) + 1, 8)
        assert torch.equal(b[1::2], p)  # blanks only add rows; the phone rows are the same
    # padding in the batch must not change a text's features
    alone = extractor.phone_features(texts[1:], word2ph[1:])[0]
    assert torch.allclose(alone, plain[1], atol=1e-6)
    # a g2p result that disagrees with the text is refused
    with pytest.raises(ValueError, match="word2ph entries"):
        extractor.phone_features(texts[:1], [word2ph[0][:-1]])


def test_calibrate_standardises_every_dimension_and_commutes_with_pooling():
    tok = _real_tokenizer()
    texts = ["佢哋去搵嘢,好.", "我喐咗少少.", "噉但係,其實嗰啲唔係黑仔嚟嘅!"]
    model = _FakeBert(vocab=max(tok.get_vocab_size(), 1))
    with torch.no_grad():
        model.emb.weight[:, 3] += 500.0  # a massive dimension like ModernBERT's
    ext = ba.BertPhoneFeatures(_HFLikeTokenizer(tok), model, layer=-3)
    raw = ext.unit_features(texts)
    ext.calibrate(texts, batch_size=2)
    assert ext.mean.shape == ext.std.shape == (8,) and ext.mean[3] > 400
    cal = ext.unit_features(texts)
    for r, c in zip(raw, cal):
        assert torch.allclose(c, (r - ext.mean) / ext.std, atol=1e-4)  # pooling is a convex combination, so the two commute
    _, enc, hidden = ext._token_states(texts)
    keep = enc["attention_mask"].bool() & ~enc["special_tokens_mask"].bool()
    tokens = ext._standardize(hidden[keep])
    assert torch.allclose(tokens.mean(0), torch.zeros(8), atol=1e-3)
    assert torch.allclose(tokens.std(0, unbiased=False), torch.ones(8), atol=1e-2)


# -------------------------------------------------------------------------------------------------- real g2p (optional)


def test_g2p_word2ph_matches_text_units():
    pytest.importorskip("pycantonese")
    pytest.importorskip("ToJyutping")
    from jyutvoice.text import text_to_sequence
    from jyutvoice.text.cleaners import clean_text

    for text in ["佢 哋 去 搵 嘢 ， 好 唔 好 ？", "專業 講解員 唔 單止 帶領 遊客 …", "噉 但係 ， 其實 嗰啲 唔係 黑仔 嚟 嘅 ！"]:
        norm_text, phones, *_, word2ph = clean_text(text, "yue", return_word2ph=True)
        assert len(word2ph) == len(ba.unit_chars(norm_text)) + 2
        assert sum(word2ph) == len(phones)
        ids, _, _, _, _, w2 = text_to_sequence(text, "yue", return_word2ph=True)
        assert w2 == word2ph and len(ids) == len(phones)
