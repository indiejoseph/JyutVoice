"""Tests for the BERT input of the text encoder, the dataset / collate plumbing and the trainer's callbacks.

These need the full training environment (lightning, datasets, whisper, ...); parts that need the Cantonese g2p skip without pycantonese.
Run with `pytest tests/test_bert_wiring.py` (add PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 if unrelated pytest plugins fail to import).
"""
import os
import sys

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jyutvoice.models.duration_predictor import DurationPredictor
from jyutvoice.models.jyutvoice_tts import JyutVoiceTTS
from jyutvoice.models.text_encoder import TextEncoder
from jyutvoice.text.bert_align import expand_for_sequence, expand_to_phones

N_CH, SPK, BD = 32, 16, 24


def make_encoder(bert_dim=BD):
    return TextEncoder(
        encoder_type="RoPE Encoder",
        encoder_params=DictConfig(
            {
                "n_feats": 80,
                "n_channels": N_CH,
                "filter_channels": 64,
                "filter_channels_dp": 16,
                "n_heads": 2,
                "n_layers": 3,
                "kernel_size": 3,
                "p_dropout": 0.0,
                "prenet": False,
                "gin_channels": SPK,
            }
        ),
        n_vocab=97,
        n_lang=4,
        n_tone=7,
        bert_dim=bert_dim,
    )


def make_tts(bert_dim=BD):
    dp = DurationPredictor(in_channels=N_CH, filter_channels=16, kernel_size=3, p_dropout=0.0)
    return JyutVoiceTTS(encoder=make_encoder(bert_dim), decoder=None, dp=dp, output_size=80, spk_embed_dim=SPK)


def make_ids(B, T):
    ids = torch.randint(0, 4, (B, T))
    return {
        "x": torch.randint(1, 97, (B, T)),
        "x_lengths": torch.full((B,), T),
        "lang": ids,
        "tone": ids.clone(),
        "word_pos": ids.clone(),
        "syllable_pos": ids.clone(),
        "spk_embed": torch.randn(B, SPK),
    }


def encode(enc, ids, bert=None):
    return enc(ids["x"], ids["x_lengths"], ids["lang"], ids["tone"], ids["word_pos"], ids["syllable_pos"], ids["spk_embed"], bert=bert)


# ------------------------------------------------------------------------------------------------------------------ encoder


def test_zero_initialised_projection_leaves_the_encoder_unchanged():
    torch.manual_seed(0)
    enc = make_encoder().eval()
    ids = make_ids(2, 10)
    ref = encode(enc, ids)
    out = encode(enc, ids, torch.randn(2, 10, BD) * 100)  # whatever the features, the model starts as the one without BERT
    assert all(torch.equal(a, b) for a, b in zip(ref, out))


def test_features_change_the_output_once_the_projection_is_nonzero():
    torch.manual_seed(0)
    enc = make_encoder().eval()
    torch.nn.init.normal_(enc.bert_proj[1].weight, std=0.1)
    ids = make_ids(2, 10)
    assert not torch.allclose(encode(enc, ids)[1], encode(enc, ids, torch.randn(2, 10, BD))[1])


def test_all_zero_rows_add_exactly_nothing():
    torch.manual_seed(0)
    enc = make_encoder().eval()
    torch.nn.init.normal_(enc.bert_proj[1].weight, std=0.1)  # a trained projection
    ids = make_ids(2, 10)
    assert all(torch.equal(a, b) for a, b in zip(encode(enc, ids), encode(enc, ids, torch.zeros(2, 10, BD))))


def test_bert_for_an_encoder_without_bert_dim_is_refused():
    with pytest.raises(ValueError, match="bert_dim=0"):
        encode(make_encoder(bert_dim=0), make_ids(1, 5), torch.zeros(1, 5, BD))


# ------------------------------------------------------------------------------------------------------------- TTS module


def test_loss_gradient_reaches_the_bert_projection():
    B, Tx, Ty = 2, 12, 40
    torch.manual_seed(0)
    model = make_tts().train()
    ids = make_ids(B, Tx)
    dur_loss, prior_loss, _ = model(
        x=ids["x"], x_lengths=ids["x_lengths"], y=torch.zeros(B, 80, Ty), y_lengths=torch.full((B,), Ty), lang=ids["lang"],
        tone=ids["tone"], word_pos=ids["word_pos"], syllable_pos=ids["syllable_pos"], spk_embed=ids["spk_embed"],
        decoder_h=torch.randn(B, Ty, 80), bert=torch.randn(B, Tx, BD),
    )  # fmt: skip
    (dur_loss + prior_loss).backward()
    assert model.encoder.bert_proj[1].weight.grad.abs().sum() > 0


def test_a_checkpoint_trained_without_bert_can_be_loaded_and_starts_from_a_zero_projection():
    old, new = make_tts(bert_dim=0), make_tts()
    ckpt = {"state_dict": old.state_dict(), "epoch": 3}
    with pytest.raises(RuntimeError, match="bert_proj"):  # a plain strict load would fail
        new.load_state_dict(ckpt["state_dict"])
    new.on_load_checkpoint(ckpt)
    new.load_state_dict(ckpt["state_dict"])
    assert new.encoder.bert_proj[1].weight.abs().sum() == 0
    assert torch.equal(new.encoder.emb.weight, old.encoder.emb.weight)


def test_init_weights_from_a_checkpoint_without_bert(tmp_path):
    from jyutvoice.train import load_init_weights

    torch.manual_seed(0)
    old, new = make_tts(bert_dim=0), make_tts()
    path = str(tmp_path / "old.ckpt")
    torch.save({"state_dict": old.state_dict(), "epoch": 7, "optimizer_states": [{"state": {}, "param_groups": []}]}, path)
    load_init_weights(new, path)
    assert torch.equal(new.encoder.proj.weight, old.encoder.proj.weight)
    assert new.encoder.bert_proj[1].weight.abs().sum() == 0
    assert new.ckpt_loaded_epoch == 0  # a warm start is not a resume
    bare = str(tmp_path / "bare.pt")
    torch.save(old.state_dict(), bare)  # a plain state dict works too
    load_init_weights(make_tts(), bare)


def test_synthesise_warns_when_a_bert_encoder_gets_no_features_and_accepts_them():
    model = make_tts().eval()
    ids = make_ids(1, 8)
    kw = dict(x=ids["x"], x_lengths=ids["x_lengths"], lang=ids["lang"], tone=ids["tone"], word_pos=ids["word_pos"],
              syllable_pos=ids["syllable_pos"], spk_embed=ids["spk_embed"])  # fmt: skip
    with pytest.warns(UserWarning, match="no BERT features"):
        model.synthesise(**kw)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.synthesise(**kw, bert=torch.randn(1, 8, BD))


# ----------------------------------------------------------------------------------------------------- dataset and collate

try:
    import jyutvoice.data.text_mel_datamodule as dm
except Exception as e:  # noqa: BLE001  (datasets / whisper / onnxruntime missing)
    dm = None
    _dm_error = repr(e)

needs_dm = pytest.mark.skipif(dm is None, reason="datamodule imports unavailable")


def bare_dataset(bert_dim=BD, **kw):
    ds = dm.TextMelDataset.__new__(dm.TextMelDataset)  # skip __init__: no audio settings are needed for the BERT logic
    ds.bert_dim = bert_dim
    return ds


@needs_dm
def test_get_bert_repeats_unit_features_for_phones_with_and_without_blanks():
    ds = bare_dataset()
    word2ph = [1, 2, 3, 1]  # pad, a 2-phone syllable, a 3-phone syllable, pad
    unit = torch.randn(4, BD)
    row = {"bert": unit.tolist()}
    assert torch.equal(ds.get_bert(row, word2ph, 7), expand_to_phones(unit, word2ph))
    blank = ds.get_bert(row, word2ph, 15)
    assert torch.equal(blank, expand_to_phones(unit, word2ph, add_blank=True))
    assert torch.equal(blank[0], unit[0]) and torch.equal(blank[-1], unit[3])


@needs_dm
def test_get_bert_without_features_is_zeros_and_bad_features_are_loud():
    ds = bare_dataset()
    assert torch.equal(ds.get_bert({}, [1, 2, 1], 4), torch.zeros(4, BD))
    assert torch.equal(ds.get_bert({"bert": None}, [1, 2, 1], 4), torch.zeros(4, BD))
    assert torch.equal(ds.get_bert({"bert": torch.randn(3, BD).tolist()}, None, 4), torch.zeros(4, BD))  # no word2ph (pre-processed rows)
    with pytest.raises(ValueError, match="expected"):
        ds.get_bert({"bert": torch.randn(3, BD + 1).tolist()}, [1, 2, 1], 4)
    with pytest.raises(ValueError, match="fit neither"):
        ds.get_bert({"bert": torch.randn(3, BD).tolist()}, [1, 2, 1], 6)  # 4 phones: 4 or 9 positions fit, 6 means different texts
    with pytest.raises(ValueError, match="word2ph entries"):
        ds.get_bert({"bert": torch.randn(4, BD).tolist()}, [1, 2, 1], 4)


@needs_dm
def test_read_audio_understands_encoded_bytes_and_decoded_dicts(tmp_path):
    import io

    import soundfile as sf

    y = (np.sin(np.linspace(0, 200, 24000)) * 0.3).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, 24000, format="WAV", subtype="PCM_16")
    a, sr, path = dm.read_audio({"bytes": buf.getvalue(), "path": None})
    assert sr == 24000 and path is None and a.dtype == np.float32 and a.shape == y.shape
    assert np.abs(a - y).max() < 1e-3  # PCM16 rounding only
    b, sr2, _ = dm.read_audio({"array": y.tolist(), "sampling_rate": 24000, "path": "x.wav"})
    assert sr2 == 24000 and np.array_equal(b, y)
    sf.write(str(tmp_path / "s.wav"), np.stack([y, y], 1), 24000)  # stereo file path: averaged to mono
    c, sr3, _ = dm.read_audio({"bytes": None, "path": str(tmp_path / "s.wav")})
    assert c.shape == y.shape


def write_sidefile(directory, ids, blocks, bad=()):
    import json

    offsets = np.concatenate([[0], np.cumsum([len(b) for b in blocks])])
    np.save(os.path.join(directory, "offsets.npy"), offsets)
    np.save(os.path.join(directory, "bert.npy"), np.concatenate(blocks).astype(np.float16))
    json.dump(list(ids), open(os.path.join(directory, "ids.json"), "w"))
    json.dump(list(bad), open(os.path.join(directory, "bad_ids.json"), "w"))


def test_phone_override_replaces_only_the_listed_rows(tmp_path):
    import json

    if dm is None:
        pytest.skip("datamodule imports unavailable")
    path = tmp_path / "ov.json"
    json.dump({"yue_1": "nei5 hou2 ."}, open(path, "w"))
    ds = dm.TextMelDataset.__new__(dm.TextMelDataset)
    ds.phone_override = json.load(open(path))
    row = {"id": "yue_1", "phone": "ni5 hou2 ."}
    assert ds.phone_override.get(row.get("id"), row["phone"]) == "nei5 hou2 ."
    assert ds.phone_override.get("yue_2", "keep") == "keep"


def test_with_punctuation_lays_syllables_over_the_characters():
    import importlib.util

    spec = importlib.util.spec_from_file_location("mpo", os.path.join(ROOT, "scripts", "make_phone_override.py"))
    mpo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mpo)
    assert mpo.with_punctuation("他現在所做的。", "taa1 jin6 zoi6 so2 zou6 dik1".split()) == "taa1 jin6 zoi6 so2 zou6 dik1 ."
    assert mpo.with_punctuation("好，嘅！", ["hou2", "ge3"]) == "hou2 . ge3 ."  # every mark becomes ".", as in dataset_processed
    with pytest.raises(ValueError):
        mpo.with_punctuation("好。", ["hou2", "ge3"])  # one syllable too many


def test_side_file_is_looked_up_by_id_and_missing_rows_get_zeros(tmp_path):
    from jyutvoice.data.bert_sidefile import BertSideFile

    blocks = [np.random.randn(4, BD), np.random.randn(6, BD)]
    write_sidefile(str(tmp_path), ["a", "b"], blocks, bad=["zz"])
    side = BertSideFile(str(tmp_path))
    assert np.allclose(side.get("b"), blocks[1].astype(np.float16)) and side.get("nope") is None and "a" in side and side.bad_ids == {"zz"}
    if dm is None:
        pytest.skip("datamodule imports unavailable")
    ds = bare_dataset()
    ds.bert_side = side
    got = ds.get_bert({"id": "a"}, [1, 2, 1, 1], 5)  # 4 unit rows (pad, 2-phone syllable, 1-phone syllable, pad), 5 phones
    assert torch.allclose(got, expand_to_phones(torch.from_numpy(blocks[0].astype(np.float16).astype(np.float32)), [1, 2, 1, 1]))
    assert torch.equal(ds.get_bert({"id": "other"}, [1, 2, 1, 1], 5), torch.zeros(5, BD))  # not in the side file: no BERT input


@needs_dm
def test_column_values_honours_selects_and_matches_the_python_formatter():
    from datasets import Dataset

    ds = Dataset.from_dict({"id": [f"r{i}" for i in range(10)], "text": ["x" * i for i in range(10)], "duration": [float(i) for i in range(10)]})
    assert dm.column_values(ds, "text") == ds["text"]
    sub = ds.select([7, 2, 5])  # a select creates an indices mapping
    assert dm.column_values(sub, "id") == ["r7", "r2", "r5"] == sub["id"]
    assert dm.column_values(sub.select([2, 0]), "duration") == [5.0, 7.0]


def test_expand_for_sequence_infers_the_blank_layout_from_the_length():
    unit = torch.randn(3, BD)
    assert expand_for_sequence(unit, [1, 2, 1], 4).shape[0] == 4
    assert expand_for_sequence(unit, [1, 2, 1], 9).shape[0] == 9


def make_item(T, Ty, bert_dim):
    return {
        "x": torch.randint(1, 97, (T,)), "y": torch.zeros(80, Ty), "lang": torch.zeros(T, dtype=torch.long),
        "tone": torch.zeros(T, dtype=torch.long), "word_pos": torch.zeros(T, dtype=torch.long),
        "syllable_pos": torch.zeros(T, dtype=torch.long), "spk_emb": [0.1] * 192, "decoder_h": torch.zeros(Ty, 80),
        "durations": None, "bert": torch.randn(T, bert_dim) if bert_dim else None,
    }  # fmt: skip


@needs_dm
def test_collate_pads_bert_like_the_phones_and_omits_it_when_unused():
    items = [make_item(5, 20, BD), make_item(9, 30, BD)]
    batch = dm.TextMelBatchCollate()(items)
    assert batch["bert"].shape == (2, 9, BD)
    assert torch.equal(batch["bert"][0, :5], items[0]["bert"]) and batch["bert"][0, 5:].abs().sum() == 0
    assert "bert" not in dm.TextMelBatchCollate()([make_item(5, 20, 0), make_item(7, 10, 0)])


@needs_dm
def test_dataset_row_to_bert_with_the_real_g2p(tmp_path):
    pytest.importorskip("pycantonese")
    pytest.importorskip("ToJyutping")
    from jyutvoice.text.cleaners import clean_text

    text = "佢 哋 去 搵 嘢 ， 好 唔 好 ？"
    norm_text, phones, *_, word2ph = clean_text(text, "yue", return_word2ph=True)
    unit = torch.randn(len(word2ph), BD)  # what bert_align.BertPhoneFeatures.unit_features returns: n_chars + 2 rows
    ds = dm.TextMelDataset([], add_blank=True, n_fft=1920, n_mels=80, sample_rate=24000, hop_length=480, win_length=1920, f_min=0,
                           f_max=8000, tmp_dir=str(tmp_path), bert_dim=BD)  # fmt: skip
    row = {
        "text": text, "lang": "yue", "phone": None,
        "audio": {"array": (np.random.randn(24000) * 0.1).astype(np.float32), "path": None, "sampling_rate": 24000},
        "decoder_h": np.random.randn(50, 80).astype(np.float32).tolist(), "spk_emb": [0.1] * 192, "bert": unit.tolist(),
    }  # fmt: skip
    out = ds.get_datapoint(row)
    assert out["x"].shape[-1] == 2 * len(phones) + 1  # add_blank
    assert torch.allclose(out["bert"], expand_to_phones(unit, word2ph, add_blank=True))


# ------------------------------------------------------------------------------------------------------- trainer callbacks


def test_configured_checkpoint_replaces_the_default_and_its_monitor_exists(tmp_path):
    L = pytest.importorskip("lightning")
    from lightning.pytorch.callbacks import ModelCheckpoint

    from jyutvoice.train import attach_callbacks

    class Toy(L.LightningModule):
        """Logs like BaseLightningClass: `loss/val` with on_step and on_epoch both set."""

        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(2, 1)

        def training_step(self, batch, batch_idx):
            loss = self.net(batch[0]).pow(2).mean()
            self.log("loss/train", loss, on_step=True, on_epoch=True)
            return loss

        def validation_step(self, batch, batch_idx):
            self.log("loss/val", self.net(batch[0]).pow(2).mean(), on_step=True, on_epoch=True)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(8, 2)), batch_size=4)
    ckpt = ModelCheckpoint(dirpath=str(tmp_path / "ck"), monitor="loss/val_epoch", save_top_k=2, mode="min", save_last=True)
    trainer = L.Trainer(max_epochs=3, accelerator="cpu", logger=False, enable_progress_bar=False, enable_model_summary=False,
                        default_root_dir=str(tmp_path), num_sanity_val_steps=0)  # fmt: skip
    attach_callbacks(trainer, [ckpt])
    assert [c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)] == [ckpt]
    trainer.fit(Toy(), data, data)
    assert "loss/val_epoch" in trainer.callback_metrics  # with on_step and on_epoch both set, Lightning logs `_step` / `_epoch` variants
    assert trainer.checkpoint_callback is ckpt and ckpt.best_model_path
    assert len([f for f in os.listdir(tmp_path / "ck") if f.endswith(".ckpt")]) == 3  # top-2 and last
