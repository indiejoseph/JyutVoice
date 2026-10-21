import glob
import io
import json
import os
import random
import warnings
from pathlib import Path
from typing import Any, Dict, Optional
import torchaudio.compliance.kaldi as kaldi
import librosa
import numpy as np
import torch
import whisper
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader
import onnxruntime
import soundfile as sf
from datasets import Audio, concatenate_datasets, load_dataset, load_from_disk
from jyutvoice.utils.audio import mel_spectrogram
from jyutvoice.utils.utils import intersperse
from jyutvoice.text import _symbol_to_id, text_to_sequence
from jyutvoice.text.symbols import punctuations
from jyutvoice.data.bert_sidefile import BertSideFile
from jyutvoice.text.bert_align import expand_for_sequence


PUNCT_IDS = torch.tensor(sorted({_symbol_to_id[p] for p in punctuations if p in _symbol_to_id}), dtype=torch.long)


def column_values(ds, name):
    """A column of a datasets.Dataset as a python list, read from the arrow table (honouring a previous select / filter), not through `ds[name]`."""
    col = ds.data.column(name)
    if ds._indices is not None:
        col = col.take(ds._indices.column(0))
    return col.to_pylist()


def read_audio(cell):
    """(float32 mono array, sampling rate, path or None) from an `audio` cell in any of its forms: encoded bytes (`Audio(decode=False)`, what
    `setup()` casts to), a decoded dict {"array", "sampling_rate"} (datasets < 4) or datasets >= 4's torchcodec AudioDecoder."""
    if isinstance(cell, dict) and cell.get("array") is not None:
        return np.asarray(cell["array"], dtype=np.float32), cell["sampling_rate"], cell.get("path")
    if isinstance(cell, dict):
        source = io.BytesIO(cell["bytes"]) if cell.get("bytes") is not None else cell["path"]
        y, sr = sf.read(source, dtype="float32")
        return (y.mean(axis=1) if y.ndim > 1 else y), sr, cell.get("path")
    samples = cell.get_all_samples()  # torchcodec AudioDecoder
    return samples.data.float().mean(dim=0).numpy(), samples.sample_rate, None


# random slice mel for reference encoder to prevent overfitting
def random_slice_tensor(x: torch.Tensor):
    length = x.size(-1)
    if length < 12:
        return x
    segmnt_size = random.randint(length // 12, length // 3)
    start = random.randint(0, length - segmnt_size)
    return x[..., start : start + segmnt_size]


class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        dataset_path,
        dataset_valid_ratio,
        batch_size,
        num_workers,
        pin_memory,
        add_blank,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        seed,
        load_durations,
        max_duration=None,
        max_text_length=None,
        bert_dim=0,
        bert_dir=None,
        phone_override=None,
        teacher_prior_weight=1.0,
        dur_weight_counts=None,
        dur_weight_min_count=0,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    @staticmethod
    def load_one(path: str):
        shards = sorted(glob.glob(path)) if any(c in path for c in "*?[") else []
        if shards:  # e.g. `<dir>/shards/shard_*`: datasets written one directory per shard
            return concatenate_datasets([load_from_disk(p) for p in shards])
        if os.path.exists(path):
            return load_from_disk(path)
        return load_dataset(path, split="train")

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        paths = self.hparams.dataset_path
        paths = [str(paths)] if isinstance(paths, (str, os.PathLike)) else [str(p) for p in paths]  # one dataset, or a list of them to mix
        parts = [self.load_one(p) for p in paths]
        if len(parts) > 1:  # sets built by different scripts have different extra columns (`source`, `duration`, `bert`): mix them on the common ones
            common = [c for c in parts[0].column_names if all(c in q.column_names for q in parts[1:])]
            parts = [q.select_columns(common) for q in parts]
        ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]

        if "audio" in ds.column_names:  # teacher rows have none
            ds = ds.cast_column("audio", Audio(decode=False))  # decoded here with soundfile: independent of the datasets version

        # Filters read the cheap columns from the underlying arrow table (`column_values`): `ds["text"]` goes through the python formatter and took
        # over ten minutes on 162 shards that also hold audio and decoder_h (this: seconds). They end in one `select`, not ds.filter, because every DDP
        # rank runs this and filter() writes cache files.
        if self.hparams.max_duration is not None and "duration" not in ds.column_names:  # no duration column: read the audio headers

            def filter_fn(batch):
                durations = []
                for cell in batch["audio"]:
                    if isinstance(cell, dict) and cell.get("bytes") is not None:
                        durations.append(sf.info(io.BytesIO(cell["bytes"])).duration)  # header only, no decoding
                    else:
                        y, sr, _ = read_audio(cell)
                        durations.append(len(y) / sr)
                return [d <= self.hparams.max_duration for d in durations]

            ds = ds.filter(
                filter_fn,
                batched=True,
                num_proc=(
                    self.hparams.num_workers if self.hparams.num_workers > 0 else 1
                ),
            )

        keep = np.ones(len(ds), dtype=bool)
        if self.hparams.max_duration is not None and "duration" in ds.column_names:
            keep &= np.asarray(column_values(ds, "duration")) <= self.hparams.max_duration
        if self.hparams.max_text_length is not None:
            keep &= np.fromiter((len(t) <= self.hparams.max_text_length for t in column_values(ds, "text")), dtype=bool, count=len(ds))
        if self.hparams.bert_dir:  # rows whose label cannot be aligned with the text (g2p would fail on them)
            bad = BertSideFile(self.hparams.bert_dir).bad_ids
            if bad:
                keep &= np.fromiter((x not in bad for x in column_values(ds, "id")), dtype=bool, count=len(ds))
        if not keep.all():
            ds = ds.select(np.flatnonzero(keep).tolist())

        if self.hparams.bert_dim > 0 and "bert" not in ds.column_names and not self.hparams.bert_dir:
            warnings.warn(
                f"bert_dim={self.hparams.bert_dim} but the dataset at {self.hparams.dataset_path} has no `bert` column: every BERT input "
                "will be zeros and the BERT projection can learn nothing"
            )

        # a fixed seed: every DDP rank calls setup() itself and must get the same split (an unseeded split gives each rank a different
        # train / validation partition, so validation rows leak into other ranks' training data)
        ds = ds.train_test_split(test_size=self.hparams.dataset_valid_ratio, seed=self.hparams.seed)

        self.trainset = (
            TextMelDataset(  # pylint: disable=attribute-defined-outside-init
                ds["train"],
                self.hparams.add_blank,
                self.hparams.n_fft,
                self.hparams.n_feats,
                self.hparams.sample_rate,
                self.hparams.hop_length,
                self.hparams.win_length,
                self.hparams.f_min,
                self.hparams.f_max,
                self.hparams.seed,
                self.hparams.load_durations,
                "tmp",
                self.hparams.bert_dim,
                self.hparams.bert_dir,
                self.hparams.phone_override,
                self.hparams.teacher_prior_weight,
                self.hparams.dur_weight_counts,
                self.hparams.dur_weight_min_count,
            )
        )
        self.validset = (
            TextMelDataset(  # pylint: disable=attribute-defined-outside-init
                ds["test"],
                self.hparams.add_blank,
                self.hparams.n_fft,
                self.hparams.n_feats,
                self.hparams.sample_rate,
                self.hparams.hop_length,
                self.hparams.win_length,
                self.hparams.f_min,
                self.hparams.f_max,
                self.hparams.seed,
                self.hparams.load_durations,
                "tmp",
                self.hparams.bert_dim,
                self.hparams.bert_dir,
                self.hparams.phone_override,
                self.hparams.teacher_prior_weight,
                self.hparams.dur_weight_counts,
                self.hparams.dur_weight_min_count,
            )
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(),
            prefetch_factor=2 if self.hparams.num_workers > 0 else None,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(),
            prefetch_factor=2 if self.hparams.num_workers > 0 else None,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        add_blank=True,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        seed=None,
        load_durations=False,
        tmp_dir="tmp",
        bert_dim=0,
        bert_dir=None,
        phone_override=None,
        teacher_prior_weight=1.0,
        dur_weight_counts=None,
        dur_weight_min_count=0,
    ):
        self.dataset = dataset
        # {character: how often the teacher saw it in training}; with `dur_weight_min_count` > 0 the tokens of characters seen less often get weight 0 in the
        # duration loss (the teacher reads them badly, so its timing there is not a target): see get_dur_weight
        self.char_counts = json.load(open(dur_weight_counts, encoding="utf-8")) if dur_weight_counts and dur_weight_min_count > 0 else None
        self.dur_weight_min_count = dur_weight_min_count
        self.teacher_prior_weight = teacher_prior_weight  # weight in the prior loss of rows whose `source` is "teacher" (they have no real audio)
        self.phone_override = json.load(open(phone_override, encoding="utf-8")) if phone_override else {}  # {row id: phone string}
        self.bert_dim = bert_dim
        self.bert_side = BertSideFile(bert_dir) if bert_dir else None
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations
        self.tmp_dir = Path(tmp_dir)

        # Create temporary directory if it does not exist
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        random.seed(seed)

    def get_datapoint(self, row):
        text = row["text"]
        lang = row["lang"]
        phone = self.phone_override.get(row.get("id"), row["phone"])  # corrected labels win over the dataset's own
        has_audio = row.get("audio") is not None  # teacher rows (text + synthetic flow-encoder states) carry no audio
        audio, sr, path = read_audio(row["audio"]) if has_audio else (None, None, None)
        audio_path = path if path is not None else str(row.get("id") or abs(hash(text))) + ".wav"

        # Check if lang/phone/tones are already pre-processed as integers (from dataset)
        # vs strings (need to be processed)
        if isinstance(lang, list) and len(lang) > 0 and isinstance(lang[0], int):
            # Already pre-computed - convert directly to tensors
            lang_ids = torch.LongTensor(lang)
            phone_ids = torch.LongTensor(phone) if isinstance(phone, list) else phone
            tone = torch.LongTensor(row.get("tones", []))
            word_pos = torch.LongTensor(row.get("word_pos", []))
            syllable_pos = torch.LongTensor(row.get("syllable_pos", []))
            word2ph = row.get("word2ph")  # only needed for BERT features
        else:
            # Need to process text
            text_result = self.get_text(
                text,
                lang,
                phone,
                add_blank=self.add_blank,
            )

            # Skip samples with empty/invalid phoneme sequences
            if text_result is None:
                return None

            text, lang_ids, phone_ids, tone, word_pos, syllable_pos, word2ph = text_result

        if has_audio:
            if audio is None:
                raise ValueError(f"Audio data is None for {audio_path}")

            # Handle both list and numpy array formats from HuggingFace datasets
            if isinstance(audio, list):
                audio = np.array(audio, dtype=np.float32)
            else:
                audio = np.array(audio, dtype=np.float32)
            # Resample audio for mel-spectrogram generation (CosyVoice2 uses 24kHz)
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            mel = self.get_mel(audio, self.sample_rate)
        else:
            mel = None  # filled in below once the flow-encoder states give the length

        # Load pre-computed speaker embedding from dataset, or provide default
        if "spk_emb" in row:
            spk_emb = (
                torch.tensor(row["spk_emb"], dtype=torch.float32)
                if not isinstance(row["spk_emb"], torch.Tensor)
                else row["spk_emb"].float()
            )
        else:
            # Provide default speaker embedding for testing
            spk_emb = torch.zeros(192, dtype=torch.float32)

        durations = self.get_durations(audio, text) if (self.load_durations and has_audio) else None

        # Load pre-computed decoder hidden state from dataset, or provide default
        if "decoder_h" in row:
            decoder_h = (
                torch.tensor(row["decoder_h"], dtype=torch.float32)
                if not isinstance(row["decoder_h"], torch.Tensor)
                else row["decoder_h"].float()
            )
        else:
            # Provide default decoder hidden state for testing
            # Shape should be (time_steps, n_feats) where n_feats matches mel features (80)
            mel_time_steps = mel.shape[1]
            decoder_h = torch.zeros(mel_time_steps, self.n_mels, dtype=torch.float32)

        if mel is None:  # no audio: the mel is only used for its shape by the encoder-only training (no decoder), so a zero one of the right length does
            mel = torch.zeros(self.n_mels, decoder_h.shape[0], dtype=torch.float32)

        bert = self.get_bert(row, word2ph, phone_ids.shape[-1]) if self.bert_dim > 0 else None
        silence = pause_mask = None
        if row.get("silence") is not None:  # per-frame silence flags (teacher rows): the alignment may give those frames to the pause tokens
            silence = torch.as_tensor(np.asarray(row["silence"]), dtype=torch.float32)
            assert silence.shape[0] == decoder_h.shape[0], (row.get("id"), silence.shape, decoder_h.shape)
            pause_mask = self.get_pause_mask(phone_ids)
        prior_weight = self.teacher_prior_weight if row.get("source") == "teacher" else 1.0
        dur_weight = self.get_dur_weight(text, lang, word2ph, phone_ids.shape[-1]) if getattr(self, "char_counts", None) is not None else None

        return {
            "x": phone_ids,
            "y": mel,
            "filepath": audio_path,
            "x_text": text,
            "durations": durations,
            "lang": lang_ids,
            "tone": tone,
            "word_pos": word_pos,
            "syllable_pos": syllable_pos,
            "spk_emb": spk_emb,
            "decoder_h": decoder_h,
            "bert": bert,
            "prior_weight": prior_weight,
            "silence": silence,
            "pause_mask": pause_mask,
            "dur_weight": dur_weight,
        }

    def get_dur_weight(self, text, lang, word2ph, n_tokens):
        """(n_tokens,) float duration-loss weight per token: 0 for the phones of characters that occur fewer than `dur_weight_min_count` times in
        `char_counts` and for the blanks next to such a phone, 1 elsewhere (pads and punctuation included). All ones when the text units cannot be matched
        to `word2ph`, the row is not Cantonese or it is code-mixed (English words have one `word2ph` entry per word, not per character)."""
        weights = np.ones(n_tokens, dtype=np.float32)
        from jyutvoice.text.script_split import has_latin

        if lang != "yue" or word2ph is None or has_latin(text):
            return torch.from_numpy(weights)
        from jyutvoice.text.bert_align import unit_chars
        from jyutvoice.text.cantonese.readings import cleaned
        from jyutvoice.text.symbols import punctuations

        chars = unit_chars(cleaned(text))
        if len(chars) != len(word2ph) - 2:  # word2ph carries the two pads
            return torch.from_numpy(weights)
        n_phones = sum(word2ph)
        phone_w = np.ones(n_phones, dtype=np.float32)
        pos = word2ph[0]
        for c, n in zip(chars, word2ph[1:-1]):
            if c not in punctuations and self.char_counts.get(c, 0) < self.dur_weight_min_count:  # punctuation is never masked: those are the pause tokens
                phone_w[pos : pos + n] = 0.0
            pos += n
        if self.add_blank:
            if n_tokens != 2 * n_phones + 1:
                return torch.from_numpy(weights)
            weights[1::2] = phone_w
            weights[2:-1:2] = np.minimum(phone_w[:-1], phone_w[1:])  # a blank between two phones: both must be trusted
        else:
            if n_tokens != n_phones:
                return torch.from_numpy(weights)
            weights[:] = phone_w
        return torch.from_numpy(weights)

    def get_pause_mask(self, phone_ids):
        """(T_text,) bool: the tokens that may take silent frames: punctuation marks, and the pad tokens at both ends (leading / trailing silence)
        with the blanks around them."""
        ids = torch.as_tensor(phone_ids, dtype=torch.long).view(-1)
        mask = torch.isin(ids, PUNCT_IDS.to(ids.device))
        edge = 3 if self.add_blank else 1
        mask[:edge] = True
        mask[-edge:] = True
        return mask

    def get_bert(self, row, word2ph, n_positions):
        """(n_positions, bert_dim) BERT features on the phone positions: the row's per-character `bert` column (n_chars + 2 rows, the two
        pads included) repeated `word2ph` times, with blanks if the phones have them. Zeros when the row has no features."""
        unit = row.get("bert")
        if unit is None and getattr(self, "bert_side", None) is not None:
            unit = self.bert_side.get(row.get("id"))  # features kept next to the dataset, looked up by row id
        if unit is None or word2ph is None:
            return torch.zeros(n_positions, self.bert_dim)
        unit = torch.as_tensor(np.asarray(unit, dtype=np.float32))
        if unit.ndim != 2 or unit.shape[1] != self.bert_dim:
            raise ValueError(f"`bert` has shape {tuple(unit.shape)}, expected (n_chars + 2, {self.bert_dim})")
        return expand_for_sequence(unit, word2ph, n_positions)

    def get_durations(self, filepath, text):
        filepath = Path(filepath)
        data_dir, name = filepath.parent.parent, filepath.stem

        try:
            dur_loc = data_dir / "durations" / f"{name}.npy"
            durs = torch.from_numpy(np.load(dur_loc).astype(int))

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}, make sure you've generate the durations first using: python matcha/utils/get_durations_from_trained_model.py \n"
            ) from e

        assert len(durs) == len(
            text
        ), f"Length of durations {len(durs)} and text {len(text)} do not match"

        return durs

    def get_mel(self, audio, sr: int):
        audio = torch.from_numpy(audio).unsqueeze(0).float()  # [1, T]
        assert sr == self.sample_rate
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        ).squeeze()

        return mel

    def get_text(
        self,
        text,
        lang,
        phone=None,
        add_blank=False,
    ):
        try:
            phone_ids, tone_ids, word_pos, syllable_pos, lang_ids, word2ph = text_to_sequence(
                text, lang, phone, return_word2ph=True
            )

            if add_blank:
                phone_ids = intersperse(phone_ids, 0)
                tone_ids = intersperse(tone_ids, 0)
                word_pos = intersperse(word_pos, 0)
                syllable_pos = intersperse(syllable_pos, 0)
                lang_ids = intersperse(lang_ids, 0)
            phone_ids = torch.LongTensor(phone_ids)
            lang_ids = torch.LongTensor(lang_ids)
            tone = torch.LongTensor(tone_ids)
            word_pos = torch.LongTensor(word_pos)
            syllable_pos = torch.LongTensor(syllable_pos)

            return text, lang_ids, phone_ids, tone, word_pos, syllable_pos, word2ph
        except Exception as e:
            raise ValueError(
                f"Error processing text: {text} with phone: {phone}. Exception: {str(e)}"
            ) from e

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.dataset[index])
        # If datapoint is None (invalid/empty phoneme), try next samples
        attempts = 0
        while datapoint is None and attempts < 10:
            index = (index + 1) % len(self.dataset)
            datapoint = self.get_datapoint(self.dataset[index])
            attempts += 1

        if datapoint is None:
            # Fallback: return first valid sample
            for i in range(len(self.dataset)):
                datapoint = self.get_datapoint(self.dataset[i])
                if datapoint is not None:
                    break

        return datapoint

    def __len__(self):
        return len(self.dataset)


class TextMelBatchCollate:
    def __init__(self, n_mels=80):
        self.n_mels = n_mels

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max(
            [item["decoder_h"].shape[0] for item in batch]
        )  # pylint: disable=consider-using-generator
        x_max_length = max(
            [item["x"].shape[-1] for item in batch]
        )  # pylint: disable=consider-using-generator
        n_feats = batch[0]["y"].shape[-2]

        # Get decoder_h dimension from the actual data (hidden state dimension)
        decoder_h_dim = (
            batch[0]["decoder_h"].shape[-1]
            if len(batch[0]["decoder_h"].shape) >= 2
            else self.n_mels
        )

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        lang = torch.zeros((B, x_max_length), dtype=torch.long)
        tone = torch.zeros((B, x_max_length), dtype=torch.long)
        word_pos = torch.zeros((B, x_max_length), dtype=torch.long)
        syllable_pos = torch.zeros((B, x_max_length), dtype=torch.long)
        durations = torch.zeros((B, x_max_length), dtype=torch.long)
        spk_embed = torch.zeros(B, 192, dtype=torch.float32)
        decoder_h = torch.zeros((B, y_max_length, decoder_h_dim), dtype=torch.float32)
        bert_dim = batch[0]["bert"].shape[-1] if batch[0].get("bert") is not None else 0
        bert = torch.zeros((B, x_max_length, bert_dim), dtype=torch.float32) if bert_dim else None

        y_lengths, x_lengths = [], []
        for i, item in enumerate(batch):
            y_, x_, lang_, tone_, word_pos_, syllable_pos_, spk_embed_, decoder_h_ = (
                item["y"],
                item["x"],
                item["lang"],
                item["tone"],
                item["word_pos"],
                item["syllable_pos"],
                item["spk_emb"],
                item["decoder_h"],
            )
            y_lengths.append(decoder_h_.shape[0])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            lang[i, : lang_.shape[-1]] = lang_
            tone[i, : tone_.shape[-1]] = tone_
            word_pos[i, : word_pos_.shape[-1]] = word_pos_
            syllable_pos[i, : syllable_pos_.shape[-1]] = syllable_pos_
            spk_embed[i] = torch.tensor(spk_embed_).float()
            decoder_h[i, : decoder_h_.shape[0], : decoder_h_.shape[1]] = decoder_h_
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]
            if bert is not None:
                bert[i, : item["bert"].shape[0]] = item["bert"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)

        batch_dict = {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "lang": lang,
            "tone": tone,
            "word_pos": word_pos,
            "syllable_pos": syllable_pos,
            "spk_embed": spk_embed,
            "decoder_h": decoder_h,
            "durations": durations if not torch.eq(durations, 0).all() else None,
        }
        if bert is not None:
            batch_dict["bert"] = bert
        if any(item.get("silence") is not None for item in batch):
            silence = torch.zeros((B, y_max_length), dtype=torch.float32)
            pause_mask = torch.zeros((B, x_max_length), dtype=torch.bool)
            for i, item in enumerate(batch):
                if item.get("silence") is not None:
                    silence[i, : item["silence"].shape[0]] = item["silence"]
                    pause_mask[i, : item["pause_mask"].shape[0]] = item["pause_mask"]
            batch_dict["silence"], batch_dict["pause_mask"] = silence, pause_mask
        if any(item.get("dur_weight") is not None for item in batch):
            dur_weight = torch.ones((B, x_max_length), dtype=torch.float32)
            for i, item in enumerate(batch):
                if item.get("dur_weight") is not None:
                    dur_weight[i, : item["dur_weight"].shape[0]] = item["dur_weight"]
            batch_dict["dur_weight"] = dur_weight
        weights = [item.get("prior_weight", 1.0) for item in batch]
        if any(w != 1.0 for w in weights):  # only when some row is down-weighted: leaves every other batch exactly as before
            batch_dict["prior_weight"] = torch.tensor(weights, dtype=torch.float32)

        return batch_dict
