import os
import random
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
from datasets import load_dataset, load_from_disk
from jyutvoice.utils.audio import mel_spectrogram
from jyutvoice.utils.model import fix_len_compatibility
from jyutvoice.utils.utils import intersperse
from jyutvoice.utils.mask import make_pad_mask
from jyutvoice.text import text_to_sequence
from jyutvoice.transformer.upsample_encoder import UpsampleConformerEncoder


def load_spk_embedding(onnx_path: str):
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(
        onnx_path, sess_options=option, providers=["CPUExecutionProvider"]
    )
    return ort_session


def get_spk_embedding(audio, onnx_session):
    audio_tensor = None

    if isinstance(audio, np.ndarray):
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(dim=0)
    elif isinstance(audio, torch.Tensor):
        if audio.dim() == 1:
            audio_tensor = audio.float().unsqueeze(dim=0)
        elif audio.dim() == 2:
            audio_tensor = audio.float()
        else:
            raise ValueError("Audio tensor must be 1D or 2D.")
    if audio_tensor is None:
        raise ValueError("Audio must be a numpy array or a torch tensor.")
    feat = kaldi.fbank(audio_tensor, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = (
        onnx_session.run(
            None,
            {onnx_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()},
        )[0]
        .flatten()
        .tolist()
    )

    return embedding


def load_speech_tokenizer(speech_tokenizer_path: str):
    """Load speech tokenizer ONNX model."""
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        speech_tokenizer_path,
        sess_options=option,
        providers=["CPUExecutionProvider"],
    )
    return session


def extract_speech_token(audio, speech_tokenizer_session):
    """
    Extract speech tokens from audio using speech tokenizer.

    Args:
        audio: audio signal (torch.Tensor or numpy.ndarray), shape (T,) at 16kHz
        speech_tokenizer_session: ONNX speech tokenizer session

    Returns:
        speech_token: tensor of shape (1, num_tokens)
        speech_token_len: tensor of shape (1,) with token sequence length
    """
    # Ensure audio is on CPU for processing
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    elif isinstance(audio, np.ndarray):
        pass
    else:
        raise ValueError("Audio must be torch.Tensor or numpy.ndarray")

    # Convert to torch tensor for mel-spectrogram
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

    # Extract mel-spectrogram (whisper format)
    feat = whisper.log_mel_spectrogram(audio_tensor, n_mels=128)

    # Run speech tokenizer
    speech_token = (
        speech_tokenizer_session.run(
            None,
            {
                speech_tokenizer_session.get_inputs()[0]
                .name: feat.detach()
                .cpu()
                .numpy(),
                speech_tokenizer_session.get_inputs()[1].name: np.array(
                    [feat.shape[2]], dtype=np.int32
                ),
            },
        )[0]
        .flatten()
        .tolist()
    )

    speech_token = torch.tensor([speech_token], dtype=torch.int32)
    speech_token_len = torch.tensor([len(speech_token[0])], dtype=torch.int32)

    return speech_token, speech_token_len


class FlowEncoder(torch.nn.Module):
    def __init__(self, vocab_size=6561, input_size=512, output_size=80, device="cpu"):
        super().__init__()
        self.device = device
        self.input_embedding = torch.nn.Embedding(vocab_size, input_size)
        # Instantiate encoder with CosyVoice2 architecture
        self.encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer="linear",
            pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn",
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
            static_chunk_size=25,
        ).to(device)
        # Project encoder output from 512 to output_size (80)
        self.encoder_proj = torch.nn.Linear(512, output_size)

    def forward(self, token, token_len):
        """
        Process speech tokens through the encoder.

        Args:
            token: speech tokens, shape (batch, seq_len)
            token_len: token sequence lengths, shape (batch,)

        Returns:
            h: encoder output, shape (batch, seq_len, 80)
            h_lengths: output lengths, shape (batch,)
        """
        mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(self.device)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        # Encode
        h, h_lengths = self.encoder(token, token_len, streaming=False)
        # Project to output size (80)
        h = self.encoder_proj(h)

        return h, h_lengths


def load_flow_encoder(flow_encoder_path, device="cpu"):
    """
    Load the UpsampleConformerEncoder from a pretrained model checkpoint.

    Args:
        flow_encoder_path (str): Path to the pretrained flow model weights
        device (str or torch.device): Device to load model on

    Returns:
        torch.nn.Module: Loaded encoder module ready for inference
    """
    if flow_encoder_path is None:
        return None

    flow_encoder = FlowEncoder(device=device)

    # Load pretrained weights
    state_dict = torch.load(flow_encoder_path, map_location=device, weights_only=True)
    flow_encoder.load_state_dict(state_dict)
    flow_encoder.eval()

    return flow_encoder


def get_decoder_hidden_state(speech_token, speech_token_len, flow_encoder, device):
    """
    Extract hidden state from the flow encoder (CosyVoice2's encoder).

    This function processes speech tokens through the encoder to get
    the hidden representation used for prior loss computation during training.

    Args:
        speech_token (torch.Tensor): Speech tokens, shape (batch, token_len)
        speech_token_len (torch.Tensor): Lengths of speech token sequences
        flow_encoder (torch.nn.Module): The flow encoder
        device (torch.device): Device to run inference on

    Returns:
        torch.Tensor: Hidden state from encoder, shape (batch, token_len, 512)
    """
    if flow_encoder is None:
        raise ValueError(
            "flow_encoder must be provided to extract decoder hidden state"
        )

    speech_token = speech_token.to(device)
    speech_token_len = speech_token_len.to(device)

    with torch.no_grad():
        # Get encoder output and lengths
        h, h_lengths = flow_encoder(speech_token, speech_token_len)
        # h shape: (batch, token_len, 512)

    return h


class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        dataset_path,
        dataset_valid_ratio,
        speaker_embedding_model_path,
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
        token_mel_ratio,
        seed,
        load_durations,
        flow_encoder_path,
        speech_tokenizer_path,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Load flow encoder if path is provided
        if self.hparams.flow_encoder_path is not None:
            self.flow_encoder = load_flow_encoder(self.hparams.flow_encoder_path)
        else:
            self.flow_encoder = None

        # Load speech tokenizer if path is provided
        if self.hparams.speech_tokenizer_path is not None:
            self.speech_tokenizer = load_speech_tokenizer(
                self.hparams.speech_tokenizer_path
            )
        else:
            self.speech_tokenizer = None

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if os.path.exists(self.hparams.dataset_path):
            ds = load_from_disk(self.hparams.dataset_path)
        else:
            ds = load_dataset(self.hparams.dataset_path, split="train")
        ds = ds.train_test_split(test_size=self.hparams.dataset_valid_ratio)

        speaker_embedding_onnx_session = load_spk_embedding(
            self.hparams.speaker_embedding_model_path
        )

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
                self.hparams.token_mel_ratio,
                self.hparams.seed,
                self.hparams.load_durations,
                "tmp",
                speaker_embedding_onnx_session,
                self.flow_encoder,
                self.speech_tokenizer,
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
                self.hparams.token_mel_ratio,
                self.hparams.seed,
                self.hparams.load_durations,
                "tmp",
                speaker_embedding_onnx_session,
                self.flow_encoder,
                self.speech_tokenizer,
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
        token_mel_ratio=0,
        seed=None,
        load_durations=False,
        tmp_dir="tmp",
        speaker_embedding_onnx_session=None,
        flow_encoder=None,
        speech_tokenizer=None,
    ):
        self.dataset = dataset
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.token_mel_ratio = token_mel_ratio
        self.load_durations = load_durations
        self.speaker_embedding_onnx_session = speaker_embedding_onnx_session
        self.flow_encoder = flow_encoder
        self.speech_tokenizer = speech_tokenizer
        self.tmp_dir = Path(tmp_dir)

        # Create temporary directory if it does not exist
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        random.seed(seed)

    def get_datapoint(self, row):
        text = row["text"]
        lang = row["lang"]
        phone = row["phone"]
        audio = row["audio"]["array"]
        decoder_h = row.get("decoder_h", None)
        spk_emb = row.get("spk_emb", None)
        audio_path = (
            row["audio"]["path"]
            if row["audio"]["path"] is not None
            else str(abs(hash(text))) + ".wav"
        )
        sr = row["audio"]["sampling_rate"]

        # Check if lang/phone/tones are already pre-processed as integers (from dataset)
        # vs strings (need to be processed)
        if isinstance(lang, list) and len(lang) > 0 and isinstance(lang[0], int):
            # Already pre-computed - convert directly to tensors
            lang_ids = torch.LongTensor(lang)
            phone_ids = torch.LongTensor(phone) if isinstance(phone, list) else phone
            tone = torch.LongTensor(row.get("tones", []))
            word_pos = torch.LongTensor(row.get("word_pos", []))
            syllable_pos = torch.LongTensor(row.get("syllable_pos", []))
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

            text, lang_ids, phone_ids, tone, word_pos, syllable_pos = text_result

        if audio is None:
            raise ValueError(f"Audio data is None for {audio_path}")

        # Handle both list and numpy array formats from HuggingFace datasets
        if isinstance(audio, list):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = np.array(audio, dtype=np.float32)
        audio16k = audio
        audio24k = audio
        # Resample audio:
        # - audio24k: for mel-spectrogram generation (CosyVoice2 uses 24kHz)
        # - audio16k: for speaker embedding extraction (CampPlus requires 16kHz)
        if sr == 16_000:
            audio24k = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        elif sr == 24_000:
            audio16k = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
        mel = self.get_mel(audio24k, self.sample_rate)

        # caching speaker embeddings
        if spk_emb is None and self.speaker_embedding_onnx_session is not None:
            spk_emb_path = self.tmp_dir / "spk_emb" / (audio_path + ".pt")

            if spk_emb_path.exists():
                spk_emb = torch.load(spk_emb_path)
            else:
                spk_emb = get_spk_embedding(
                    audio16k, self.speaker_embedding_onnx_session
                )
                spk_emb_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(spk_emb, spk_emb_path)

        durations = self.get_durations(audio, text) if self.load_durations else None

        # Extract decoder hidden state for prior loss computation
        if (
            decoder_h is None
            and self.flow_encoder is not None
            and self.speech_tokenizer is not None
        ):
            decoder_h_path = self.tmp_dir / "decoder_h" / (audio_path + ".pt")
            # caching decoder hidden states
            if decoder_h_path.exists():
                decoder_h = torch.load(decoder_h_path)
            else:
                # Extract speech tokens from audio (16kHz required for speech tokenizer)
                speech_token, speech_token_len = extract_speech_token(
                    audio16k, self.speech_tokenizer
                )

                # Pass speech tokens to flow encoder to get hidden state
                decoder_h = get_decoder_hidden_state(
                    speech_token, speech_token_len, self.flow_encoder, "cpu"
                )
                # decoder_h shape: (1, token_len, 512)
                # Save for caching: shape should be (1, token_len, 512)
                decoder_h_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(decoder_h.squeeze(0), decoder_h_path)
                decoder_h = decoder_h.squeeze(0)  # Remove batch dim -> (token_len, 512)

        if self.token_mel_ratio != 0:
            decoder_h_len = decoder_h.shape[0]
            token_len = int(min(mel.shape[1] / self.token_mel_ratio, decoder_h_len))
            mel_len = self.token_mel_ratio * token_len
            mel = mel[:, :mel_len]

            if mel_len != decoder_h_len:
                decoder_h = decoder_h[:mel_len, :]

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
        }

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
            phone_ids, tone_ids, word_pos, syllable_pos, lang_ids = text_to_sequence(
                text, lang, phone
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

            return text, lang_ids, phone_ids, tone, word_pos, syllable_pos
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
    def __call__(self, batch):
        B = len(batch)
        y_max_length = max(
            [item["y"].shape[-1] for item in batch]
        )  # pylint: disable=consider-using-generator
        x_max_length = max(
            [item["x"].shape[-1] for item in batch]
        )  # pylint: disable=consider-using-generator
        n_feats = batch[0]["y"].shape[-2]

        # Check if decoder_h is present and get its hidden dimension
        has_decoder_h = batch[0].get("decoder_h") is not None
        decoder_h_dim = None
        if has_decoder_h:
            decoder_h_dim = batch[0]["decoder_h"].shape[-1]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        lang = torch.zeros((B, x_max_length), dtype=torch.long)
        tone = torch.zeros((B, x_max_length), dtype=torch.long)
        word_pos = torch.zeros((B, x_max_length), dtype=torch.long)
        syllable_pos = torch.zeros((B, x_max_length), dtype=torch.long)
        durations = torch.zeros((B, x_max_length), dtype=torch.long)
        spk_embed = torch.zeros(B, 192, dtype=torch.float32)
        decoder_h = None
        if has_decoder_h:
            decoder_h = torch.zeros(
                (B, y_max_length, decoder_h_dim), dtype=torch.float32
            )

        y_lengths, x_lengths, decoder_h_lengths = [], [], []
        filepaths, x_texts = [], []
        for i, item in enumerate(batch):
            y_, x_, lang_, tone_, word_pos_, syllable_pos_, spk_embed_ = (
                item["y"],
                item["x"],
                item["lang"],
                item["tone"],
                item["word_pos"],
                item["syllable_pos"],
                item["spk_emb"],
            )
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            lang[i, : lang_.shape[-1]] = lang_
            tone[i, : tone_.shape[-1]] = tone_
            word_pos[i, : word_pos_.shape[-1]] = word_pos_
            syllable_pos[i, : syllable_pos_.shape[-1]] = syllable_pos_
            if spk_embed_ is not None:
                spk_embed[i] = torch.tensor(spk_embed_).float()
            if has_decoder_h and item.get("decoder_h") is not None:
                decoder_h_ = item["decoder_h"]
                decoder_h[i, : decoder_h_.shape[0]] = decoder_h_
                decoder_h_lengths.append(decoder_h_.shape[0])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]

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
            "filepaths": filepaths,
            "x_texts": x_texts,
            "spk_embed": spk_embed,
            "durations": durations if not torch.eq(durations, 0).all() else None,
        }

        if has_decoder_h:
            batch_dict["decoder_h"] = decoder_h
            batch_dict["decoder_h_lengths"] = torch.tensor(
                decoder_h_lengths, dtype=torch.long
            )

        return batch_dict
