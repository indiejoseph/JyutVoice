"""The TTS service behind the API: the exported ONNX acoustic model, the CosyVoice2 flow decoder and HiFT vocoder (PyTorch, CPU), and the CAMPPlus speaker model.

Same division of work as the SoulX-Podcast service: `prepare_speech_context` does what can fail on bad input (reference clip, text front end) so that the HTTP layer can answer
400/404 before any audio is sent, `generate_speech_bytes` returns the whole file, `stream_speech_bytes` yields it chunk by chunk. One synthesis at a time (a lock): the
decoder and the vocoder already use all the cores the container is given.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import numpy as np
import torch  # before onnxruntime (see infer_onnx.py)
from hyperpyyaml import load_hyperpyyaml

from api.audio import CEILING, apply_level, decode_audio, level_gain, tensor_to_pcm16_bytes, wav_bytes_from_pcm, wav_header
from api.config import APIConfig, config
from api.models import SpeechRequest
from api.text_split import split_text
from jyutvoice.utils.onnx_infer import extract_spk_embedding, load_flow_decoder, prepare_text, synthesise_wave, to_16k_mono
from jyutvoice.utils.onnx_synth import OnnxAcousticModel, load_session, text_tokens

logger = logging.getLogger(__name__)

_VOCODER_UNLIMITED = 100.0  # HiFT clamps its output to +-audio_limit (0.99); for some voices it would be 1.2-1.3 x full scale, which that clamp flattens

_DATA_URI = re.compile(r"^data:[^,;]*(?:;[^,;=]+=[^,;]*)*;base64,", re.IGNORECASE)


def _in_phrase(text: str, i: int) -> bool:
    """The space at `i` is inside an English phrase (ASCII letters or digits on both sides): not a place to cut."""
    return 0 < i < len(text) - 1 and text[i - 1].isascii() and text[i - 1].isalnum() and text[i + 1].isascii() and text[i + 1].isalnum()


class PromptCacheMiss(LookupError):
    """prompt_cache_id names no voice in the cache (never sent, or evicted): the client sends prompt_audio again."""


@dataclass
class Chunk:
    text: str  # the prepared (normalised, word-segmented) text
    tokens: Dict[str, np.ndarray]


@dataclass
class SpeechPlan:
    chunks: List[Chunk]
    spk_embed: torch.Tensor  # (1, 192), L2-normalised x-vector
    c: torch.Tensor  # the flow decoder's speaker conditioning
    seed: int
    flow_steps: int
    length_scale: float
    dp_temperature: float
    dp_pause_temperature: float
    prosody_temperature: float
    use_prosody: bool
    level_dbfs: Optional[float] = None  # speech RMS to level the output to; None leaves the vocoder's own level (limited at CEILING)
    gain: Optional[float] = None  # set by the first chunk, used by all of them


def _split_prepared(text: str, tokens_of, max_tokens: int) -> List[Tuple[str, Dict[str, np.ndarray]]]:
    """`text` (word-segmented) as pieces whose token arrays fit the text encoder: cut at the word break nearest to the middle, recursively. English phrases stay whole."""
    tokens = tokens_of(text)
    if tokens["x"].shape[1] <= max_tokens:
        return [(text, tokens)]
    breaks = [m.start() for m in re.finditer(" ", text) if not _in_phrase(text, m.start())]
    if not breaks:
        raise ValueError(f"a stretch of text without any word break gives {tokens['x'].shape[1]} tokens, more than the model takes ({max_tokens}); add punctuation or spaces")
    cut = min(breaks, key=lambda i: abs(i - len(text) / 2))
    return _split_prepared(text[:cut].strip(), tokens_of, max_tokens) + _split_prepared(text[cut:].strip(), tokens_of, max_tokens)


class TTSService:
    def __init__(self, cfg: APIConfig = config):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self._lock = threading.Lock()  # one synthesis at a time
        self._cache: "OrderedDict[str, Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._loaded = False
        self._load()

    # ---- loading ------------------------------------------------------------------------------------------------------------------------------------

    def _load(self) -> None:
        cfg = self.cfg
        if cfg.torch_threads > 0:
            torch.set_num_threads(cfg.torch_threads)
        t0 = time.time()
        with open(cfg.decoder_config, "r") as f:
            configs = load_hyperpyyaml(f)
        self.acoustic = OnnxAcousticModel(cfg.onnx_dir, threads=cfg.onnx_threads)
        m = self.acoustic.manifest
        logger.info(
            "acoustic model: %s duration model, ProsodyCFM %s, max %d tokens, exported from %s",
            self.acoustic.dp_kind, "yes" if self.acoustic.has_prosody else "no", self.acoustic.max_tokens, m.get("source", {}).get("checkpoint"),
        )  # fmt: skip
        self.decoder, self.spk_affine = load_flow_decoder(configs, os.path.join(cfg.pretrained_dir, "flow_decoder.pt"), self.device)
        self.hift = configs["hift"].eval().to(self.device)
        self.hift.load_state_dict(torch.load(os.path.join(cfg.pretrained_dir, "hift.pt"), map_location="cpu"))
        self.hift.audio_limit = _VOCODER_UNLIMITED  # `_level` limits and levels the audio instead, from its true level
        self.spk_model = load_session(os.path.join(cfg.pretrained_dir, "campplus.onnx"), threads=1)
        logger.info("models loaded in %.1f s", time.time() - t0)
        if cfg.warmup:
            self._warmup()
        self._loaded = True

    def _warmup(self) -> None:
        """One short synthesis: loads the pydips model and the WeTextProcessing graphs, and warms ONNX Runtime and PyTorch, so the first request is not slow."""
        t0 = time.time()
        g = torch.Generator().manual_seed(0)
        spk = torch.nn.functional.normalize(torch.randn(1, 192, generator=g), dim=1)
        with torch.no_grad():
            c = self.spk_affine(spk)
        plan = self._plan_text("你好，我係 JyutVoice。今日係2026年10月21日。", spk, c, self._defaults(), normalize=self.cfg.text_normalize)
        synthesise_wave(self.acoustic, self.decoder, self.hift, spk, c, plan.chunks[0].tokens, device=self.device, seed=0, n_timesteps=2)
        logger.info("warm-up done in %.1f s", time.time() - t0)

    def is_loaded(self) -> bool:
        return self._loaded

    def info(self) -> dict:
        with self._cache_lock:
            cached = len(self._cache)
        return {"duration_model": self.acoustic.dp_kind, "prosody": self.acoustic.has_prosody, "cached_prompts": cached}

    # ---- the reference voice ------------------------------------------------------------------------------------------------------------------------

    def _read_prompt_bytes(self, prompt_audio: str) -> bytes:
        """prompt_audio -> the audio file's bytes: a file:// URI under PROMPT_AUDIO_ROOT, a data: URI, or raw base64."""
        limit = self.cfg.max_prompt_bytes
        value = prompt_audio.strip()
        if value.lower().startswith("file://"):
            root = self.cfg.prompt_audio_root
            if not root:
                raise ValueError("file:// prompt_audio is not enabled on this server (PROMPT_AUDIO_ROOT is not set); send the clip as base64")
            path = os.path.realpath(unquote(urlparse(value).path))
            root = os.path.realpath(root)
            if os.path.commonpath([path, root]) != root:
                raise ValueError("prompt_audio file is outside PROMPT_AUDIO_ROOT")
            if not os.path.isfile(path):
                raise ValueError("prompt_audio file does not exist")
            if os.path.getsize(path) > limit:
                raise ValueError(f"prompt_audio is larger than {limit} bytes")
            with open(path, "rb") as f:
                return f.read()
        match = _DATA_URI.match(value)
        if match:
            value = value[match.end() :]
        value = re.sub(r"\s+", "", value)
        if len(value) * 3 // 4 > limit:
            raise ValueError(f"prompt_audio is larger than {limit} bytes")
        try:
            value += "=" * (-len(value) % 4)
            data = base64.urlsafe_b64decode(value) if ("-" in value or "_" in value) else base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("prompt_audio is not valid base64 (send the audio file's bytes, base64-encoded, or a data:audio/...;base64, URI)") from None
        if not data:
            raise ValueError("prompt_audio is empty")
        return data

    def _speaker(self, payload: bytes) -> Tuple[torch.Tensor, torch.Tensor]:
        """Audio file bytes -> (L2-normalised x-vector (1, 192), the decoder's conditioning from it)."""
        waveform, sr = decode_audio(payload)
        seconds = waveform.shape[1] / sr
        if seconds < self.cfg.prompt_min_seconds:
            raise ValueError(f"prompt_audio is {seconds:.2f} s long, at least {self.cfg.prompt_min_seconds:g} s is needed")
        waveform = waveform[:, : int(self.cfg.prompt_max_seconds * sr)]
        speech = to_16k_mono(waveform, sr)
        spk_embed = torch.nn.functional.normalize(extract_spk_embedding(self.spk_model, speech), dim=1)  # the model normalises it first
        with torch.no_grad():
            c = self.spk_affine(spk_embed)
        return spk_embed, c

    def _voice(self, request: SpeechRequest) -> Tuple[str, torch.Tensor, torch.Tensor]:
        if request.prompt_audio and request.prompt_cache_id:
            raise ValueError("send either prompt_audio or prompt_cache_id, not both")
        if request.prompt_cache_id:
            key = request.prompt_cache_id
            with self._cache_lock:
                if key not in self._cache:
                    raise PromptCacheMiss(f"prompt_cache_id {key!r} is not in the cache; send prompt_audio again")
                self._cache.move_to_end(key)
                spk_embed, c = self._cache[key]
            return key, spk_embed, c
        if not request.prompt_audio:
            raise ValueError("prompt_audio (base64 of the reference clip) or prompt_cache_id is required")
        payload = self._read_prompt_bytes(request.prompt_audio)
        key = "pc_" + hashlib.sha256(payload).hexdigest()[:32]
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
        if hit is None:
            hit = self._speaker(payload)
            with self._cache_lock:
                self._cache[key] = hit
                while len(self._cache) > max(self.cfg.prompt_cache_size, 1):
                    self._cache.popitem(last=False)
        return key, hit[0], hit[1]

    # ---- the text -----------------------------------------------------------------------------------------------------------------------------------

    def _defaults(self) -> dict:
        cfg = self.cfg
        return {
            "flow_steps": cfg.flow_steps, "length_scale": cfg.length_scale, "dp_temperature": cfg.dp_temperature,
            "dp_pause_temperature": cfg.dp_pause_temperature, "prosody_temperature": cfg.prosody_temperature, "use_prosody": self.acoustic.has_prosody, "seed": 0,
            "level_dbfs": cfg.output_level_dbfs if cfg.normalize_loudness else None,
        }  # fmt: skip

    def _plan_text(self, text: str, spk_embed: torch.Tensor, c: torch.Tensor, settings: dict, normalize: bool) -> SpeechPlan:
        add_blank = self.acoustic.add_blank
        tokens_of = lambda t: text_tokens(t, lang="yue", add_blank=add_blank)  # noqa: E731
        chunks: List[Chunk] = []
        for piece in split_text(text, self.cfg.max_chunk_chars):
            prepared = prepare_text(piece, "yue", "pydips", normalize).strip()
            if not any(ch.isalnum() for ch in prepared):  # nothing but marks (or nothing): it would be a pause, not speech
                continue
            for part, tokens in _split_prepared(prepared, tokens_of, self.acoustic.max_tokens):
                chunks.append(Chunk(part, tokens))
        if not chunks:
            raise ValueError("input has nothing to read")
        return SpeechPlan(chunks=chunks, spk_embed=spk_embed, c=c, **settings)

    def prepare_speech_context(self, request: SpeechRequest) -> dict:
        """Everything that can be wrong with the request: the reference voice (and its cache id), the text. Returns {"prompt_cache_id", "prepared"}."""
        if len(request.input) > self.cfg.max_input_chars:
            raise ValueError(f"input is {len(request.input)} characters, the limit is {self.cfg.max_input_chars}")
        cache_id, spk_embed, c = self._voice(request)
        settings = self._defaults()
        if request.flow_steps is not None:
            settings["flow_steps"] = request.flow_steps
        if request.speed is not None:
            settings["length_scale"] = self.cfg.length_scale / request.speed
        if request.dp_temperature is not None:
            settings["dp_temperature"] = request.dp_temperature
        if request.prosody_temperature is not None:
            settings["prosody_temperature"] = request.prosody_temperature
        if request.prosody is not None:
            if request.prosody and not self.acoustic.has_prosody:
                raise ValueError("this model was exported without ProsodyCFM: prosody=true is not available")
            settings["use_prosody"] = request.prosody
        if request.normalize_loudness is not None:
            settings["level_dbfs"] = self.cfg.output_level_dbfs if request.normalize_loudness else None
        settings["seed"] = request.seed if request.seed is not None else secrets.randbelow(2**31)
        normalize = request.normalize_text if request.normalize_text is not None else self.cfg.text_normalize
        plan = self._plan_text(request.input, spk_embed, c, settings, normalize)
        return {"prompt_cache_id": cache_id, "prepared": plan}

    # ---- synthesis ----------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _level(plan: SpeechPlan, wav: torch.Tensor) -> torch.Tensor:
        """The vocoder's waveform (limiter off) as it is sent: levelled by the request's one gain (taken from its first chunk, so chunks keep their level relative to
        each other), or, with levelling off, limited at CEILING as HiFT used to."""
        if plan.level_dbfs is None:
            return torch.clamp(wav, -CEILING, CEILING)
        if plan.gain is None:
            plan.gain = level_gain(wav, plan.level_dbfs)
            logger.info("level: speech RMS of the first chunk brought to %.1f dBFS with a gain of %+.1f dB", plan.level_dbfs, 20 * float(np.log10(plan.gain)))
        return apply_level(wav, plan.gain)

    def _synthesise(self, plan: SpeechPlan, index: int) -> torch.Tensor:
        """Chunk `index` of the plan -> waveform (1, T), 24 kHz. Chunk i is drawn with seed + i, so a one-chunk request equals `infer_onnx.py --seed N` on that text,
        up to the level (`normalize_loudness=false` gives that audio exactly)."""
        chunk = plan.chunks[index]
        with self._lock:
            t0 = time.time()
            wav, info = synthesise_wave(
                self.acoustic, self.decoder, self.hift, plan.spk_embed, plan.c, chunk.tokens, device=self.device, seed=plan.seed + index, n_timesteps=plan.flow_steps,
                length_scale=plan.length_scale, dp_temperature=plan.dp_temperature, dp_pause_temperature=plan.dp_pause_temperature,
                prosody_temperature=plan.prosody_temperature, use_prosody=plan.use_prosody,
            )  # fmt: skip
        wav = self._level(plan, wav)
        seconds = wav.shape[1] / 24000
        logger.info(
            "chunk %d/%d: %d tokens -> %.2f s of audio in %.2f s (acoustic %.2f s, decoder+vocoder %.2f s, RTF %.2f)",
            index + 1, len(plan.chunks), chunk.tokens["x"].shape[1], seconds, time.time() - t0, info["t_acoustic"], info["t_vocoder"], (time.time() - t0) / max(seconds, 1e-6),
        )  # fmt: skip
        return wav

    def generate_pcm(self, plan: SpeechPlan) -> bytes:
        return b"".join(tensor_to_pcm16_bytes(self._synthesise(plan, i)) for i in range(len(plan.chunks)))

    def generate_speech_bytes(self, request: SpeechRequest, plan: SpeechPlan) -> bytes:
        pcm = self.generate_pcm(plan)
        return wav_bytes_from_pcm(pcm) if request.output_format == "wav" else pcm

    def stream_speech_bytes(self, request: SpeechRequest, plan: SpeechPlan) -> Iterator[bytes]:
        """The audio chunk by chunk: for wav the placeholder-length header first, then the PCM of each chunk as soon as it is synthesised."""
        if request.output_format == "wav":
            yield wav_header(None)
        try:
            for i in range(len(plan.chunks)):
                yield tensor_to_pcm16_bytes(self._synthesise(plan, i))
        except Exception:  # the status line is long sent: all that is left is to end the stream
            logger.error("streaming synthesis failed", exc_info=True)


_service: Optional[TTSService] = None
_service_lock = threading.Lock()


def get_service() -> TTSService:
    global _service
    with _service_lock:
        if _service is None:
            _service = TTSService()
        return _service
