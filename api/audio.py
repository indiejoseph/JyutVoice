"""Audio helpers: PCM16 / WAV bytes for the responses, and decoding of the reference clip."""
from __future__ import annotations

import io
import os
import struct
import subprocess
import tempfile

import numpy as np
import torch

SAMPLE_RATE = 24000


CEILING = 0.99  # HiFT's own output limit (audio_limit): nothing leaves the API above it
SPEECH_FLOOR_DB = -40.0  # a 20 ms frame this far under the loudest one is a pause or the noise floor, not speech


def speech_rms(wav: torch.Tensor, sample_rate: int = SAMPLE_RATE) -> float:
    """Root-mean-square of the speech in `wav` (linear, full scale 1.0): only the 20 ms frames within 40 dB of the loudest one count, so pauses do not pull the level
    down. 0.0 for silence; a clip shorter than a frame is taken whole."""
    x = wav.detach().float().cpu().reshape(-1)
    if x.numel() == 0:
        return 0.0
    frame = int(0.02 * sample_rate)
    n = x.numel() // frame
    power = (x[: n * frame].reshape(n, frame) if n else x.reshape(1, -1)).pow(2).mean(dim=1)
    return float(power[power >= float(power.max()) * 10 ** (SPEECH_FLOOR_DB / 10)].mean().sqrt())  # silence: every frame passes the 0 threshold and the mean is 0


def level_gain(wav: torch.Tensor, target_dbfs: float) -> float:
    """The factor that brings the speech RMS of `wav` to `target_dbfs`; 1.0 when there is no speech."""
    rms = speech_rms(wav)
    return 1.0 if rms <= 0.0 else 10 ** (target_dbfs / 20) / rms


def apply_level(wav: torch.Tensor, gain: float) -> torch.Tensor:
    """`wav` times `gain`, the gain lowered if needed so that the loudest sample stays at the ceiling: the chunk is scaled as a whole, never flat-topped."""
    peak = float(wav.abs().max()) if wav.numel() else 0.0
    return wav * min(gain, CEILING / peak) if peak > 0.0 else wav


def tensor_to_pcm16_bytes(wav: torch.Tensor) -> bytes:
    """A `[1, T]` or `[T]` float tensor as little-endian PCM16 bytes."""
    audio = wav.detach().float().cpu().squeeze().numpy()
    return np.rint(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def wav_header(data_size: int | None, sample_rate: int = SAMPLE_RATE) -> bytes:
    """PCM16 mono WAV header. `data_size` None: the placeholder length for chunked transfer, else the exact PCM byte count."""
    if data_size is None:
        data_size = riff_size = 0xFFFFFFFF
    else:
        riff_size = 36 + data_size
    channels, bits = 1, 16
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI", b"RIFF", riff_size, b"WAVE", b"fmt ", 16, 1, channels, sample_rate, sample_rate * channels * bits // 8, channels * bits // 8, bits, b"data", data_size
    )


def wav_bytes_from_pcm(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    return wav_header(len(pcm), sample_rate=sample_rate) + pcm


def decode_audio(payload: bytes) -> tuple[torch.Tensor, int]:
    """Audio file bytes -> ((channels, samples) float32 tensor, sample rate). libsndfile first (wav, flac, ogg, mp3); anything else (m4a, webm, ...) through ffmpeg, from a
    temporary file (an m4a with its index at the end cannot be read from a pipe)."""
    try:
        return decode_audio_wav(payload)
    except Exception:
        pass
    try:
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "prompt"), os.path.join(tmp, "prompt.wav")
            with open(src, "wb") as f:
                f.write(payload)
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", src, "-ac", "1", "-acodec", "pcm_s16le", dst], capture_output=True, timeout=60, check=True)
            with open(dst, "rb") as f:
                return decode_audio_wav(f.read())
    except Exception:
        raise ValueError("prompt_audio is not an audio file this server can decode (wav, flac, ogg, mp3, m4a, ...)") from None


def decode_audio_wav(payload: bytes) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    data, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(data.T)), int(sr)
