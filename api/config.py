"""API configuration, from environment variables (docker/docker-compose.yml sets them)."""
import os
from dataclasses import dataclass
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str) -> tuple:
    value = os.getenv(name, "")
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass
class APIConfig:
    # Model files: the graphs and manifest.json written by scripts/export_onnx.py, and the pretrained flow decoder, HiFT vocoder and CAMPPlus
    onnx_dir: str = os.getenv("ONNX_DIR", "onnx_models")
    pretrained_dir: str = os.getenv("PRETRAINED_DIR", "pretrained_models")
    decoder_config: str = os.getenv("DECODER_CONFIG", "configs/onnx_infer.yaml")

    # CPU threads: the flow decoder and the vocoder (PyTorch) take almost all of the time; 0 leaves PyTorch's default (all cores)
    torch_threads: int = int(os.getenv("TORCH_NUM_THREADS", "0"))
    onnx_threads: int = int(os.getenv("ONNX_NUM_THREADS", "2"))

    # API security (same names as the SoulX-Podcast API)
    api_key: str = (os.getenv("API_KEY") or "").strip()
    require_api_key: bool = _env_bool("REQUIRE_API_KEY", True)
    cors_allowed_origins: tuple = _env_csv("CORS_ALLOWED_ORIGINS")
    cors_allow_credentials: bool = _env_bool("CORS_ALLOW_CREDENTIALS", False)

    # Reference audio: base64 (raw or a data: URI) in the request; file:// only under PROMPT_AUDIO_ROOT (unset: not accepted)
    prompt_audio_root: Optional[str] = os.getenv("PROMPT_AUDIO_ROOT") or None
    prompt_cache_size: int = int(os.getenv("PROMPT_CACHE_SIZE", "64"))
    max_prompt_bytes: int = int(os.getenv("MAX_PROMPT_BYTES", str(20 * 1024 * 1024)))
    prompt_min_seconds: float = float(os.getenv("PROMPT_MIN_SECONDS", "0.5"))
    prompt_max_seconds: float = float(os.getenv("PROMPT_MAX_SECONDS", "30"))
    max_request_bytes: int = int(os.getenv("MAX_REQUEST_BYTES", str(40 * 1024 * 1024)))

    # Text
    max_input_chars: int = int(os.getenv("MAX_INPUT_CHARS", "5000"))
    max_chunk_chars: int = int(os.getenv("MAX_CHUNK_CHARS", "100"))  # sentences are packed into chunks of at most this many characters, one synthesis each
    text_normalize: bool = _env_bool("TEXT_NORMALIZE", True)

    # Loudness: without a decoder prompt the vocoder's level swings 12 dB with the voice (speech RMS -9.8 ... -21.7 dBFS) and clips for some; the speech of each request is
    # levelled to this (the prompted demo renders sit at about -20.5), by one gain per request so that streamed chunks do not step
    normalize_loudness: bool = _env_bool("NORMALIZE_LOUDNESS", True)
    output_level_dbfs: float = float(os.getenv("OUTPUT_LEVEL_DBFS", "-20"))

    # Synthesis defaults (a request can override flow_steps, speed, seed, prosody, normalize_text, normalize_loudness, dp_temperature, prosody_temperature)
    flow_steps: int = int(os.getenv("FLOW_STEPS", "10"))  # flow decoder steps: on a CPU the cost is linear in it (5 took 0.6 of the time of 10)
    length_scale: float = float(os.getenv("LENGTH_SCALE", "1.1"))  # 1.0 renders the English-capable model about 10% faster than the teacher (median length ratio 0.90), 1.1 matches it (0.99)
    dp_temperature: float = float(os.getenv("DP_TEMPERATURE", "0.3"))
    dp_pause_temperature: float = float(os.getenv("DP_PAUSE_TEMPERATURE", "1.0"))
    prosody_temperature: float = float(os.getenv("PROSODY_TEMPERATURE", "0.5"))
    warmup: bool = _env_bool("WARMUP", True)

    # Service
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))

    def __post_init__(self):
        if "*" in self.cors_allowed_origins and self.cors_allow_credentials:
            raise ValueError("CORS_ALLOW_CREDENTIALS=true cannot be used with CORS_ALLOWED_ORIGINS=*")

    def validate_runtime_security(self):
        """Called at server start (not on import, so tests and scripts need no API_KEY)."""
        if not self.require_api_key:
            return
        if not self.api_key:
            raise RuntimeError("API_KEY is required when REQUIRE_API_KEY=true")
        weak = self.api_key.lower()
        if weak in {"changeme", "change-me", "dev", "test"} or weak.startswith("change-me") or weak.endswith("-dev-key"):
            raise RuntimeError("API_KEY must be changed from the development placeholder")


config = APIConfig()
