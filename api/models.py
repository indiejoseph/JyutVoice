"""Pydantic models of the API. `/v1/audio/speech` takes the request shape of the SoulX-Podcast API (itself the OpenAI `/v1/audio/speech` convention): JSON in, audio
bytes out, the reference clip as base64. Fields of that API that make no sense for this model (temperature, top_k, top_p, repetition_penalty, pronunciation_fix, ...)
are accepted and ignored."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    """`/v1/audio/speech` request body."""

    model: str = Field(default="jyutvoice", description="Model name. Accepted for API compatibility.")
    input: str = Field(..., min_length=1, description=(
        "Text to synthesize: Cantonese, with English words or phrases inside it (code-mixing). Numbers, dates and money are read out (text normalisation). Long text is cut at "
        "sentence ends and synthesised in chunks of at most MAX_CHUNK_CHARS characters, which `stream=true` returns as they are ready."
    ))
    prompt_audio: Optional[str] = Field(default=None, description=(
        "Reference clip of the voice to speak with: raw base64 audio bytes or a data:audio/*;base64 URI (wav, flac, ogg, mp3, ...; at least 0.5 s, only the first "
        "30 s are used). A file:// URI of a server-local file is accepted only under PROMPT_AUDIO_ROOT. Only the speaker's x-vector is taken from it: no transcript is needed."
    ))
    prompt_text: Optional[str] = Field(default=None, description="Transcript of prompt_audio. Accepted for compatibility with the SoulX-Podcast API and ignored.")
    prompt_cache_id: Optional[str] = Field(default=None, description=(
        "Opaque id returned in the Prompt-Cache-Id response header for an earlier prompt_audio. When set, omit prompt_audio: the cached voice is used. "
        "The cache holds PROMPT_CACHE_SIZE voices; an unknown id is answered with 404 and the clip has to be sent again."
    ))
    voice: Optional[str] = Field(default=None, description="OpenAI compatibility field, ignored: the voice comes from prompt_audio.")
    language: Optional[str] = Field(default=None, description=(
        "Dialect hint, accepted for compatibility and ignored: the front end is always Cantonese, and English words inside the text are read in English."
    ))
    speed: Optional[float] = Field(default=None, ge=0.25, le=4.0, description="Speaking speed, 1.0 = as the model speaks (LENGTH_SCALE). 0.5 is half as fast.")
    format: Literal["wav", "pcm"] = Field(default="wav", description="Output audio: wav (24 kHz mono PCM16) or pcm (raw PCM16LE, 24 kHz mono)")
    response_format: Optional[Literal["wav", "pcm"]] = Field(default=None, description="OpenAI-compatible alias. Overrides `format` when set.")
    stream: bool = Field(default=False, description=(
        "Return the audio chunk by chunk as it is synthesised (chunked transfer). WAV streaming uses a placeholder-length header; use format=pcm or stream=false for "
        "stricter clients. Unlike the SoulX-Podcast API the default is false."
    ))
    seed: Optional[int] = Field(default=None, ge=0, le=2**62, description="Sampling seed (durations, ProsodyCFM, vocoder noise; the flow decoder's noise is fixed). Omitted: random per request. Chunk i uses seed + i.")
    flow_steps: Optional[int] = Field(default=None, ge=1, le=50, description="Flow decoder steps (default FLOW_STEPS, 10). The cost on a CPU is linear in it; only 10 was scored for quality.")
    prosody: Optional[bool] = Field(default=None, description="Sample the decoder's input features with ProsodyCFM (default: yes when the exported model has it).")
    normalize_text: Optional[bool] = Field(default=None, description="Read out numbers, dates, money and symbols before synthesis (default TEXT_NORMALIZE, true).")
    normalize_loudness: Optional[bool] = Field(default=None, description=(
        "Level the speech to OUTPUT_LEVEL_DBFS (default -20 dBFS RMS) with one gain per request (default NORMALIZE_LOUDNESS, true). false: the vocoder's own level, "
        "which varies by about 12 dB with the voice and clips for some."
    ))
    dp_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Noise scale of the duration sampler for speech tokens (default 0.3).")
    prosody_temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Noise scale of ProsodyCFM (default 0.5).")

    @property
    def output_format(self) -> str:
        return self.response_format or self.format


class HealthResponse(BaseModel):
    status: str = Field(default="healthy")
    model_loaded: bool
    device: str = Field(default="cpu")
    duration_model: Optional[str] = Field(default=None, description="flow (sampler) or regression")
    prosody: Optional[bool] = Field(default=None, description="Whether the exported model has ProsodyCFM")
    cached_prompts: int = Field(default=0, description="Voices in the prompt cache")
    version: str = Field(default="1.0.0")


class ErrorResponse(BaseModel):
    error: str
    message: str
