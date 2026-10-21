"""FastAPI application of the JyutVoice TTS API.

Endpoints (the first three follow the SoulX-Podcast API, `/v1/audio/speech` is the OpenAI convention):
    GET  /                   name, version, status
    GET  /health             503 until the models are loaded
    POST /v1/audio/speech    JSON in (text, reference clip as base64), audio out (wav or pcm), optionally streamed
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPBearer
from starlette.concurrency import run_in_threadpool

from api.audio import SAMPLE_RATE
from api.config import config
from api.models import ErrorResponse, HealthResponse, SpeechRequest
from api.service import PromptCacheMiss, get_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting JyutVoice TTS API...")
    config.validate_runtime_security()
    logger.info("Loading models (CPU)...")
    service = await run_in_threadpool(get_service)
    if not service.is_loaded():
        raise RuntimeError("Failed to load model")
    logger.info("API started successfully!")
    yield
    logger.info("API shutdown completed")


app = FastAPI(
    title="JyutVoice TTS API",
    description="Cantonese (and Cantonese-English code-mixed) text-to-speech with the voice of a reference clip, on the CPU.",
    version=VERSION,
    lifespan=lifespan,
)

# CORS is off unless configured, as in the SoulX-Podcast API.
if config.cors_allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_allowed_origins),
        allow_credentials=config.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Prompt-Cache-Id", "X-Audio-Duration", "X-TTS-Model"],
    )


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """A base64 reference clip makes the body large: refuse what is over MAX_REQUEST_BYTES before it is read."""
    length = request.headers.get("content-length")
    if length is not None and length.isdigit() and int(length) > config.max_request_bytes:
        resp = ErrorResponse(error="RequestEntityTooLarge", message=f"request body is larger than {config.max_request_bytes} bytes")
        return JSONResponse(status_code=413, content=resp.model_dump(mode="json"))
    return await call_next(request)


# Declares the bearer scheme so /docs has an Authorize button; the check below reads the raw header (auto_error=False: documentation only).
_bearer_scheme = HTTPBearer(
    auto_error=False,
    description="API key as `Authorization: Bearer <key>`. Enforced when REQUIRE_API_KEY is true (the default); the key is API_KEY.",
)


def require_api_key(
    authorization: str | None = Header(default=None, include_in_schema=False),
    _bearer=Depends(_bearer_scheme),
) -> None:
    """Bearer-token guard of the generation endpoint."""
    if not config.require_api_key:
        return
    if not config.api_key:
        raise HTTPException(status_code=500, detail="API_KEY is required but not configured")
    if authorization != f"Bearer {config.api_key}":
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/", tags=["Health"])
async def root():
    return {"name": "JyutVoice TTS API", "version": VERSION, "status": "running", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    service = get_service()
    loaded = service.is_loaded()
    response = HealthResponse(status="healthy" if loaded else "unhealthy", model_loaded=loaded, version=VERSION, **(service.info() if loaded else {}))
    if not loaded:
        raise HTTPException(status_code=503, detail=response.model_dump())
    return response


@app.post("/v1/audio/speech", tags=["OpenAI Compatible"])
async def openai_audio_speech(request: SpeechRequest, _: None = Depends(require_api_key)):
    """Text to speech in the voice of a reference clip.

    JSON in, audio bytes out. The reference clip goes in `prompt_audio` as base64 (raw, or a `data:audio/...;base64,` URI); the response carries a `Prompt-Cache-Id`
    header, and later requests may send that id as `prompt_cache_id` instead of the clip. `stream=true` returns the audio chunk by chunk while later chunks are still being
    synthesised. Authentication: `Authorization: Bearer <key>` when REQUIRE_API_KEY is true.
    """
    try:
        service = get_service()
        # Blocking work runs off the event loop (see the SoulX-Podcast API: called inline it hangs /health while a stream holds the lock).
        context = await run_in_threadpool(service.prepare_speech_context, request)
        plan = context["prepared"]
        media_type = "audio/wav" if request.output_format == "wav" else "audio/pcm"
        headers = {
            "Content-Disposition": f'attachment; filename="speech.{request.output_format}"',
            "X-TTS-Model": request.model,
            "Prompt-Cache-Id": context["prompt_cache_id"],
        }
        if request.stream:
            return StreamingResponse(service.stream_speech_bytes(request, plan), media_type=media_type, headers=headers)
        t0 = time.time()
        content = await run_in_threadpool(service.generate_speech_bytes, request, plan)
        pcm_bytes = len(content) - (44 if request.output_format == "wav" else 0)
        seconds = pcm_bytes / 2 / SAMPLE_RATE
        headers["X-Audio-Duration"] = f"{seconds:.3f}"
        headers["X-Synthesis-Time"] = f"{time.time() - t0:.3f}"
        return Response(content=content, media_type=media_type, headers=headers)
    except HTTPException:
        raise
    except PromptCacheMiss as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("speech request failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    resp = ErrorResponse(error="InternalServerError", message=str(exc))
    return JSONResponse(status_code=500, content=resp.model_dump(mode="json"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=config.host, port=config.port, log_level="info")
