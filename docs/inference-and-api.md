# Inference, ONNX export and the HTTP API

Three ways to synthesise, from most to least reference-like:

| path | file | what runs where | use for |
|---|---|---|---|
| PyTorch | `JyutVoiceTTS.synthesise`, `scripts/synth_compare.py` | everything in PyTorch; can pass a decoder prompt clip (`--prompt`) | experiments, listening pages, evaluation |
| ONNX CLI | `infer_onnx.py` | encoder, duration sampler and ProsodyCFM in ONNX Runtime; flow decoder + HiFT in PyTorch; x-vector only, no clip prompt | one-off synthesis, deployment code reference |
| HTTP API | `api/` (+ `docker/`) | same as the ONNX CLI, wrapped in FastAPI, CPU only | serving |

`infer.py` is the older PyTorch script and does not use ProsodyCFM.

## ONNX export

```bash
python scripts/export_onnx.py                       # defaults from configs/export_onnx.yaml -> onnx_models/ (git-ignored, about 51 MB)
python scripts/export_onnx.py --checkpoint checkpoints/best/jyutvoice_flowdp_epoch019.ckpt \
    --prosody-ckpt checkpoints/best/prosody_cfm_mix1.pt --output-dir onnx_models_best     # the Cantonese-only best
```

Writes `text_encoder.onnx`, `flow_duration_sampler.onnx` (or `duration_predictor.onnx` for a regression model), `prosody_sampler.onnx` and `manifest.json` (graph file names,
`add_blank`, `max_tokens`, duration rounding, source checkpoint). It **checks every graph and the whole chain against PyTorch** (`JyutVoiceTTS.synthesise`, same noise) at three
lengths and fails above `atol` (1e-3; observed about 2e-6). `--tensorrt` is untested.

How the pieces split (`models/onnx_wrappers.py`, `utils/onnx_synth.py`):
- Four graphs with dynamic batch/length: text encoder, regression duration predictor, flow duration sampler (10 Euler steps **unrolled** in the graph), ProsodyCFM sampler (unrolled).
- **Random noise is a graph input**, drawn by the caller already scaled by the temperature (0.3 phones / 1.0 pauses for durations, 0.5 for ProsodyCFM). That is what makes a render
  seedable and the graphs comparable with PyTorch on identical noise.
- The expansion of tokens to frames (the alignment path) is numpy in `onnx_synth.py` because its output length depends on the predicted durations: `round_durations`,
  `frame_counts`, `expand_frames`, `phone_position`, `pause_token_mask`, `token_temperatures`. `OnnxAcousticModel` is driven by `manifest.json` and refuses more than `max_tokens`.
- `OnnxAcousticModel.__call__(..., seed=)` draws the duration noise first, then ProsodyCFM's, from `numpy.random.default_rng(seed)`.

Gotchas (each cost a debugging session; tests cover them):
1. The encoder's RoPE tables are built lazily to the longest sequence seen and `torch.onnx.export` bakes them at the trace length: a graph traced on 93 tokens crashed on 97.
   `warm_rotary_caches` builds them to `max_tokens` (1024, about 30 s of speech); longer text must be split at punctuation.
2. `torch.cumsum` on CPU accumulates float32 in double, so numpy must too (`_cumsum`), or `ceil` frame boundaries shift by one frame. Torch's float32 `sum` can still differ by one
   frame in the total for non-binary length scales; the tests use 0.5 / 1.25 / 1.5 where sums are exact.
3. `import torch` before `onnxruntime` (libstdc++ `CXXABI` error otherwise).
4. The reference for the check must run the flow sampler with the exported step count (`dp.infer_steps`).

## `infer_onnx.py`

```bash
python infer_onnx.py --onnx_dir onnx_models --text "我喺Zoom開會，但係個microphone有問題。" --ref_audio ref.wav --output out.wav --seed 7
```

Front end: normalise -> pydips -> g2p (`--segmentation given` takes already-segmented text and needs no pydips, `--no_text_normalize` skips WeTextProcessing). Useful options:
`--length_scale` (default 1.0 here; the API defaults to 1.1, which matches the teacher's pace for the English-capable model), `--n_timesteps` (flow decoder steps, default 10), `--no_prosody`, `--dp_temperature`,
`--dp_pause_temperature`, `--prosody_temperature`, `--threads`, `--use_gpu`. The shared steps (`prepare_text`, `load_flow_decoder`, `to_16k_mono`, `extract_spk_embedding`,
`synthesise_wave`) are in `jyutvoice/utils/onnx_infer.py`; the API uses the same functions, which is why the two agree bit for bit.

The decoder gets only the speaker x-vector, no clip prompt (the demos had one, and the pitch is smoother with it; see decisions-and-known-issues.md). `seed` seeds durations, ProsodyCFM and the HiFT source noise; the flow decoder's own noise is a fixed buffer
(`CausalConditionalCFM.rand_noise`, filled after `set_all_random_seed(0)`), so it does not vary with the seed. Same seed + same thread count = same waveform in any process.

Speed (CPU): the ONNX part is 0.03-0.06 s per sentence; the flow decoder and HiFT in PyTorch take nearly all the time. Real-time factor was 0.75-1.15 at 10 flow steps and about 0.45 at
5 steps on a shared 12-core desktop with 6 PyTorch threads. Only 10 steps were scored for quality.

## HTTP API (`api/`)

FastAPI app `api.main:app` over the ONNX path. Request shape and auth follow the SoulX-Podcast API; the full field list, errors and examples are in `api/README.md`.

| | auth | |
|---|---|---|
| `GET /` | no | name, version, status |
| `GET /health` | no | 200 with `duration_model`, `prosody`, `cached_prompts`; 503 until models are loaded |
| `POST /v1/audio/speech` | `Authorization: Bearer <API_KEY>` | JSON in, `audio/wav` or `audio/pcm` out |

Module map: `config.py` (env-driven `APIConfig`; `validate_runtime_security` refuses to start without a real key), `models.py` (pydantic `SpeechRequest`, extra fields ignored so SoulX/OpenAI
clients work), `service.py` (`TTSService`: loads models, resolves the reference clip, caches x-vectors, plans chunks, synthesises under one lock), `text_split.py` (sentence/chunk
splitting), `audio.py` (wav/pcm helpers, clip decoding with an ffmpeg fallback), `main.py` (routes, auth, CORS, size limit), `test_client.py` (stdlib client).

Behaviour to remember:
- **Reference clip as base64** in `prompt_audio` (raw or `data:audio/...;base64,`); `file://` only under `PROMPT_AUDIO_ROOT`; 0.5 s minimum, first 30 s used, 20 MB cap. Only the x-vector is
  taken, so `prompt_text` is accepted and ignored. The response header `Prompt-Cache-Id` (`pc_` + sha256[:32] of the clip bytes) can replace the clip in later requests
  (`prompt_cache_id`; LRU of `PROMPT_CACHE_SIZE`, unknown id -> 404, both -> 400).
- Text is cut at sentence ends into chunks of at most `MAX_CHUNK_CHARS` (100); each chunk is one synthesis, chunk *i* uses `seed + i`; a chunk over the encoder's token limit is split again
  at word breaks (never inside an English phrase). Chunks made only of punctuation are skipped; input with nothing to read is a 400.
- `stream` defaults to **false** (unlike SoulX). Streaming sends a placeholder-length WAV header (0xFFFFFFFF) then PCM per chunk; streamed audio equals non-streamed audio for the same seed.
- One synthesis at a time (a lock); `/health` stays responsive meanwhile. Blocking work runs in the thread pool, never on the event loop.
- Errors: 400 bad input, 401 key, 404 unknown cache id, 413 body over `MAX_REQUEST_BYTES` (40 MB), 422 validation, 500 other.
- Output PCM16 is `rint(clip(x) * 32767)`. With `normalize_loudness=false` API output is bit-identical to `infer_onnx.py` for the same seed, text, clip, thread counts (checked in the image).
- **Loudness.** The API switches HiFT's `audio_limit` (0.99) off and levels each request itself (`api/audio.py`: `speech_rms`, `level_gain`, `apply_level`; `service._level`): the RMS of the speech frames (20 ms
  frames within 40 dB of the loudest) is brought to `OUTPUT_LEVEL_DBFS` (-20), with **one gain per request** taken from the first chunk (a gain per chunk would step in level between streamed chunks), lowered when a chunk
  would pass 0.99. Reason: with no decoder prompt the vocoder's level swings 12 dB with the voice and clips for some voices (see decisions-and-known-issues.md). Apart from the peaks the old limiter flattened, the levelled audio is the same signal as before,
  only louder or quieter (checked on the live service: a two-chunk streamed request of 27 s equals the unlevelled one times a constant, spread 0.1%; all eight preset voices come out at -20.0 dBFS with no sample at the
  ceiling), so comparisons of durations, pitch or ASR are unaffected; tone (the high band is 4-13 dB hotter than with a prompt) is not changed by it.
  `infer_onnx.py` is unchanged and still limits at 0.99.
- Not mirrored from SoulX: voice conversion, multipart upload endpoints, the Redis task queue (`/generate*`, `/task`, `/download`).

### Docker (CPU)

```bash
cp docker/.env.example docker/.env          # set API_KEY (openssl rand -hex 32); the file is git-ignored
docker compose -f docker/docker-compose.yml up -d --build
curl -s http://127.0.0.1:8000/health
python api/test_client.py --ref ref.wav --text "今日係2026年10月21日，我哋喺 interview room 開會。" --output out.wav   # reads API_KEY from the environment
```

- `docker/Dockerfile`: python:3.11-slim, torch/torchaudio 2.6.0 from the **CPU** wheel index (the default wheel brings about 3 GB of CUDA libraries), `api/requirements.txt`, the WeTextProcessing
  fork at a pinned commit, NLTK data and the compiled normaliser graphs **baked in** (so a start needs no network, about 14 s with warm-up), runs as user `app`, one uvicorn worker.
  Image is about 3.1 GB. The build context is the repository root; `.dockerignore` is an allow-list (`api`, `jyutvoice`, `bert`, `configs/onnx_infer.yaml`, `infer_onnx.py`).
- Weights are **mounted read-only**, not in the image: `onnx_models/` and `pretrained_models/` (needs `flow_decoder.pt`, `hift.pt`, `campplus.onnx`). Override the host paths with
  `ONNX_MODELS_DIR` and `PRETRAINED_MODELS_DIR`.
- Compose publishes `127.0.0.1:${API_PORT:-8000}`; set `API_BIND_HOST=0.0.0.0` only behind a TLS proxy. `API_KEY` is required (`${API_KEY:?}`); the server also refuses placeholder keys.
  Set `TORCH_NUM_THREADS` to the CPUs the container may use (it cannot see a `cpus:` quota).
- Gotchas found building it: the NLTK downloader writes zips as 0600 root, so a non-root user re-downloads them at every start (fixed with `chmod -R a+rX`); a shell-level
  `API_KEY=x docker compose build` does not affect `up`, which reads `docker/.env`; transformers in the image (5.20) differs from the development machine's (5.9) but English g2p token arrays were identical on
  the test sentences.
- Verified in the image: CLI parity, repeat/seed behaviour, streaming == non-streaming, wav/flac/mp3/ogg/m4a/data-URI clips, every error status, start with `--network none`, refusal to start without a key,
  two simultaneous requests served in turn. Tests: `python -m pytest tests/test_api.py` (a stub service, no weights).

## Playground (`playground/`)

A browser page for the API, modelled on the SoulX-Podcast playground (`tts-playground` on the GPU host). `playground/server.py` is a standard-library proxy (`make_server()` is what the tests
start) and `index.html` the page (no build step); `voices/` holds four CC0 Common Voice Cantonese clips as presets (they give slow, pause-heavy speech with this model, see decisions-and-known-issues.md) and `voices.local/` (git-ignored) clips listed first. Run: `JYUTVOICE_API_KEY=... python playground/server.py` (port 8898), or
`docker compose -f playground/docker-compose.yml --env-file docker/.env up -d --build` from the repository root (it joins the API's network `jyutvoice_default`). Details: `playground/README.md`.

- The proxy holds the API key, so the browser never sees it; it has **no login of its own** and binds to `127.0.0.1` by default. Settings the page may set are whitelisted
  (`seed, speed, flow_steps, dp_temperature, prosody_temperature, prosody, normalize_text`); everything else in a request is dropped.
- The page has no multi-speaker controls (removed at the maintainer's request); the proxy still supports `[S1]..[S4]` dialogues (`extra_voices` in the request body gives a speaker its own voice): they are cut into turns, one API call each (the API reads one voice per call), joined with 0.25 s of silence; turn *i* uses `seed + 1000 i`. A seed the page does not send
  is drawn by the proxy and returned as `X-Seed` so a take can be reproduced. Streaming relays the API's `format=pcm` stream.
- API errors are mapped to one message; a 401 from the API becomes a 502 ("the playground's API key was rejected") because the visitor cannot fix it.
- Facts that shaped the page: the API streams per *chunk* (up to 100 characters), so a short text arrives all at once after the whole synthesis; chunks start and end in silence (join step 0.003), so the
  reference playground's click-removal at stream joins was left out (it fired on 20% of ordinary samples and would have altered continuous speech). Same text, voice and seed through the playground
  and straight to the API give identical PCM (checked, also streamed).
- The page was driven in headless Chromium against the real API (synthesis, seed handling, streaming, dialogue, upload, error display, the cURL dialog). Fonts: the development machine has no CJK font,
  so screenshots made there show Cantonese as boxes; that is the machine, not the page.
