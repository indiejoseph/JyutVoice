# JyutVoice TTS API (CPU)

HTTP API for the ONNX inference path of JyutVoice (Cantonese, with English words inside it): the text encoder, duration sampler and ProsodyCFM run in ONNX Runtime, the
CosyVoice2 flow decoder and the HiFT vocoder in PyTorch, all on the CPU. The voice comes from a **reference clip sent as base64** in the request.

The request shape and the auth follow the SoulX-Podcast API (`../SoulX-Podcast/api`), which in turn follows OpenAI's `/v1/audio/speech`.

## Run it

```bash
python scripts/export_onnx.py                       # writes onnx_models/ (see "ONNX inference" in the README)
cp docker/.env.example docker/.env                  # set API_KEY (openssl rand -hex 32)
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml logs -f # "API started successfully!" once the models are loaded and warmed up (about a minute)
```

The weights are mounted, not baked into the image: `onnx_models/` and `pretrained_models/` (`flow_decoder.pt`, `hift.pt`, `campplus.onnx`). Without Docker:
`pip install -r api/requirements.txt` (plus torch, torchaudio and the WeTextProcessing fork, see `docker/Dockerfile`), then `API_KEY=... uvicorn api.main:app` from the
repository root.

## Endpoints

| | | |
|---|---|---|
| `GET /` | no key | name, version, status |
| `GET /health` | no key | `200` with the model facts, `503` while the models are not loaded |
| `POST /v1/audio/speech` | Bearer key | JSON in, audio out |

`Authorization: Bearer <API_KEY>`. The server refuses to start without a real `API_KEY` (not empty, not `changeme`/`dev`/`test`/`change-me*`/`*-dev-key`) unless
`REQUIRE_API_KEY=false`. Interactive docs: `/docs`.

### `POST /v1/audio/speech`

```bash
REF=$(base64 -w0 prompt.wav)
curl -sS http://127.0.0.1:8000/v1/audio/speech \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d "{\"input\": \"我喺Zoom開會，但係個microphone有問題。\", \"prompt_audio\": \"$REF\", \"seed\": 7}" \
  -o speech.wav -D headers.txt
grep -i prompt-cache-id headers.txt
```

(A long clip makes a long command line: use `-d @request.json` or `python api/test_client.py`.)

```python
import base64, requests

clip = base64.b64encode(open("prompt.wav", "rb").read()).decode()
r = requests.post(
    "http://127.0.0.1:8000/v1/audio/speech",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"input": "今日係2026年10月21日，我哋喺 interview room 開會。", "prompt_audio": clip},
)
r.raise_for_status()
open("speech.wav", "wb").write(r.content)
voice = r.headers["Prompt-Cache-Id"]   # send it instead of the clip next time
```

| field | meaning |
|---|---|
| `input` (required) | Text: Cantonese with English words/phrases inside it. Numbers, dates, money, symbols are read out. Up to `MAX_INPUT_CHARS` (5000). |
| `prompt_audio` | The reference clip: raw base64 of an audio file (wav, flac, ogg, mp3; other formats such as m4a through ffmpeg, which decodes the clip from a temporary file), or a `data:audio/...;base64,` URI. Lossless formats give exactly the voice of the wav; lossy ones work but move the x-vector a little. At least 0.5 s, the first 30 s are used (`PROMPT_MIN_SECONDS`, `PROMPT_MAX_SECONDS`), at most `MAX_PROMPT_BYTES` (20 MB) decoded. A `file://` URI is accepted only for files under `PROMPT_AUDIO_ROOT` (unset: not accepted). |
| `prompt_cache_id` | The `Prompt-Cache-Id` of an earlier response, instead of `prompt_audio` (not both). The server keeps the last `PROMPT_CACHE_SIZE` (64) voices; an unknown id gets **404** and the clip has to be sent again. The id is derived from the clip's bytes, so the same clip always gives the same id. |
| `format` / `response_format` | `wav` (24 kHz mono PCM16, default) or `pcm` (raw PCM16LE). `response_format` wins if both are given. |
| `stream` | `true`: chunked response, audio is sent as each chunk is synthesised. WAV streaming uses a placeholder-length header (as in the SoulX API). **Default `false`** (SoulX defaults to true). |
| `seed` | Sampling seed of the durations, ProsodyCFM and the vocoder noise (the flow decoder's own noise is a fixed buffer, as in CosyVoice2); random if omitted. Chunk *i* of the text uses `seed + i`; the same seed, text and clip give the same audio again, and a one-chunk request with `normalize_loudness=false` equals `python infer_onnx.py --seed N` for the same text and clip (checked: bit-identical PCM16 in the image; the default levelling only changes the gain). |
| `speed` | 0.25–4, 1.0 = the model's own pace (`length_scale = LENGTH_SCALE / speed`). |
| `flow_steps` | Flow decoder steps, default `FLOW_STEPS` (10). On a CPU the cost is linear in it (5 took 0.6 of the time of 10); only 10 steps were scored for quality. |
| `prosody`, `normalize_text`, `dp_temperature`, `prosody_temperature` | Switch ProsodyCFM / text normalisation; noise scales of the duration sampler (0.3) and ProsodyCFM (0.5). |
| `normalize_loudness` | Level the speech to `OUTPUT_LEVEL_DBFS` (-20 dBFS RMS of the speech frames) with **one gain per request**, taken from its first chunk, so streamed chunks keep their level relative to each other; the gain is lowered if a chunk would pass 0.99. Default `NORMALIZE_LOUDNESS` (true). `false`: the vocoder's own level, limited at 0.99 as HiFT does, which swings about 12 dB with the voice and clips for some. |
| `model`, `voice`, `language`, `prompt_text`, and the SoulX sampling fields (`temperature`, `top_k`, ..., `pronunciation_fix`) | Accepted and ignored, so that SoulX/OpenAI clients work unchanged. |

Response headers: `Prompt-Cache-Id`, `X-TTS-Model`, `Content-Disposition`, and for non-streamed responses `X-Audio-Duration` and `X-Synthesis-Time` (seconds).

Errors: `400` bad input (invalid base64, clip too short, text with nothing to read, ...), `401` wrong key, `404` unknown `prompt_cache_id`, `413` body over
`MAX_REQUEST_BYTES` (40 MB), `422` field validation, `500` anything else.

**Long text** is cut at sentence ends (`。！？!?；;…`, line breaks, a full stop before a space) and the sentences are packed into chunks of at most `MAX_CHUNK_CHARS` (100)
characters; each chunk is one synthesis (and one streamed piece). A chunk that exceeds the text encoder's token limit is cut again at word breaks.

## What differs from the SoulX-Podcast API

Mirrored: `GET /`, `GET /health`, `POST /v1/audio/speech` (fields, Bearer auth, `Prompt-Cache-Id` cache, CORS opt-in, wav/pcm, streaming, one synthesis at a time).
Not mirrored: `/v1/audio/conversions` (voice conversion), `/generate`, `/generate-stream`, `/generate-async`, `/task/{id}`, `/download/{file}` (multi-speaker dialogue,
multipart uploads and the Redis task queue). JyutVoice takes only the speaker's x-vector from the clip, so `prompt_text` is not needed (and ignored); there is no
`pronunciation_fix`/SSML.

## Speed

On a CPU the flow decoder and the vocoder take nearly all the time (the ONNX acoustic part is 0.03-0.06 s). Measured in the container on this 12-core desktop, shared with
other work, with `TORCH_NUM_THREADS=6`: real-time factor 0.75-1.15 at 10 flow steps (it varies with the sentence and the load) and 0.45 at 5 steps. Two requests at once are
served one after the other (the second waits for the first), while `/health` stays responsive. Use `stream=true` so that the first chunk plays while the rest is synthesised
(the first audio arrives after the first chunk: about 5 s for a 100-character chunk, so a smaller `MAX_CHUNK_CHARS` gives an earlier start), and set `TORCH_NUM_THREADS` to
the CPUs the container may use. The streamed audio is identical to the non-streamed one for the same seed.

## Settings (environment)

`API_KEY`, `REQUIRE_API_KEY`, `CORS_ALLOWED_ORIGINS`, `CORS_ALLOW_CREDENTIALS`, `API_HOST`, `API_PORT`, `ONNX_DIR`, `PRETRAINED_DIR`, `DECODER_CONFIG`, `TORCH_NUM_THREADS`,
`ONNX_NUM_THREADS`, `FLOW_STEPS`, `LENGTH_SCALE`, `DP_TEMPERATURE`, `DP_PAUSE_TEMPERATURE`, `PROSODY_TEMPERATURE`, `TEXT_NORMALIZE`, `NORMALIZE_LOUDNESS`, `OUTPUT_LEVEL_DBFS`, `MAX_INPUT_CHARS`, `MAX_CHUNK_CHARS`,
`MAX_PROMPT_BYTES`, `MAX_REQUEST_BYTES`, `PROMPT_MIN_SECONDS`, `PROMPT_MAX_SECONDS`, `PROMPT_AUDIO_ROOT`, `PROMPT_CACHE_SIZE`, `WARMUP`. Defaults are in `api/config.py`;
`docker/.env.example` lists the ones docker compose passes on. `LENGTH_SCALE` defaults to **1.1** (the English-capable model renders about 10% faster than the teacher at 1.0: median length ratio 0.90, 4.6 against 4.2 syllables/s; at 1.1 it is 0.99); `speed` in a request is relative to it.

## Tests

`python -m pytest tests/test_api.py` (the endpoints run against a stub service, no weights needed).
