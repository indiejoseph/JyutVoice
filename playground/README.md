# JyutVoice playground

A web page for trying the [JyutVoice API](../api/README.md): type Cantonese (with English words in it), pick or upload a reference voice, listen, download, copy the `curl`
call. It looks and works like the SoulX-Podcast playground, adapted to this model: the reference clip needs **no transcript** (only its voice print is used), and the settings
are this model's (seed, speed, flow steps, duration and prosody noise, ProsodyCFM, text normalisation).

```
browser  ->  playground/server.py  ->  JyutVoice API  (POST /v1/audio/speech, Bearer key)
             (holds the key, no login)
```

`server.py` is a standard-library proxy: the browser never sees the API key. It also turns the preset voices into reference clips, cuts `[S1]… [S2]…` dialogues into turns
(one API call per turn, joined with 0.25 s of silence; `extra_voices` in the request gives a speaker its own voice, the page does not use it), draws a seed when the page sends none and returns it in `X-Seed` ("Keep seed" in the page reuses it), and relays streams.

## Run

With the API already running on `127.0.0.1:8000` (see `docker/`):

```bash
JYUTVOICE_API_KEY=... python playground/server.py        # http://127.0.0.1:8898
```

or in Docker, reusing the key in `docker/.env` (run from the repository root):

```bash
docker compose -f docker/docker-compose.yml up -d --build                               # the API, once
docker compose -f playground/docker-compose.yml --env-file docker/.env up -d --build    # the playground
```

The compose file joins the network `jyutvoice_default` that the API's project creates and talks to `http://jyutvoice-api:8000`.

| variable | default | |
|---|---|---|
| `JYUTVOICE_URL` | `http://127.0.0.1:8000` | where the proxy reaches the API |
| `JYUTVOICE_API_KEY` | empty | Bearer key; empty sends no `Authorization` header (an API started with `REQUIRE_API_KEY=false`) |
| `JYUTVOICE_PUBLIC_URL` | `JYUTVOICE_URL` | the address the page's cURL example shows |
| `HOST`, `PORT` | `127.0.0.1`, `8898` | where the playground listens (compose publishes on `127.0.0.1:${PLAYGROUND_PORT:-8898}`; `PLAYGROUND_BIND_HOST` changes the address) |
| `MAX_BODY_BYTES` | 60 MB | request limit of the proxy (a dialogue can carry four uploaded clips) |

**Security:** whoever can reach the playground can synthesise with the key it holds, and it has no login. Keep it on localhost, or behind a login or a private network.

## What the page does

- **Voices:** preset voices, or upload any clip. `voices.local/` (git-ignored; same layout as `voices/`, an optional `note` per entry) is listed first and a voice there replaces a shipped one with the same id: put clips of
  corpus speakers there. The four shipped clips (`voices/`, Mozilla Common Voice Cantonese, CC0; source of each in `voices/voices.json`) sound slow and pausey with this model because their x-vectors are unlike the training voices;
  the pacing follows the clip's x-vector, so the same applies to many uploads. Upload any clip
  (wav, flac, ogg, mp3, m4a; at least 0.5 s, the first 30 s are used).
- **Dialogue markers:** the page has no multi-speaker controls. Text with `[S1]… [S2]…` markers is still cut into turns by the proxy and made in the one chosen voice (the markers are not read
  aloud); `extra_voices` in a `POST /api/synthesize` body gives a speaker its own voice (`{"2": {"voice_id": "yue-high-1"}}` or `{"2": {"prompt_audio": "data:..."}}`).
- **Advanced:** an empty box means the server's default; only filled boxes are sent. Streaming plays the pieces as the API finishes them (the API makes one chunk of up to 100
  characters at a time, so a short text arrives all at once).
- **After a take:** time, audio length, real-time factor, seed (with "Keep seed"), download as WAV.

## Files

`server.py` (proxy; `make_server()` is what the tests start), `index.html` (the page, no build step), `voices/` (preset clips and `voices.json`), `Dockerfile`, `docker-compose.yml`.
Tests: `python -m pytest tests/test_playground.py` (a stub API, no models); the page itself was driven in a headless browser against the real API (synthesis, streaming, dialogue,
upload, errors, the cURL dialog).
