"""Playground for the JyutVoice API: a web page plus the small proxy behind it. Standard library only.

The proxy exists so the browser never sees the API key: the page posts to /api/synthesize, which adds the Bearer key and calls POST /v1/audio/speech of the API.
It also turns preset voices (voices/voices.json) into reference clips, cuts [S1]..[S4] dialogues into turns (the API reads one voice per call, so each turn is one call
and the turns are joined with a short pause), draws a seed when the page sends none (and reports it in X-Seed so a good take can be reproduced), and relays streams.

    JYUTVOICE_API_KEY=... python playground/server.py        # http://127.0.0.1:8898, API at JYUTVOICE_URL (default http://127.0.0.1:8000)

Whoever can reach this server can synthesise with the key it holds: it binds to 127.0.0.1 unless HOST says otherwise, and has no login of its own.
"""
import base64
import io
import json
import os
import re
import secrets
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(HERE, "voices")
LOCAL_VOICES_DIR = os.path.join(HERE, "voices.local")  # git-ignored: clips that must not enter the public repository; listed first, same id replaces a shipped voice
INDEX_PATH = os.path.join(HERE, "index.html")

# Settings of the model the page may set; everything else in a request is dropped (the key and the API's own routing fields stay the proxy's business).
SETTINGS = ("speed", "flow_steps", "dp_temperature", "prosody_temperature", "prosody", "normalize_text")
TURN_RE = re.compile(r"\[S([1-4])\]")
TURN_GAP_S = 0.25
# The API draws chunk i of a text with seed + i, so the turns of a dialogue start this far apart to never share a chunk seed.
TURN_SEED_STRIDE = 1000
UPSTREAM_TIMEOUT_S = 300
MAX_BODY_BYTES = 60 * 1024 * 1024  # a dialogue can carry four uploaded clips; the API caps one request at 40 MB itself


def load_voices(voices_dir=VOICES_DIR):
    """{id: {"id", "name", "note", "path"}} from voices.json; entries whose file is missing or outside the directory are skipped."""
    try:
        with open(os.path.join(voices_dir, "voices.json"), encoding="utf-8") as f:
            entries = json.load(f)
    except FileNotFoundError:
        return {}
    root = os.path.realpath(voices_dir)
    voices = {}
    for e in entries:
        path = os.path.realpath(os.path.join(root, e["file"]))
        if os.path.commonpath([root, path]) != root or not os.path.isfile(path):
            continue
        note = e["note"] if "note" in e else f"F0 {e['f0']} Hz · “{e['text']}”" if "f0" in e and "text" in e else ""
        voices[e["id"]] = {"id": e["id"], "name": e["name"], "note": note, "path": path}
    return voices


def clip_data_uri(voice):
    with open(voice["path"], "rb") as f:
        return "data:audio/wav;base64," + base64.b64encode(f.read()).decode()


def split_turns(text):
    """[(speaker, text)] for text containing [S#] markers, else None. Text before the first marker belongs to speaker 1; empty turns are dropped."""
    marks = list(TURN_RE.finditer(text))
    if not marks:
        return None
    turns = []
    head = text[:marks[0].start()].strip()
    if head:
        turns.append((1, head))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip()
        if body:
            turns.append((int(m.group(1)), body))
    return turns


def describe_error(status, raw):
    """(status, message) for an error answer of the API: FastAPI's {"detail": str | [{loc, msg}]}, the API's own {"error", "message"}, or anything else."""
    if status == 401:
        return 502, "the playground's API key was rejected by the TTS API (check JYUTVOICE_API_KEY)"
    text = raw.decode("utf-8", errors="replace")
    try:
        body = json.loads(text)
    except ValueError:
        return status, text.strip()[:300] or f"TTS API answered HTTP {status}"
    detail = body.get("detail", body.get("message", text)) if isinstance(body, dict) else text
    if isinstance(detail, list):
        detail = "; ".join(f"{(d.get('loc') or ['?'])[-1]}: {d.get('msg', d)}" if isinstance(d, dict) else str(d) for d in detail)
    elif isinstance(detail, dict):
        detail = detail.get("message") or json.dumps(detail, ensure_ascii=False)
    return status, str(detail)


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


class Playground:
    """What the handler needs: where the API is, the key, the preset voices."""

    def __init__(self, api_url, api_key, voices, public_api_url, max_body):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.voices = voices
        self.public_api_url = (public_api_url or api_url).rstrip("/")
        self.max_body = max_body

    def open_api(self, payload):
        """The open response of POST /v1/audio/speech; ApiError (with the API's status and message) when it fails."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.api_url}/v1/audio/speech", data=json.dumps(payload).encode(), headers=headers, method="POST")
        try:
            return urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT_S)
        except urllib.error.HTTPError as e:
            raise ApiError(*describe_error(e.code, e.read()))
        except Exception as e:
            raise ApiError(502, f"cannot reach the TTS API: {e}")

    def reference(self, spec):
        """data-URI clip of a voice spec {"voice_id"} or {"prompt_audio"}; ApiError 400 when it names no usable voice."""
        if spec.get("voice_id"):
            voice = self.voices.get(spec["voice_id"])
            if not voice:
                raise ApiError(400, f"unknown voice_id: {spec['voice_id']}")
            return clip_data_uri(voice)
        if spec.get("prompt_audio"):
            return spec["prompt_audio"]
        raise ApiError(400, "choose a preset voice (voice_id) or upload a reference clip (prompt_audio)")


def make_server(api_url, api_key="", host="127.0.0.1", port=8898, voices_dir=VOICES_DIR, public_api_url=None, max_body=MAX_BODY_BYTES, local_voices_dir=LOCAL_VOICES_DIR):
    voices = load_voices(local_voices_dir) if local_voices_dir else {}
    voices.update({k: v for k, v in load_voices(voices_dir).items() if k not in voices})
    app = Playground(api_url, api_key, voices, public_api_url, max_body)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        # ---- replies
        def send_bytes(self, status, body, ctype, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, obj, extra=None):
            self.send_bytes(status, json.dumps(obj, ensure_ascii=False).encode(), "application/json", extra)

        def send_error_json(self, status, message):
            self.send_json(status, {"error": message})

        # ---- routes
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(INDEX_PATH, "rb") as f:
                    self.send_bytes(200, f.read(), "text/html; charset=utf-8")
            elif self.path == "/api/voices":
                self.send_json(200, {"voices": [{k: v[k] for k in ("id", "name", "note")} for v in app.voices.values()]})
            elif self.path == "/api/config":
                self.send_json(200, {"api_url": app.public_api_url})
            elif self.path == "/api/health":
                try:
                    with urllib.request.urlopen(f"{app.api_url}/health", timeout=5) as r:
                        self.send_bytes(200, r.read(), "application/json")
                except urllib.error.HTTPError as e:
                    self.send_error_json(502, f"TTS API answered HTTP {e.code}")
                except Exception as e:
                    self.send_error_json(502, f"cannot reach the TTS API: {e}")
            else:
                self.send_error_json(404, "not found")

        def do_POST(self):
            if self.path != "/api/synthesize":
                self.send_error_json(404, "not found")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > app.max_body:
                self.close_connection = True  # the body is not read, so the connection cannot be reused
                self.send_error_json(413, f"request body over {app.max_body // (1024 * 1024)} MB")
                return
            try:
                request = json.loads(self.rfile.read(length).decode())
                if not isinstance(request, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as e:
                self.send_error_json(400, f"invalid JSON body: {e}")
                return
            self.streaming = False
            try:
                self.synthesize(request)
            except ApiError as e:
                self.send_error_json(e.status, e.message)
            except Exception as e:
                if self.streaming:  # the audio has begun, so a JSON error would corrupt it: drop the connection
                    self.close_connection = True
                else:
                    self.send_error_json(500, f"internal error: {type(e).__name__}: {e}")

        # ---- synthesis
        def payload_of(self, request, text, reference, seed, stream):
            payload = {"model": "jyutvoice", "input": text, "prompt_audio": reference, "format": "pcm" if stream else "wav", "stream": stream, "seed": seed}
            payload.update({k: request[k] for k in SETTINGS if request.get(k) is not None})
            return payload

        def synthesize(self, request):
            text = request.get("text")
            text = text.strip() if isinstance(text, str) else ""
            if not text:
                raise ApiError(400, "text is required")
            reference = app.reference(request)
            seed = request.get("seed")
            if seed is None:
                seed = secrets.randbelow(2**31)
            elif isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ApiError(400, "seed must be a non-negative integer")
            turns = split_turns(text)
            if turns is not None:
                self.synthesize_dialogue(request, turns, reference, seed)
                return
            stream = bool(request.get("stream", False))
            with app.open_api(self.payload_of(request, text, reference, seed, stream)) as r:
                extra = {"X-Seed": str(seed)}
                for name in ("Prompt-Cache-Id", "X-Audio-Duration", "X-Synthesis-Time"):
                    if r.headers.get(name):
                        extra[name] = r.headers[name]
                if stream:
                    self.relay_stream(r, extra)
                else:
                    self.send_bytes(200, r.read(), "audio/wav", extra)

        def relay_stream(self, upstream, extra):
            """Chunked relay of a PCM stream."""
            self.streaming = True
            self.send_response(200)
            self.send_header("Content-Type", upstream.headers.get("Content-Type", "audio/pcm"))
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-cache")
            for k, v in extra.items():
                self.send_header(k, v)
            self.end_headers()
            try:
                while True:
                    chunk = upstream.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def synthesize_dialogue(self, request, turns, speaker1_reference, seed):
            """One call per turn (never streamed), the turns joined with a short pause into one WAV."""
            if not turns:
                raise ApiError(400, "no text after the [S#] markers")
            voices = {1: speaker1_reference}
            for key, spec in (request.get("extra_voices") or {}).items():
                if str(key) in ("2", "3", "4"):
                    voices[int(key)] = app.reference(spec)
            fallback = sorted({sp for sp, _ in turns if sp not in voices})
            frames, fmt, seconds = [], None, []
            for i, (speaker, turn_text) in enumerate(turns):
                payload = self.payload_of(request, turn_text, voices.get(speaker, speaker1_reference), seed + i * TURN_SEED_STRIDE, False)
                try:
                    with app.open_api(payload) as r:
                        wav_bytes = r.read()
                        if r.headers.get("X-Synthesis-Time"):
                            seconds.append(float(r.headers["X-Synthesis-Time"]))
                except ApiError as e:
                    raise ApiError(e.status, f"turn {i + 1} (S{speaker}): {e.message}")
                try:
                    with wave.open(io.BytesIO(wav_bytes)) as w:
                        this = (w.getnchannels(), w.getsampwidth(), w.getframerate())
                        frames.append(w.readframes(w.getnframes()))
                except (wave.Error, EOFError):
                    raise ApiError(502, f"turn {i + 1} (S{speaker}): the TTS API did not return WAV audio")
                fmt = fmt or this
                if this != fmt:
                    raise ApiError(502, f"turn {i + 1} (S{speaker}): audio format {this} differs from the first turn's {fmt}")
            channels, width, rate = fmt
            gap = b"\x00" * (int(rate * TURN_GAP_S) * channels * width)
            pcm = gap.join(frames)
            out = io.BytesIO()
            with wave.open(out, "wb") as w:
                w.setnchannels(channels)
                w.setsampwidth(width)
                w.setframerate(rate)
                w.writeframes(pcm)
            extra = {"X-Seed": str(seed), "X-Dialogue-Turns": str(len(turns)), "X-Audio-Duration": f"{len(pcm) / (channels * width * rate):.3f}"}
            if seconds:
                extra["X-Synthesis-Time"] = f"{sum(seconds):.3f}"
            if fallback:
                extra["X-Dialogue-Fallback"] = ",".join(str(s) for s in fallback)
            self.send_bytes(200, out.getvalue(), "audio/wav", extra)

    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    api_url = os.getenv("JYUTVOICE_URL", "http://127.0.0.1:8000")
    api_key = os.getenv("JYUTVOICE_API_KEY", "")
    host, port = os.getenv("HOST", "127.0.0.1"), int(os.getenv("PORT", "8898"))
    server = make_server(api_url, api_key, host, port, public_api_url=os.getenv("JYUTVOICE_PUBLIC_URL") or None, max_body=int(os.getenv("MAX_BODY_BYTES", MAX_BODY_BYTES)))
    print(f"JyutVoice playground at http://{host}:{port} -> API {api_url} ({'with' if api_key else 'WITHOUT'} an API key)", flush=True)
    server.serve_forever()
