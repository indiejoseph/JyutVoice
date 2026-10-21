"""The playground (playground/server.py): a stdlib proxy between a browser page and the JyutVoice API. It holds the API key, turns preset voices into reference clips,
cuts [S1]..[S4] dialogues into turns, and relays audio. The API is replaced by a stub HTTP server here, so no models are needed."""
import base64
import importlib.util
import io
import json
import os
import socket
import threading
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PLAYGROUND = os.path.join(ROOT, "playground")
_spec = importlib.util.spec_from_file_location("playground_server", os.path.join(PLAYGROUND, "server.py"))
playground = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(playground)

KEY = "unit-test-key-0123456789"
RATE = 24000


def wav_of(samples: int, value: int = 1000) -> bytes:
    """Mono PCM16 WAV at 24 kHz with `samples` samples of a constant value."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(value.to_bytes(2, "little", signed=True) * samples)
    return buf.getvalue()


def frames_of(wav_bytes: bytes):
    with wave.open(io.BytesIO(wav_bytes)) as w:
        return w.readframes(w.getnframes()), w.getframerate(), w.getnchannels(), w.getsampwidth()


class StubApi:
    """Stands in for the JyutVoice API: records every request, answers from a queue of canned replies (or the default one)."""

    def __init__(self):
        self.requests = []
        self.replies = []
        self.default = {"status": 200, "body": wav_of(2400), "headers": {"Content-Type": "audio/wav", "Prompt-Cache-Id": "pc_abc", "X-Audio-Duration": "0.100", "X-Synthesis-Time": "0.250"}}
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _reply(self, reply):
                chunks = reply.get("chunks")
                self.send_response(reply["status"])
                for k, v in reply.get("headers", {}).items():
                    self.send_header(k, v)
                if chunks is not None:
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for c in chunks:
                        self.wfile.write(f"{len(c):X}\r\n".encode() + c + b"\r\n")
                    if reply.get("abort"):  # hang up without the closing chunk
                        self.wfile.flush()
                        self.close_connection = True
                        return
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    self.send_header("Content-Length", str(len(reply["body"])))
                    self.end_headers()
                    self.wfile.write(reply["body"])

            def do_GET(self):
                stub.requests.append({"method": "GET", "path": self.path, "auth": self.headers.get("Authorization"), "json": None})
                self._reply(stub.replies.pop(0) if stub.replies else {"status": 200, "body": b'{"status":"healthy","model_loaded":true}', "headers": {"Content-Type": "application/json"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                stub.requests.append({"method": "POST", "path": self.path, "auth": self.headers.get("Authorization"), "json": body})
                self._reply(stub.replies.pop(0) if stub.replies else stub.default)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class Playground:
    def __init__(self, api_url, **kw):
        kw.setdefault("api_key", KEY)
        kw.setdefault("local_voices_dir", None)  # this machine's own voices.local must not change what the tests see
        self.httpd = playground.make_server(api_url, host="127.0.0.1", port=0, **kw)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, method, path, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={"Content-Type": "application/json"} if data is not None else {})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def post(self, body):
        return self.request("POST", "/api/synthesize", body)

    def get(self, path):
        return self.request("GET", path)


@pytest.fixture
def stub():
    s = StubApi()
    yield s
    s.close()


@pytest.fixture
def pg(stub):
    p = Playground(stub.url)
    yield p
    p.close()


def clip_bytes(voice_id):
    entry = next(v for v in json.load(open(os.path.join(PLAYGROUND, "voices", "voices.json"))) if v["id"] == voice_id)
    return open(os.path.join(PLAYGROUND, "voices", entry["file"]), "rb").read()


def data_uri_bytes(uri):
    head, b64 = uri.split(",", 1)
    assert head.startswith("data:audio/") and head.endswith(";base64")
    return base64.b64decode(b64)


# ---- dialogue turns ---------------------------------------------------------------------------------------------------------------------------------------


def test_text_without_markers_is_not_a_dialogue():
    assert playground.split_turns("今日天氣好好") is None


def test_dialogue_is_cut_at_markers_and_text_before_the_first_one_is_speaker_one():
    assert playground.split_turns("前言[S2]你好[S1]再見") == [(1, "前言"), (2, "你好"), (1, "再見")]


def test_dialogue_turns_are_stripped_and_empty_turns_dropped():
    assert playground.split_turns("[S1]  甲  [S2]   [S3]乙") == [(1, "甲"), (3, "乙")]


def test_only_speakers_one_to_four_are_markers():
    assert playground.split_turns("[S5]x") is None


# ---- voices -----------------------------------------------------------------------------------------------------------------------------------------------


def test_voice_list_has_ids_and_names_but_no_file_paths(pg):
    status, _, body = pg.get("/api/voices")
    voices = json.loads(body)["voices"]
    assert status == 200
    assert [v["id"] for v in voices] == ["yue-low-1", "yue-high-1", "yue-low-2", "yue-high-2"]
    assert all(v["name"] and "file" not in v for v in voices)
    assert "wav" not in body.decode()


def test_preset_voice_is_sent_as_the_reference_clip(pg, stub):
    pg.post({"text": "你好", "voice_id": "yue-high-1"})
    assert data_uri_bytes(stub.requests[0]["json"]["prompt_audio"]) == clip_bytes("yue-high-1")


def test_voice_file_outside_the_voices_directory_is_refused(tmp_path):
    voices = tmp_path / "voices"
    voices.mkdir()
    (tmp_path / "secret.txt").write_bytes(b"outside")  # a real file, one level above the voices directory
    (voices / "voices.json").write_text(json.dumps([{"id": "evil", "name": "x", "file": "../secret.txt"}]))
    assert playground.load_voices(str(voices)) == {}


def test_voice_whose_file_is_missing_is_left_out_and_the_rest_kept(tmp_path):
    (tmp_path / "ok.wav").write_bytes(wav_of(10))
    (tmp_path / "voices.json").write_text(json.dumps([{"id": "gone", "name": "x", "file": "gone.wav"}, {"id": "ok", "name": "Ok", "file": "ok.wav"}]))
    assert list(playground.load_voices(str(tmp_path))) == ["ok"]


def test_without_voices_json_there_are_no_voices(tmp_path):
    assert playground.load_voices(str(tmp_path)) == {}


def local_voices(tmp_path, entries):
    d = tmp_path / "local"
    d.mkdir()
    for e in entries:
        (d / e["file"]).write_bytes(wav_of(50, e.pop("value", 7)))
    (d / "voices.json").write_text(json.dumps(entries))
    return str(d)


def test_local_voices_are_listed_before_the_shipped_ones(stub, tmp_path):
    local = local_voices(tmp_path, [{"id": "mine-1", "name": "Mine 1", "file": "m1.wav", "note": "my own clip"}, {"id": "mine-2", "name": "Mine 2", "file": "m2.wav"}])
    p = Playground(stub.url, local_voices_dir=local)
    try:
        voices = json.loads(p.get("/api/voices")[2])["voices"]
        p.post({"text": "你好", "voice_id": "mine-1"})
        p.post({"text": "你好", "voice_id": "yue-low-1"})
    finally:
        p.close()
    assert [v["id"] for v in voices] == ["mine-1", "mine-2", "yue-low-1", "yue-high-1", "yue-low-2", "yue-high-2"]
    assert voices[0]["note"] == "my own clip"
    assert data_uri_bytes(stub.requests[0]["json"]["prompt_audio"]) == wav_of(50, 7)
    assert data_uri_bytes(stub.requests[1]["json"]["prompt_audio"]) == clip_bytes("yue-low-1")


def test_a_local_voice_replaces_a_shipped_one_with_the_same_id(stub, tmp_path):
    local = local_voices(tmp_path, [{"id": "yue-low-1", "name": "Replaced", "file": "x.wav"}])
    p = Playground(stub.url, local_voices_dir=local)
    try:
        voices = json.loads(p.get("/api/voices")[2])["voices"]
        p.post({"text": "你好", "voice_id": "yue-low-1"})
    finally:
        p.close()
    assert [v["name"] for v in voices if v["id"] == "yue-low-1"] == ["Replaced"] and len(voices) == 4
    assert data_uri_bytes(stub.requests[0]["json"]["prompt_audio"]) == wav_of(50, 7)


def test_a_missing_local_voices_directory_changes_nothing(stub, tmp_path):
    p = Playground(stub.url, local_voices_dir=str(tmp_path / "does-not-exist"))
    try:
        voices = json.loads(p.get("/api/voices")[2])["voices"]
    finally:
        p.close()
    assert [v["id"] for v in voices] == ["yue-low-1", "yue-high-1", "yue-low-2", "yue-high-2"]


# ---- one synthesis ----------------------------------------------------------------------------------------------------------------------------------------


def test_synthesis_is_forwarded_with_the_key_and_the_audio_comes_back(pg, stub):
    status, headers, body = pg.post({"text": "我喺Zoom開會", "voice_id": "yue-low-1", "seed": 7})
    sent = stub.requests[0]
    assert status == 200 and headers["Content-Type"] == "audio/wav"
    assert body == stub.default["body"]
    assert sent["path"] == "/v1/audio/speech" and sent["auth"] == f"Bearer {KEY}"
    assert sent["json"]["input"] == "我喺Zoom開會"
    assert (sent["json"]["format"], sent["json"]["stream"], sent["json"]["seed"]) == ("wav", False, 7)


def test_api_headers_the_page_shows_are_passed_on(pg):
    _, headers, _ = pg.post({"text": "你好", "voice_id": "yue-low-1", "seed": 3})
    assert headers["Prompt-Cache-Id"] == "pc_abc"
    assert headers["X-Audio-Duration"] == "0.100" and headers["X-Synthesis-Time"] == "0.250"
    assert headers["X-Seed"] == "3"


def test_a_missing_seed_is_drawn_once_sent_upstream_and_reported(pg, stub):
    _, h1, _ = pg.post({"text": "你好", "voice_id": "yue-low-1"})
    _, h2, _ = pg.post({"text": "你好", "voice_id": "yue-low-1", "seed": None})
    s1, s2 = stub.requests[0]["json"]["seed"], stub.requests[1]["json"]["seed"]
    assert str(s1) == h1["X-Seed"] and str(s2) == h2["X-Seed"]
    assert 0 <= s1 < 2**31 and s1 != s2


def test_seed_zero_is_a_real_seed(pg, stub):
    pg.post({"text": "你好", "voice_id": "yue-low-1", "seed": 0})
    assert stub.requests[0]["json"]["seed"] == 0


def test_only_the_models_settings_are_forwarded(pg, stub):
    pg.post({"text": "你好", "voice_id": "yue-low-1", "seed": 1, "speed": 0.9, "flow_steps": 5, "dp_temperature": 0.4, "prosody_temperature": 0.6, "prosody": False,
             "normalize_text": False, "temperature": 9, "api_key": "x", "model": "other", "format": "pcm", "prompt_cache_id": "pc_other"})
    sent = stub.requests[0]["json"]
    assert (sent["speed"], sent["flow_steps"], sent["dp_temperature"], sent["prosody_temperature"], sent["prosody"], sent["normalize_text"]) == (0.9, 5, 0.4, 0.6, False, False)
    assert not {"temperature", "api_key", "prompt_cache_id"} & set(sent)
    assert sent["model"] == "jyutvoice" and sent["format"] == "wav"


def test_unset_settings_are_left_to_the_api_defaults(pg, stub):
    pg.post({"text": "你好", "voice_id": "yue-low-1", "seed": 1, "speed": None, "flow_steps": None})
    assert not {"speed", "flow_steps", "prosody", "dp_temperature"} & set(stub.requests[0]["json"])


def test_an_uploaded_clip_is_forwarded_untouched(pg, stub):
    uri = "data:audio/mpeg;base64," + base64.b64encode(b"not really mp3").decode()
    status, _, _ = pg.post({"text": "你好", "prompt_audio": uri})
    assert status == 200 and stub.requests[0]["json"]["prompt_audio"] == uri


def test_no_authorization_header_when_no_key_is_configured(stub):
    p = Playground(stub.url, api_key="")
    try:
        p.post({"text": "你好", "voice_id": "yue-low-1"})
    finally:
        p.close()
    assert stub.requests[0]["auth"] is None


# ---- bad requests -----------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("body, fragment", [
    ({"voice_id": "yue-low-1"}, "text is required"),
    ({"text": "   ", "voice_id": "yue-low-1"}, "text is required"),
    ({"text": "你好"}, "voice"),
    ({"text": "你好", "voice_id": "nobody"}, "unknown voice_id"),
])
def test_bad_requests_are_refused_before_reaching_the_api(pg, stub, body, fragment):
    status, _, out = pg.post(body)
    assert status == 400 and fragment in json.loads(out)["error"]
    assert stub.requests == []


def test_a_body_that_is_not_json_is_a_400(pg):
    status, _, out = pg.request("POST", "/api/synthesize", raw=b"{nope")
    assert status == 400 and "invalid JSON" in json.loads(out)["error"]


def test_an_oversized_body_is_a_413(stub):
    p = Playground(stub.url, max_body=1000)
    try:
        status, _, out = p.post({"text": "你好" * 1000, "voice_id": "yue-low-1"})
    finally:
        p.close()
    assert status == 413 and stub.requests == []


def test_unknown_paths_are_404(pg):
    assert pg.get("/nope")[0] == 404
    assert pg.request("POST", "/api/other", {})[0] == 404


# ---- errors of the API ------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("reply, status, fragment", [
    ({"status": 400, "body": b'{"detail": "reference clip is too short"}'}, 400, "reference clip is too short"),
    ({"status": 404, "body": b'{"detail": "unknown prompt_cache_id"}'}, 404, "unknown prompt_cache_id"),
    ({"status": 500, "body": b'{"error": "internal_error", "message": "boom"}'}, 500, "boom"),
    ({"status": 422, "body": b'{"detail": [{"loc": ["body", "speed"], "msg": "ensure this value is greater than or equal to 0.25"}]}'}, 422, "speed: ensure this value"),
    ({"status": 502, "body": b"<html>bad gateway</html>"}, 502, "bad gateway"),
])
def test_errors_of_the_api_reach_the_page_as_one_message(pg, stub, reply, status, fragment):
    stub.replies.append({**reply, "headers": {"Content-Type": "application/json"}})
    got, _, out = pg.post({"text": "你好", "voice_id": "yue-low-1"})
    assert got == status and fragment in json.loads(out)["error"]


def test_a_rejected_key_is_the_playgrounds_problem_not_the_users(pg, stub):
    stub.replies.append({"status": 401, "body": b'{"detail": "Invalid API key"}', "headers": {"Content-Type": "application/json"}})
    status, _, out = pg.post({"text": "你好", "voice_id": "yue-low-1"})
    message = json.loads(out)["error"]
    assert status == 502 and "API key" in message and KEY not in message


def test_an_unreachable_api_is_a_502(stub):
    p = Playground("http://127.0.0.1:9")  # nothing listens on the discard port
    try:
        status, _, out = p.post({"text": "你好", "voice_id": "yue-low-1"})
    finally:
        p.close()
    assert status == 502 and "error" in json.loads(out)


# ---- streaming --------------------------------------------------------------------------------------------------------------------------------------------


def test_a_stream_is_asked_for_as_pcm_and_relayed_chunk_by_chunk(pg, stub):
    chunks = [b"\x01\x00" * 500, b"\x02\x00" * 700]
    stub.replies.append({"status": 200, "chunks": chunks, "headers": {"Content-Type": "audio/pcm", "Prompt-Cache-Id": "pc_s"}})
    status, headers, body = pg.post({"text": "你好", "voice_id": "yue-low-1", "stream": True, "seed": 4})
    sent = stub.requests[0]["json"]
    assert (sent["format"], sent["stream"]) == ("pcm", True)
    assert status == 200 and headers["Content-Type"] == "audio/pcm" and headers["Prompt-Cache-Id"] == "pc_s" and headers["X-Seed"] == "4"
    assert body == b"".join(chunks)


def test_a_stream_that_breaks_off_is_cut_short_not_followed_by_a_json_error(pg, stub):
    stub.replies.append({"status": 200, "chunks": [b"\x01\x00" * 400], "abort": True, "headers": {"Content-Type": "audio/pcm"}})
    payload = json.dumps({"text": "你好", "voice_id": "yue-low-1", "stream": True, "seed": 1}).encode()
    host, port = pg.httpd.server_address
    with socket.create_connection((host, port), timeout=20) as conn:
        conn.sendall(b"POST /api/synthesize HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: %d\r\n\r\n" % len(payload) + payload)
        received = b""
        while True:
            piece = conn.recv(65536)
            if not piece:
                break
            received += piece
    assert received.startswith(b"HTTP/1.1 200") and b"audio/pcm" in received
    assert b"error" not in received.split(b"\r\n\r\n", 1)[1]


# ---- dialogue ---------------------------------------------------------------------------------------------------------------------------------------------


def dialogue_replies(stub, lengths):
    stub.replies.extend({"status": 200, "body": wav_of(n, 100 * (i + 1)), "headers": {"Content-Type": "audio/wav", "X-Synthesis-Time": "0.5"}} for i, n in enumerate(lengths))


def test_every_turn_is_one_call_in_its_own_speakers_voice(pg, stub):
    dialogue_replies(stub, [1000, 2000, 3000])
    status, headers, _ = pg.post({"text": "[S1]甲一[S2]乙一[S1]甲二", "voice_id": "yue-low-1", "extra_voices": {"2": {"voice_id": "yue-high-1"}}, "seed": 10})
    calls = [r["json"] for r in stub.requests]
    assert status == 200 and headers["X-Dialogue-Turns"] == "3"
    assert [c["input"] for c in calls] == ["甲一", "乙一", "甲二"]
    assert [data_uri_bytes(c["prompt_audio"]) for c in calls] == [clip_bytes("yue-low-1"), clip_bytes("yue-high-1"), clip_bytes("yue-low-1")]
    assert all(c["format"] == "wav" and c["stream"] is False for c in calls)


def test_turns_are_joined_with_a_quarter_second_of_silence(pg, stub):
    dialogue_replies(stub, [1000, 2000])
    _, headers, body = pg.post({"text": "[S1]甲[S2]乙", "voice_id": "yue-low-1", "seed": 1})
    pcm, rate, channels, width = frames_of(body)
    gap = int(RATE * 0.25) * 2
    first, second = 1000 * 2, 2000 * 2
    assert (rate, channels, width) == (RATE, 1, 2)
    assert len(pcm) == first + gap + second
    assert pcm[:first] == (100).to_bytes(2, "little") * 1000 and pcm[first:first + gap] == b"\x00" * gap and pcm[first + gap:] == (200).to_bytes(2, "little") * 2000
    assert float(headers["X-Audio-Duration"]) == pytest.approx(len(pcm) / 2 / RATE, abs=1e-3)
    assert float(headers["X-Synthesis-Time"]) == pytest.approx(1.0)


def test_each_turn_gets_its_own_seed_and_a_streaming_request_is_not_streamed(pg, stub):
    dialogue_replies(stub, [100, 100, 100])
    pg.post({"text": "[S1]甲[S2]乙[S1]丙", "voice_id": "yue-low-1", "seed": 50, "stream": True})
    seeds = [r["json"]["seed"] for r in stub.requests]
    assert len(set(seeds)) == 3 and seeds[0] == 50
    assert all(r["json"]["stream"] is False for r in stub.requests)


def test_a_speaker_without_a_voice_falls_back_to_speaker_one_and_says_so(pg, stub):
    dialogue_replies(stub, [100, 100, 100])
    _, headers, _ = pg.post({"text": "[S1]甲[S3]乙[S4]丙", "voice_id": "yue-low-2", "extra_voices": {"4": {"voice_id": "yue-high-2"}}})
    calls = [data_uri_bytes(r["json"]["prompt_audio"]) for r in stub.requests]
    assert calls == [clip_bytes("yue-low-2"), clip_bytes("yue-low-2"), clip_bytes("yue-high-2")]
    assert headers["X-Dialogue-Fallback"] == "3"


def test_extra_voices_cannot_replace_speaker_ones_voice(pg, stub):
    dialogue_replies(stub, [100, 100])
    pg.post({"text": "[S1]甲[S2]乙", "voice_id": "yue-low-1", "extra_voices": {"1": {"voice_id": "yue-high-1"}}})
    assert data_uri_bytes(stub.requests[0]["json"]["prompt_audio"]) == clip_bytes("yue-low-1")


def test_an_uploaded_clip_can_be_a_speakers_voice(pg, stub):
    dialogue_replies(stub, [100, 100])
    uri = "data:audio/wav;base64," + base64.b64encode(wav_of(50)).decode()
    pg.post({"text": "[S1]甲[S2]乙", "voice_id": "yue-low-1", "extra_voices": {"2": {"prompt_audio": uri}}})
    assert stub.requests[1]["json"]["prompt_audio"] == uri


def test_an_error_in_one_turn_names_the_turn(pg, stub):
    stub.replies.append({"status": 200, "body": wav_of(100), "headers": {}})
    stub.replies.append({"status": 400, "body": b'{"detail": "input has nothing to read"}', "headers": {"Content-Type": "application/json"}})
    status, _, out = pg.post({"text": "[S1]甲[S2]。", "voice_id": "yue-low-1"})
    message = json.loads(out)["error"]
    assert status == 400 and "turn 2" in message and "S2" in message and "nothing to read" in message


def test_a_dialogue_with_no_text_after_the_markers_is_a_400(pg, stub):
    status, _, out = pg.post({"text": "[S1][S2]", "voice_id": "yue-low-1"})
    assert status == 400 and stub.requests == []


def test_a_turn_that_is_not_audio_is_reported_not_joined(pg, stub):
    stub.replies.append({"status": 200, "body": wav_of(100), "headers": {}})
    stub.replies.append({"status": 200, "body": b"<html>not a wav</html>", "headers": {}})
    status, _, out = pg.post({"text": "[S1]甲[S2]乙", "voice_id": "yue-low-1"})
    message = json.loads(out)["error"]
    assert status == 502 and "turn 2" in message and "S2" in message and "audio" in message


def test_turns_in_different_audio_formats_are_not_joined(pg, stub):
    other_rate = io.BytesIO()
    with wave.open(other_rate, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 100)
    stub.replies.append({"status": 200, "body": wav_of(100), "headers": {}})
    stub.replies.append({"status": 200, "body": other_rate.getvalue(), "headers": {}})
    status, _, out = pg.post({"text": "[S1]甲[S2]乙", "voice_id": "yue-low-1"})
    assert status == 502 and "turn 2" in json.loads(out)["error"] and "format" in json.loads(out)["error"]


def test_text_that_is_not_a_string_is_a_400(pg, stub):
    status, _, out = pg.post({"text": 5, "voice_id": "yue-low-1"})
    assert status == 400 and "text is required" in json.loads(out)["error"] and stub.requests == []


def test_a_seed_that_is_not_an_integer_is_a_400(pg, stub):
    for bad in ("abc", 1.5, True, -1):
        status, _, out = pg.post({"text": "你好", "voice_id": "yue-low-1", "seed": bad})
        assert status == 400 and "seed" in json.loads(out)["error"], bad
    assert stub.requests == []


def test_an_unexpected_failure_is_a_json_500_and_the_server_keeps_serving(pg, stub):
    status, _, out = pg.post({"text": "[S1]甲[S2]乙", "voice_id": "yue-low-1", "extra_voices": "not a mapping"})
    assert status == 500 and "error" in json.loads(out)
    assert pg.post({"text": "你好", "voice_id": "yue-low-1"})[0] == 200


# ---- the page and its helpers -----------------------------------------------------------------------------------------------------------------------------


def test_health_is_relayed_from_the_api(pg, stub):
    status, _, body = pg.get("/api/health")
    assert status == 200 and json.loads(body)["model_loaded"] is True
    assert stub.requests[0]["path"] == "/health"


def test_health_of_an_unreachable_api_is_a_502():
    p = Playground("http://127.0.0.1:9")
    try:
        assert p.get("/api/health")[0] == 502
    finally:
        p.close()


def test_the_page_tells_the_curl_example_where_the_api_is_and_never_the_key(stub):
    p = Playground(stub.url, public_api_url="https://tts.example.org")
    try:
        status, _, body = p.get("/api/config")
    finally:
        p.close()
    assert status == 200 and json.loads(body)["api_url"] == "https://tts.example.org" and KEY not in body.decode()


def test_the_page_is_served_at_the_root(pg):
    status, headers, body = pg.get("/")
    assert status == 200 and headers["Content-Type"].startswith("text/html") and b"<title>" in body
