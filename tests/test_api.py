"""The HTTP API (api/): text splitting, WAV helpers, the reference-clip handling of the service (base64, data URI, file://, the cache of voices), and the endpoints with a
stub in place of the models (no weights are needed; `docker/` runs the real thing)."""
import base64
import io
import math
import os
import shutil
import struct
import subprocess
import sys
import threading
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("hyperpyyaml")
pytest.importorskip("torchaudio")
from fastapi.testclient import TestClient

import api.main as api_main
from api.audio import apply_level, decode_audio, level_gain, speech_rms, tensor_to_pcm16_bytes, wav_bytes_from_pcm, wav_header
from api.config import APIConfig
from api.models import SpeechRequest
from api.service import Chunk, PromptCacheMiss, SpeechPlan, TTSService, _split_prepared
from api.text_split import sentences, split_text

KEY = "unit-test-key-0123456789"


# ---- text splitting ---------------------------------------------------------------------------------------------------------------------------------------


def test_sentences_end_at_marks_and_keep_closers():
    assert sentences("你好。我係Joseph！今日好嗎？") == ["你好。", "我係Joseph！", "今日好嗎？"]
    assert sentences("佢話「好啊。」然後走咗。") == ["佢話「好啊。」", "然後走咗。"]
    assert sentences("真係咩…？好。") == ["真係咩…？", "好。"]
    assert sentences("一\n二\n") == ["一\n", "二\n"]


def test_full_stop_needs_space_or_end_and_no_digit_before():
    assert sentences("It is 3.5 degrees. Fine.") == ["It is 3.5 degrees. ", "Fine."]
    assert sentences("visit google.com now") == ["visit google.com now"]
    assert sentences("1. one 2. two") == ["1. one 2. two"]  # list numbers are not sentence ends


def test_split_text_packs_sentences_up_to_the_limit():
    text = "甲乙丙丁戊。" * 10  # 6 characters per sentence
    chunks = split_text(text, 20)
    assert all(len(c) <= 20 for c in chunks)
    assert "".join(chunks) == text and len(chunks) == 4  # three sentences (18 characters) per chunk
    assert split_text("好。", 100) == ["好。"]
    assert split_text("   \n ", 100) == []


def test_long_sentence_is_cut_at_commas_then_spaces_then_anywhere():
    clauses = "呢個係一段好長嘅句子，" * 8
    chunks = split_text(clauses, 30)
    assert all(len(c) <= 30 for c in chunks) and "".join(chunks) == clauses
    assert all(c.endswith("，") for c in chunks)
    words = " ".join(["hello"] * 30)
    chunks = split_text(words, 25)
    assert all(len(c) <= 25 for c in chunks) and " ".join(chunks).split() == words.split()
    chunks = split_text("a" * 100, 30)
    assert [len(c) for c in chunks] == [30, 30, 30, 10]


# ---- audio helpers ----------------------------------------------------------------------------------------------------------------------------------------


def test_wav_header_exact_and_placeholder():
    pcm = b"\x01\x00" * 100
    wav = wav_bytes_from_pcm(pcm)
    riff, size, wave, fmt, fmt_len, tag, ch, sr, byte_rate, align, bits, data, data_len = struct.unpack("<4sI4s4sIHHIIHH4sI", wav[:44])
    assert (riff, wave, fmt, data) == (b"RIFF", b"WAVE", b"fmt ", b"data")
    assert (tag, ch, sr, byte_rate, align, bits) == (1, 1, 24000, 48000, 2, 16)
    assert data_len == 200 and size == 36 + 200 and len(wav) == 244
    open_ended = wav_header(None)
    assert len(open_ended) == 44 and struct.unpack("<I", open_ended[4:8])[0] == 0xFFFFFFFF and struct.unpack("<I", open_ended[40:44])[0] == 0xFFFFFFFF


def test_pcm16_rounds_clips_and_has_two_bytes_per_sample():
    pcm = tensor_to_pcm16_bytes(torch.tensor([[0.0, 0.25, -0.25, 2.0, -2.0]]))
    assert struct.unpack("<5h", pcm) == (0, 8192, -8192, 32767, -32767)  # rounded, not truncated (8191.75)


def sine(amplitude, seconds=1.0, sr=24000, hz=220):
    t = torch.arange(int(seconds * sr)) / sr
    return (amplitude * torch.sin(2 * math.pi * hz * t)).unsqueeze(0)


def dbfs(linear):
    return 20 * math.log10(linear)


def test_speech_rms_ignores_pauses_and_the_noise_floor():
    torch.manual_seed(0)
    speech = sine(0.3)
    padded = torch.cat([torch.zeros(1, 24000), speech, torch.zeros(1, 48000), 0.0005 * torch.randn(1, 24000)], dim=1)  # the noise is 52 dB under the speech
    assert speech_rms(speech) == pytest.approx(0.3 / math.sqrt(2), rel=0.01)
    assert speech_rms(padded) == pytest.approx(speech_rms(speech), rel=0.02)


def test_speech_rms_counts_quiet_speech_down_to_40_db_under_the_loudest_frame_and_nothing_below():
    loud, soft, faint = sine(0.3), sine(0.01), sine(0.001)  # 29.5 dB and 49.5 dB under the loud one
    power = lambda *parts: float(torch.cat(parts, dim=1).pow(2).mean())  # noqa: E731
    assert speech_rms(torch.cat([loud, soft], dim=1)) == pytest.approx(math.sqrt(power(loud, soft)), rel=0.01)
    assert speech_rms(torch.cat([loud, faint], dim=1)) == pytest.approx(math.sqrt(power(loud)), rel=0.01)


def test_speech_rms_of_silence_is_zero_and_a_clip_shorter_than_a_frame_still_counts():
    assert speech_rms(torch.zeros(1, 24000)) == 0.0
    assert speech_rms(torch.zeros(1, 100)) == 0.0
    assert speech_rms(torch.zeros(1, 0)) == 0.0
    assert speech_rms(sine(0.3, seconds=0.01, hz=500)) == pytest.approx(0.3 / math.sqrt(2), rel=0.02)  # 240 samples, half a frame, whole periods


def test_level_gain_brings_speech_to_the_target_whatever_it_started_at():
    for amplitude in (0.05, 0.3, 1.2):
        wav = sine(amplitude)
        assert dbfs(speech_rms(wav * level_gain(wav, -20.0))) == pytest.approx(-20.0, abs=0.05)
    assert dbfs(speech_rms(sine(0.3) * level_gain(sine(0.3), -14.0))) == pytest.approx(-14.0, abs=0.05)
    assert level_gain(torch.zeros(1, 24000), -20.0) == 1.0  # nothing to level: no inf or nan


def test_apply_level_lowers_the_gain_rather_than_clip():
    spiky = sine(0.1)
    spiky[0, 5000] = 0.9
    out = apply_level(spiky, 5.0)  # 5 x 0.9 would be 4.5
    assert out.abs().max() == pytest.approx(0.99, abs=1e-6)
    assert torch.allclose(out, spiky * (0.99 / 0.9), atol=1e-6)  # the whole chunk is scaled, nothing is flat-topped
    assert torch.allclose(apply_level(sine(0.1), 2.0), sine(0.1) * 2.0, atol=1e-6)  # below the ceiling the gain is exactly the one asked for


def sine_wav(seconds=1.0, sr=16000):
    import numpy as np
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * np.arange(int(seconds * sr)) / sr)).astype("float32"), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_decode_audio_wav_and_junk():
    wav, sr = decode_audio(sine_wav(1.0, 16000))
    assert sr == 16000 and wav.shape == (1, 16000) and wav.dtype == torch.float32
    with pytest.raises(ValueError, match="not an audio file"):
        decode_audio(b"this is not audio")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
def test_decode_audio_m4a_through_ffmpeg(tmp_path):
    src, dst = tmp_path / "a.wav", tmp_path / "a.m4a"
    src.write_bytes(sine_wav(1.0, 16000))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), str(dst)], check=True)  # the index (moov) is at the end: not readable from a pipe
    wav, sr = decode_audio(dst.read_bytes())
    assert abs(wav.shape[1] / sr - 1.0) < 0.1 and wav.abs().max() > 0.1


# ---- the service: reference clip and cache of voices -------------------------------------------------------------------------------------------------------


def make_service(tmp_path=None, **cfg):
    svc = TTSService.__new__(TTSService)  # no models: only the parts that deal with the request
    svc.cfg = APIConfig(**cfg)
    svc._cache = OrderedDict()
    svc._cache_lock = threading.Lock()
    svc._lock = threading.Lock()
    svc.calls = []

    def speaker(payload):
        svc.calls.append(payload)
        return torch.full((1, 192), float(len(payload))), torch.zeros(1, 80)

    svc._speaker = speaker
    return svc


def b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


CLIP = bytes(range(256)) * 4


def test_prompt_audio_as_raw_base64_data_uri_and_wrapped_base64():
    svc = make_service()
    for value in (b64(CLIP), "data:audio/wav;base64," + b64(CLIP), "data:audio/wav;codecs=pcm;base64," + b64(CLIP), base64.encodebytes(CLIP).decode(), b64(CLIP).rstrip("=")):
        assert svc._read_prompt_bytes(value) == CLIP
    urlsafe = base64.urlsafe_b64encode(bytes([251, 255, 254, 250] * 10)).decode()
    assert "-" in urlsafe or "_" in urlsafe
    assert svc._read_prompt_bytes(urlsafe) == bytes([251, 255, 254, 250] * 10)


@pytest.mark.parametrize("value", ["not base64 at all!!", "ab$cd", "", "   ", "data:audio/wav;base64,"])
def test_invalid_base64_is_a_value_error(value):
    with pytest.raises(ValueError):
        make_service()._read_prompt_bytes(value)


def test_prompt_size_limit():
    svc = make_service(max_prompt_bytes=1000)
    assert svc._read_prompt_bytes(b64(b"x" * 900)) == b"x" * 900
    with pytest.raises(ValueError, match="larger"):
        svc._read_prompt_bytes(b64(b"x" * 2000))


def test_file_uri_only_under_the_configured_root(tmp_path):
    root = tmp_path / "prompts"
    root.mkdir()
    (root / "a.wav").write_bytes(CLIP)
    outside = tmp_path / "secret.wav"
    outside.write_bytes(b"secret")
    (root / "link.wav").symlink_to(outside)

    assert "not enabled" in str(pytest.raises(ValueError, make_service()._read_prompt_bytes, f"file://{root}/a.wav").value)
    svc = make_service(prompt_audio_root=str(root))
    assert svc._read_prompt_bytes(f"file://{root}/a.wav") == CLIP
    for bad in (f"file://{outside}", f"file://{root}/../secret.wav", f"file://{root}/link.wav", f"file://{root}-evil/a.wav"):
        with pytest.raises(ValueError, match="outside"):
            svc._read_prompt_bytes(bad)
    with pytest.raises(ValueError, match="does not exist"):
        svc._read_prompt_bytes(f"file://{root}/missing.wav")


def req(**kw):
    kw.setdefault("input", "你好")
    return SpeechRequest(**kw)


def test_voice_cache_by_content_and_by_id():
    svc = make_service()
    key, spk, c = svc._voice(req(prompt_audio=b64(CLIP)))
    assert key.startswith("pc_") and len(key) == 3 + 32
    key2, spk2, _ = svc._voice(req(prompt_audio="data:audio/wav;base64," + b64(CLIP)))  # the same clip in another notation: the same voice
    assert key2 == key and torch.equal(spk, spk2) and len(svc.calls) == 1
    key3, spk3, _ = svc._voice(req(prompt_cache_id=key))
    assert key3 == key and torch.equal(spk, spk3) and len(svc.calls) == 1
    assert svc._voice(req(prompt_audio=b64(CLIP + b"1")))[0] != key and len(svc.calls) == 2


def test_voice_errors():
    svc = make_service()
    with pytest.raises(PromptCacheMiss):
        svc._voice(req(prompt_cache_id="pc_" + "0" * 32))
    with pytest.raises(ValueError, match="not both"):
        svc._voice(req(prompt_audio=b64(CLIP), prompt_cache_id="pc_x"))
    with pytest.raises(ValueError, match="required"):
        svc._voice(req())
    assert not issubclass(PromptCacheMiss, ValueError)  # 404, not 400


def test_voice_cache_is_lru():
    svc = make_service(prompt_cache_size=2)
    a, b, c = (svc._voice(req(prompt_audio=b64(bytes([i]) * 64)))[0] for i in range(3))
    with pytest.raises(PromptCacheMiss):
        svc._voice(req(prompt_cache_id=a))
    svc._voice(req(prompt_cache_id=b))  # b is now the most recent: the next new voice evicts c
    d = svc._voice(req(prompt_audio=b64(b"\x09" * 64)))[0]
    svc._voice(req(prompt_cache_id=b))
    svc._voice(req(prompt_cache_id=d))
    with pytest.raises(PromptCacheMiss):
        svc._voice(req(prompt_cache_id=c))


def test_input_length_limit():
    svc = make_service(max_input_chars=10)
    with pytest.raises(ValueError, match="limit"):
        svc.prepare_speech_context(req(input="好" * 11, prompt_audio=b64(CLIP)))


def tokens_of(text):
    import numpy as np

    words = text.split(" ")
    return {"x": np.zeros((1, 2 * len(words) + 1), dtype=np.int64)}


def test_text_that_exceeds_the_encoder_is_cut_at_word_breaks():
    text = " ".join(["我", "喺", "machine learning", "公司", "做", "嘢"] * 4)
    pieces = _split_prepared(text, tokens_of, max_tokens=17)
    assert len(pieces) > 1 and all(t["x"].shape[1] <= 17 for _, t in pieces)
    assert " ".join(p for p, _ in pieces).split(" ") == text.split(" ") and all("machine learning" in p or "machine" not in p for p, _ in pieces)
    assert _split_prepared("我 喺", tokens_of, 17)[0][0] == "我 喺"
    with pytest.raises(ValueError, match="word break"):
        _split_prepared("a" * 40, lambda t: {"x": __import__("numpy").zeros((1, 40))}, 17)
    with pytest.raises(ValueError, match="word break"):
        _split_prepared("machine learning", lambda t: {"x": __import__("numpy").zeros((1, 40))}, 17)  # the only space is inside an English phrase


def test_plan_skips_chunks_without_anything_to_read(monkeypatch):
    import numpy as np

    import api.service as service

    svc = make_service(max_chunk_chars=10)
    svc.acoustic = SimpleNamespace(add_blank=True, max_tokens=100)
    monkeypatch.setattr(service, "prepare_text", lambda text, lang, segmentation, normalize: text)  # no pydips or WeTextProcessing here
    monkeypatch.setattr(service, "text_tokens", lambda text, lang, add_blank: {"x": np.zeros((1, 2 * len(text) + 1), dtype=np.int64)})
    settings = {"flow_steps": 10, "length_scale": 1.0, "dp_temperature": 0.3, "dp_pause_temperature": 1.0, "prosody_temperature": 0.5, "use_prosody": True, "seed": 0}
    plan = lambda text: svc._plan_text(text, torch.zeros(1, 192), torch.zeros(1, 80), settings, normalize=True)  # noqa: E731
    assert [c.text for c in plan("你好。我係Joseph。").chunks] == ["你好。", "我係Joseph。"]
    svc.cfg.max_chunk_chars = 3
    assert [c.text for c in plan("你好。 。。 ！！").chunks] == ["你好。"]  # the chunks "。。" and "！！" are marks alone: no speech
    svc.cfg.max_chunk_chars = 10
    for text in ("。。。 ！！", "   ", "…"):
        with pytest.raises(ValueError, match="nothing to read"):
            plan(text)


def leveled_service(monkeypatch, waves):
    """A service whose vocoder gives `waves[seed]` (what HiFT returns with its limiter off), chunk i being drawn with seed i."""
    import api.service as service

    svc = make_service()
    svc.acoustic = svc.decoder = svc.hift = None
    svc.device = torch.device("cpu")
    monkeypatch.setattr(service, "synthesise_wave", lambda *args, seed, **kwargs: (waves[seed], {"frames": 1, "t_acoustic": 0.0, "t_vocoder": 0.0}))
    return svc


def plan_of(chunks, level_dbfs):
    import numpy as np

    return SpeechPlan(
        chunks=[Chunk("x", {"x": np.zeros((1, 3), dtype=np.int64)}) for _ in range(chunks)], spk_embed=torch.zeros(1, 192), c=torch.zeros(1, 80), seed=0, flow_steps=10,
        length_scale=1.0, dp_temperature=0.3, dp_pause_temperature=1.0, prosody_temperature=0.5, use_prosody=True, level_dbfs=level_dbfs,
    )  # fmt: skip


def test_speech_is_levelled_to_the_target_and_never_clips(monkeypatch):
    hot = sine(1.2)  # the vocoder's own output for some voices: 1.2 x full scale (its limiter would have flattened it)
    out = leveled_service(monkeypatch, {0: hot})._synthesise(plan_of(1, -20.0), 0)
    assert dbfs(speech_rms(out)) == pytest.approx(-20.0, abs=0.1) and out.abs().max() <= 0.99


def test_a_silent_chunk_stays_silent(monkeypatch):
    out = leveled_service(monkeypatch, {0: torch.zeros(1, 2400)})._synthesise(plan_of(1, -20.0), 0)
    assert out.shape == (1, 2400) and float(out.abs().max()) == 0.0


def test_one_gain_serves_every_chunk_of_a_request(monkeypatch):
    loud, quiet = sine(1.2), sine(0.05)
    svc = leveled_service(monkeypatch, {0: loud, 1: quiet})
    plan = plan_of(2, -20.0)
    first, second = svc._synthesise(plan, 0), svc._synthesise(plan, 1)
    gain = float(first.abs().max() / loud.abs().max())
    assert second.abs().max() == pytest.approx(float(quiet.abs().max()) * gain, rel=1e-4)  # a quiet chunk stays quieter by the same ratio: no steps between chunks
    assert plan.gain == pytest.approx(gain, rel=1e-4)


def test_a_later_chunk_that_would_pass_the_ceiling_is_lowered_alone(monkeypatch):
    quiet, loud = sine(0.05), sine(1.2)
    svc = leveled_service(monkeypatch, {0: quiet, 1: loud})
    plan = plan_of(2, -20.0)
    svc._synthesise(plan, 0)  # the gain is about 20 dB up
    assert svc._synthesise(plan, 1).abs().max() == pytest.approx(0.99, abs=1e-6)


def test_without_levelling_the_output_is_the_vocoders_own_with_its_old_ceiling(monkeypatch):
    svc = leveled_service(monkeypatch, {0: sine(1.2), 1: sine(0.3)})
    plan = plan_of(2, None)
    assert torch.equal(svc._synthesise(plan, 0), torch.clamp(sine(1.2), -0.99, 0.99))
    assert torch.equal(svc._synthesise(plan, 1), sine(0.3))


def planned(monkeypatch, cfg=None, **request):
    import numpy as np

    import api.service as service

    svc = make_service(**(cfg or {}))
    svc.acoustic = SimpleNamespace(add_blank=True, max_tokens=100, has_prosody=True)
    monkeypatch.setattr(service, "prepare_text", lambda text, lang, segmentation, normalize: text)
    monkeypatch.setattr(service, "text_tokens", lambda text, lang, add_blank: {"x": np.zeros((1, 2 * len(text) + 1), dtype=np.int64)})
    return svc.prepare_speech_context(req(input="你好。", prompt_audio=b64(CLIP), **request))["prepared"]


def test_loudness_is_levelled_by_default_and_the_request_and_server_can_switch_it(monkeypatch):
    assert planned(monkeypatch).level_dbfs == -20.0
    assert planned(monkeypatch, normalize_loudness=False).level_dbfs is None
    assert planned(monkeypatch, cfg={"normalize_loudness": False}).level_dbfs is None
    assert planned(monkeypatch, cfg={"normalize_loudness": False}, normalize_loudness=True).level_dbfs == -20.0
    assert planned(monkeypatch, cfg={"output_level_dbfs": -17.5}).level_dbfs == -17.5
    assert planned(monkeypatch, cfg={"output_level_dbfs": -17.5}, normalize_loudness=False).level_dbfs is None


def test_wav_and_pcm_outputs_and_streaming_order():
    svc = make_service()
    svc._synthesise = lambda plan, i: torch.full((1, 240 * (i + 1)), 0.25)
    plan = SimpleNamespace(chunks=[object(), object(), object()])
    wav_request, pcm_request = SimpleNamespace(output_format="wav"), SimpleNamespace(output_format="pcm")
    samples = 240 + 480 + 720
    whole = svc.generate_speech_bytes(wav_request, plan)
    assert len(whole) == 44 + 2 * samples and struct.unpack("<I", whole[40:44])[0] == 2 * samples
    assert len(svc.generate_speech_bytes(pcm_request, plan)) == 2 * samples
    parts = list(svc.stream_speech_bytes(wav_request, plan))
    assert parts[0] == wav_header(None) and [len(p) for p in parts[1:]] == [480, 960, 1440]
    assert b"".join(parts[1:]) == whole[44:]
    assert [len(p) for p in svc.stream_speech_bytes(pcm_request, plan)] == [480, 960, 1440]


def test_streaming_ends_quietly_when_a_chunk_fails():
    svc = make_service()

    def synth(plan, i):
        if i == 1:
            raise RuntimeError("boom")
        return torch.zeros(1, 240)

    svc._synthesise = synth
    parts = list(svc.stream_speech_bytes(SimpleNamespace(output_format="pcm"), SimpleNamespace(chunks=[0, 1, 2])))
    assert [len(p) for p in parts] == [480]


# ---- the endpoints ----------------------------------------------------------------------------------------------------------------------------------------


class FakeService:
    """Stands in for TTSService: records the requests, answers the way the real one reports errors."""

    def __init__(self):
        self.loaded = True
        self.requests = []

    def is_loaded(self):
        return self.loaded

    def info(self):
        return {"duration_model": "flow", "prosody": True, "cached_prompts": 2}

    def prepare_speech_context(self, request):
        self.requests.append(request)
        if request.input == "unknown voice":
            raise PromptCacheMiss("prompt_cache_id 'pc_x' is not in the cache")
        if request.input == "bad input":
            raise ValueError("input has nothing to read")
        if request.input == "crash":
            raise RuntimeError("model exploded")
        return {"prompt_cache_id": "pc_" + "a" * 32, "prepared": SimpleNamespace(n=2)}

    def generate_speech_bytes(self, request, plan):
        pcm = b"\x00\x01" * 2400
        return wav_bytes_from_pcm(pcm) if request.output_format == "wav" else pcm

    def stream_speech_bytes(self, request, plan):
        if request.output_format == "wav":
            yield wav_header(None)
        yield b"\x00\x01" * 1200
        yield b"\x00\x01" * 1200


@pytest.fixture
def client(monkeypatch):
    fake = FakeService()
    monkeypatch.setattr(api_main, "get_service", lambda: fake)
    monkeypatch.setattr(api_main.config, "api_key", KEY)
    monkeypatch.setattr(api_main.config, "require_api_key", True)
    c = TestClient(api_main.app, raise_server_exceptions=False)  # no `with`: the lifespan would load the models
    c.fake = fake
    return c


AUTH = {"Authorization": f"Bearer {KEY}"}
BODY = {"input": "你好，我係 Joseph。", "prompt_audio": b64(CLIP)}


def test_root_and_health_need_no_key(client):
    assert client.get("/").json()["status"] == "running"
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "healthy", "model_loaded": True, "device": "cpu", "duration_model": "flow", "prosody": True, "cached_prompts": 2, "version": "1.0.0"}
    client.fake.loaded = False
    assert client.get("/health").status_code == 503


def test_speech_requires_the_bearer_key(client, monkeypatch):
    assert client.post("/v1/audio/speech", json=BODY).status_code == 401
    assert client.post("/v1/audio/speech", json=BODY, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/v1/audio/speech", json=BODY, headers={"Authorization": KEY}).status_code == 401  # no "Bearer"
    assert client.post("/v1/audio/speech", json=BODY, headers=AUTH).status_code == 200
    assert client.fake.requests and len(client.fake.requests) == 1  # the refused requests never reached the service
    monkeypatch.setattr(api_main.config, "require_api_key", False)
    assert client.post("/v1/audio/speech", json=BODY).status_code == 200


def test_speech_returns_wav_with_headers(client):
    r = client.post("/v1/audio/speech", json=BODY, headers=AUTH)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    assert r.content[:4] == b"RIFF" and len(r.content) == 44 + 4800
    assert r.headers["prompt-cache-id"] == "pc_" + "a" * 32 and r.headers["x-tts-model"] == "jyutvoice"
    assert r.headers["content-disposition"] == 'attachment; filename="speech.wav"'
    assert r.headers["x-audio-duration"] == "0.100"
    assert client.fake.requests[0].prompt_audio == BODY["prompt_audio"] and client.fake.requests[0].stream is False


def test_pcm_format_and_response_format_alias(client):
    r = client.post("/v1/audio/speech", json={**BODY, "format": "pcm"}, headers=AUTH)
    assert r.headers["content-type"] == "audio/pcm" and len(r.content) == 4800 and r.headers["x-audio-duration"] == "0.100"
    r = client.post("/v1/audio/speech", json={**BODY, "format": "wav", "response_format": "pcm"}, headers=AUTH)
    assert r.headers["content-type"] == "audio/pcm"
    assert client.post("/v1/audio/speech", json={**BODY, "format": "mp3"}, headers=AUTH).status_code == 422


def test_streaming_response(client):
    r = client.post("/v1/audio/speech", json={**BODY, "stream": True}, headers=AUTH)
    assert r.status_code == 200 and r.headers["prompt-cache-id"].startswith("pc_")
    assert r.content[:44] == wav_header(None) and len(r.content) == 44 + 4800
    r = client.post("/v1/audio/speech", json={**BODY, "stream": True, "format": "pcm"}, headers=AUTH)
    assert len(r.content) == 4800


def test_errors_map_to_status_codes(client):
    post = lambda text: client.post("/v1/audio/speech", json={**BODY, "input": text}, headers=AUTH)  # noqa: E731
    assert post("unknown voice").status_code == 404 and "cache" in post("unknown voice").json()["detail"]
    assert post("bad input").status_code == 400 and post("bad input").json()["detail"] == "input has nothing to read"
    assert post("crash").status_code == 500


def test_request_validation(client):
    post = lambda body: client.post("/v1/audio/speech", json=body, headers=AUTH).status_code  # noqa: E731
    assert post({"prompt_audio": b64(CLIP)}) == 422  # no input
    assert post({**BODY, "input": ""}) == 422
    assert post({**BODY, "speed": 0}) == 422 and post({**BODY, "speed": 9}) == 422 and post({**BODY, "speed": 1.5}) == 200
    assert post({**BODY, "seed": -1}) == 422 and post({**BODY, "seed": 7}) == 200
    assert post({**BODY, "flow_steps": 0}) == 422 and post({**BODY, "flow_steps": 5}) == 200


def test_fields_of_the_soulx_api_are_tolerated(client):
    body = {**BODY, "model": "soulx", "prompt_text": "transcript", "voice": "alloy", "language": "yue", "temperature": 0.6, "top_k": 100, "top_p": 0.9, "repetition_penalty": 1.1,
            "pronunciation_fix": True, "normalize_loudness": True, "chunk_size": 25}  # fmt: skip
    r = client.post("/v1/audio/speech", json=body, headers=AUTH)
    assert r.status_code == 200 and r.headers["x-tts-model"] == "soulx"


def test_normalize_loudness_is_a_real_field_that_reaches_the_service(client):
    assert client.post("/v1/audio/speech", json={**BODY, "normalize_loudness": False}, headers=AUTH).status_code == 200
    assert client.fake.requests[-1].normalize_loudness is False
    assert client.post("/v1/audio/speech", json=BODY, headers=AUTH).status_code == 200
    assert client.fake.requests[-1].normalize_loudness is None  # the server's default applies
    assert client.post("/v1/audio/speech", json={**BODY, "normalize_loudness": "loud"}, headers=AUTH).status_code == 422


def test_prompt_cache_id_request_and_oversized_body(client, monkeypatch):
    r = client.post("/v1/audio/speech", json={"input": "你好", "prompt_cache_id": "pc_" + "a" * 32}, headers=AUTH)
    assert r.status_code == 200 and client.fake.requests[-1].prompt_audio is None
    monkeypatch.setattr(api_main.config, "max_request_bytes", 100)
    r = client.post("/v1/audio/speech", json=BODY, headers=AUTH)
    assert r.status_code == 413 and r.json()["error"] == "RequestEntityTooLarge"


# ---- configuration ----------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.skipif("LENGTH_SCALE" in os.environ, reason="the default is read from the environment when the module is imported")
def test_the_default_pace_is_the_one_that_matches_the_teachers_for_the_shipped_model():
    """The English-capable sampler renders about 10% short at a length scale of 1.0 (median length ratio 0.90 against the teacher, 4.6 against 4.2 syllables/s);
    1.1 gives 0.99 (docs/training.md)."""
    assert APIConfig().length_scale == 1.1


@pytest.mark.skipif("OUTPUT_LEVEL_DBFS" in os.environ or "NORMALIZE_LOUDNESS" in os.environ, reason="the defaults are read from the environment when the module is imported")
def test_the_default_output_level_is_the_one_the_prompted_demo_renders_have():
    """Without a decoder prompt the vocoder's level swings 12 dB with the voice and clips for some (speech RMS -9.8 ... -21.7 dBFS, peaks to +3.8); the prompted
    renders sit at about -20.5 (docs/decisions-and-known-issues.md)."""
    assert APIConfig().normalize_loudness is True and APIConfig().output_level_dbfs == -20.0


@pytest.mark.parametrize("key", ["", "changeme", "change-me-please", "dev", "test", "my-dev-key"])
def test_weak_or_missing_key_stops_the_server(key):
    with pytest.raises(RuntimeError):
        APIConfig(api_key=key, require_api_key=True).validate_runtime_security()


def test_real_key_or_disabled_auth_is_accepted():
    APIConfig(api_key=KEY, require_api_key=True).validate_runtime_security()
    APIConfig(api_key="", require_api_key=False).validate_runtime_security()


def test_wildcard_cors_with_credentials_is_refused():
    with pytest.raises(ValueError):
        APIConfig(cors_allowed_origins=("*",), cors_allow_credentials=True)
