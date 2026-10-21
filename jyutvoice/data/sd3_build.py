"""Worker side of scripts/build_sd3_dataset.py: per-process models, audio cuts, feature extraction and shard writing.

Kept in an importable module, not in the script: `Dataset.from_generator` fingerprints its generator with dill, and functions defined in
`__main__` are pickled by value together with every global they use (here the ONNX sessions, which cannot be pickled).
"""
import io
import os
import shutil
import subprocess
import time

import numpy as np

MIN_DUR = 2.0


_W = {}  # per-process models and shared read-only data
FAILURES = {}  # reasons for skipped turns of the shard being built


def decode_audio(path, rate):
    """Whole file as mono int16 at `rate` (ffmpeg runs outside the GIL)."""
    ffmpeg = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.int16)


def init_worker(pretrained, out_dir):
    import torch

    from jyutvoice.utils import flow_features as ff

    torch.set_num_threads(1)
    _W["ff"] = ff
    _W["spk"] = ff.load_spk_embedding_model(os.path.join(pretrained, "campplus.onnx"))
    _W["tok"] = ff.load_speech_tokenizer(os.path.join(pretrained, "speech_tokenizer_v2.onnx"))
    _W["flow"] = ff.load_flow_encoder(os.path.join(pretrained, "flow_encoder.pt"), "cpu")
    _W["bert"] = np.load(os.path.join(out_dir, "bert", "bert.npy"), mmap_mode="r")
    _W["offsets"] = np.load(os.path.join(out_dir, "bert", "offsets.npy"))


def wav_bytes(x24):
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, x24, 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def make_row(turn, index, x16, x24):
    """One dataset row for a turn, or a reason string when it cannot be built."""
    import torch

    ff = _W["ff"]
    i16, j16 = int(round(turn["start"] * 16000)), int(round(turn["end"] * 16000))
    a16 = x16[i16:j16].astype(np.float32) / 32768.0
    if len(a16) < 16000 * MIN_DUR * 0.9:
        return "audio shorter than the turn"
    token, token_len = ff.extract_speech_token(a16, _W["tok"])
    n = int(token.shape[1])
    h = ff.get_decoder_hidden_state(token, token_len, _W["flow"], "cpu").squeeze(0)
    if h.shape[0] != 2 * n:
        return f"decoder_h has {h.shape[0]} frames for {n} tokens"
    spk = ff.get_spk_embedding(a16, _W["spk"])
    i24 = int(round(turn["start"] * 24000))
    a24 = x24[i24: i24 + n * 960]  # 2 mel frames (480 samples each) per S3 token
    if len(a24) < n * 960:
        a24 = np.pad(a24, (0, n * 960 - len(a24)))
    offsets = _W["offsets"]
    return {
        "audio": {"bytes": wav_bytes(a24), "path": None}, "id": turn["id"], "lang": "yue", "text": turn["text"], "text_valid": True, "phone": None,
        "spk_emb": np.asarray(spk, dtype=np.float32), "decoder_h": h.numpy().astype(np.float32),
        "bert": np.asarray(_W["bert"][offsets[index]: offsets[index + 1]]),
        "source": f"sd3/{turn['channel']}", "duration": float(len(a24) / 24000),
    }  # fmt: skip


def features():
    from datasets import Audio, Features, Sequence, Value

    return Features({
        "audio": Audio(sampling_rate=24000), "id": Value("string"), "lang": Value("string"), "text": Value("string"), "text_valid": Value("bool"),
        "phone": Value("string"), "spk_emb": Sequence(Value("float32")), "decoder_h": Sequence(Sequence(Value("float32"))),
        "bert": Sequence(Sequence(Value("float16"))), "source": Value("string"), "duration": Value("float32"),
    })  # fmt: skip


def shard_rows(videos, turns_by_video, index_of, vocals_dir, limit):
    """Rows of one shard (a generator for `Dataset.from_generator`); skipped turns are counted in FAILURES."""
    count = 0
    for video in videos:
        path = os.path.join(vocals_dir, f"{video}_vocal.mp3")
        try:
            x16, x24 = decode_audio(path, 16000), decode_audio(path, 24000)
        except Exception as e:  # noqa: BLE001
            key = f"decode failed: {repr(e)[:60]}"
            FAILURES[key] = FAILURES.get(key, 0) + len(turns_by_video[video])
            continue
        for turn in turns_by_video[video]:
            try:
                row = make_row(turn, index_of[turn["id"]], x16, x24)
            except Exception as e:  # noqa: BLE001
                row = f"error: {repr(e)[:80]}"
            if isinstance(row, str):
                FAILURES[row] = FAILURES.get(row, 0) + 1
                continue
            yield row
            count += 1
            if limit and count >= limit:
                return


def build_shard(task):
    """Worker: build one shard. Returns (shard id, rows written or -1 if it already existed, {reason: count}, seconds)."""
    from datasets import Dataset

    shard_id, videos, turns_by_video, index_of, vocals_dir, out_dir, limit = task
    final = os.path.join(out_dir, "shards", f"shard_{shard_id:03d}")
    if os.path.exists(os.path.join(final, "state.json")):
        return shard_id, -1, {}, 0.0
    t0 = time.time()
    FAILURES.clear()
    tmp = final + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    cache = os.path.join(out_dir, "cache", f"shard_{shard_id:03d}")
    shutil.rmtree(cache, ignore_errors=True)
    ds = Dataset.from_generator(
        shard_rows,
        gen_kwargs={"videos": videos, "turns_by_video": turns_by_video, "index_of": index_of, "vocals_dir": vocals_dir, "limit": limit},
        features=features(),
        cache_dir=cache,
        writer_batch_size=64,
    )
    ds.save_to_disk(tmp)
    os.rename(tmp, final)
    shutil.rmtree(cache, ignore_errors=True)
    return shard_id, len(ds), dict(FAILURES), time.time() - t0
