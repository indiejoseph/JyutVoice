"""Worker side of scripts/build_mce_dataset.py: one clip of the MCE parquet -> one dataset row (24 kHz audio, CampPlus speaker embedding, S3 tokens, flow-encoder
`decoder_h`), shards written like the existing dataset (`TextMelDataModule` reads them directly).

Same feature extraction as jyutvoice/data/sd3_build.py (kept in an importable module for the same reason: `Dataset.from_generator` pickles its generator, and
functions defined in `__main__` drag the unpicklable ONNX sessions along), without the BERT features and the ffmpeg cuts: the clips are already single utterances
of 16 kHz WAV, the stored 24 kHz audio is that clip upsampled by 3/2.
"""
import io
import os
import shutil
import time

import numpy as np

_W = {}  # per-process models
FAILURES = {}  # reasons for skipped clips of the shard being built


def init_worker(pretrained):
    import torch

    from jyutvoice.utils import flow_features as ff

    torch.set_num_threads(1)
    _W["ff"] = ff
    _W["spk"] = ff.load_spk_embedding_model(os.path.join(pretrained, "campplus.onnx"))
    _W["tok"] = ff.load_speech_tokenizer(os.path.join(pretrained, "speech_tokenizer_v2.onnx"))
    _W["flow"] = ff.load_flow_encoder(os.path.join(pretrained, "flow_encoder.pt"), "cpu")


def read_wav16(wav_bytes):
    """Mono int16 samples of a 16 kHz WAV."""
    import soundfile as sf

    x, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16", always_2d=True)
    if sr != 16000:
        raise ValueError(f"sample rate {sr}")
    return x.mean(axis=1).astype(np.int16) if x.shape[1] > 1 else x[:, 0]


def upsample_24k(x16):
    from scipy.signal import resample_poly

    return np.clip(np.round(resample_poly(x16.astype(np.float32), 3, 2)), -32768, 32767).astype(np.int16)


def make_row(turn, wav_bytes):
    """One dataset row, or a reason string when it cannot be built."""
    from jyutvoice.data.sd3_build import wav_bytes as encode_wav

    ff = _W["ff"]
    x16 = read_wav16(wav_bytes)
    a16 = x16.astype(np.float32) / 32768.0
    token, token_len = ff.extract_speech_token(a16, _W["tok"])
    n = int(token.shape[1])
    h = ff.get_decoder_hidden_state(token, token_len, _W["flow"], "cpu").squeeze(0)
    if h.shape[0] != 2 * n:
        return f"decoder_h has {h.shape[0]} frames for {n} tokens"
    spk = ff.get_spk_embedding(a16, _W["spk"])
    a24 = upsample_24k(x16)[: n * 960]  # 2 mel frames (480 samples each) per S3 token
    if len(a24) < n * 960:
        a24 = np.pad(a24, (0, n * 960 - len(a24)))
    return {
        "audio": {"bytes": encode_wav(a24), "path": None}, "id": turn["id"], "lang": "yue", "text": turn["text"], "text_valid": True, "phone": None,
        "spk_emb": np.asarray(spk, dtype=np.float32), "decoder_h": h.numpy().astype(np.float32),
        "source": "mce", "duration": float(len(a24) / 24000),
    }  # fmt: skip


def features():
    from datasets import Audio, Features, Sequence, Value

    return Features({
        "audio": Audio(sampling_rate=24000), "id": Value("string"), "lang": Value("string"), "text": Value("string"), "text_valid": Value("bool"),
        "phone": Value("string"), "spk_emb": Sequence(Value("float32")), "decoder_h": Sequence(Sequence(Value("float32"))),
        "source": Value("string"), "duration": Value("float32"),
    })  # fmt: skip


def shard_rows(parquet_path, turns_by_row, limit):
    """Rows of one task: the clips of `parquet_path` whose row index is a key of `turns_by_row` (a generator for `Dataset.from_generator`)."""
    import pyarrow.parquet as pq

    count, row, last = 0, 0, max(turns_by_row)
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=32, columns=["audio"]):
        for audio in batch.column("audio").to_pylist():
            if row in turns_by_row:
                try:
                    out = make_row(turns_by_row[row], audio["bytes"])
                except Exception as e:  # noqa: BLE001
                    out = f"error: {repr(e)[:80]}"
                if isinstance(out, str):
                    FAILURES[out] = FAILURES.get(out, 0) + 1
                else:
                    yield out
                    count += 1
                    if limit and count >= limit:
                        return
            row += 1
            if row > last:
                return


def build_shard(task):
    """Worker: build one task into a shard. Returns (task id, rows written or -1 if it already existed, {reason: count}, seconds)."""
    from datasets import Dataset

    task_id, parquet_path, turns_by_row, out_dir, limit = task
    final = os.path.join(out_dir, "shards", f"shard_{task_id:03d}")
    if os.path.exists(os.path.join(final, "state.json")):
        return task_id, -1, {}, 0.0
    t0 = time.time()
    FAILURES.clear()
    tmp = final + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    cache = os.path.join(out_dir, "cache", f"shard_{task_id:03d}")
    shutil.rmtree(cache, ignore_errors=True)
    ds = Dataset.from_generator(
        shard_rows,
        gen_kwargs={"parquet_path": parquet_path, "turns_by_row": turns_by_row, "limit": limit},
        features=features(),
        cache_dir=cache,
        writer_batch_size=32,
    )
    ds.save_to_disk(tmp)
    os.rename(tmp, final)
    shutil.rmtree(cache, ignore_errors=True)
    return task_id, len(ds), dict(FAILURES), time.time() - t0
