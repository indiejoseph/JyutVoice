"""Turn the speech-dataset3 transcripts into training shards in the layout `TextMelDataModule` reads, with per-character BERT features.

    python scripts/build_sd3_dataset.py plan   --transcripts 'results/full_nonnews/shard*.jsonl' --vocals-dir <speech-dataset3/downloads/vocals> --out-dir D
    python scripts/build_sd3_dataset.py bert   --out-dir D --model <ModernCantoneseBert-large dir>            # GPU: calibration + features
    python scripts/build_sd3_dataset.py build  --out-dir D --pretrained pretrained_models --workers 28       # CPU pool, one shard per task
    python scripts/build_sd3_dataset.py verify --out-dir D

* plan:   keeps Cantonese-tagged turns of 2-30 s (CosyVoice2's limit) whose text is valid (the checks `prepare_dataset.word_seg` makes), segments
          the text into words with pydips like the existing dataset, and groups videos into shards (a video is decoded once).
* bert:   (GPU or CPU) ModernCantoneseBert-large features per cleaned character (n_chars + 2 rows, `jyutvoice.text.bert_align`), standardised with statistics
          measured on the first `--calibrate-n` texts; written to one float16 memory map so the build workers need no GPU.
* build:  per turn: ffmpeg cut of the separated vocals (24 kHz mono for the stored audio, 16 kHz for the features), CampPlus speaker embedding,
          S3 tokens, flow-encoder `decoder_h`. The 24 kHz audio is trimmed to exactly 2 mel frames (960 samples) per S3 token, so that the target mel
          and `decoder_h` have the same number of frames. Rows are written like the existing dataset (`audio` as 24 kHz PCM16 WAV bytes, `phone` null,
          `text` word-segmented) plus `bert`, `source` and `duration`. Resumable: finished shards are skipped.
* verify: reloads the shards and checks shapes and alignments.

`TextMelDataModule(dataset_path='<out-dir>/shards/shard_*', bert_dim=1024)` reads the shards directly.
"""
import argparse
import functools
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

VALID_PUNCT = "。，、！？；：、…—·「」『』（）"  # prepare_dataset.word_seg's validity rule
MIN_DUR, MAX_DUR = 2.0, 30.0
BERT_DIM = 1024


def plan_dir(a):
    return os.path.join(a.out_dir, "plan")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ----------------------------------------------------------------------------------------------------------------------------- plan


def valid_text(text):
    from jyutvoice.text.cleaners import is_chinese

    return bool(text) and all(is_chinese(c) or c.isspace() or c in VALID_PUNCT for c in text)


def balance_shards(videos, num_shards):
    """videos: [(video, total seconds)] -> list of video lists, greedy by total duration (largest first), so shards take similar time."""
    shards = [[] for _ in range(num_shards)]
    load = [0.0] * num_shards
    for video, seconds in sorted(videos, key=lambda v: -v[1]):
        i = min(range(num_shards), key=load.__getitem__)
        shards[i].append(video)
        load[i] += seconds
    return [sorted(s) for s in shards]


_WS = None


def _init_segmenter():
    global _WS  # noqa: PLW0603
    import torch
    from pydips import BertModel

    torch.set_num_threads(1)
    _WS = BertModel()
    # pydips (bert.cpp) cuts every text with 8 native threads by default: 24 workers made load average 190 on winrex's 56 threads
    _WS.cut_batch = functools.partial(_WS.cut_batch, n_threads=1)


def _segment(text):
    """pydips word segmentation exactly as prepare_dataset.word_seg does it."""
    return " ".join(_WS.cut(text.replace(" ", ""), mode="coarse"))


def plan(a):
    from jyutvoice.text.bert_align import unit_chars
    from jyutvoice.text.cantonese.readings import cleaned

    rows, stats = [], {"records": 0, "not_cantonese": 0, "duration": 0, "invalid_text": 0, "no_audio": 0}
    have_audio = {}
    for path in sorted(glob.glob(a.transcripts)):
        for r in read_jsonl(path):
            stats["records"] += 1
            if r.get("error") or not r.get("text"):
                continue
            if r.get("asr_lang") != "Cantonese":
                stats["not_cantonese"] += 1
            elif not a.min_dur <= r["duration"] <= a.max_dur:
                stats["duration"] += 1
            elif not valid_text(r["text"]):
                stats["invalid_text"] += 1
            else:
                audio = os.path.join(a.vocals_dir, f"{r['video']}_vocal.mp3")
                if r["video"] not in have_audio:
                    have_audio[r["video"]] = os.path.exists(audio)
                if have_audio[r["video"]]:
                    rows.append(r)
                else:
                    stats["no_audio"] += 1
    if a.limit_videos:
        keep = sorted({r["video"] for r in rows})[: a.limit_videos]
        rows = [r for r in rows if r["video"] in set(keep)]
    print(f"{stats}; kept {len(rows)} turns of {len({r['video'] for r in rows})} videos, {sum(r['duration'] for r in rows) / 3600:.1f} h", flush=True)

    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers, initializer=_init_segmenter) as pool:
        segmented = pool.imap(_segment, [r["text"] for r in rows], chunksize=200)
        turns = []
        for k, (r, text) in enumerate(zip(rows, segmented)):
            norm = cleaned(text)
            turns.append({
                "id": f"{r['video']}_{r['turn']:04d}", "video": r["video"], "turn": r["turn"], "start": r["start"], "end": r["end"],
                "duration": r["duration"], "channel": r.get("channel"), "speaker": r.get("speaker"), "text": text,
                "n_units": len(unit_chars(norm)),
            })  # fmt: skip
            if (k + 1) % 20000 == 0:
                print(f"segmented {k + 1}/{len(rows)} ({(k + 1) / (time.time() - t0):.0f}/s)", flush=True)
    turns.sort(key=lambda t: (t["video"], t["turn"]))
    seconds = {}
    for t in turns:
        seconds[t["video"]] = seconds.get(t["video"], 0.0) + t["duration"]
    shards = balance_shards(list(seconds.items()), min(a.num_shards, max(1, len(seconds))))
    os.makedirs(plan_dir(a), exist_ok=True)
    with open(os.path.join(plan_dir(a), "turns.jsonl"), "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    json.dump({"vocals_dir": a.vocals_dir, "shards": shards, "stats": stats, "turns": len(turns), "hours": sum(seconds.values()) / 3600},
              open(os.path.join(plan_dir(a), "plan.json"), "w"), indent=1)
    print(f"plan written: {len(turns)} turns, {sum(seconds.values()) / 3600:.1f} h in {len(shards)} shards ({len(turns) / len(shards):.0f} turns each)")


# ----------------------------------------------------------------------------------------------------------------------------- bert


def bert_dir(a):
    return os.path.join(a.out_dir, "bert")


def bert(a):
    import torch

    from jyutvoice.text.bert_align import BertPhoneFeatures
    from jyutvoice.text.cantonese.readings import cleaned

    turns = read_jsonl(os.path.join(plan_dir(a), "turns.jsonl"))
    norms = [cleaned(t["text"]) for t in turns]
    os.makedirs(bert_dir(a), exist_ok=True)
    if a.fake:  # shapes only, for testing the build without a GPU
        rows = np.array([t["n_units"] + 2 for t in turns])
        offsets = np.concatenate([[0], np.cumsum(rows)])
        out = np.lib.format.open_memmap(os.path.join(bert_dir(a), "bert.npy"), mode="w+", dtype=np.float16, shape=(int(offsets[-1]), BERT_DIM))
        out[:] = np.random.default_rng(0).standard_normal((len(out), BERT_DIM)).astype(np.float16)
        out.flush()
        np.save(os.path.join(bert_dir(a), "offsets.npy"), offsets)
        print(f"FAKE bert features written for {len(turns)} texts ({int(offsets[-1])} rows)")
        return
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[a.dtype]
    device = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(a.threads or os.cpu_count())
    ext = BertPhoneFeatures.from_pretrained(a.model, device=device, dtype=dtype)
    print(f"bert on {device} ({torch.get_num_threads()} threads), {a.dtype}", flush=True)
    t0 = time.time()
    rng = np.random.default_rng(0)
    sample = [norms[i] for i in rng.choice(len(norms), min(a.calibrate_n, len(norms)), replace=False)]
    ext.calibrate(sample, batch_size=a.batch_size)
    np.savez(os.path.join(bert_dir(a), "stats.npz"), mean=ext.mean.numpy(), std=ext.std.numpy())
    print(f"calibrated on {len(sample)} texts in {time.time() - t0:.0f} s; per-dim std min/median/max "
          f"{ext.std.min():.2f}/{ext.std.median():.2f}/{ext.std.max():.0f}", flush=True)
    rows = np.array([t["n_units"] + 2 for t in turns])
    offsets = np.concatenate([[0], np.cumsum(rows)])
    out = np.lib.format.open_memmap(os.path.join(bert_dir(a), "bert.npy"), mode="w+", dtype=np.float16, shape=(int(offsets[-1]), BERT_DIM))
    order = np.argsort([len(n) for n in norms])  # similar lengths together: little padding
    t0 = time.time()
    for k in range(0, len(order), a.batch_size):
        idx = order[k: k + a.batch_size]
        feats = ext.unit_features([norms[i] for i in idx])
        for i, f in zip(idx, feats):
            assert f.shape[0] == rows[i], (turns[i]["id"], f.shape, rows[i])
            out[offsets[i]: offsets[i + 1]] = f.numpy().astype(np.float16)
        if (k // a.batch_size) % 50 == 0:
            print(f"bert {k + len(idx)}/{len(order)} ({(k + len(idx)) / (time.time() - t0):.0f} texts/s)", flush=True)
    out.flush()
    np.save(os.path.join(bert_dir(a), "offsets.npy"), offsets)
    finite = bool(np.isfinite(out[:: max(1, len(out) // 100000)].astype(np.float32)).all())
    json.dump({"model": a.model, "layer": ext.layer, "dtype": a.dtype, "calibrate_n": len(sample), "rows": int(offsets[-1]), "finite_sample": finite},
              open(os.path.join(bert_dir(a), "meta.json"), "w"))
    print(f"bert features done: {len(turns)} texts, {int(offsets[-1])} rows, {time.time() - t0:.0f} s, finite={finite}")


# ----------------------------------------------------------------------------------------------------------------------------- build

from jyutvoice.data import sd3_build as sb  # noqa: E402


def build(a):
    plan_json = json.load(open(os.path.join(plan_dir(a), "plan.json")))
    turns = read_jsonl(os.path.join(plan_dir(a), "turns.jsonl"))
    index_of = {t["id"]: i for i, t in enumerate(turns)}
    by_video = {}
    for t in turns:
        by_video.setdefault(t["video"], []).append(t)
    os.makedirs(os.path.join(a.out_dir, "shards"), exist_ok=True)
    ids = range(len(plan_json["shards"])) if not a.shards else [int(s) for s in a.shards.split(",")]
    tasks = [(i, plan_json["shards"][i], {v: by_video[v] for v in plan_json["shards"][i]}, {t["id"]: index_of[t["id"]] for v in plan_json["shards"][i] for t in by_video[v]},
              plan_json["vocals_dir"], a.out_dir, a.limit_rows) for i in ids]  # fmt: skip
    print(f"{len(tasks)} shards, {a.workers} workers", flush=True)
    t0 = time.time()
    done_rows, failures = 0, {}
    with mp.get_context("fork").Pool(a.workers, initializer=sb.init_worker, initargs=(a.pretrained, a.out_dir)) as pool:
        for k, (sid, n, fail, sec) in enumerate(pool.imap_unordered(sb.build_shard, tasks)):
            done_rows += max(n, 0)
            for key, v in fail.items():
                failures[key] = failures.get(key, 0) + v
            print(f"[{k + 1}/{len(tasks)}] shard {sid}: " + ("already done" if n < 0 else f"{n} rows in {sec:.0f} s") + f"; elapsed {time.time() - t0:.0f} s", flush=True)
    print(f"build finished: {done_rows} rows written in {time.time() - t0:.0f} s; failures: {failures}")


# ----------------------------------------------------------------------------------------------------------------------------- verify


def verify(a):
    from datasets import load_from_disk

    paths = sorted(glob.glob(os.path.join(a.out_dir, "shards", "shard_*")))
    paths = [p for p in paths if not p.endswith(".tmp")]
    plan_json = json.load(open(os.path.join(plan_dir(a), "plan.json")))
    turns = {t["id"]: t for t in read_jsonl(os.path.join(plan_dir(a), "turns.jsonl"))}
    offsets = np.load(os.path.join(bert_dir(a), "offsets.npy"))
    n_rows, seconds, ids, bad = 0, 0.0, set(), []
    for p in paths:
        ds = load_from_disk(p)
        n_rows += len(ds)
        seconds += float(sum(ds["duration"]))
        for k in range(0, len(ds), max(1, len(ds) // a.sample)):
            row = ds[k]
            a24 = row["audio"]["array"]
            h = np.asarray(row["decoder_h"])
            b = np.asarray(row["bert"])
            t = turns[row["id"]]
            problems = []
            if len(a24) != h.shape[0] * 480:
                problems.append(f"audio {len(a24)} samples vs decoder_h {h.shape[0]} frames")
            if h.shape[1] != 80 or len(row["spk_emb"]) != 192 or b.shape != (t["n_units"] + 2, BERT_DIM):
                problems.append(f"shapes decoder_h {h.shape}, spk {len(row['spk_emb'])}, bert {b.shape}")
            if not (np.isfinite(h).all() and np.isfinite(b.astype(np.float32)).all()):
                problems.append("non-finite values")
            if abs(len(a24) / 24000 - t["duration"]) > 0.3:
                problems.append(f"duration {len(a24) / 24000:.2f} vs turn {t['duration']:.2f}")
            if problems:
                bad.append((row["id"], problems))
        ids.update(ds["id"])
    print(f"{len(paths)} of {len(plan_json['shards'])} shards, {n_rows} rows ({len(ids)} distinct ids), {seconds / 3600:.1f} h of {plan_json['hours']:.1f} h planned")
    print(f"checked a sample of rows per shard: {len(bad)} with problems" + ("" if not bad else f"; first: {bad[:3]}"))
    missing = len(turns) - len(ids)
    print(f"turns without a row: {missing} ({100 * missing / max(1, len(turns)):.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--transcripts", required=True, help="glob of transcribe_turns.py jsonl files")
    p.add_argument("--vocals-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-shards", type=int, default=120)
    p.add_argument("--min-dur", type=float, default=MIN_DUR)
    p.add_argument("--max-dur", type=float, default=MAX_DUR)
    p.add_argument("--limit-videos", type=int, default=0, help="testing: only the first N videos")
    p.add_argument("--workers", type=int, default=8, help="processes for the pydips word segmentation")
    p.set_defaults(fn=plan)
    b = sub.add_parser("bert")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--model", required=False, default=None)
    b.add_argument("--fake", action="store_true", help="testing: random features of the right shapes, no model, no GPU")
    b.add_argument("--calibrate-n", type=int, default=4000)
    b.add_argument("--batch-size", type=int, default=128)
    b.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    b.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="cpu is enough: ~60 texts/s on 48 threads in fp32")
    b.add_argument("--threads", type=int, default=0, help="torch threads (default: all cores)")
    b.set_defaults(fn=bert)
    c = sub.add_parser("build")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--pretrained", default="pretrained_models")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--shards", default="", help="comma-separated shard ids (default: all)")
    c.add_argument("--limit-rows", type=int, default=0, help="testing: rows per shard")
    c.set_defaults(fn=build)
    v = sub.add_parser("verify")
    v.add_argument("--out-dir", required=True)
    v.add_argument("--sample", type=int, default=5, help="rows checked per shard")
    v.set_defaults(fn=verify)
    args = ap.parse_args()
    args.fn(args)
