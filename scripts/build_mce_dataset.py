"""Turn the MCE corpus (AlienKevin/mixed_cantonese_and_english_speech: volunteers reading code-mixed Cantonese/English scripts) into training shards in the
layout `TextMelDataModule` reads. These are the only code-mixed rows there are: the 478 h dataset has no Latin letter in any yue row.

    python scripts/build_mce_dataset.py plan   --parquet '~/data/mce_raw/data/*.parquet' --asr ~/data/mce_asr.jsonl --out-dir D
    python scripts/build_mce_dataset.py build  --out-dir D --pretrained ~/sd3_build/pretrained_models --workers 28
    python scripts/build_mce_dataset.py verify --out-dir D

* plan:   the reference scripts are noisy (wrapping quotes, Simplified / Traditional mixed, typos, `我地`, `d` for 啲), so a row is kept only when the R8 ASR
          transcript (scripts/asr_transcribe_parquet.py, no forced language) agrees with the reference: both go through the text processor (numbers read out,
          OpenCC s2hk) and the character error rate between them must be <= --max-cer. The training text is the ASR transcript (`--text-source asr`: it is what
          was said, 既/嘅 and d/啲 included) or the normalised reference (`ref`). It is cut into words by pydips like the existing dataset, English phrases
          keep their spaces. Prints the CER histogram: calibrate --max-cer from it.
* build:  jyutvoice/data/mce_build.py, one shard per task (a chunk of rows of one parquet file); resumable.
* verify: reloads the shards and checks shapes and alignments.

The rows are lang "yue" with Latin words in the text: `clean_text` sends such a text to the code-mixed g2p. `source` is "mce" (mix it with the other sets by it).
"""
import argparse
import functools
import glob
import json
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

VALID_PUNCT = "。，、！？；：、…—·「」『』（）,.!?'-"  # prepare_dataset.word_seg's marks plus the ASCII ones of English phrases
MIN_DUR, MAX_DUR = 2.0, 30.0
_QUOTES = "\"“”"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def plan_dir(a):
    return os.path.join(a.out_dir, "plan")


# ----------------------------------------------------------------------------------------------------------------------------- text


def unwrap(text):
    """The scripts come wrapped in literal quotes (`"…"`, `"…",`): remove them (and curly ones) anywhere, plus the trailing comma they leave."""
    text = text.strip()
    text = re.sub(r'",?\s*$', "", text) if text.startswith('"') else text
    text = text.translate({ord(c): None for c in _QUOTES})
    if len(text) > 2 and text[0] == "'" and text[-1] == "'":
        text = text[1:-1]
    return text.strip()


def squash(text):
    """The characters that are compared: lower-case ASCII letters and digits and the Chinese characters."""
    return "".join(c for c in text.lower() if ("a" <= c <= "z") or c.isdigit() or "一" <= c <= "鿿" or "㐀" <= c <= "䶿")


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref, hyp):
    return edit_distance(ref, hyp) / max(1, len(ref))


def valid_text(text):
    from jyutvoice.text.cleaners import is_chinese

    return bool(text.strip()) and all(is_chinese(c) or ("a" <= c.lower() <= "z") or c.isspace() or c in VALID_PUNCT for c in text)


_WS = None


def _init_segmenter():
    global _WS  # noqa: PLW0603
    import torch
    from pydips import BertModel

    torch.set_num_threads(1)
    _WS = BertModel()
    _WS.cut_batch = functools.partial(_WS.cut_batch, n_threads=1)  # pydips cuts with 8 native threads by default, see build_sd3_dataset


def segment(text):
    """pydips word segmentation like prepare_dataset.word_seg for the Chinese parts; English phrases stay whole (pydips cuts inside English words)."""
    from jyutvoice.text.text_processor import segment_mixed

    return " ".join(segment_mixed(text, lambda t: _WS.cut(t, mode="coarse")))


def prepare_row(args):
    """Worker of plan: (id, reference, ASR text, max cer, text source) -> dict with the normalised texts, cer, validity and the segmented training text."""
    from jyutvoice.text.text_processor import normalize_text

    cid, ref, asr, max_cer, source = args
    ref_n = normalize_text(unwrap(ref), "yue", simple_to_traditional=True)
    asr_n = normalize_text(unwrap(asr), "yue", simple_to_traditional=True)
    c = cer(squash(ref_n), squash(asr_n))
    text = asr_n if source == "asr" else ref_n
    out = {"id": cid, "ref": ref_n, "asr": asr_n, "cer": round(c, 4), "valid": valid_text(text), "text": None}
    if c <= max_cer and out["valid"]:
        out["text"] = segment(text)
    return out


# ----------------------------------------------------------------------------------------------------------------------------- plan


def plan(a):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.expanduser(a.parquet)))
    refs = {}
    for shard, f in enumerate(files):
        sentences = pq.read_table(f, columns=["sentence"]).column("sentence").to_pylist()
        for row, s in enumerate(sentences):
            refs[f"{a.id_prefix}_{shard:02d}_{row:05d}"] = (shard, row, s or "")
    asr = {}
    for r in read_jsonl(os.path.expanduser(a.asr)):
        asr[r["id"]] = r
    stats = {"clips": len(refs), "no_asr": 0, "duration": 0, "empty": 0, "invalid_text": 0, "high_cer": 0}
    todo = []
    for cid, (shard, row, s) in refs.items():
        r = asr.get(cid)
        if r is None:
            stats["no_asr"] += 1
        elif not a.min_dur <= r["dur"] <= a.max_dur:
            stats["duration"] += 1
        elif not r["asr"].strip() or not unwrap(s):
            stats["empty"] += 1
        else:
            todo.append((cid, s, r["asr"], a.max_cer, a.text_source))
    if a.limit:
        todo = todo[: a.limit]

    from jyutvoice.text.text_processor import normalize_text

    normalize_text("測試3個test", "yue", simple_to_traditional=True)  # compile / load the graphs once here, the forked workers inherit them
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers, initializer=_init_segmenter) as pool:
        out = []
        for k, o in enumerate(pool.imap(prepare_row, todo, chunksize=50)):
            out.append(o)
            if (k + 1) % 2000 == 0:
                print(f"prepared {k + 1}/{len(todo)} ({(k + 1) / (time.time() - t0):.0f}/s)", flush=True)
    cers = np.array([o["cer"] for o in out])
    edges = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1e9]
    hist = np.histogram(cers, bins=edges)[0]
    print("cer histogram (all rows with a transcript): " + ", ".join(f"<{e:g}: {h}" for e, h in zip(edges[1:], hist)), flush=True)

    kept = []
    for o in out:
        if not o["valid"]:
            stats["invalid_text"] += 1
        elif o["cer"] > a.max_cer:
            stats["high_cer"] += 1
        else:
            kept.append(o)
    turns = []
    for o in kept:
        shard, row, _ = refs[o["id"]]
        dur = asr[o["id"]]["dur"]
        words = o["text"].split()
        turns.append({"id": o["id"], "shard": shard, "row": row, "text": o["text"], "ref": o["ref"], "asr": o["asr"], "cer": o["cer"], "duration": dur,
                      "n_english": sum(any("a" <= c.lower() <= "z" for c in w) for w in words)})  # fmt: skip
    turns.sort(key=lambda t: (t["shard"], t["row"]))
    hours = sum(t["duration"] for t in turns) / 3600
    with_en = sum(t["n_english"] > 0 for t in turns)
    print(f"{stats}; kept {len(turns)} clips, {hours:.1f} h, {with_en} with English ({sum(t['n_english'] for t in turns)} English words)", flush=True)

    tasks = []  # (parquet file, [turn ids]) chunks of consecutive kept rows
    by_shard = {}
    for t in turns:
        by_shard.setdefault(t["shard"], []).append(t["id"])
    for shard, ids in sorted(by_shard.items()):
        for k in range(0, len(ids), a.rows_per_shard):
            tasks.append({"parquet": files[shard], "ids": ids[k : k + a.rows_per_shard]})
    os.makedirs(plan_dir(a), exist_ok=True)
    with open(os.path.join(plan_dir(a), "turns.jsonl"), "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    json.dump({"tasks": tasks, "stats": stats, "turns": len(turns), "hours": hours, "max_cer": a.max_cer, "text_source": a.text_source},
              open(os.path.join(plan_dir(a), "plan.json"), "w"), indent=1)
    print(f"plan written: {len(turns)} clips in {len(tasks)} shards")


# ----------------------------------------------------------------------------------------------------------------------------- build

from jyutvoice.data import mce_build as mb  # noqa: E402


def build(a):
    plan_json = json.load(open(os.path.join(plan_dir(a), "plan.json")))
    turns = {t["id"]: t for t in read_jsonl(os.path.join(plan_dir(a), "turns.jsonl"))}
    os.makedirs(os.path.join(a.out_dir, "shards"), exist_ok=True)
    ids = range(len(plan_json["tasks"])) if not a.shards else [int(s) for s in a.shards.split(",")]
    tasks = []
    for i in ids:
        t = plan_json["tasks"][i]
        tasks.append((i, t["parquet"], {turns[c]["row"]: {"id": c, "text": turns[c]["text"]} for c in t["ids"]}, a.out_dir, a.limit_rows))
    print(f"{len(tasks)} shards, {a.workers} workers", flush=True)
    t0 = time.time()
    done_rows, failures = 0, {}
    with mp.get_context("fork").Pool(a.workers, initializer=mb.init_worker, initargs=(os.path.expanduser(a.pretrained),)) as pool:
        for k, (sid, n, fail, sec) in enumerate(pool.imap_unordered(mb.build_shard, tasks)):
            done_rows += max(n, 0)
            for key, v in fail.items():
                failures[key] = failures.get(key, 0) + v
            print(f"[{k + 1}/{len(tasks)}] shard {sid}: " + ("already done" if n < 0 else f"{n} rows in {sec:.0f} s") + f"; elapsed {time.time() - t0:.0f} s", flush=True)
    print(f"build finished: {done_rows} rows written in {time.time() - t0:.0f} s; failures: {failures}")


# ----------------------------------------------------------------------------------------------------------------------------- verify


def verify(a):
    from datasets import load_from_disk

    paths = [p for p in sorted(glob.glob(os.path.join(a.out_dir, "shards", "shard_*"))) if not p.endswith(".tmp")]
    plan_json = json.load(open(os.path.join(plan_dir(a), "plan.json")))
    turns = {t["id"]: t for t in read_jsonl(os.path.join(plan_dir(a), "turns.jsonl"))}
    n_rows, seconds, ids, bad = 0, 0.0, set(), []
    for p in paths:
        ds = load_from_disk(p)
        n_rows += len(ds)
        seconds += float(sum(ds["duration"]))
        for k in range(0, len(ds), max(1, len(ds) // a.sample)):
            row = ds[k]
            a24 = row["audio"]["array"]
            h = np.asarray(row["decoder_h"])
            t = turns[row["id"]]
            problems = []
            if len(a24) != h.shape[0] * 480:
                problems.append(f"audio {len(a24)} samples vs decoder_h {h.shape[0]} frames")
            if h.shape[1] != 80 or len(row["spk_emb"]) != 192:
                problems.append(f"shapes decoder_h {h.shape}, spk {len(row['spk_emb'])}")
            if not np.isfinite(h).all():
                problems.append("non-finite values")
            if row["text"] != t["text"] or row["lang"] != "yue" or row["source"] != "mce":
                problems.append("text / lang / source differ from the plan")
            if abs(len(a24) / 24000 - t["duration"]) > 0.3:
                problems.append(f"duration {len(a24) / 24000:.2f} vs clip {t['duration']:.2f}")
            if problems:
                bad.append((row["id"], problems))
        ids.update(ds["id"])
    print(f"{len(paths)} of {len(plan_json['tasks'])} shards, {n_rows} rows ({len(ids)} distinct ids), {seconds / 3600:.1f} h of {plan_json['hours']:.1f} h planned")
    print(f"checked a sample of rows per shard: {len(bad)} with problems" + ("" if not bad else f"; first: {bad[:3]}"))
    missing = len(turns) - len(ids)
    print(f"clips without a row: {missing} ({100 * missing / max(1, len(turns)):.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--parquet", required=True, help="glob of the HF parquet shards (audio, sentence columns)")
    p.add_argument("--asr", required=True, help="asr_transcribe_parquet.py output")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--id-prefix", default="mce")
    p.add_argument("--max-cer", type=float, default=0.15)
    p.add_argument("--text-source", choices=["asr", "ref"], default="asr")
    p.add_argument("--rows-per-shard", type=int, default=200)
    p.add_argument("--min-dur", type=float, default=MIN_DUR)
    p.add_argument("--max-dur", type=float, default=MAX_DUR)
    p.add_argument("--limit", type=int, default=0, help="testing: only the first N clips with a transcript")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(fn=plan)
    b = sub.add_parser("build")
    b.add_argument("--out-dir", required=True)
    b.add_argument("--pretrained", default="pretrained_models")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--shards", default="", help="comma-separated shard ids (default: all)")
    b.add_argument("--limit-rows", type=int, default=0, help="testing: rows per shard")
    b.set_defaults(fn=build)
    v = sub.add_parser("verify")
    v.add_argument("--out-dir", required=True)
    v.add_argument("--sample", type=int, default=5, help="rows checked per shard")
    v.set_defaults(fn=verify)
    args = ap.parse_args()
    args.fn(args)
