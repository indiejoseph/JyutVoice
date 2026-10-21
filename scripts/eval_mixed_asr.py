"""Score code-mixed renders of scripts/synth_compare.py with a Qwen3-ASR endpoint (no forced language).

    python scripts/eval_mixed_asr.py --out <synth_compare out dir> --base-url http://127.0.0.1:18000        (run where the endpoint is reachable: winrex)

Per utterance: `english` = share of the English words of the sentence (>= 2 letters, case-insensitive) that the transcript contains, `cer` = character error
rate of the Chinese characters only (insertions and deletions included). The summary is per model, for the code-mixed sentences and for the sentences without
English (the controls: they must not get worse). It tells how intelligible the words are, not how natural they sound.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asr_transcribe_parquet import transcribe  # noqa: E402


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cjk(text):
    return [c for c in text if "一" <= c <= "鿿" or "㐀" <= c <= "䶿"]


def score(target, said):
    words = [w.lower() for w in re.findall(r"[A-Za-z]{2,}", target)]
    low = re.sub(r"[^a-z]", "", said.lower())
    found = [w for w in words if w in low]
    ref = cjk(target)
    return (len(found) / len(words) if words else None), edit_distance(ref, cjk(said)) / max(1, len(ref)), words, found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:18000")
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--json", help="write the per-utterance scores here")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    rows = [json.loads(line) for line in open(os.path.join(out, "results.jsonl"), encoding="utf-8")]
    scored = []
    for r in rows:
        wav = open(os.path.join(out, r["wav"]), "rb").read()
        said = transcribe(a.base_url, a.model, wav, 300) or ""
        eng, cer, words, found = score(r["text"], said)
        scored.append({"model": r["model"], "spk": r["spk"], "sid": r["sid"], "text": r["text"], "said": said, "english": eng, "cer": cer, "missed": [w for w in words if w not in found]})
    if a.json:
        json.dump(scored, open(os.path.expanduser(a.json), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for name in sorted({s["model"] for s in scored}):
        mine = [s for s in scored if s["model"] == name]
        mixed = [s for s in mine if s["english"] is not None]
        plain = [s for s in mine if s["english"] is None]
        n_words = sum(len(re.findall(r"[A-Za-z]{2,}", s["text"])) for s in mixed)
        n_miss = sum(len(s["missed"]) for s in mixed)
        print(f"{name}: code-mixed ({len(mixed)} renders) English words recognised {n_words - n_miss}/{n_words} ({100 * (n_words - n_miss) / max(1, n_words):.0f}%), "
              f"Chinese CER {100 * sum(s['cer'] for s in mixed) / max(1, len(mixed)):.1f}%; controls ({len(plain)} renders) CER "
              f"{100 * sum(s['cer'] for s in plain) / max(1, len(plain)):.1f}%")


if __name__ == "__main__":
    main()
