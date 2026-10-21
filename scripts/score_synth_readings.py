"""Does the synthesised audio say the syllables the model was given? Decode synth_compare.py's wavs with a Jyutping recogniser and compare.

    ~/.venvs/w2v-bert-cuda/bin/python scripts/score_synth_readings.py --synth-dir <dir with results.jsonl and wavs/> \\
        --project ~/projects/wav2vec2bert-jyutping --model-dir ~/projects/wav2vec2bert-jyutping/checkpoints/run22_8000

For every utterance the recogniser's syllables are aligned (edit distance) with `intended` (what g2p gave the encoder). Reported per model: syllable error
rate (substitutions + deletions + insertions over the intended syllables), the same without the documented Hong Kong merger pairs (tone 3/6, 4/6, 2/5, onset
n/l, ng/-, gw/g, which the recogniser's own labels blur), the share of substitutions that only change the tone, mean duration and speaking rate.

This measures how faithfully the audio realises the INPUT phones. It says nothing about whether g2p chose the right phones, nor about naturalness, and
the recogniser has its own error (about 0.5% unrelated syllables on clean synthetic speech in an earlier probe).
"""
import argparse
import collections
import json
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import correct_readings as cr  # noqa: E402  (merger_only)
import reading_probe as rp  # noqa: E402  (decode, align, recognised_syllables, swap_kind)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--reuse", action="store_true", help="reuse decoded.jsonl if it exists")
    ap.add_argument("--ignore-ref", default="ei6", help="comma-separated intended syllables left out of the adjusted SER (default: ei6, the filler 誒, whose g2p reading the recogniser never agrees with)")
    ap.add_argument("--referee-dir", help="a scored synth_compare.py dir (results.jsonl + decoded.jsonl) of ONE reference system, e.g. the teacher: adds a column with the SER on only the syllables "
                    "that system was heard to say as intended (a student error there is not the g2p's, the recogniser's or a reading choice's fault)")
    ap.add_argument("--equiv", default="nei5:lei5,ni1:nei1", help="comma-separated a:b syllable pairs the adjusted SER counts as the same either way round (colloquial variants)")
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(os.path.join(a.synth_dir, "results.jsonl"), encoding="utf-8")]
    decoded_path = os.path.join(a.synth_dir, "decoded.jsonl")
    if not (a.reuse and os.path.exists(decoded_path)):
        clips = os.path.join(a.synth_dir, "clips.jsonl")
        with open(clips, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"video": f"{r['model']}/spk{r['spk']}", "turn": r["sid"], "channel": "synth", "duration": r["dur"], "text": r["text"],
                                    "wav": r["wav"]}, ensure_ascii=False) + "\n")  # fmt: skip
        rp.decode(SimpleNamespace(project=a.project, model_dir=a.model_dir, clips=clips, audio_root=a.synth_dir, out=decoded_path))
    decoded = [json.loads(line) for line in open(decoded_path, encoding="utf-8")]
    assert len(decoded) == len(rows)

    ignore_ref = {x for x in a.ignore_ref.split(",") if x}
    equiv = {frozenset(p.split(":")) for p in a.equiv.split(",") if p}
    referee = {}  # sid -> {syllable index: heard syllable or None} of the reference system
    if a.referee_dir:
        for rr, dd in zip((json.loads(line) for line in open(os.path.join(a.referee_dir, "results.jsonl"), encoding="utf-8")),
                          (json.loads(line) for line in open(os.path.join(a.referee_dir, "decoded.jsonl"), encoding="utf-8"))):  # fmt: skip
            rhyp = rp.recognised_syllables(dd["phonemes"])
            referee[rr["sid"]] = {i: (rhyp[j] if j is not None else None) for i, j in rp.align(rr["intended"], rhyp) if i is not None}
    stats = collections.defaultdict(lambda: collections.Counter())
    per_sentence = collections.defaultdict(dict)
    for r, d in zip(rows, decoded):
        hyp = rp.recognised_syllables(d["phonemes"])
        ref = r["intended"]
        for key in ((r["model"], "all"), (r["model"], f"spk{r['spk']}")):
            s = stats[key]
            s["n"] += len(ref)
            s["dur"] += r["dur"]
            s["chars"] += len(ref)
            s["utts"] += 1
        errs = merger_ok = tone_only = adj_errs = ignored = ref_n = ref_errs = 0
        for i, j in rp.align(ref, hyp):
            if referee and i is not None and ref[i] not in ignore_ref and referee[r["sid"]].get(i) == ref[i]:
                ref_n += 1
                ref_errs += j is None or (ref[i] != hyp[j] and frozenset((ref[i], hyp[j])) not in equiv)
            if i is not None and ref[i] in ignore_ref:
                ignored += 1  # left out of the adjusted SER (numerator and denominator)
            elif i is None or j is None or (ref[i] != hyp[j] and frozenset((ref[i], hyp[j])) not in equiv):
                adj_errs += 1
            if i is None or j is None:
                errs += 1
            elif ref[i] != hyp[j]:
                errs += 1
                if cr.merger_only(ref[i], hyp[j]):
                    merger_ok += 1
                if rp.swap_kind(ref[i], hyp[j]) == "tone":
                    tone_only += 1
        for key in ((r["model"], "all"), (r["model"], f"spk{r['spk']}")):
            s = stats[key]
            s["err"] += errs
            s["err_nomerger"] += errs - merger_ok
            s["tone_only"] += tone_only
            s["adj_err"] += adj_errs
            s["adj_n"] += len(ref) - ignored
            s["ignored"] += ignored
            s["ref_n"] += ref_n
            s["ref_err"] += ref_errs
        per_sentence[r["sid"]].setdefault(r["model"], []).append((errs, len(ref)))

    models = sorted({k[0] for k in stats}, key=lambda m: [r["model"] for r in rows].index(m))
    print(f"{len(rows)} utterances decoded with {os.path.basename(a.model_dir.rstrip('/'))}\n")
    print(f"{'model':8s} {'subset':6s} {'utts':>4s} {'syll':>5s} {'SER %':>7s} {'no-merger %':>12s} {'tone-only share':>16s} {'adjusted %':>11s}{' referee-agreed %' if referee else ''} {'sec/utt':>8s} {'syll/s':>7s}")
    for m in models:
        for sub in ["all"] + sorted(k[1] for k in stats if k[0] == m and k[1] != "all"):
            s = stats[(m, sub)]
            ref_cell = f" {100 * s['ref_err'] / max(1, s['ref_n']):10.2f} ({s['ref_n']} syll)" if referee else ""
            print(f"{m:8s} {sub:6s} {s['utts']:4d} {s['n']:5d} {100 * s['err'] / s['n']:7.2f} {100 * s['err_nomerger'] / s['n']:12.2f} "
                  f"{(100 * s['tone_only'] / s['err'] if s['err'] else 0):15.0f}% {100 * s['adj_err'] / max(1, s['adj_n']):11.2f}{ref_cell} {s['dur'] / s['utts']:8.2f} {s['n'] / s['dur']:7.2f}")  # fmt: skip
    print(f"\nadjusted % = SER without the intended syllables {sorted(ignore_ref)} (also out of the denominator) and with {sorted(sorted(p) for p in equiv)} counted as equal")
    print("\nper sentence, errors / syllables summed over speakers:")
    texts = {r["sid"]: r["text"] for r in rows}
    print("  " + "  ".join(f"{m:>10s}" for m in models))
    for sid in sorted(per_sentence):
        cells = []
        for m in models:
            e = sum(x for x, _ in per_sentence[sid].get(m, []))
            n = sum(y for _, y in per_sentence[sid].get(m, []))
            cells.append(f"{e:4d}/{n:<5d}")
        print("  " + "  ".join(f"{c:>10s}" for c in cells), "  ", texts[sid])


if __name__ == "__main__":
    main()
