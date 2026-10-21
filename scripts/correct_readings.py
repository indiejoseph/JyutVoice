"""Reading overrides from Jyutping recognisers: where models that listened to the audio agree on a different syllable than g2p gives.

g2p (ToJyutping + colloquial.py) reads many words differently from how they are spoken: 分 in 百分之 (fan6, not fan1), 行 in 銀行
(hong4), 長 in 局長 (zoeng2), 樣 in 點樣 (joeng2) ... The student is trained on phones made from g2p's readings, so for those positions it sees
phones that do not match the audio. For clips with audio (teacher output, real speech) this script collects the syllables decoded by several
checkpoints of the wav2vec2bert-jyutping project and keeps a correction only when enough models agree on it.

    # 1. decode the clips with each model (scripts/reading_probe.py decode, in the recogniser project's environment)
    # 2. merge:
    python scripts/correct_readings.py merge --clips clips.jsonl \\
        --decode run22=probe_run22.jsonl --decode run21=probe_run21.jsonl --decode ckpt3048=probe_3048.jsonl \\
        --out corrected.jsonl --rules rules.tsv

Output, one JSON line per clip: `norm_text` (the text after cleaners.replace_punctuation, spaces removed; unit i is its i-th character) and
`overrides` {unit index: syllable}. The dataset builder turns them into the `phone` column with
`jyutvoice.text.cantonese.readings.jyutping_with_overrides(segmented_text, overrides)`, which `g2p(jyutping=...)` then consumes.

A correction is kept when (a) the model's decoded syllable is aligned to the character's g2p syllable and shares its onset or rime, (b) at least
`--min-agree` models decoded the same different syllable, and (c) if the two differ only by a documented Hong Kong merger (tone 3/6, 4/6, 2/5,
onset n/l, ng/-, gw/g) all models agree: the recognisers' labels are canonical readings, so those pairs are where they are least reliable.
Two more filters keep precision above recall: an agreed correction that only undoes a colloquial.py reading is not applied (the recognisers
print base readings and base tones and cannot tell), and a (character, g2p reading, heard reading) correction must cover at least `--min-rate`
of the character's occurrences in the processed corpus and occur `--min-count` times (variation and recogniser noise are scattered;
g2p errors are systematic).

The segmenter is pydips when installed (genkai's ~/venv/jc has it), as in the dataset build; without it colloquial.py's single-character-word
rules fire inside compounds and the results are biased.
"""
import argparse
import ast
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import reading_probe as rp  # noqa: E402

TONE_MERGERS = [{3, 6}, {4, 6}, {2, 5}]
ONSET_MERGERS = [{"n", "l"}, {"ng", ""}, {"gw", "g"}]


def _literals(path, names):
    """Module-level literal assignments of a source file, read with `ast` (importing jyutvoice.text pulls in the whole g2p stack)."""
    out = {}
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


REP_MAP = _literals(os.path.join(ROOT, "jyutvoice", "text", "cleaners.py"), {"rep_map"})["rep_map"]
PUNCTUATIONS = _literals(os.path.join(ROOT, "jyutvoice", "text", "symbols.py"), {"punctuations"})["punctuations"]
_REP = re.compile("|".join(re.escape(p) for p in REP_MAP))


def clean(text):
    """`cleaners.replace_punctuation(text, 'yue')` with spaces removed: the characters g2p sees, one unit each."""
    text = _REP.sub(lambda m: REP_MAP[m.group()], text)
    return "".join(c for c in text if (rp.HAN.match(c) or c in PUNCTUATIONS) and not c.isspace())


def parse(syllable):
    m = rp.SYL.match(syllable)
    return (m.group(1) or "", m.group(2), int(m.group(3))) if m else None


def merger_only(a, b):
    """True when the two syllables differ only by one of the documented Hong Kong mergers."""
    pa, pb = parse(a), parse(b)
    if not pa or not pb:
        return False
    (oa, ra, ta), (ob, rb, tb) = pa, pb
    if oa == ob and ra == rb and {ta, tb} in TONE_MERGERS:
        return True
    return ra == rb and ta == tb and {oa, ob} in ONSET_MERGERS


def decide(default, votes, min_agree=2, merger_agree=None):
    """The syllable to use instead of `default`, or None. `votes`: {model: decoded syllable} for the models that aligned a syllable to this
    character (a model that agrees with g2p votes for `default`). Returns (syllable, number of models that decoded it)."""
    if not votes:
        return None
    ranked = collections.Counter(votes.values()).most_common()
    heard, n = ranked[0]
    if heard == default or n < min_agree or (len(ranked) > 1 and ranked[1][1] == n):
        return None
    if merger_only(default, heard) and n < (merger_agree or len(votes)):
        return None
    return heard, n


def clip_overrides(text, decodes, min_agree=2, merger_agree=None, segmenter=None, keep_colloquial=True):
    """(norm_text, {unit: syllable}, changes, the clip's Chinese characters, reverted) for one clip. `decodes`: {model: decoded phoneme
    string}. `reverted` lists agreed corrections that were not applied because they only undo a colloquial.py reading (the recognisers print
    base readings and base tones, so they cannot tell a colloquial reading from a g2p error); `keep_colloquial=False` applies them."""
    norm = clean(text)
    units = rp.text_to_syllables(text, True, segmenter, all_units=True, clean=clean)
    assert "".join(c for c, _ in units) == norm, (units, norm)
    base = rp.text_to_syllables(text, False, segmenter, all_units=True, clean=clean)
    han = [(u, c, r) for u, (c, r) in enumerate(units) if rp.HAN.match(c) and r != "?"]
    votes = collections.defaultdict(dict)
    for model, phonemes in decodes.items():
        hyp = rp.recognised_syllables(phonemes)
        for i, j in rp.align([r for _, _, r in han], hyp):
            if i is None or j is None:
                continue
            u, _, r = han[i]
            if hyp[j] == r or rp.swap_kind(r, hyp[j]) != "unrelated":  # unrelated = alignment slip or recogniser error, no vote
                votes[u][model] = hyp[j]
    overrides, changes, reverted = {}, [], []
    for u, c, r in han:
        d = decide(r, votes.get(u, {}), min_agree, merger_agree)
        if not d:
            continue
        change = {
            "idx": u, "char": c, "prev": units[u - 1][0] if u else "", "next": units[u + 1][0] if u + 1 < len(units) else "",
            "default": r, "heard": d[0], "agree": d[1], "votes": votes[u], "merger": merger_only(r, d[0]),
        }  # fmt: skip
        if keep_colloquial and r != base[u][1] and d[0] == base[u][1]:
            reverted.append(change)
            continue
        overrides[u] = d[0]
        changes.append(change)
    return norm, overrides, changes, [c for _, c, _ in han], reverted


def merge(a):
    clips = [json.loads(line) for line in open(a.clips)]
    decodes = {}
    for spec in a.decode:
        name, path = spec.split("=", 1)
        rows = [json.loads(line) for line in open(path)]
        if len(rows) != len(clips) or any((r["video"], r["turn"]) != (c["video"], c["turn"]) for r, c in zip(rows, clips)):
            raise SystemExit(f"{path}: not the same clips, in the same order, as {a.clips}")
        decodes[name] = rows
    names = list(decodes)
    segmenter = rp.get_segmenter()
    results = []
    for k, c in enumerate(clips):
        results.append(clip_overrides(c["text"], {m: decodes[m][k]["phonemes"] for m in names}, a.min_agree, a.merger_agree, segmenter,
                                      not a.apply_colloquial_reversions))  # fmt: skip
    chars = collections.Counter(ch for r in results for ch in r[3])
    pairs = collections.Counter((ch["char"], ch["default"], ch["heard"]) for r in results for ch in r[2])
    # corpus-level consistency: a character whose g2p reading is contradicted only now and then is variation or recogniser noise;
    # one contradicted for a large share of its occurrences is a systematic g2p error
    keep = {p for p, n in pairs.items() if n >= a.min_count and n / chars[p[0]] >= a.min_rate}
    n_syl = sum(len(r[3]) for r in results)
    n_agreed = sum(len(r[2]) for r in results)
    n_reverted = sum(len(r[4]) for r in results)
    n_changed = n_clips_changed = 0
    kinds, agree, contexts = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
    final = collections.Counter()
    with open(a.out, "w") as out:
        for c, (norm, overrides, changes, han, reverted) in zip(clips, results):
            changes = [ch for ch in changes if (ch["char"], ch["default"], ch["heard"]) in keep]
            overrides = {ch["idx"]: ch["heard"] for ch in changes}
            out.write(json.dumps({**{x: c[x] for x in ("video", "turn", "channel", "text") if x in c}, "norm_text": norm, "n_syllables": len(han),
                                  "overrides": {str(u): s for u, s in overrides.items()}, "changes": changes}, ensure_ascii=False) + "\n")  # fmt: skip
            n_changed += len(changes)
            n_clips_changed += bool(changes)
            for ch in changes:
                final[(ch["char"], ch["default"], ch["heard"])] += 1
                contexts[(ch["char"], ch["default"], ch["heard"])][ch["prev"] + "_" + ch["next"]] += 1
                kinds["merger-pair (unanimous)" if ch["merger"] else rp.swap_kind(ch["default"], ch["heard"])] += 1
                agree[ch["agree"]] += 1
    print(f"{len(clips)} clips, {n_syl} syllables; models: {', '.join(names)}; segmenter: {'pydips' if segmenter else 'lexicon stand-in'}")
    print(f"  agreed by >= {a.min_agree} models: {n_agreed} syllables ({100 * n_agreed / max(1, n_syl):.1f}%); not applied because they only undo a colloquial.py "
          f"reading: {n_reverted}")
    print(f"  kept after the corpus filter (>= {a.min_count} occurrences and >= {a.min_rate:.0%} of the character's occurrences): "
          f"{n_changed} syllables ({100 * n_changed / max(1, n_syl):.1f}%) in {n_clips_changed} clips ({100 * n_clips_changed / len(clips):.0f}%)")
    print(f"  kept, by kind: {dict(kinds)}; by number of agreeing models: {dict(sorted(agree.items()))}")
    per_char = collections.Counter()
    for (ch, _, _), v in final.items():
        per_char[ch] += v
    print("  kept corrections per character (corrections / occurrences: default -> heard x count):")
    for ch, v in per_char.most_common(a.top):
        top = [(f"{d}->{h}", n) for (c2, d, h), n in final.most_common() if c2 == ch][:2]
        print(f"    {ch}  {v:4d} / {chars[ch]:4d}  {top}")
    dropped = collections.Counter({p: n for p, n in pairs.items() if p not in keep})
    print(f"  dropped by the corpus filter: {sum(dropped.values())} syllables over {len(dropped)} (character, reading) pairs; largest: "
          + ", ".join(f"{c}:{d}->{h} {n}/{chars[c]}" for (c, d, h), n in dropped.most_common(8)))
    if a.rules:
        with open(a.rules, "w") as f:
            f.write("char\tdefault\theard\tcount\toccurrences_of_char\ttop_contexts(prev_next)\n")
            for (ch, d, h), n in final.most_common():
                ctx = ",".join(f"{k}x{v}" for k, v in contexts[(ch, d, h)].most_common(5))
                f.write(f"{ch}\t{d}\t{h}\t{n}\t{chars[ch]}\t{ctx}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--clips", required=True, help="clips.jsonl (video, turn, text, ...): the order every --decode file follows")
    m.add_argument("--decode", action="append", required=True, help="name=path of a decode (reading_probe.py decode output); repeat per model")
    m.add_argument("--out", required=True)
    m.add_argument("--rules", help="also write a tsv of (character, g2p, heard, count, contexts)")
    m.add_argument("--min-agree", type=int, default=2)
    m.add_argument("--merger-agree", type=int, default=None, help="models that must agree on a merger-pair difference (default: all)")
    m.add_argument("--min-count", type=int, default=5, help="corpus filter: occurrences of a (character, g2p, heard) correction")
    m.add_argument("--min-rate", type=float, default=0.25, help="corpus filter: share of the character's occurrences it covers")
    m.add_argument("--apply-colloquial-reversions", action="store_true", help="also apply agreed corrections that only undo a colloquial.py reading")
    m.add_argument("--top", type=int, default=25)
    m.set_defaults(fn=merge)
    args = ap.parse_args()
    args.fn(args)
