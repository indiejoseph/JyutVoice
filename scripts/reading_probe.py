"""Measure how often the front end's Jyutping (ToJyutping + colloquial.py) disagrees with what a speaker actually said.

Two steps, because the audio and the recogniser live on winrex while the front end runs wherever JyutVoice is installed:

    # on winrex: sample turns, cut the audio, send it to the wav2vec2-bert Jyutping service (127.0.0.1:8091)
    python scripts/reading_probe.py collect --records 'out/full_nonnews/shard*.jsonl' --audio-root full/vocals \
        --n 300 --out probe.jsonl
    python scripts/reading_probe.py collect --manifest runs/listen_len/manifest.tsv --wav-root runs/listen_len/wavs \
        --out teacher.jsonl                      # teacher_bench outputs instead of real speech (add --append for more runs)

    # anywhere with ToJyutping: align g2p syllables to the recognised ones and report the mismatches
    python scripts/reading_probe.py compare --probe probe.jsonl [--no-colloquial] [--dump mismatches.tsv]

`collect` needs only the standard library and ffmpeg. `compare` aligns the two syllable sequences with edit distance, so the
numbers are per syllable: how many g2p syllables were confirmed, replaced (a different reading or a different word), or missing.

    # with a local checkpoint of the wav2vec2bert-jyutping project instead of the service (needs that project's CUDA/CPU environment):
    ~/.venvs/w2v-bert-cuda/bin/python scripts/reading_probe.py decode --project ~/projects/wav2vec2bert-jyutping \
        --model-dir ~/projects/wav2vec2bert-jyutping/checkpoints/run22_8000 --clips clips.jsonl --audio-root <dir> --out probe_run22.jsonl
    # clips.jsonl: one {video, turn, channel, duration, text, wav} per line, `wav` relative to --audio-root
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import subprocess
import sys
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9]")


def cut_wav(path, start, dur):
    """16 kHz mono wav bytes of [start, start+dur) from an audio file."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "-"]
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def post_wav(url, wav):
    """multipart/form-data POST with a single `file` field (the service's /phoneme contract)."""
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode()
    body += wav + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def collect_manifest(a):
    """Teacher outputs: teacher_bench's manifest.tsv (columns id, mode, audio_s, wav, text) and its wavs/<mode>/<wav> files."""
    rows = [dict(zip(h := open(a.manifest).readline().rstrip("\n").split("\t"), l.rstrip("\n").split("\t"))) for l in open(a.manifest).readlines()[1:]]
    tag = os.path.basename(os.path.dirname(os.path.abspath(a.manifest)))

    def one(r):
        base = {"video": f"{tag}/{r['id']}", "turn": r["mode"], "channel": "teacher", "duration": float(r["audio_s"]), "text": r["text"]}
        if a.no_latin and LATIN_OR_DIGIT.search(r["text"]):
            return None
        try:
            res = post_wav(a.url, open(os.path.join(a.wav_root, r["mode"], r["wav"]), "rb").read())
        except Exception as e:  # noqa: BLE001
            return {**base, "error": repr(e)}
        return {**base, "phonemes": res["phonemes"], "rtf": res.get("rtf")}

    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a" if a.append else "w") as out:
        for r in ex.map(one, rows):
            if r:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")


def collect(a):
    if a.manifest:
        return collect_manifest(a)
    rng = random.Random(a.seed)
    pool = []
    for f in sorted(glob.glob(a.records)):
        for line in open(f):
            d = json.loads(line)
            t = d.get("text") or ""
            if d.get("asr_lang") != "Cantonese" or not t or not a.min_dur <= d["duration"] <= a.max_dur:
                continue
            if a.no_latin and LATIN_OR_DIGIT.search(t):
                continue
            pool.append(d)
    rng.shuffle(pool)
    pool = pool[: a.n]
    print(f"{len(pool)} turns sampled", file=sys.stderr)

    def one(d):
        path = os.path.join(a.audio_root, f"{d['video']}_vocal.mp3")
        try:
            res = post_wav(a.url, cut_wav(path, d["start"], d["duration"]))
        except Exception as e:  # noqa: BLE001
            return {**{k: d[k] for k in ("video", "turn", "channel", "duration", "text")}, "error": repr(e)}
        return {**{k: d[k] for k in ("video", "turn", "channel", "duration", "text")}, "phonemes": res["phonemes"], "rtf": res.get("rtf")}

    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "w") as out:
        for i, r in enumerate(ex.map(one, pool)):
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(pool)}", file=sys.stderr)


def decode(a):
    """Decode clips with a local checkpoint of the wav2vec2bert-jyutping project (its own `make_predictor`), same output format as `collect`."""
    sys.path.insert(0, a.project)
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    import numpy as np
    import soundfile as sf
    from error_consistency import make_predictor

    predict = make_predictor("ours", a.model_dir)
    with open(a.out, "w") as out:
        for i, line in enumerate(open(a.clips)):
            r = json.loads(line)
            y, rate = sf.read(os.path.join(a.audio_root, r["wav"]), dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if rate != 16000:
                import librosa

                y = librosa.resample(y, orig_sr=rate, target_sr=16000)
            rec = {k: r[k] for k in ("video", "turn", "channel", "duration", "text")}
            rec["phonemes"] = " ".join(predict(y.astype(np.float32)))  # toned syllables, every token ends with its tone digit
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"{i + 1} clips", file=sys.stderr)


# ---------------------------------------------------------------------------------------------------------------- compare


def align(ref, hyp):
    """Levenshtein alignment of two token lists -> list of (ref_index|None, hyp_index|None)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


def load_colloquial():
    """colloquial.py by file path: importing it through the jyutvoice package pulls in pycantonese and the whole text frontend, and the
    module itself only needs `re`."""
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jyutvoice", "text", "cantonese", "colloquial.py")
    spec = importlib.util.spec_from_file_location("colloquial", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PUNCT_MAP = str.maketrans({"，": ",", "。": ".", "？": "?", "！": "!", "、": ",", "：": ",", "；": ",", "…": "…", "—": "-"})
HAN = re.compile(r"[\u3400-\u9fff]")


def lexicon_segmenter(colloquial):
    """Stand-in for the pydips word segmenter when it is not installed: one word per character, except the multi-character words in
    colloquial.TOKENS. Single-character words are what the colloquial rules test (tok[i] == ch), so this errs on the side of letting the
    rules fire; names of 3+ characters (the 無綫電視 exception) will not be detected."""
    multi = sorted((w for w in colloquial.TOKENS if len(w) > 1), key=len, reverse=True)

    def cut(text):
        out, i = [], 0
        while i < len(text):
            for w in multi:
                if text.startswith(w, i):
                    out.append(w)
                    i += len(w)
                    break
            else:
                out.append(text[i])
                i += 1
        return out

    return cut


def get_segmenter(verbose=True):
    """The pydips word segmenter the dataset is built with (`ws_model.cut(text, mode="coarse")`), or None when it is not installed.
    The lexicon stand-in treats every character as its own word, which lets colloquial.py's single-character-word rules fire inside
    compounds (房屋, 聲明), so numbers about colloquial.py are only trustworthy with the real segmenter."""
    try:
        from pydips import BertModel

        ws = BertModel()
        return lambda t: ws.cut(t.replace(" ", ""), mode="coarse")
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"pydips unavailable ({type(e).__name__}): using the lexicon stand-in segmenter, colloquial results are biased", file=sys.stderr)
        return None


def text_to_syllables(text, use_colloquial, segmenter=None, all_units=False, clean=None):
    """[(char, jyutping)] for every Chinese character, built the way jyutvoice/text/cantonese/g2p.py builds it: punctuation is normalised to
    the ASCII set, each segmented word gets one entry per character (punctuation stays as itself), then colloquial.apply corrects readings.

    `all_units=True` returns every character of the text, punctuation included (its reading is the mark itself) and "?" where a joint
    reading of a multi-character chunk cannot be split; the list index is then the unit index of the cleaned text without spaces.

    The raw `text` is segmented first and every word cleaned afterwards (`clean`, default: punctuation to the ASCII set), the order the
    dataset is built in (word_seg on the raw text, then clean_text per word)."""
    import ToJyutping

    colloquial = load_colloquial()
    clean = clean or (lambda w: w.translate(PUNCT_MAP))
    words = [w for w in (clean(x) for x in (segmenter or lexicon_segmenter(colloquial))(text)) if w]
    chars_per_word, read_per_word = [], []
    for w in words:
        chars, reads = [], []
        for chunk, syl in ToJyutping.get_jyutping_list(w):
            if syl is None:
                chars.extend(chunk)
                reads.extend(chunk)  # punctuation / unknown: the character itself, as g2p.get_jyutping does
            elif len(syl.split()) == len(chunk):
                chars.extend(chunk)
                reads.extend(syl.split())
            else:  # a multi-character chunk with a joint reading: cannot attribute syllables to characters
                chars.extend(chunk)
                reads.extend(["?"] * len(chunk))
        assert "".join(chars) == w, (chars, w)
        chars_per_word.append(chars)
        read_per_word.append(reads)
    if use_colloquial and colloquial.COLLOQUIAL_READINGS:
        read_per_word = colloquial.apply(["".join(c) for c in chars_per_word], read_per_word)
    pairs = [(c, r) for cs, rs in zip(chars_per_word, read_per_word) for c, r in zip(cs, rs)]
    return pairs if all_units else [(c, r) for c, r in pairs if HAN.match(c) and r != "?"]


SYL = re.compile(r"^(ng|gw|kw|[bpmfdtnlgkhwzcsj])?([a-z]+?)([1-6])$")


def swap_kind(g2p, heard):
    """How a contradicted syllable differs: 'tone' (same onset+rime), 'onset', 'rime', or 'unrelated' (nothing shared; almost always an
    alignment slip, a recogniser error or an ASR text error rather than a different reading of the same character)."""
    a, b = SYL.match(g2p), SYL.match(heard)
    if not a or not b:
        return "unrelated"
    (ao, ar, _), (bo, br, _) = a.groups(), b.groups()
    if ao == bo and ar == br:
        return "tone"
    if ar == br:
        return "onset"
    if ao == bo:
        return "rime"
    return "unrelated"


def recognised_syllables(phonemes):
    """The service writes onset / rime+tone tokens separated by spaces (e.g. 't eng1 d aai6'); a syllable closes at every token that ends
    with a tone digit. CTC repeats ('j j au5') are collapsed."""
    syl, cur = [], []
    for tok in phonemes.split():
        if not (cur and cur[-1] == tok and not tok[-1].isdigit()):
            cur.append(tok)
        if tok[-1].isdigit():
            syl.append("".join(cur))
            cur = []
    return syl


def compare(a):
    rows = [json.loads(l) for l in open(a.probe)]
    rows = [r for r in rows if "phonemes" in r]
    seg = get_segmenter()
    tot = collections.Counter()
    ab = collections.defaultdict(collections.Counter)  # char -> {"gain": rule fixed it, "loss": rule broke it, "neither": changed, still wrong}
    by_char = collections.defaultdict(collections.Counter)  # char -> {"match": n, "swap": n}
    swaps = collections.Counter()  # (char, g2p, heard)
    tsv = []
    for r in rows:
        ref = text_to_syllables(r["text"], not a.no_colloquial, seg)
        hyp = recognised_syllables(r["phonemes"])
        tot["turns"] += 1
        if not a.no_colloquial:  # which syllables did colloquial.py change, and did the change agree with the recogniser?
            base = text_to_syllables(r["text"], False, seg)
            hit_on = {i: hyp[j] == ref[i][1] for i, j in align([s for _, s in ref], hyp) if i is not None and j is not None}
            hit_off = {i: hyp[j] == base[i][1] for i, j in align([s for _, s in base], hyp) if i is not None and j is not None}
            for i, (ch, rc) in enumerate(ref):
                if rc != base[i][1] and i in hit_on and i in hit_off:
                    ab[ch]["gain" if hit_on[i] and not hit_off[i] else "loss" if hit_off[i] and not hit_on[i] else "neither"] += 1
        for i, j in align([s for _, s in ref], hyp):
            if i is not None and j is not None:
                ch, rc = ref[i]
                hc = hyp[j]
                if rc == hc:
                    tot["match"] += 1
                    by_char[ch]["match"] += 1
                else:
                    kind = swap_kind(rc, hc)
                    tot["swap"] += 1
                    tot["swap_" + kind] += 1
                    if kind != "unrelated":
                        by_char[ch]["swap"] += 1
                        swaps[(ch, rc, hc)] += 1
                    else:
                        by_char[ch]["match"] += 0  # keep the char listed; unrelated swaps are not counted as reading evidence
                    tsv.append((r["video"], r["turn"], ch, rc, hc, kind))
            elif i is not None:
                tot["g2p_only"] += 1  # the recogniser heard nothing for this syllable (often fillers it skips, e.g. 啊)
            else:
                tot["recogniser_only"] += 1
    n_ref = tot["match"] + tot["swap"] + tot["g2p_only"]
    print(f"{tot['turns']} turns, {n_ref} g2p syllables" + ("  [colloquial rules OFF]" if a.no_colloquial else ""))
    for k in ("match", "swap", "g2p_only", "recogniser_only"):
        print(f"  {k:16s} {tot[k]:7d}  {100 * tot[k] / max(1, n_ref):5.1f}% of g2p syllables")
    paired = tot["match"] + tot["swap"]
    print(f"  among {paired} paired syllables {100 * tot['swap'] / max(1, paired):.1f}% differ: " + ", ".join(
        f"{k} {100 * tot['swap_' + k] / max(1, paired):.1f}%" for k in ("tone", "onset", "rime", "unrelated")))
    close = paired - tot["swap_unrelated"]
    base = (tot["swap"] - tot["swap_unrelated"]) / max(1, close)
    print(f"\ncharacters whose reading is contradicted, ranked by rate x sqrt(count); the average is {100 * base:.1f}% average (>= {a.min_count} paired):")
    rank = [(ch, c["swap"] / (c["swap"] + c["match"]), c["swap"] + c["match"]) for ch, c in by_char.items() if c["swap"] + c["match"] >= a.min_count]
    print("(unrelated swaps are left out of these per-character rates)")
    for ch, rate, n in sorted(rank, key=lambda x: -x[1] * x[2] ** 0.5)[:30]:
        top = collections.Counter({(rc, hc): c for (cc, rc, hc), c in swaps.items() if cc == ch}).most_common(2)
        print(f"  {ch}  {100 * rate:5.1f}% of {n:4d}   g2p->heard {top}")
    if ab:
        g = sum(c["gain"] for c in ab.values())
        l = sum(c["loss"] for c in ab.values())
        n = sum(c["neither"] for c in ab.values())
        print(f"\ncolloquial.py changed {g + l + n} syllables: {g} now agree with the recogniser, {l} no longer agree, {n} unchanged verdict (wrong both ways)")
        for ch, c in sorted(ab.items(), key=lambda kv: -sum(kv[1].values()))[:12]:
            print(f"  {ch}  gain {c['gain']:3d}  loss {c['loss']:3d}  neither {c['neither']:3d}")
    if a.dump:
        with open(a.dump, "w") as f:
            for row in tsv:
                f.write("\t".join(map(str, row)) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--records", help="glob of transcribe_turns.py jsonl files (real speech)")
    c.add_argument("--audio-root", help="directory of <video>_vocal.mp3 (real speech)")
    c.add_argument("--manifest", help="teacher_bench manifest.tsv instead of --records (teacher outputs)")
    c.add_argument("--wav-root", help="the run's wavs/ directory, used with --manifest")
    c.add_argument("--append", action="store_true")
    c.add_argument("--url", default="http://127.0.0.1:8091/phoneme")
    c.add_argument("--n", type=int, default=300)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--min-dur", type=float, default=3.0)
    c.add_argument("--max-dur", type=float, default=12.0)
    c.add_argument("--no-latin", action=argparse.BooleanOptionalAction, default=True)
    c.add_argument("--workers", type=int, default=2)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=collect)
    d = sub.add_parser("decode")
    d.add_argument("--project", required=True, help="the wav2vec2bert-jyutping checkout (error_consistency.make_predictor is imported from it)")
    d.add_argument("--model-dir", required=True)
    d.add_argument("--clips", required=True)
    d.add_argument("--audio-root", required=True)
    d.add_argument("--out", required=True)
    d.set_defaults(fn=decode)
    p = sub.add_parser("compare")
    p.add_argument("--probe", required=True)
    p.add_argument("--no-colloquial", action="store_true")
    p.add_argument("--dump")
    p.add_argument("--min-count", type=int, default=15)
    p.set_defaults(fn=compare)
    args = ap.parse_args()
    args.fn(args)
