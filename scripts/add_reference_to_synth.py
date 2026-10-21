"""Add recordings of the same sentences (e.g. the teacher's renders) to a synth_compare.py output directory as one more "model", so that
make_listening_page.py / pause_stats.py / score_synth_readings.py can treat them alike.

    python scripts/add_reference_to_synth.py --synth-dir tmp/synth_dp --manifest tmp/teacher_render60/manifest.jsonl \\
        --audio-root tmp/teacher_render60 --name teacher

The manifest has one {wav, text} per line; every line is matched to the sentence of results.jsonl with the same `text` (all other lines of the
manifest are ignored; run score_synth_readings.py BEFORE adding, the reference rows have no `intended` syllables) and copied to <synth-dir>/wavs/<name>/spk0/sNN.wav (same speaker slot 0 as the synthesised rows). Existing rows of that name are replaced.
"""
import argparse
import json
import os
import shutil

import soundfile as sf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    path = os.path.join(a.synth_dir, "results.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    rows = [r for r in rows if r["model"] != a.name]
    sid_of = {r["text"]: r["sid"] for r in rows}
    out_dir = os.path.join(a.synth_dir, "wavs", a.name, "spk0")
    os.makedirs(out_dir, exist_ok=True)
    added = []
    for line in open(a.manifest, encoding="utf-8"):
        m = json.loads(line)
        if m["text"] not in sid_of:
            continue
        sid = sid_of[m["text"]]
        rel = os.path.join("wavs", a.name, "spk0", f"s{sid:02d}.wav")
        src = os.path.join(a.audio_root, m["wav"])
        shutil.copyfile(src, os.path.join(a.synth_dir, rel))
        info = sf.info(src)
        added.append({"model": a.name, "spk": 0, "spk_id": a.name, "sid": sid, "text": m["text"], "seg_text": "", "intended": [], "frames": 0,
                      "dur": info.duration, "wav": rel})  # fmt: skip
    # the reference goes first: the listening page lists models in file order
    with open(path, "w", encoding="utf-8") as f:
        for r in added + rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(added)} of {len(sid_of)} sentences added as '{a.name}'")


if __name__ == "__main__":
    main()
