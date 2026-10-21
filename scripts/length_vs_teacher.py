"""Length of each student render against the teacher's render of the same sentence (the pacing check of the duration predictors).

    python scripts/length_vs_teacher.py --teacher tmp/render_ref1 --student old=tmp/match1_flow --student new=tmp/match1_flowdp_en

`--teacher` is a scripts/teacher_render.py output dir (manifest.jsonl with {wav, text}); a `--student` is a scripts/synth_compare.py output dir (results.jsonl with
`text` and `wav`, any number of models and voices). Sentences are paired by their text with the spaces removed. Per student: median ratio student / teacher length,
std of the log ratio, share within +-20%, and the mean length in seconds next to the teacher's on the same sentences.
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf


def key(text):
    return text.replace(" ", "")


def seconds(path):
    info = sf.info(path)
    return info.frames / info.samplerate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", action="append", required=True, help="name=dir")
    a = ap.parse_args()
    teacher = {}
    for line in open(os.path.join(os.path.expanduser(a.teacher), "manifest.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        teacher[key(r["text"])] = seconds(os.path.join(os.path.expanduser(a.teacher), r["wav"]))
    for spec in a.student:
        name, path = spec.split("=", 1)
        path = os.path.expanduser(path)
        ratios, mine, theirs = [], [], []
        for line in open(os.path.join(path, "results.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            t = teacher.get(key(r["text"]))
            if t is None:
                continue
            d = seconds(os.path.join(path, r["wav"]))
            ratios.append(d / t)
            mine.append(d)
            theirs.append(t)
        ratios = np.asarray(ratios)
        print(f"{name}: {len(ratios)} renders, median ratio {np.median(ratios):.3f}, std log ratio {np.std(np.log(ratios)):.3f}, "
              f"within 20% {100 * np.mean(np.abs(ratios - 1) <= 0.2):.0f}%, mean {np.mean(mine):.2f} s vs teacher {np.mean(theirs):.2f} s")


if __name__ == "__main__":
    main()
