"""Sentence pool for teaching the student the teacher's punctuation behaviour: every base sentence appears three times, ending in 。 ？ and ！ (and the clause
sentences have a comma inside), so the only difference between the three renders is the mark.

    python scripts/build_punct_pool.py --out ~/teacher/pool_punct_train.jsonl --exclude ~/teacher/punct_sentences.txt [--n-simple 200 --n-clause 40]

Written as pool rows ({"id", "text", "n_chars", "base", "mark", "prompt"}) for scripts/teacher_gen.py. The three variants of a base share ONE prompt (voice), so a
pair differs in the mark only; bases are spread over the 4 prompts of the teacher's prompts/meta.json (`prompt` column; run teacher_gen.py once per prompt with
--prompt-index and --prompt-mode zero_shot). Bases whose text (without the mark) occurs in --exclude (the held-out probe sentences) are left out.
"""
import argparse
import itertools
import json
import os
import random

SUBJECTS = ["你", "佢", "我哋", "佢哋", "呢個人", "阿媽", "阿爸", "老師", "老細", "個仔", "阿哥", "隔離嘅人"]
PREDICATES = [
    "好叻", "好攰", "成功咗", "遲到咗", "返咗屋企", "走咗", "病咗", "贏咗", "中咗獎", "識咗新朋友", "買咗部新電話", "搬咗屋", "瞓著咗", "唔嚟", "唔識", "唔使去",
    "煮咗好多嘢", "拎咗第一", "學識咗游水", "好有錢", "知道", "聽到", "睇到", "揾到", "忘記咗", "答應咗", "同意", "唔同意", "反對", "輸咗",
]
FREE = ["落雨喇", "好靚呀", "今日好熱", "又塞車", "呢度好多人", "呢個價錢好平", "架車壞咗", "電話冇電", "門開咗", "燈著咗", "湯凍咗", "個波入咗", "時間到喇", "有蟑螂"]
CONDITIONS = ["如果落雨", "你再唔嚟", "聽日有時間", "如果佢唔同意", "呢次考得好", "你有咩事", "天氣好啲", "我哋遲到", "佢肯幫手", "錢夠嘅話"]
RESULTS = ["我哋就唔去", "我就走", "一齊食飯啦", "我就開心", "就唔使擔心", "你話點算", "打俾我啦", "我哋改日再約"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", help="text file of sentences (one per line) whose base text must not appear")
    ap.add_argument("--n-simple", type=int, default=200)
    ap.add_argument("--n-clause", type=int, default=40)
    ap.add_argument("--n-prompts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    banned = set()
    if a.exclude:
        banned = {line.strip()[:-1] for line in open(os.path.expanduser(a.exclude), encoding="utf-8") if line.strip()}
        banned |= {b.replace("，", "") for b in banned}
    simple = [s + p for s, p in itertools.product(SUBJECTS, PREDICATES)] + FREE
    clauses = [c + "，" + r for c, r in itertools.product(CONDITIONS, RESULTS)]
    simple = [t for t in simple if t not in banned]
    clauses = [t for t in clauses if t not in banned and t.replace("，", "") not in banned]
    rng.shuffle(simple)
    rng.shuffle(clauses)
    bases = simple[: a.n_simple] + clauses[: a.n_clause]
    out = open(os.path.expanduser(a.out), "w", encoding="utf-8")
    n = 0
    for b, base in enumerate(bases):
        for mark_name, mark in (("period", "。"), ("question", "？"), ("exclaim", "！")):
            text = base + mark
            out.write(json.dumps({"id": f"pm{b:03d}{mark_name[0]}", "text": text, "n_chars": len(text), "base": base, "mark": mark_name, "prompt": b % a.n_prompts}, ensure_ascii=False) + "\n")
            n += 1
    out.close()
    print(f"{len(bases)} bases ({min(a.n_simple, len(simple))} simple, {min(a.n_clause, len(clauses))} with a comma) x 3 marks = {n} sentences -> {a.out}")


if __name__ == "__main__":
    main()
