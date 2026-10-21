"""Add one more system (e.g. the noise-0.7 render) to a listening page directory made by make_listening_page.py, for the sentences it already has.

    python scripts/add_variant_to_page.py --page-dir tmp/listen_prosody --src tmp/match1_prosody_t07 --model student_prosody --name prosody_noise0.7 \\
        --after prosody_noise0.5 && python scripts/make_listening_page.py --synth-dir tmp/listen_prosody

Copies <src>/<wav> of every `--model` row whose `sid` is on the page to <page-dir>/wavs/<name>/spk1/sNN.wav, appends the rows to <page-dir>/results.jsonl under the
player name `--name`, and puts the new player right after `--after` (default: last) for each sentence. Re-run make_listening_page.py afterwards to rebuild index.html.
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page-dir", required=True)
    ap.add_argument("--src", required=True, help="a synth_compare.py output directory (results.jsonl + wavs/)")
    ap.add_argument("--model", required=True, help="the model name inside --src")
    ap.add_argument("--name", required=True, help="the player name on the page")
    ap.add_argument("--after", help="place the new player after this existing player (default: last)")
    a = ap.parse_args()

    path = os.path.join(a.page_dir, "results.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert not any(r["model"] == a.name for r in rows), f"{a.name} is already on the page"
    order = list(dict.fromkeys(r["model"] for r in rows))  # players in their current order
    order.insert(order.index(a.after) + 1 if a.after else len(order), a.name)
    texts = {r["sid"]: r["text"] for r in rows}
    spk = {r["sid"]: r["spk"] for r in rows}
    src = {}
    for line in open(os.path.join(a.src, "results.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r["model"] == a.model:
            src[r["sid"]] = r
    for sid in sorted(texts):
        rel = f"wavs/{a.name}/spk{spk[sid]}/s{sid:02d}.wav"
        os.makedirs(os.path.dirname(os.path.join(a.page_dir, rel)), exist_ok=True)
        shutil.copyfile(os.path.join(a.src, src[sid]["wav"]), os.path.join(a.page_dir, rel))
        rows.append({"model": a.name, "spk": spk[sid], "sid": sid, "text": texts[sid], "wav": rel, "dur": 0, "intended": []})
    rows.sort(key=lambda r: (r["sid"], order.index(r["model"])))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"added {len(texts)} clips as '{a.name}' (players now: {', '.join(order)})")


if __name__ == "__main__":
    main()
