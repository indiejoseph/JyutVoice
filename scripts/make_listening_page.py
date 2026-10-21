"""Static HTML page with every sentence and speaker as a row and one audio player per model, in the order the models were synthesised.

    python scripts/make_listening_page.py --synth-dir tmp/synth_ab [--blind]

--blind hides the model names behind A / B / C in a fixed shuffled order per row (the key is written to <synth-dir>/blind_key.json), so that the
listener does not know which player is which. Open <synth-dir>/index.html in a browser; the wav paths are relative.
"""
import argparse
import collections
import html
import json
import os
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(os.path.join(a.synth_dir, "results.jsonl"), encoding="utf-8")]
    models = list(dict.fromkeys(r["model"] for r in rows))
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["sid"], r["spk"])][r["model"]] = r
    rng = random.Random(a.seed)
    key = {}
    body = []
    for (sid, spk), per in sorted(by.items()):
        order = list(models)
        if a.blind:
            rng.shuffle(order)
            key[f"{sid}/spk{spk}"] = {chr(65 + i): m for i, m in enumerate(order)}
        cells = []
        for i, m in enumerate(order):
            r = per[m]
            label = chr(65 + i) if a.blind else m
            cells.append(f'<td><div class="m">{html.escape(label)} <span>{r["dur"]:.1f}s</span></div><audio controls preload="none" src="{r["wav"]}"></audio></td>')
        text = next(iter(per.values()))["text"]
        body.append(f'<tr><td class="t">{sid + 1}. {html.escape(text)}<br><span>speaker {spk}</span></td>{"".join(cells)}</tr>')
    head = "".join(f"<th>{'player ' + chr(65 + i) if a.blind else html.escape(m)}</th>" for i, m in enumerate(models))
    page = f"""<!doctype html><meta charset="utf-8"><title>Synthesis comparison</title>
<style>body{{font-family:sans-serif;margin:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 8px;vertical-align:top}}
.t{{max-width:34em}}.t span,.m span{{color:#888;font-size:12px}}audio{{width:240px}}</style>
<h3>{'Blind' if a.blind else 'Labelled'} comparison: {len(by)} rows, {len(models)} models</h3><table><tr><th>sentence</th>{head}</tr>{''.join(body)}</table>"""
    open(os.path.join(a.synth_dir, "index.html"), "w", encoding="utf-8").write(page)
    if a.blind:
        json.dump(key, open(os.path.join(a.synth_dir, "blind_key.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{len(by)} rows -> {os.path.join(a.synth_dir, 'index.html')}" + (" (blind key in blind_key.json)" if a.blind else ""))


if __name__ == "__main__":
    main()
