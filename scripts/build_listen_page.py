"""Listening page for a synth_compare.py output dir: one row per sentence, one audio player per (model, voice), the ASR transcript under it when scores exist.

    python scripts/build_listen_page.py --out tmp/mixed_eval [--scores tmp/mixed_eval/scores.json]
    cd tmp/mixed_eval && python -m http.server 8774 --bind 0.0.0.0

`scores.json` is the --json output of scripts/eval_mixed_asr.py (the transcript is what the recogniser heard, an intelligibility check, not a naturalness one).
"""
import argparse
import html
import json
import os


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores")
    ap.add_argument("--title", default="Mixed Cantonese / English renders")
    ap.add_argument("--models", help="comma-separated model names to show, in this order (default: all, in the order of results.jsonl)")
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    rows = [json.loads(line) for line in open(os.path.join(out, "results.jsonl"), encoding="utf-8")]
    said = {}
    if a.scores:
        for s in json.load(open(os.path.expanduser(a.scores), encoding="utf-8")):
            said[(s["model"], s["spk"], s["sid"])] = s
    models = a.models.split(",") if a.models else list(dict.fromkeys(r["model"] for r in rows))
    spks = sorted({r["spk"] for r in rows})
    sids = sorted({r["sid"] for r in rows})
    by = {(r["model"], r["spk"], r["sid"]): r for r in rows}
    head = "".join(f"<th>{html.escape(m)} · voice {s}</th>" for m in models for s in spks)
    body = []
    for sid in sids:
        first = next(r for r in rows if r["sid"] == sid)
        cells = []
        for m in models:
            for s in spks:
                r = by.get((m, s, sid))
                if r is None:
                    cells.append("<td></td>")
                    continue
                sc = said.get((m, s, sid))
                heard = ""
                if sc:
                    bad = " bad" if sc["missed"] else ""
                    heard = f'<div class="said{bad}">{html.escape(sc["said"])}</div>'
                cells.append(f'<td><audio controls preload="none" src="{html.escape(r["wav"])}"></audio>{heard}</td>')
        body.append(f'<tr><th class="text">{sid + 1}. {html.escape(first["text"].replace(" ", ""))}</th>{"".join(cells)}</tr>')
    page = f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(a.title)}</title>
<style>body{{font:15px system-ui,sans-serif;margin:16px}}table{{border-collapse:collapse}}td,th{{border:1px solid #8884;padding:6px 8px;vertical-align:top}}
th.text{{text-align:left;max-width:22em}}audio{{width:200px;height:34px}}.said{{font-size:13px;opacity:.75;margin-top:2px}}.said.bad{{color:#c0392b;opacity:1}}</style>
<h2>{html.escape(a.title)}</h2><p>Red transcript: the recogniser missed an English word of the sentence.</p>
<table><tr><th></th>{head}</tr>{"".join(body)}</table>"""
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    print(os.path.join(out, "index.html"), len(sids), "sentences")


if __name__ == "__main__":
    main()
