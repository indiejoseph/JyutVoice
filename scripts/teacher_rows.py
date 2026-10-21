#!/usr/bin/env python3
"""Turn the teacher's generated S3 tokens into training rows: the text, and the flow-encoder states (`decoder_h`) of the teacher's tokens. No audio.

    python scripts/teacher_rows.py --pool pool.jsonl --tokens 'out/tokens_*.jsonl' --prompts ~/teacher/prompts/meta.json \\
        --pretrained ~/sd3_build/code/pretrained_models --out rows_dir

For the encoder-only training the target of a row is `decoder_h`, the CosyVoice2 flow encoder's output for the token sequence at 50 Hz, exactly what the real
audio's tokens give for a recording. The teacher's tokens therefore stand in for a recording of the sentence: MAS aligns them with the text encoder's output and
the durations it finds (token counts per phone, pauses at punctuation) are what the duration predictor is trained on. The speaker embedding is the prompt
speaker's CampPlus vector (the flow decoder and the student use it as the voice). Rows are dropped when the generation ran away (hit the token cap) or has an odd
number of speech tokens per character.

Columns: id, lang (yue), text (pydips word-segmented, like the corpus), text_valid, phone (null: g2p reads the text), spk_emb, decoder_h, source ("teacher"),
duration (s), mode, prompt, and with --silence-probe `silence` (uint8 per frame). Saved with Dataset.save_to_disk; `TextMelDataModule(dataset_path=<out>)` reads it (rows without audio are supported).
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def main():
    import soundfile as sf
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--tokens", required=True, help="glob of teacher_gen.py outputs")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--pretrained", required=True, help="dir with campplus.onnx and flow_encoder.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-tok-per-char", type=float, default=2.0)
    ap.add_argument("--max-tok-per-char", type=float, default=9.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--silence-probe", help="SilenceProbe npz (scripts/silence_probe.py --save): adds a per-frame `silence` column for the pause-aware alignment")
    a = ap.parse_args()

    from datasets import Dataset, Features, Sequence, Value
    from pydips import BertModel

    from jyutvoice.utils import flow_features as ff
    from jyutvoice.utils.silence_probe import SilenceProbe

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    pool = {}
    for line in open(a.pool, encoding="utf-8"):
        r = json.loads(line)
        pool[r["id"]] = r
    metas = json.load(open(a.prompts, encoding="utf-8"))["prompts"]
    spk = ff.load_spk_embedding_model(os.path.join(a.pretrained, "campplus.onnx"))
    prompt_emb = []
    for m in metas:
        y, sr = sf.read(str(Path(a.prompts).parent / Path(m["wav"]).name), dtype="float32")
        assert sr == 16000
        prompt_emb.append(np.asarray(ff.get_spk_embedding(y if y.ndim == 1 else y.mean(1), spk), dtype=np.float32))
    flow = ff.load_flow_encoder(os.path.join(a.pretrained, "flow_encoder.pt"), str(dev))
    ws = BertModel()
    probe = SilenceProbe.load(a.silence_probe) if a.silence_probe else None

    gen = {}
    for fn in sorted(glob.glob(a.tokens)):
        for line in open(fn, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                gen[r["id"]] = r
    stats = {"generated": len(gen), "hit_max": 0, "ratio": 0, "kept": 0}
    keep = []
    for sid, g in gen.items():
        n_chars = pool[sid]["n_chars"]
        ratio = len(g["tokens"]) / max(1, n_chars)
        if g.get("hit_max"):
            stats["hit_max"] += 1
        elif not a.min_tok_per_char <= ratio <= a.max_tok_per_char:
            stats["ratio"] += 1
        else:
            keep.append(g)
    if a.limit:
        keep = keep[: a.limit]
    stats["kept"] = len(keep)
    print(stats, flush=True)

    def rows():
        for g in keep:
            tok = torch.tensor([g["tokens"]], dtype=torch.int32)
            with torch.no_grad():
                h = ff.get_decoder_hidden_state(tok, torch.tensor([tok.shape[1]], dtype=torch.int32), flow, dev).squeeze(0).float().cpu().numpy()
            text = " ".join(ws.cut(pool[g["id"]]["text"], mode="coarse"))
            row = {"id": g["id"], "lang": "yue", "text": text, "text_valid": True, "phone": None, "spk_emb": prompt_emb[g["prompt"]], "decoder_h": h,
                   "source": "teacher", "duration": float(h.shape[0] / 50), "mode": g["mode"], "prompt": int(g["prompt"])}  # fmt: skip
            if probe is not None:
                row["silence"] = probe.mask(h).astype(np.uint8)
            yield row

    feats = Features({"id": Value("string"), "lang": Value("string"), "text": Value("string"), "text_valid": Value("bool"), "phone": Value("string"),
                      "spk_emb": Sequence(Value("float32")), "decoder_h": Sequence(Sequence(Value("float32"))), "source": Value("string"),
                      "duration": Value("float32"), "mode": Value("string"), "prompt": Value("int32"),
                      **({"silence": Sequence(Value("uint8"))} if probe is not None else {})})  # fmt: skip
    ds = Dataset.from_generator(rows, features=feats, cache_dir=os.path.join(a.out, ".cache"))
    ds.save_to_disk(a.out)
    print(f"{len(ds)} rows, {sum(ds['duration']) / 3600:.2f} h of teacher 'speech' -> {a.out}")


if __name__ == "__main__":
    main()
