"""What does the pause-aware alignment (JyutVoiceTTS.pause_bonus) do to the durations MAS finds? Sweep the bonus over rows that carry silence flags.

    python scripts/diag_pause_alignment.py --data-config configs/teacher_dp_eval.yaml \\
        --config configs/teacher_dp.yaml --ckpt checkpoints_yue_nobert/checkpoint_epoch=009.ckpt --bonus 0 3 6 10 20 --max-rows 1500

For every bonus B the frozen text encoder aligns each row's text to the row's `decoder_h` (MAS, as in training, with `silence` + `pause_mask`) and the script
reports, over the rows:

    tokens      mean frames of the pause tokens (punctuation marks; the pad tokens at both ends) and of the other tokens (phones; blanks)
    runs        for the silent runs of the flag (>= --run-min frames, not touching the clip ends): the share of their frames that the alignment gives to
                pause tokens / to blanks / to phones (a phone holding silence is what teaches the duration predictor "short phone" next to a pause);
                the edge runs (touching an end) likewise
    purity      the share of the frames given to pause tokens that are silent (a pause token should not swallow speech)
    prior       the prior loss of the alignment (how well mu_x fits decoder_h; the cost of moving frames off the best-fitting tokens)
    long        pause tokens longer than 10 frames per row (the pauses the predictor would learn to insert)
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def runs_of(flags, min_len):
    """[(start, stop)] of the maximal runs of 1 in a 1-D array that are at least min_len long."""
    f = np.concatenate([[0], np.asarray(flags, dtype=np.int8), [0]])
    d = np.diff(f)
    starts, stops = np.where(d == 1)[0], np.where(d == -1)[0]
    return [(s, e) for s, e in zip(starts, stops) if e - s >= min_len]


def main():
    from hyperpyyaml import load_hyperpyyaml
    from torch.utils.data import DataLoader

    from jyutvoice.data.text_mel_datamodule import PUNCT_IDS, TextMelBatchCollate

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--config", required=True, help="model config (tts)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bonus", type=float, nargs="+", default=[0, 3, 6, 10, 20])
    ap.add_argument("--max-rows", type=int, default=1500)
    ap.add_argument("--run-min", type=int, default=10, help="shortest silent run (frames, 20 ms each) that counts")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    with open(a.data_config) as f:
        dm = load_hyperpyyaml(f)["data"]
    dm.setup()
    ds = dm.validset
    n = min(a.max_rows, len(ds))
    sub = torch.utils.data.Subset(ds, range(n))
    batches = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
               for b in DataLoader(sub, batch_size=16, shuffle=False, num_workers=2, collate_fn=TextMelBatchCollate())]
    assert "silence" in batches[0], "the rows have no `silence` column (teacher_rows.py --silence-probe)"
    print(f"{n} rows, {sum(int(b['y_lengths'].sum()) for b in batches) / 50 / 3600:.2f} h; silent frames "
          f"{100 * sum(float(b['silence'].sum()) for b in batches) / sum(float(b['y_lengths'].sum()) for b in batches):.1f}%\n")

    with open(a.config) as f:
        tts = load_hyperpyyaml(f)["tts"]
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    tts.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    tts = tts.eval().to(dev)
    punct = PUNCT_IDS.to(dev)

    print(f"{'B':>5s} | {'pause tok':>9s} {'other tok':>9s} {'long/row':>8s} | {'internal runs: pause':>20s} {'blank':>6s} {'phone':>6s} | "
          f"{'edge runs: pause':>16s} | {'purity':>6s} {'prior':>7s}")
    for B in a.bonus:
        tts.pause_bonus = B
        pause_d, other_d, longs = [], [], []
        agg = {"int": np.zeros(3), "edge": np.zeros(3)}
        pure_num = pure_den = prior = 0.0
        rows = 0
        with torch.no_grad():
            for b in batches:
                dur_loss, prior_loss, attn = tts(
                    x=b["x"], x_lengths=b["x_lengths"], y=b["y"], y_lengths=b["y_lengths"], lang=b["lang"], tone=b["tone"], word_pos=b["word_pos"],
                    syllable_pos=b["syllable_pos"], spk_embed=b["spk_embed"], decoder_h=b["decoder_h"], silence=b["silence"], pause_mask=b["pause_mask"],
                )  # fmt: skip
                prior += float(prior_loss) * b["x"].shape[0]
                dur = attn.sum(-1)  # (B, T_text)
                for i in range(b["x"].shape[0]):
                    T, Ty = int(b["x_lengths"][i]), int(b["y_lengths"][i])
                    pm = b["pause_mask"][i, :T].bool()
                    ids = b["x"][i, :T]
                    is_blank = (ids == 0) & ~pm
                    is_phone = ~pm & ~is_blank
                    d = dur[i, :T]
                    pause_d += d[pm & torch.isin(ids, punct)].tolist()
                    other_d += d[~pm].tolist()
                    longs.append(float((d[pm & torch.isin(ids, punct)] > 10).sum()))
                    sil = b["silence"][i, :Ty].cpu().numpy()
                    tok_of = attn[i, :T, :Ty].argmax(0)  # token index of every frame
                    kind = torch.where(pm[tok_of], 0, torch.where(is_blank[tok_of], 1, 2)).cpu().numpy()  # 0 pause, 1 blank, 2 phone
                    for s, e in runs_of(sil, a.run_min):
                        key = "edge" if (s == 0 or e == Ty) else "int"
                        agg[key] += np.bincount(kind[s:e], minlength=3)
                    sel = kind == 0
                    pure_num += float(sil[sel].sum())
                    pure_den += float(sel.sum())
                    rows += 1
        share = {k: v / max(1.0, v.sum()) for k, v in agg.items()}
        print(f"{B:5.1f} | {np.mean(pause_d):9.2f} {np.mean(other_d):9.2f} {np.mean(longs):8.2f} | {100 * share['int'][0]:19.0f}% {100 * share['int'][1]:5.0f}% "
              f"{100 * share['int'][2]:5.0f}% | {100 * share['edge'][0]:15.0f}% | {100 * pure_num / max(1.0, pure_den):5.0f}% {prior / rows:7.4f}"
              f"   (runs: {int(agg['int'].sum())} internal / {int(agg['edge'].sum())} edge frames)")


if __name__ == "__main__":
    main()
