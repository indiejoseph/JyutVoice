"""How well does the duration predictor match the real timing? Predicted vs true total length on the validation split, for several checkpoints.

    python scripts/diag_duration.py --data-config configs/yue_encoder.yaml \\
        --model old=configs/yue_encoder_nobert.yaml:pretrained_models/20260102_encoder_only.ckpt:0 --model bert=configs/yue_encoder.yaml:checkpoints_yue/checkpoint_epoch=009.ckpt:1

The predicted length is what synthesise() uses: sum over tokens of ceil(exp(logw)); the true one is the clip's mel length (decoder_h frames, 50 Hz).
Reported per model: median ratio predicted / true, mean |log ratio| (0.1 ~ 10% off), share of clips within +-10% and +-20%, and the correlation of the logs.
The last field of --model is 1 if the checkpoint takes BERT features, 0 otherwise.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    from hyperpyyaml import load_hyperpyyaml
    from torch.utils.data import DataLoader

    from jyutvoice.data.text_mel_datamodule import TextMelBatchCollate

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=config:checkpoint:uses_bert")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    with open(a.data_config) as f:
        dm = load_hyperpyyaml(f)["data"]
    dm.setup()
    batches = [{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
               for b in DataLoader(dm.validset, batch_size=16, shuffle=False, num_workers=2, collate_fn=TextMelBatchCollate())]
    true = np.concatenate([b["y_lengths"].cpu().numpy() for b in batches]).astype(float)
    print(f"{len(true)} validation clips, true length median {np.median(true) / 50:.2f} s\n")
    print(f"{'model/rounding':13s} {'median pred/true':>17s} {'mean |log ratio|':>17s} {'within 10%':>11s} {'within 20%':>11s} {'corr(log)':>10s}")
    for spec in a.model:
        name, rest = spec.split("=", 1)
        cfg_path, ckpt, uses_bert = rest.rsplit(":", 2)
        with open(cfg_path) as f:
            tts = load_hyperpyyaml(f)["tts"]
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        tts.load_state_dict(state["state_dict"] if "state_dict" in state else state)
        tts = tts.eval().to(dev)
        preds = {"ceil": [], "round": [], "exact": []}
        frac = []
        with torch.no_grad():
            for b in batches:
                spk = F.normalize(b["spk_embed"], dim=1)
                kw = {"bert": b["bert"]} if uses_bert == "1" else {}
                x_enc, _, x_mask = tts.encoder(b["x"], b["x_lengths"], b["lang"], b["tone"], b["word_pos"], b["syllable_pos"], spk, **kw)
                w = torch.exp(tts.dp(x_enc, x_mask)) * x_mask
                preds["ceil"].append(torch.ceil(w).sum([1, 2]).cpu().numpy())  # what synthesise() does today
                preds["round"].append((torch.clamp_min(torch.round(w), 1) * x_mask).sum([1, 2]).cpu().numpy())  # every real token keeps >= 1 frame
                preds["exact"].append(w.sum([1, 2]).cpu().numpy())  # no rounding at all
                frac.append(w[x_mask.bool()].cpu().numpy())
        frac = np.concatenate(frac)
        print(f"  [{name}] predicted per-token duration: median {np.median(frac):.2f} frames, mean {frac.mean():.2f}; {100 * np.mean(frac <= 1):.0f}% of tokens <= 1 frame, "
              f"{100 * np.mean((frac > 1) & (frac <= 2)):.0f}% in (1, 2], so ceil() adds on average {np.mean(np.ceil(frac) - frac):.2f} frames per token")
        for mode, pr in preds.items():
            pred = np.concatenate(pr)
            lr = np.log(pred / true)
            print(f"{name + '/' + mode:13s} {np.median(pred / true):12.3f} {np.abs(lr).mean():17.3f} {100 * np.mean(np.abs(lr) < np.log(1.1)):10.0f}% "
                  f"{100 * np.mean(np.abs(lr) < np.log(1.2)):10.0f}% {np.corrcoef(np.log(pred), np.log(true))[0, 1]:10.3f}")


if __name__ == "__main__":
    main()
