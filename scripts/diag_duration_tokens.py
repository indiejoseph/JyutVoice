"""Where does the duration predictor go wrong: in the phones, or in the blanks and the pauses at punctuation?

    python scripts/diag_duration_tokens.py --config configs/yue_encoder_nobert.yaml --ckpt checkpoints_yue_nobert/checkpoint_epoch=009.ckpt --uses-bert 0

For every validation token the predicted duration exp(logw) is compared with the duration MAS found from the real audio (frames at 50 Hz), grouped into
phone / blank (the interspersed 0 and the pads) / punctuation. Reported per class: share of tokens, mean true and predicted frames, correlation, R^2
(1 - SSE / variance of the true durations), and per utterance the signed and the absolute contribution of each class to the total-length error.
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
    from jyutvoice.models.flow_duration_predictor import FlowDurationPredictor
    from jyutvoice.text import _symbol_to_id
    from jyutvoice.text.symbols import punctuations

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-config", default=None, help="config for the validation split (default: --config)")
    ap.add_argument("--uses-bert", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cfg-scale", type=float, default=None, help="flow-matching duration predictor: guidance scale (default: the model's)")
    ap.add_argument("--temperature", type=float, default=None, help="flow-matching duration predictor: noise temperature (default: the model's)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    with open(a.config) as f:
        tts = load_hyperpyyaml(f)["tts"]
    with open(a.data_config or a.config) as f:
        dm = load_hyperpyyaml(f)["data"]
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    tts.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    tts = tts.eval().to(dev)
    torch.manual_seed(a.seed)
    dm.setup()
    punct_ids = torch.tensor(sorted({_symbol_to_id[p] for p in punctuations if p in _symbol_to_id}), device=dev)

    true_all = {c: [] for c in ("phone", "blank", "punct")}
    pred_all = {c: [] for c in true_all}
    per_utt = {c: [] for c in true_all}
    totals = []
    for b in DataLoader(dm.validset, batch_size=16, shuffle=False, num_workers=2, collate_fn=TextMelBatchCollate()):
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        kw = {"bert": b["bert"]} if a.uses_bert else {}
        align_kw = {k: b[k] for k in ("silence", "pause_mask") if k in b}  # pause-aware alignment when the rows carry silence flags and the config a bonus
        with torch.no_grad():
            _, _, attn = tts(x=b["x"], x_lengths=b["x_lengths"], y=b["y"], y_lengths=b["y_lengths"], lang=b["lang"], tone=b["tone"], word_pos=b["word_pos"],
                             syllable_pos=b["syllable_pos"], spk_embed=b["spk_embed"], decoder_h=b["decoder_h"], **kw, **align_kw)  # fmt: skip
            true = attn.squeeze(1).sum(-1) if attn.dim() == 4 else attn.sum(-1)  # (B, T_text) frames per token
            spk = F.normalize(b["spk_embed"], dim=1)
            x_enc, _, x_mask = tts.encoder(b["x"], b["x_lengths"], b["lang"], b["tone"], b["word_pos"], b["syllable_pos"], spk, **kw)
            if isinstance(tts.dp, FlowDurationPredictor):  # a sampler: one draw per token, conditioned like in training
                pred = (torch.exp(tts.dp(x_enc, x_mask, g=spk, cfg_scale=a.cfg_scale, temperature=a.temperature, **kw)) * x_mask).squeeze(1)
            else:
                pred = (torch.exp(tts.dp(x_enc, x_mask)) * x_mask).squeeze(1)
        valid = x_mask.squeeze(1).bool()
        is_blank = (b["x"] == 0) & valid
        is_punct = torch.isin(b["x"], punct_ids) & valid
        is_phone = valid & ~is_blank & ~is_punct
        for name, m in (("phone", is_phone), ("blank", is_blank), ("punct", is_punct)):
            true_all[name].append(true[m].cpu().numpy())
            pred_all[name].append(pred[m].cpu().numpy())
            per_utt[name].append(((pred - true) * m).sum(1).cpu().numpy())
        totals.append((true * valid).sum(1).cpu().numpy())
    n_tok = sum(len(np.concatenate(v)) for v in true_all.values())
    totals = np.concatenate(totals)
    print(f"{len(totals)} validation clips, {n_tok} tokens, mean true length {totals.mean():.0f} frames\n")
    print(f"{'class':7s} {'tokens':>7s} {'true fr':>8s} {'pred fr':>8s} {'corr':>6s} {'R^2':>6s}   per-utterance error (frames): signed mean, mean |.|")
    for name in true_all:
        t, p = np.concatenate(true_all[name]), np.concatenate(pred_all[name])
        u = np.concatenate(per_utt[name])
        r2 = 1 - ((p - t) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-9)
        corr = np.corrcoef(t, p)[0, 1] if len(t) > 2 and t.std() > 0 else float("nan")
        print(f"{name:7s} {len(t):7d} {t.mean():8.2f} {p.mean():8.2f} {corr:6.2f} {r2:6.2f}   {u.mean():+7.2f}  {np.abs(u).mean():6.2f}")
    allu = sum(np.concatenate(per_utt[c]) for c in per_utt)
    print(f"{'total':7s} {'':>7s} {'':>8s} {'':>8s} {'':>6s} {'':>6s}   {allu.mean():+7.2f}  {np.abs(allu).mean():6.2f}  ({100 * np.abs(allu).mean() / totals.mean():.1f}% of the length)")
    p_true, p_pred = np.concatenate(true_all["punct"]), np.concatenate(pred_all["punct"])
    if len(p_true):
        print(f"\npunctuation tokens (pauses): true duration mean {p_true.mean():.1f} frames, std {p_true.std():.1f}, 25/50/75/95 percentiles "
              f"{np.percentile(p_true, [25, 50, 75, 95]).round(1).tolist()}")
        print(f"                              predicted: std {p_pred.std():.1f}, 25/50/75/95 percentiles {np.percentile(p_pred, [25, 50, 75, 95]).round(1).tolist()}; "
              f"{100 * np.mean(p_true > 10):.1f}% of the true pauses are > 10 frames (200 ms), {100 * np.mean(p_pred > 10):.1f}% of the predictions")


if __name__ == "__main__":
    main()
