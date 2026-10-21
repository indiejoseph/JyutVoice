"""How much frame-level detail does the text encoder's output lose compared with the real flow-encoder states (decoder_h)?

    python scripts/diag_mu_smoothing.py --data-config configs/teacher_flowdp_masked_cpu.yaml --config configs/teacher_flowdp_masked_cpu.yaml \\
        --ckpt checkpoints_yue_nobert/checkpoint_epoch=009.ckpt --device cpu

For held-out rows the frozen encoder's output per token is expanded with the MAS alignment (the true timing, as in training), giving mu_y (80 channels, 50 Hz), which is
compared with decoder_h of the same row. The frozen CosyVoice2 flow decoder was trained on real decoder_h, so how much sharper decoder_h is than mu_y is how much
the regression average costs the decoder. Mean over rows of:

    R2          1 - sum (h - mu)^2 / sum (h - mean_t h)^2 : the share of the within-row variance of decoder_h that mu_y explains
    corr        mean over channels of the correlation over time between mu_y and decoder_h
    std ratio   mean over channels of std_t(mu_y) / std_t(decoder_h): below 1 = mu_y moves less than the real states
    delta ratio |mu_y[t+1] - mu_y[t]| / |h[t+1] - h[t]|, summed over the row: below 1 = smoother frame to frame (the share of fast change that is missing is 1 - ratio)
    within-token R2   the same R2 but only about each token's own frames: how much of the variation INSIDE a phone (transitions, pitch movement) mu_y has
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
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max-rows", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    with open(a.config) as f:
        tts = load_hyperpyyaml(f)["tts"]
    with open(a.data_config) as f:
        dm = load_hyperpyyaml(f)["data"]
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    state = state["state_dict"] if "state_dict" in state else state
    state = {k: v for k, v in state.items() if k.startswith(("encoder.", "spk_embed_affine_layer."))}  # only the encoder matters here (the duration predictor may differ)
    missing, unexpected = tts.load_state_dict(state, strict=False)
    assert not [k for k in missing if k.startswith("encoder.")] and not unexpected, (missing, unexpected)
    tts = tts.eval().to(dev)
    dm.setup()
    sub = torch.utils.data.Subset(dm.validset, range(min(a.max_rows, len(dm.validset))))
    acc = {k: [] for k in ("r2", "corr", "std", "delta", "r2_in")}
    with torch.no_grad():
        for b in DataLoader(sub, batch_size=16, shuffle=False, num_workers=2, collate_fn=TextMelBatchCollate()):
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            _, _, attn = tts(x=b["x"], x_lengths=b["x_lengths"], y=b["y"], y_lengths=b["y_lengths"], lang=b["lang"], tone=b["tone"], word_pos=b["word_pos"],
                             syllable_pos=b["syllable_pos"], spk_embed=b["spk_embed"], decoder_h=b["decoder_h"])  # fmt: skip
            spk = F.normalize(b["spk_embed"], dim=1)
            _, mu_x, _ = tts.encoder(b["x"], b["x_lengths"], b["lang"], b["tone"], b["word_pos"], b["syllable_pos"], spk)
            attn = attn.squeeze(1) if attn.dim() == 4 else attn  # (B, T_text, T_mel)
            mu_y = torch.matmul(attn.transpose(1, 2), mu_x.transpose(1, 2))  # (B, T_mel, 80)
            tok_of = attn.argmax(1)  # (B, T_mel) token of every frame
            for i in range(mu_y.shape[0]):
                T = int(b["y_lengths"][i])
                h, m = b["decoder_h"][i, :T].double(), mu_y[i, :T].double()
                if T < 8:
                    continue
                var = ((h - h.mean(0, keepdim=True)) ** 2).sum()
                acc["r2"].append(float(1 - ((h - m) ** 2).sum() / var))
                hc, mc = h - h.mean(0, keepdim=True), m - m.mean(0, keepdim=True)
                acc["corr"].append(float(((hc * mc).sum(0) / (hc.norm(dim=0) * mc.norm(dim=0) + 1e-9)).mean()))
                acc["std"].append(float((m.std(0) / (h.std(0) + 1e-9)).mean()))
                acc["delta"].append(float((m[1:] - m[:-1]).norm() / ((h[1:] - h[:-1]).norm() + 1e-9)))
                # variation inside each token: subtract the per-token mean of both
                tok = tok_of[i, :T]
                onehot = F.one_hot(tok, attn.shape[1]).double()  # (T, T_text)
                cnt = onehot.sum(0).clamp_min(1.0)
                hm = onehot @ ((onehot.T @ h) / cnt[:, None])
                mm = onehot @ ((onehot.T @ m) / cnt[:, None])
                v_in = ((h - hm) ** 2).sum()
                acc["r2_in"].append(float(1 - (((h - hm) - (m - mm)) ** 2).sum() / v_in.clamp_min(1e-9)))
    n = len(acc["r2"])
    print(f"{n} rows of {a.data_config}, checkpoint {a.ckpt}")
    print(f"  R2 {np.mean(acc['r2']):.3f} | corr {np.mean(acc['corr']):.3f} | std ratio {np.mean(acc['std']):.3f} | delta ratio {np.mean(acc['delta']):.3f} | within-token R2 {np.mean(acc['r2_in']):.3f}")


if __name__ == "__main__":
    main()
