"""Sweep the sampling settings of a trained ProsodyCFM against the real features of the validation rows.

    python scripts/eval_prosody_cfm.py --cache ~/teacher/prosody_cache_teacher.pt --ckpt ~/teacher/prosody_ckpt/run1/prosody_cfm_best.pt --rows 128

For every (temperature, cfg scale, steps) the EMA model samples decoder_h for the validation rows; reported are the mean R2 of one sample, the temporal-std ratio and the
frame-to-frame-change ratio ("delta"; 1.0 = as much change as the real states, above 1 = rougher), and the R2 of the mean of 4 samples. The encoder's own mean is the
reference line. Pick settings whose ratios are near 1.0 without losing R2.
"""
import argparse
import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from train_prosody_cfm import sample_stats  # noqa: E402

from jyutvoice.models.prosody_cfm import ProsodyCFM  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--temperature", type=float, nargs="+", default=[0.5, 0.7, 0.85, 1.0])
    ap.add_argument("--cfg", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    ap.add_argument("--steps", type=int, nargs="+", default=[10])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    cache = torch.load(a.cache, map_location="cpu", weights_only=False)
    valid = cache["valid"][: a.rows]
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = ProsodyCFM(**ck["model_config"]).to(dev)
    model.load_state_dict(ck["ema"])
    model.eval()
    print(f"{a.ckpt}: epoch {ck['epoch']}, val {ck['val_loss']:.4f}; {len(valid)} validation rows\n")
    print(f"{'temp':>5s} {'cfg':>4s} {'steps':>5s} | {'sample R2':>9s} {'std':>6s} {'delta':>6s} | {'mean of 4 R2':>12s}")
    ref = None
    for temp, cfg, steps in itertools.product(a.temperature, a.cfg, a.steps):
        st = sample_stats(model, valid, dev, steps, cfg, temp)
        ref = ref or st["mu"]
        print(f"{temp:5.2f} {cfg:4.1f} {steps:5d} | {st['sample'][0]:9.3f} {st['sample'][1]:6.3f} {st['sample'][2]:6.3f} | {st['mean_k'][0]:12.3f}", flush=True)
    print(f"\nencoder mean mu_y: R2 {ref[0]:.3f}, std ratio {ref[1]:.3f}, delta ratio {ref[2]:.3f}")


if __name__ == "__main__":
    main()
