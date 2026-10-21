"""Train the frame-level prosody model (ProsodyCFM) on cached encoder features (scripts/build_prosody_cache.py).

    python scripts/train_prosody_cfm.py --cache ~/teacher/prosody_cache_teacher.pt [more caches ...] --out-dir checkpoints_prosody --epochs 60 --device cuda

With several caches (e.g. real corpus rows and teacher rows) the training rows are pooled, each cache keeps its own validation set (named by file) and the model kept as
"best" has the lowest mean validation loss over them; `--repeat 2 1` repeats the first cache's training rows twice per epoch (one number per cache).

The text encoder predicts one average vector per token, which explains only ~65-75% of the variance of the real CosyVoice2 flow-encoder states (decoder_h) and moves
~12% less per frame (scripts/diag_mu_smoothing.py). ProsodyCFM is a conditional flow-matching model that generates decoder_h at 50 Hz from the frozen encoder's expanded
output (mu_y), its hidden states (h_text_y), the position inside the phone, the phone's length and the speaker x-vector, i.e. it samples the missing detail instead of
averaging it. The result replaces mu_y in `JyutVoiceTTS.synthesise`.

Every `--eval-every` epochs the EMA model is sampled on validation rows and compared with the real features and with the encoder's mean mu_y:
    R2 / std ratio / delta ratio as in diag_mu_smoothing.py, where the ratios of a good sample are near 1.0 (the mean's are 0.85-0.9) and the R2 of one sample is below the
    mean's (a sample is not the mean); `mean of K` is the R2 of the average of K samples, which approaches what the regression would give.
Writes <out-dir>/prosody_cfm_best.pt (lowest validation flow-matching loss, EMA weights) and prosody_cfm_last.pt: {"model_config", "ema", "epoch", "val_loss"}.
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from jyutvoice.models.prosody_cfm import ProsodyCFM, mark_class_table, mark_features  # noqa: E402


def attn_from_durations(dur, t_mel):
    """(B, T_text) integer frames per token -> (B, T_text, t_mel) 0/1 monotonic alignment (padding tokens have 0 frames)."""
    cum = dur.cumsum(-1)
    start = cum - dur
    frames = torch.arange(t_mel, device=dur.device)[None, None, :]
    return ((frames >= start[:, :, None]) & (frames < cum[:, :, None])).float()


SYMBOLS_JSON = None  # set from --symbols-json: the symbol list as JSON, for machines without the text package's dependencies (pycantonese, ...)


def class_table():
    """Punctuation class of every symbol id (the text package is only imported when a punctuation-aware model asks for it and no --symbols-json was given)."""
    if SYMBOLS_JSON:
        with open(SYMBOLS_JSON, encoding="utf-8") as f:
            return mark_class_table(json.load(f))
    from jyutvoice.text.symbols import symbols

    return mark_class_table(symbols)


def make_batch(rows, device, table=None):
    """Pad a list of cached rows into the tensors ProsodyCFM.compute_loss takes; with `table` (the punctuation class of every symbol id) also `mark_y` / `dist_y`
    from the rows' `ids` (cache built by scripts/build_prosody_cache.py with token ids)."""
    B = len(rows)
    tt = max(r["x"].shape[0] for r in rows)
    ty = max(r["h"].shape[0] for r in rows)
    x = torch.zeros(B, 192, tt)
    mu = torch.zeros(B, 80, tt)
    dur = torch.zeros(B, tt, dtype=torch.long)
    h = torch.zeros(B, 80, ty)
    y_lengths = torch.zeros(B, dtype=torch.long)
    for i, r in enumerate(rows):
        n, m = r["x"].shape[0], r["h"].shape[0]
        x[i, :, :n] = r["x"].float().T
        mu[i, :, :n] = r["mu"].float().T
        dur[i, :n] = r["dur"].long()
        h[i, :, :m] = r["h"].float().T
        y_lengths[i] = m
    spk = F.normalize(torch.stack([r["spk"] for r in rows]), dim=1)
    y_mask = (torch.arange(ty)[None, :] < y_lengths[:, None]).float().unsqueeze(1)
    attn = attn_from_durations(dur, ty)
    mu_y = torch.matmul(mu, attn)  # (B, 80, T_mel): each token's vector repeated over its frames
    h_text_y = torch.matmul(x, attn)  # (B, 192, T_mel)
    out = dict(h=h, mu_y=mu_y, h_text_y=h_text_y, attn=attn, y_mask=y_mask, y_lengths=y_lengths, spk=spk)
    if table is not None:
        ids = torch.zeros(B, tt, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, : r["ids"].shape[0]] = r["ids"].long()
        out["mark_y"], out["dist_y"] = mark_features(ids, attn, table)
    return {k: v.to(device) for k, v in out.items()}


def length_batches(rows, batch_size, shuffle, rng):
    """Batches of rows of similar mel length (sorted, cut into consecutive groups, group order shuffled)."""
    order = sorted(range(len(rows)), key=lambda i: rows[i]["h"].shape[0] + (rng.random() * 20 if shuffle else 0))
    groups = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(groups)
    return groups


def frame_stats(h, m):
    """R2, temporal-std ratio and frame-to-frame-change ratio of m against h, both (T, 80)."""
    h, m = h.double(), m.double()
    var = ((h - h.mean(0, keepdim=True)) ** 2).sum().clamp_min(1e-9)
    r2 = 1 - ((h - m) ** 2).sum() / var
    std = (m.std(0) / (h.std(0) + 1e-9)).mean()
    delta = (m[1:] - m[:-1]).norm() / ((h[1:] - h[:-1]).norm() + 1e-9)
    return float(r2), float(std), float(delta)


@torch.no_grad()
def sample_stats(model, rows, device, steps, cfg_scale, temperature, k_mean=4, batch_size=16):
    """Mean over rows of frame_stats for: the encoder's mu_y, one sample, and the mean of k_mean samples."""
    acc = {"mu": [], "sample": [], "mean_k": []}
    table = class_table() if getattr(model, "use_mark", False) else None
    for g in length_batches(rows, batch_size, False, random.Random(0)):
        b = make_batch([rows[i] for i in g], device, table)
        samples = [model(b["mu_y"], b["h_text_y"], b["attn"], b["y_mask"], g=b["spk"], n_steps=steps, cfg_scale=cfg_scale, temperature=temperature, mark_y=b.get("mark_y"), dist_y=b.get("dist_y")) for _ in range(k_mean)]
        for j in range(len(g)):
            t = int(b["y_lengths"][j])
            h = b["h"][j, :, :t].T.cpu()
            acc["mu"].append(frame_stats(h, b["mu_y"][j, :, :t].T.cpu()))
            acc["sample"].append(frame_stats(h, samples[0][j, :, :t].T.cpu()))
            acc["mean_k"].append(frame_stats(h, torch.stack([s[j, :, :t].T.cpu() for s in samples]).mean(0)))
    return {k: torch.tensor(v).mean(0).tolist() for k, v in acc.items()}


@torch.no_grad()
def val_loss(model, rows, device, batch_size=16, draws=2):
    model.eval()
    torch.manual_seed(0)
    tot, n = 0.0, 0
    table = class_table() if getattr(model, "use_mark", False) else None
    for _ in range(draws):
        for g in length_batches(rows, batch_size, False, random.Random(0)):
            b = make_batch([rows[i] for i in g], device, table)
            tot += float(model.compute_loss(decoder_h=b["h"], mu_y=b["mu_y"], h_text_y=b["h_text_y"], attn=b["attn"], y_mask=b["y_mask"], y_lengths=b["y_lengths"], g=b["spk"], mark_y=b.get("mark_y"), dist_y=b.get("dist_y")))
            n += 1
    return tot / max(1, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, nargs="+")
    ap.add_argument("--repeat", type=int, nargs="+", help="times each cache's training rows are used per epoch (default 1 each)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--kernel", type=int, default=7)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--cfg-dropout", type=float, default=0.15)
    ap.add_argument("--symbols-json", help="JSON list of the symbols (jyutvoice.text.symbols.symbols) for --marks where the text package cannot be imported")
    ap.add_argument("--marks", action="store_true", help="punctuation-aware: every frame also gets the class of the next punctuation mark and the frames until it (needs a cache with token ids)")
    ap.add_argument("--residual", action="store_true", help="generate decoder_h - mu_y instead of decoder_h (ProsodyCFM(residual=True))")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-rows", type=int, default=96)
    ap.add_argument("--sample-steps", type=int, default=10)
    ap.add_argument("--cfg-scale", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    a = ap.parse_args()
    global SYMBOLS_JSON
    SYMBOLS_JSON = a.symbols_json
    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    caches = [torch.load(c, map_location="cpu", weights_only=False) for c in a.cache]
    names = [os.path.basename(c) for c in a.cache]
    reps = a.repeat or [1] * len(caches)
    assert len(reps) == len(caches)
    train = [r for c, k in zip(caches, reps) for _ in range(k) for r in c["train"]]
    valids = {n: c["valid"] for n, c in zip(names, caches)}
    for n, c in zip(names, caches):
        print(f"{n}: {len(c['train'])} train rows ({sum(r['h'].shape[0] for r in c['train']) / 50 / 3600:.1f} h), {len(c['valid'])} valid rows, meta {c.get('meta')}", flush=True)
    print(f"pooled training rows per epoch: {len(train)}", flush=True)
    cfgd = dict(in_channels=80, text_channels=192, hidden_channels=a.hidden, n_blocks=a.blocks, kernel_size=a.kernel, p_dropout=a.dropout, spk_embed_dim=192, cfg_dropout=a.cfg_dropout, residual=a.residual, use_mark=a.marks)
    model = ProsodyCFM(**cfgd).to(dev)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"ProsodyCFM {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)
    table = class_table() if a.marks else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = len(length_batches(train, a.batch_size, False, rng))
    total = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))  # warmup, cosine to 5%
    os.makedirs(a.out_dir, exist_ok=True)
    best, step, t0 = float("inf"), 0, time.time()
    amp = dev.type == "cuda"
    for epoch in range(1, a.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for g in length_batches(train, a.batch_size, True, rng):
            b = make_batch([train[i] for i in g], dev, table)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                loss = model.compute_loss(decoder_h=b["h"], mu_y=b["mu_y"], h_text_y=b["h_text_y"], attn=b["attn"], y_mask=b["y_mask"], y_lengths=b["y_lengths"], g=b["spk"], mark_y=b.get("mark_y"), dist_y=b.get("dist_y"))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            decay = min(a.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
            tot += float(loss)
            n += 1
        vls = {nm: val_loss(ema, v, dev) for nm, v in valids.items()}
        vl = sum(vls.values()) / len(vls)
        line = f"epoch {epoch:3d} train {tot / n:.4f} val(ema) {vl:.4f}" + ("" if len(vls) == 1 else " [" + " ".join(f"{nm.split('.')[0][-14:]} {v:.4f}" for nm, v in vls.items()) + "]") + f" lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s"
        if vl < best:
            best = vl
            torch.save({"model_config": cfgd, "ema": ema.state_dict(), "epoch": epoch, "val_loss": vl}, os.path.join(a.out_dir, "prosody_cfm_best.pt"))
            line += " *"
        print(line, flush=True)
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            fmt = lambda v: f"R2 {v[0]:.3f} std ratio {v[1]:.3f} delta ratio {v[2]:.3f}"  # noqa: E731
            for nm, v in valids.items():
                st = sample_stats(ema, v[: a.eval_rows], dev, a.sample_steps, a.cfg_scale, a.temperature)
                print(f"   [{nm}] encoder mean mu_y: {fmt(st['mu'])}\n   [{nm}] one sample       : {fmt(st['sample'])}\n   [{nm}] mean of 4 samples: {fmt(st['mean_k'])}", flush=True)
        if a.max_minutes and (time.time() - t0) / 60 > a.max_minutes:
            print("time limit reached", flush=True)
            break
    torch.save({"model_config": cfgd, "ema": ema.state_dict(), "epoch": epoch, "val_loss": vl}, os.path.join(a.out_dir, "prosody_cfm_last.pt"))
    json.dump({"args": vars(a), "best_val": best, "epochs_run": epoch}, open(os.path.join(a.out_dir, "run.json"), "w"))
    print(f"done; best val {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
