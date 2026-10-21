"""Cache what the frozen text encoder gives for every row, so the frame-level prosody model (ProsodyCFM) can be trained without the text front end or the encoder.

    python scripts/build_prosody_cache.py --config configs/teacher_flowdp_masked_cpu.yaml --ckpt checkpoints_yue_nobert/checkpoint_epoch=009.ckpt \\
        --out ~/teacher/prosody_cache_teacher.pt --device cpu

For every row of the config's data (its train and validation split) the encoder is run once and the alignment between its tokens and the row's `decoder_h` is found
by MAS (with the pause-aware bonus and the silence flags when the config and the rows have them, exactly like training). Saved per row, in fp16 where it is a
feature: `x` (T_text, 192) encoder hidden states, `mu` (T_text, 80) its output, `dur` (T_text,) frames per token, `h` (T_mel, 80) the real flow-encoder states,
`spk` (192,) the stored x-vector. File: {"train": [...], "valid": [...], "meta": {...}}, read by scripts/train_prosody_cfm.py.
"""
import argparse
import os
import sys
import time

import torch
import torch.multiprocessing
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy("file_system")  # many batches from the loader workers: the default one passes a file descriptor per tensor and runs out of them

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    from hyperpyyaml import load_hyperpyyaml
    from torch.utils.data import DataLoader

    from jyutvoice.data.text_mel_datamodule import TextMelBatchCollate

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-rows", type=int, default=0, help="testing: first N rows of each split")
    ap.add_argument("--dataset-path", nargs="+", help="use these datasets (globs allowed) instead of the config's data.dataset_path, e.g. one cache per kind of row")
    a = ap.parse_args()
    dev = torch.device(a.device)
    with open(a.config) as f:
        cfg = load_hyperpyyaml(f)
    tts, dm = cfg["tts"], cfg["data"]
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    state = state["state_dict"] if "state_dict" in state else state
    state = {k: v for k, v in state.items() if k.startswith(("encoder.", "spk_embed_affine_layer."))}
    missing, unexpected = tts.load_state_dict(state, strict=False)
    assert not [k for k in missing if k.startswith("encoder.")] and not unexpected, (missing, unexpected)
    tts = tts.eval().to(dev)
    if a.dataset_path:
        dm.hparams.dataset_path = [os.path.expanduser(p) for p in a.dataset_path]
    dm.setup()
    out = {"train": [], "valid": [], "meta": {"config": a.config, "ckpt": a.ckpt, "pause_bonus": float(getattr(tts, "pause_bonus", 0.0))}}
    t0 = time.time()
    for split, ds in (("train", dm.trainset), ("valid", dm.validset)):
        n = min(len(ds), a.max_rows) if a.max_rows else len(ds)
        sub = torch.utils.data.Subset(ds, range(n))
        for bi, b in enumerate(DataLoader(sub, batch_size=16, shuffle=False, num_workers=4, collate_fn=TextMelBatchCollate())):
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.no_grad():
                _, _, attn = tts(x=b["x"], x_lengths=b["x_lengths"], y=b["y"], y_lengths=b["y_lengths"], lang=b["lang"], tone=b["tone"], word_pos=b["word_pos"],
                                 syllable_pos=b["syllable_pos"], spk_embed=b["spk_embed"], decoder_h=b["decoder_h"], silence=b.get("silence"), pause_mask=b.get("pause_mask"))  # fmt: skip
                spk = F.normalize(b["spk_embed"], dim=1)
                x_enc, mu_x, _ = tts.encoder(b["x"], b["x_lengths"], b["lang"], b["tone"], b["word_pos"], b["syllable_pos"], spk)
            attn = attn.squeeze(1) if attn.dim() == 4 else attn
            for i in range(b["x"].shape[0]):
                tt, ty = int(b["x_lengths"][i]), int(b["y_lengths"][i])
                dur = attn[i, :tt, :ty].sum(-1)
                assert int(dur.sum()) == ty, (split, bi, i, int(dur.sum()), ty)  # a full monotonic path: every frame belongs to one token
                out[split].append({
                    "x": x_enc[i, :, :tt].T.half().cpu(), "mu": mu_x[i, :, :tt].T.half().cpu(), "dur": dur.round().to(torch.int16).cpu(),
                    "h": b["decoder_h"][i, :ty].half().cpu(), "spk": b["spk_embed"][i].float().cpu(), "ids": b["x"][i, :tt].to(torch.int16).cpu(),
                })  # fmt: skip
            if bi % 20 == 0:
                print(f"{split}: {len(out[split])}/{n} rows ({time.time() - t0:.0f} s)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(out, a.out)
    frames = sum(r["h"].shape[0] for r in out["train"])
    print(f"saved {len(out['train'])} train + {len(out['valid'])} valid rows ({frames} train frames = {frames / 50 / 3600:.1f} h) -> {a.out}, {os.path.getsize(a.out) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
