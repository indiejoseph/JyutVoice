"""Does a trained encoder actually use its BERT input? Validation losses with the real features, zeros, and another sentence's features.

    python scripts/diag_bert_use.py --config configs/yue_encoder.yaml --ckpt checkpoints_yue/checkpoint_epoch=009.ckpt [--corpus-f0 60]

Runs the checkpoint over the configured validation split (the same rows the training validated on). `zeros` removes the BERT information, `other` gives every
sentence the features of its neighbour in the batch (cut or zero-padded to its own length): if `other` is worse than `zeros`, the model reads
sentence-specific information from the features; if `zeros` equals `real`, it does not use them. Also prints how large the BERT term is next to the phone
embeddings, and with --corpus-f0 N the pitch spread (std of voiced F0 in semitones) of N real validation clips for comparison with synthesised audio.
"""
import argparse
import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    from hyperpyyaml import load_hyperpyyaml
    from torch.utils.data import DataLoader

    from jyutvoice.data.text_mel_datamodule import TextMelBatchCollate, read_audio

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus-f0", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    with open(a.config) as f:
        cfg = load_hyperpyyaml(f)
    model, dm = cfg["tts"], cfg["data"]
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model = model.eval().to(dev)
    dm.setup()
    loader = DataLoader(dm.validset, batch_size=16, shuffle=False, num_workers=2, collate_fn=TextMelBatchCollate())
    assert model.encoder.bert_dim > 0, "this checkpoint has no BERT input"

    tot = {k: np.zeros(2) for k in ("real", "zeros", "other")}
    n_tok = 0
    mu_diff, mu_std, ratio = [], [], []
    for batch in loader:
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        B = batch["x"].shape[0]
        real = batch["bert"]
        zeros = torch.zeros_like(real)
        other = torch.roll(real, 1, 0) * (batch["x"] != 0).any(-1, keepdim=True).unsqueeze(-1)  # neighbour's features
        lens = batch["x_lengths"]
        mask = (torch.arange(real.shape[1], device=dev)[None] < lens[:, None]).unsqueeze(-1)
        other = other * mask
        with torch.no_grad():
            for name, feats in (("real", real), ("zeros", zeros), ("other", other)):
                b = dict(batch, bert=feats)
                losses = model.get_losses(b)
                tot[name] += B * np.array([losses["dur_loss"].item(), losses["prior_loss"].item()])
            spk = F.normalize(batch["spk_embed"], dim=1)
            args = (batch["x"], lens, batch["lang"], batch["tone"], batch["word_pos"], batch["syllable_pos"], spk)
            _, mu_r, xm = model.encoder(*args, bert=real)
            _, mu_z, _ = model.encoder(*args, bert=zeros)
            m = xm.bool().expand_as(mu_r)
            mu_diff.append((mu_r - mu_z).abs()[m].mean().item())
            mu_std.append(mu_r[m].std().item())
            enc = model.encoder
            emb = (enc.emb(batch["x"]) + enc.tone_emb(batch["tone"]) + enc.lang_emb(batch["lang"]) + enc.word_pos_emb(batch["word_pos"])
                   + enc.syllable_pos(batch["syllable_pos"])) * enc.n_channels**0.5
            br = enc.bert_proj(real)
            ratio.append((br[mask.expand_as(br)].std() / emb[mask.expand_as(emb)].std()).item())
        n_tok += B
    print(f"{n_tok} validation clips from {a.ckpt}")
    for name in ("real", "zeros", "other"):
        d, p = tot[name] / n_tok
        print(f"  bert={name:5s}: duration loss {d:.4f}  prior loss {p:.4f}  total {d + p:.4f}")
    print(f"  mean |mu(real) - mu(zeros)| = {np.mean(mu_diff):.4f} (std of mu {np.mean(mu_std):.3f}) -> the BERT term moves the encoder output by {100 * np.mean(mu_diff) / np.mean(mu_std):.1f}% of its spread")
    print(f"  std of the BERT term / std of the phone embedding sum = {np.mean(ratio):.3f}")

    if a.corpus_f0:
        import librosa

        ds = dm.validset.dataset
        f0s = []
        for i in range(min(a.corpus_f0, len(ds))):
            y, sr, _ = read_audio(ds[i]["audio"])
            y = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
            f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=400, sr=16000, frame_length=1024, hop_length=160)
            v = f0[voiced & ~np.isnan(f0)]
            if len(v) >= 10:
                f0s.append(float((12 * np.log2(v / np.median(v))).std()))
        print(f"  corpus clips: pitch spread (std of voiced F0, semitones) mean {np.mean(f0s):.2f}, median {np.median(f0s):.2f}, n={len(f0s)}")


if __name__ == "__main__":
    main()
