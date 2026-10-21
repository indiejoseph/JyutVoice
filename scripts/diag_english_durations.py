"""Do the duration predictors give English phones the frames they have in real recordings, and the stress contrast? On held-out rows with English words.

    python scripts/diag_english_durations.py --data-config configs/yue_en_flowdp.yaml --dataset-path '/home/joseph/data_local/mce_ds/shards_sil/shard_*' \\
        --model flow=configs/synth_flowdp.yaml:checkpoints_yue_en_flowdp/checkpoint_epoch=018.ckpt --model regression=configs/base.yaml:checkpoints_yue_en/checkpoint_epoch=014.ckpt

For every row of the validation split the alignment (MAS, with the pause-aware bonus when the config has it) gives the frames of each token; each `--model name=config:ckpt` predicts them
from the text alone (flow sampler: mean of `--samples` draws at the render temperatures 0.3 / 1.0; regression: exp(logw)). Reported over the English tokens (language id 2),
per group: English consonants, English vowels by stress (unstressed / primary / secondary tone ids 8 / 9 / 10) and the blanks next to English tokens: mean aligned frames,
mean predicted frames and their ratio. Then the stress contrast (primary / unstressed vowel) in both, and the frames of whole English words (first phone to last phone, blanks
between them included) aligned vs predicted. Frames are 50 Hz (20 ms).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

VOWEL_STRESS = {8: "unstressed", 9: "primary", 10: "secondary"}


def main():
    import synth_compare as sc
    from hyperpyyaml import load_hyperpyyaml
    from torch.utils.data import DataLoader

    from jyutvoice.data.text_mel_datamodule import TextMelBatchCollate
    from jyutvoice.models.flow_duration_predictor import FlowDurationPredictor

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--dataset-path", nargs="+", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=config:ckpt")
    ap.add_argument("--max-rows", type=int, default=400)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--dp-temperature", type=float, default=0.3)
    ap.add_argument("--dp-pause-temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = torch.device(a.device)
    with open(a.data_config) as f:
        cfg = load_hyperpyyaml(f)
    tts_ref, dm = cfg["tts"], cfg["data"]
    dm.hparams.dataset_path = [os.path.expanduser(p) for p in a.dataset_path]
    dm.setup()
    flow_decoder = os.path.join(HERE, "..", "pretrained_models", "flow_decoder.pt")
    models = {}
    for spec in a.model:
        name, rest = spec.split("=", 1)
        config, ckpt = rest.split(":", 1)
        models[name] = sc.build_tts(config, ckpt, flow_decoder, 0, dev)[0]
    # the alignment comes from the first model's encoder (all models here share one frozen encoder)
    first = next(iter(models.values()))
    ref = {"mas": {}, **{n: {} for n in models}}
    n_rows = min(len(dm.validset), a.max_rows)
    loader = DataLoader(torch.utils.data.Subset(dm.validset, range(n_rows)), batch_size=16, shuffle=False, num_workers=4, collate_fn=TextMelBatchCollate())
    words = {k: [] for k in ["mas", *models]}
    shape = {k: [] for k in ["mas", *models]}  # (first vowel, last vowel) frames of words with >= 2 vowels and primary stress on the first
    tokens = {k: {} for k in ["mas", *models]}
    for b in loader:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.no_grad():
            _, _, attn = first(x=b["x"], x_lengths=b["x_lengths"], y=b["y"], y_lengths=b["y_lengths"], lang=b["lang"], tone=b["tone"], word_pos=b["word_pos"],
                               syllable_pos=b["syllable_pos"], spk_embed=b["spk_embed"], decoder_h=b["decoder_h"], silence=b.get("silence"), pause_mask=b.get("pause_mask"))  # fmt: skip
            attn = attn.squeeze(1) if attn.dim() == 4 else attn
            mas = attn.sum(-1)  # (B, T_text)
            preds = {}
            spk = F.normalize(b["spk_embed"], dim=1)
            for name, m in models.items():
                x_enc, _, x_mask = m.encoder(b["x"], b["x_lengths"], b["lang"], b["tone"], b["word_pos"], b["syllable_pos"], spk)
                if isinstance(m.dp, FlowDurationPredictor):
                    pm = b["pause_mask"].bool() if b.get("pause_mask") is not None else torch.zeros_like(b["x"], dtype=torch.bool)
                    draws = []
                    for k in range(a.samples):
                        torch.manual_seed(k)
                        temp = torch.where(pm.unsqueeze(1), torch.full_like(x_mask, a.dp_pause_temperature), torch.full_like(x_mask, a.dp_temperature))
                        draws.append((torch.exp(m.dp(x_enc, x_mask, g=spk, temperature=temp)) * x_mask)[:, 0])
                    preds[name] = torch.stack(draws).mean(0)
                else:
                    preds[name] = (torch.exp(m.dp(x_enc, x_mask)) * x_mask)[:, 0]
        for i in range(b["x"].shape[0]):
            tt = int(b["x_lengths"][i])
            lang, tone = b["lang"][i, :tt].cpu().numpy(), b["tone"][i, :tt].cpu().numpy()
            en = lang == 2
            if not en.any():
                continue
            series = {"mas": mas[i, :tt].cpu().numpy(), **{n: p[i, :tt].cpu().numpy() for n, p in preds.items()}}
            for key, d in series.items():
                for g, sel in [("consonant", en & (tone == 7)), *[(f"vowel {nm}", en & (tone == t)) for t, nm in VOWEL_STRESS.items()],
                               ("blank before English", (lang == 0) & (np.roll(en, -1))), ("blank after English", (lang == 0) & np.roll(en, 1))]:
                    tokens[key].setdefault(g, []).extend(d[sel].tolist())
                # whole English words: maximal runs of English phones with the blanks between them
                idx = np.flatnonzero(en)
                if len(idx):
                    cuts = np.flatnonzero(np.diff(idx) > 2) + 1  # a gap of more than one blank = another word (or a word separator token)
                    for run in np.split(idx, cuts):
                        words[key].append(float(d[run[0]:run[-1] + 1].sum()))
                        v = [j for j in run if tone[j] in VOWEL_STRESS]
                        if len(v) >= 2 and tone[v[0]] == 9:
                            shape[key].append((float(d[v[0]]), float(d[v[-1]])))
    names = ["mas", *models]
    print(f"{n_rows} validation rows, English tokens per group (aligned): " + ", ".join(f"{g} {len(v)}" for g, v in tokens["mas"].items()))
    print(f"{'group':26}" + "".join(f"{n:>12}" for n in names) + "".join(f"{n + '/mas':>16}" for n in names[1:]))
    for g in tokens["mas"]:
        m = {n: float(np.mean(tokens[n][g])) for n in names}
        print(f"{g:26}" + "".join(f"{m[n]:>12.2f}" for n in names) + "".join(f"{m[n] / m['mas']:>16.2f}" for n in names[1:]))
    cont = {n: np.mean(tokens[n]["vowel primary"]) / np.mean(tokens[n]["vowel unstressed"]) for n in names}
    print("stress contrast (primary / unstressed vowel): " + ", ".join(f"{n} {cont[n]:.2f}" for n in names))
    for n in names:
        fl = np.asarray(shape[n])
        print(f"words stressed on the first syllable ({len(fl)}), first vowel / last vowel frames, {n}: {fl[:, 0].mean():.2f} / {fl[:, 1].mean():.2f}, last is longer in {100 * np.mean(fl[:, 1] > fl[:, 0]):.0f}%")
    wm = {n: float(np.mean(words[n])) for n in names}
    print(f"English word frames (first to last phone): " + ", ".join(f"{n} {wm[n]:.1f}" for n in names) + f" over {len(words['mas'])} words")


if __name__ == "__main__":
    main()
