"""Frames the duration sampler gives each token of one sentence, next to what the alignment gave the same phone sequence in real recordings.

    python scripts/diag_token_durations.py --sentence "我 喺 Zoom 開會 ，" --ckpt checkpoints_yue_en_flowdp/checkpoint_epoch=018.ckpt \\
        --config configs/synth_flowdp.yaml --dataset-dir <dir with data-*.arrow> --cache /home/joseph/data_local/prosody/cache_mce.pt --word Zoom

Per token of the sentence: symbol, language id, mean +- std of the predicted frames (50 Hz, `--samples` draws of the sampler with the temperatures of the renders,
before rounding). With `--word` (a word of the sentence) and `--cache` (a scripts/build_prosody_cache.py file, which keeps `ids` and the aligned `dur` of every row) the
same phone sequence is searched in the cached rows: how often it occurs and the mean aligned frames per token there. A phone far shorter than in the recordings, or a
word that never occurs in them, is where an unnatural word is to be looked for.
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))


def main():
    import synth_compare as sc
    from jyutvoice.text import text_to_sequence
    from jyutvoice.models.flow_duration_predictor import FlowDurationPredictor
    from jyutvoice.text.symbols import punctuations, symbols
    from jyutvoice.utils.utils import intersperse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sentence", required=True, help="word-segmented sentence (English phrases whole)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--regression", action="store_true", help="the checkpoint has the regression predictor: use configs/base.yaml")
    ap.add_argument("--config", default=os.path.join(HERE, "..", "configs", "synth_flowdp.yaml"))
    ap.add_argument("--dataset-dir", required=True, help="for the speaker embedding")
    ap.add_argument("--cache", help="prosody cache with ids and dur (build_prosody_cache.py)")
    ap.add_argument("--word", help="word of the sentence whose phone sequence is looked up in --cache")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--dp-temperature", type=float, default=0.3)
    ap.add_argument("--dp-pause-temperature", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    if a.regression:
        a.config = os.path.join(HERE, "..", "configs", "base.yaml")
    tts, _ = sc.build_tts(a.config, a.ckpt, os.path.join(HERE, "..", "pretrained_models", "flow_decoder.pt"), 0, dev)
    ids, tones, word_pos, syl_pos, lang, _ = text_to_sequence(a.sentence, "yue", return_word2ph=True)
    t = lambda v: torch.tensor([intersperse(v, 0)], device=dev)  # noqa: E731
    x, tone, wp, sp, lg = t(ids), t(tones), t(word_pos), t(syl_pos), t(lang)
    spk = sc.pick_speakers(a.dataset_dir, 1)[0][1].to(dev)
    x_len = torch.tensor([x.shape[1]], device=dev)
    punct_ids = torch.tensor([symbols.index(p) for p in punctuations if p in symbols], device=dev)
    pm = torch.isin(x[0], punct_ids)
    pm[:3] = True
    pm[-3:] = True
    with torch.no_grad():
        x_enc, _, x_mask = tts.encoder(x, x_len, lg, tone, wp, sp, torch.nn.functional.normalize(spk, dim=1))
        draws = []
        if isinstance(tts.dp, FlowDurationPredictor):
            for k in range(a.samples):
                torch.manual_seed(k)
                temp = torch.where(pm.view(1, 1, -1), torch.full_like(x_mask, a.dp_pause_temperature), torch.full_like(x_mask, a.dp_temperature))
                logw = tts.dp(x_enc, x_mask, g=spk, temperature=temp)
                draws.append((torch.exp(logw) * x_mask)[0, 0].cpu().numpy())
        else:  # regression predictor: one deterministic prediction
            draws.append((torch.exp(tts.dp(x_enc, x_mask)) * x_mask)[0, 0].cpu().numpy())
    draws = np.stack(draws)
    names = [symbols[i] if i < len(symbols) else "?" for i in x[0].tolist()]
    print(f"{'#':>3} {'symbol':8} {'lang':>4} {'frames':>7} {'std':>5}")
    for i, nm in enumerate(names):
        print(f"{i:>3} {nm if i % 2 else '(blank)':8} {lg[0, i].item():>4} {draws[:, i].mean():>7.2f} {draws[:, i].std():>5.2f}")
    print(f"total {draws.sum(1).mean():.1f} frames")
    if a.cache and a.word:
        from jyutvoice.text.english.g2p import en_G2p

        phones = [p for p in en_G2p()(a.word)[0][1]]
        print("word", a.word, "phones", phones)
        want = [symbols.index(p.lower()) if p.lower() in symbols else None for p in phones]
        print("symbol ids", want, "(vowel stress is a separate tone id; the symbol is the bare phone)")
        base = [symbols.index(p.rstrip("012").lower()) for p in phones]
        pat = intersperse(base, 0)
        cache = torch.load(a.cache, map_location="cpu", weights_only=False)
        found = []
        for split in ("train", "valid"):
            for r in cache[split]:
                row = r["ids"].numpy().tolist()
                for s in range(0, len(row) - len(pat) + 1):
                    if row[s:s + len(pat)] == pat:
                        found.append(r["dur"][s:s + len(pat)].numpy().astype(float))
        print(f"occurrences of the phone sequence in {a.cache}: {len(found)}")
        if found:
            m = np.stack(found)
            print("mean aligned frames per token:", " ".join(f"{v:.1f}" for v in m.mean(0)), "| median:", " ".join(f"{v:.0f}" for v in np.median(m, 0)))


if __name__ == "__main__":
    main()
