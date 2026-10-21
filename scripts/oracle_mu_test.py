"""Is the flat pitch caused by the encoder's smoothed output, or by the frozen decoder / vocoder?

    python scripts/oracle_mu_test.py --data-config configs/yue_encoder.yaml --ckpt checkpoints_yue/checkpoint_epoch=009.ckpt --n 40

For held-out clips of the corpus (2.5-8 s) the frozen CosyVoice2 flow decoder + HiFT render the same utterance from different conditioning, all with the
same speaker, the same noise seed and (except the last row) the true timing found by MAS:

    real             the recording itself
    oracle           mu = the true flow-encoder states (decoder_h) of the real audio: what the decoder was trained on; an upper bound for this recipe
    student_true_dur mu = the trained text encoder's output, expanded with the true durations
    student_pred_dur the normal synthesis path (predicted durations; timing differs from the recording, so only spread statistics apply)

Per condition (mean over clips): pitch spread (std of voiced F0 in semitones around the clip's median), and for the equal-timing rows the correlation of the
semitone contour with the recording's (frames voiced in both) and the mel L1 distance to the recording. Also the temporal std of mu per channel
(student / true): below 1 means the encoder output moves less over time than the real states.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


def f0_semitones(wav16):
    import librosa

    f0, voiced, _ = librosa.pyin(wav16, fmin=70, fmax=400, sr=16000, frame_length=1024, hop_length=160)
    ok = voiced & ~np.isnan(f0)
    st = np.full(len(f0), np.nan)
    if ok.sum() >= 10:
        st[ok] = 12 * np.log2(f0[ok] / np.median(f0[ok]))
    return st


def to16(wav24):
    import librosa

    return librosa.resample(np.asarray(wav24, dtype=np.float32), orig_sr=24000, target_sr=16000)


def main():
    from hyperpyyaml import load_hyperpyyaml

    from jyutvoice.data.text_mel_datamodule import TextMelBatchCollate, read_audio
    from jyutvoice.utils.mask import make_pad_mask
    from synth_compare import build_tts

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-sec", type=float, default=2.5)
    ap.add_argument("--max-sec", type=float, default=8.0)
    ap.add_argument("--base-config", default=os.path.join(ROOT, "configs", "base.yaml"))
    ap.add_argument("--flow-decoder", default=os.path.join(ROOT, "pretrained_models", "flow_decoder.pt"))
    ap.add_argument("--hift", default=os.path.join(ROOT, "pretrained_models", "hift.pt"))
    ap.add_argument("--bert-dim", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-wavs", help="directory for the rendered wavs (to listen to)")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    with open(a.data_config) as f:
        dm = load_hyperpyyaml(f)["data"]
    dm.setup()
    tts, cfg = build_tts(a.base_config, a.ckpt, a.flow_decoder, a.bert_dim, dev)
    hift = cfg["hift"].eval().to(dev)
    hift.load_state_dict(torch.load(a.hift, map_location="cpu"))
    collate = TextMelBatchCollate()
    ds = dm.validset

    rows = {k: [] for k in ("real", "oracle", "student_true_dur", "student_pred_dur")}
    contour_r = {"oracle": [], "student_true_dur": []}
    mel_l1 = {"oracle": [], "student_true_dur": []}
    mu_ratio = []
    saved = 0
    for i in range(len(ds.dataset)):
        if len(rows["real"]) >= a.n:
            break
        row = ds.dataset[i]
        item = ds.get_datapoint(row)
        if item is None or not (a.min_sec <= item["decoder_h"].shape[0] / 50 <= a.max_sec):
            continue
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate([item]).items()}
        bert = batch["bert"]
        T = int(batch["y_lengths"][0])
        with torch.no_grad():
            spk = F.normalize(batch["spk_embed"], dim=1)
            c = tts.spk_embed_affine_layer(spk)
            _, mu_x, _ = tts.encoder(batch["x"], batch["x_lengths"], batch["lang"], batch["tone"], batch["word_pos"], batch["syllable_pos"], spk, bert=bert)
            _, _, attn = tts(x=batch["x"], x_lengths=batch["x_lengths"], y=batch["y"], y_lengths=batch["y_lengths"], lang=batch["lang"], tone=batch["tone"],
                             word_pos=batch["word_pos"], syllable_pos=batch["syllable_pos"], spk_embed=batch["spk_embed"], decoder_h=batch["decoder_h"], bert=bert)  # fmt: skip
            mu_student = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)  # (1, 80, T)
            mu_true = batch["decoder_h"].transpose(1, 2)
            mask = (~make_pad_mask(batch["y_lengths"])).to(mu_true.dtype).unsqueeze(1)
            mu_ratio.append((mu_student[0, :, :T].std(dim=1) / mu_true[0, :, :T].std(dim=1).clamp_min(1e-6)).mean().item())
            wavs, mels = {}, {}
            for name, mu in (("oracle", mu_true), ("student_true_dur", mu_student)):
                torch.manual_seed(a.seed + i)
                mel, _ = tts.decoder(mu=mu, mask=mask, spks=c, cond=torch.zeros_like(mu), n_timesteps=10, temperature=1.0, streaming=False)
                wavs[name], _ = hift.inference(mel)
                mels[name] = mel
            torch.manual_seed(a.seed + i)
            out = tts.synthesise(x=batch["x"], x_lengths=batch["x_lengths"], lang=batch["lang"], tone=batch["tone"], word_pos=batch["word_pos"],
                                 syllable_pos=batch["syllable_pos"], spk_embed=batch["spk_embed"], n_timesteps=10, bert=bert)  # fmt: skip
            wavs["student_pred_dur"], _ = hift.inference(out["mel"])
        real, sr, _ = read_audio(row["audio"])
        import librosa

        real16 = librosa.resample(real, orig_sr=sr, target_sr=16000) if sr != 16000 else real
        st = {"real": f0_semitones(real16[: T * 320])}  # 50 Hz mel frames = 320 samples at 16 kHz
        for name, w in wavs.items():
            st[name] = f0_semitones(to16(w.squeeze().cpu().numpy()))
        if any(np.isnan(v).all() for v in st.values()):
            continue
        for name, v in st.items():
            rows[name].append(np.nanstd(v))
        for name in contour_r:
            n = min(len(st["real"]), len(st[name]))
            both = ~np.isnan(st["real"][:n]) & ~np.isnan(st[name][:n])
            contour_r[name].append(np.corrcoef(st["real"][:n][both], st[name][:n][both])[0, 1] if both.sum() > 10 else np.nan)
            mel_l1[name].append((mels[name][0, :, :T] - batch["y"][0, :, :T]).abs().mean().item())
        if a.save_wavs and saved < 8:
            import soundfile as sf

            os.makedirs(a.save_wavs, exist_ok=True)
            sf.write(os.path.join(a.save_wavs, f"c{saved}_real.wav"), real[: T * 480], sr)
            for name, w in wavs.items():
                sf.write(os.path.join(a.save_wavs, f"c{saved}_{name}.wav"), w.squeeze().cpu().numpy(), 24000)
            saved += 1

    n = len(rows["real"])
    print(f"{n} held-out clips ({a.min_sec}-{a.max_sec} s), checkpoint {a.ckpt}\n")
    print(f"{'condition':18s} {'F0 spread (semitones)':>22s} {'contour r vs real':>18s} {'mel L1 vs real':>15s}")
    for name in ("real", "oracle", "student_true_dur", "student_pred_dur"):
        r = f"{np.nanmean(contour_r[name]):.3f}" if name in contour_r else "-"
        m = f"{np.mean(mel_l1[name]):.3f}" if name in mel_l1 else "-"
        print(f"{name:18s} {np.mean(rows[name]):22.2f} {r:>18s} {m:>15s}")
    print(f"\ntemporal std of mu per channel, student / true: {np.mean(mu_ratio):.3f} (1.0 = as much movement as the real flow-encoder states)")


if __name__ == "__main__":
    main()
