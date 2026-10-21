"""Synthesise the same Cantonese sentences with several encoder checkpoints for listening and recogniser checks.

    python scripts/synth_compare.py --out synth_out --dataset-dir ~/jyutvoice_data/dataset_yue --bert-model <ModernCantoneseBert dir> \\
        --bert-stats ~/jyutvoice_data/bert_yue/stats.npz \\
        --model old=pretrained_models/20260102_encoder_only.ckpt:0 --model bert=checkpoints_yue/checkpoint_epoch=009.ckpt:1024

Everything except the encoder is shared: the CosyVoice2 flow decoder (flow_decoder.pt, also its speaker layer), the HiFT vocoder, the speaker
embeddings (taken from clips of the training corpus, picked to be far apart) and the random seed per utterance. A checkpoint trained without BERT is
`:0` (add `@cumulative` or `@round` to change how predicted durations become frames, default `ceil`, and `!noblank` for a model trained with
`add_blank: false`: the phone sequence then has no blank tokens, e.g. `nb=ckpt:0@round!noblank`); a BERT checkpoint needs `--bert-model` and the statistics file written by build_bert_sidefile.py (the features must be standardised the same way
as in training). Models and phones are fed the same input: the text is word-segmented with pydips and read by g2p (no label overrides).

`--prompt` gives the frozen decoder each speaker's reference clip as a CosyVoice-style zero-shot prompt (the clip's mel as `cond` and its flow-encoder
states `decoder_h` in front of the generated ones; the clip is the one the speaker's x-vector came from, at most `--prompt-max-sec` long). Without it the
decoder only gets the x-vector.

`--match-teacher MANIFEST --prompts-meta META` reproduces the teacher's voices (scripts/teacher_render.py): each sentence is synthesised only with the speaker
of the prompt the teacher used for it (the manifest's `prompt` index into META's prompts, which are rows of the corpus: shard + row), with that row's stored
x-vector and, as the teacher's flow decoder gets its prompt clip in both modes, its whole clip as decoder prompt (add `--prompt` and `--prompt-max-sec 30`).
`spk` in results.jsonl is then the prompt index.

Writes <out>/wavs/<model>/<speaker>/<sentence>.wav (24 kHz) and <out>/results.jsonl, one line per utterance with the intended syllables (what g2p
gave the model), the number of mel frames and the duration. The noise realisation differs between models whenever their predicted lengths differ.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def pick_speakers(dataset_dir, n):
    """n speaker embeddings from the corpus, chosen by farthest-point sampling (cosine) over the first clip of 40 spread-out shards."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    files = sorted(glob.glob(os.path.join(dataset_dir, "data-*.arrow")))
    cands = []
    for fn in files[:: max(1, len(files) // 40)][:40]:
        b = next(iter(ipc.open_stream(pa.memory_map(fn)))).select(["id", "spk_emb"])
        v = np.asarray(b.column("spk_emb")[0].as_py(), dtype=np.float32)
        cands.append((b.column("id")[0].as_py(), v / np.linalg.norm(v)))
    chosen = [0]
    while len(chosen) < n:
        best = max((i for i in range(len(cands)) if i not in chosen), key=lambda i: min(1 - float(cands[i][1] @ cands[j][1]) for j in chosen))
        chosen.append(best)
    return [(cands[i][0], torch.from_numpy(cands[i][1]).unsqueeze(0)) for i in chosen]


def load_prompts(dataset_dir, ids, max_sec):
    """{row id: (mel (T, 80), decoder_h (T, 80))} of the real clips `pick_speakers` took the x-vectors from, for use as decoder prompts.

    The mel is the flow decoder's (24 kHz, n_fft 1920, hop 480, fmax 8 kHz, no centering: the datamodule's); both are cut to the shorter of the two lengths and
    to `max_sec` seconds (50 frames per second)."""
    import io

    import librosa
    import pyarrow as pa
    import pyarrow.ipc as ipc

    from jyutvoice.utils.audio import mel_spectrogram

    files = sorted(glob.glob(os.path.join(dataset_dir, "data-*.arrow")))
    out = {}
    for fn in files[:: max(1, len(files) // 40)][:40]:
        b = next(iter(ipc.open_stream(pa.memory_map(fn)))).select(["id", "audio", "decoder_h"])
        rid = b.column("id")[0].as_py()
        if rid not in ids:
            continue
        y, sr = sf.read(io.BytesIO(b.column("audio")[0].as_py()["bytes"]), dtype="float32")
        y = y.mean(-1) if y.ndim > 1 else y
        if sr != 24000:
            y = librosa.resample(y, orig_sr=sr, target_sr=24000)
        mel = mel_spectrogram(torch.from_numpy(y).unsqueeze(0), 1920, 80, 24000, 480, 1920, 0, 8000, center=False).squeeze(0).T  # (T, 80)
        h = torch.tensor(b.column("decoder_h")[0].as_py(), dtype=torch.float32)  # (T, 80)
        n = min(mel.shape[0], h.shape[0], int(max_sec * 50))
        print(f"prompt {rid}: mel {mel.shape[0]} frames, decoder_h {h.shape[0]} frames, using {n} ({n / 50:.1f} s)", flush=True)
        out[rid] = (mel[:n], h[:n])
    missing = set(ids) - set(out)
    assert not missing, f"no reference clip found for {missing}"
    return out


def load_prompt_rows(dataset_dir, metas, max_sec):
    """The teacher's prompt rows: [(row id, normalised stored x-vector (1, 192), mel (T, 80), decoder_h (T, 80))] for META's prompts (file `shard`, row number `row`)."""
    import io

    import librosa
    import pyarrow as pa
    import pyarrow.ipc as ipc

    from jyutvoice.utils.audio import mel_spectrogram

    out = []
    for m in metas:
        n_seen, found = 0, None
        for b in ipc.open_stream(pa.memory_map(os.path.join(dataset_dir, m["shard"]))):
            if n_seen + b.num_rows > m["row"]:
                found = b.slice(m["row"] - n_seen, 1).select(["id", "spk_emb", "audio", "decoder_h"])
                break
            n_seen += b.num_rows
        assert found is not None and found.column("id")[0].as_py() == m["id"], (m["id"], m["shard"], m["row"])
        y, sr = sf.read(io.BytesIO(found.column("audio")[0].as_py()["bytes"]), dtype="float32")
        y = y.mean(-1) if y.ndim > 1 else y
        if sr != 24000:
            y = librosa.resample(y, orig_sr=sr, target_sr=24000)
        mel = mel_spectrogram(torch.from_numpy(y).unsqueeze(0), 1920, 80, 24000, 480, 1920, 0, 8000, center=False).squeeze(0).T
        h = torch.tensor(found.column("decoder_h")[0].as_py(), dtype=torch.float32)
        n = min(mel.shape[0], h.shape[0], int(max_sec * 50))
        e = torch.tensor(found.column("spk_emb")[0].as_py(), dtype=torch.float32)
        print(f"teacher prompt {m['id']}: mel {mel.shape[0]} frames, decoder_h {h.shape[0]} frames, using {n} ({n / 50:.1f} s)", flush=True)
        out.append((m["id"], (e / e.norm()).unsqueeze(0), mel[:n], h[:n]))
    return out


def build_tts(config, ckpt, flow_decoder, bert_dim, device):
    from hyperpyyaml import load_hyperpyyaml

    with open(config) as f:
        cfg = load_hyperpyyaml(f, overrides={"bert_dim": bert_dim})
    tts = cfg["tts"]
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = dict(state["state_dict"] if "state_dict" in state else state)
    state.update(torch.load(flow_decoder, map_location="cpu"))  # the pretrained flow decoder and its speaker layer
    tts.load_state_dict(state)  # strict: a missing or unexpected key (e.g. bert_proj) is an error
    return tts.eval().to(device), cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", required=True, help="name=checkpoint:bert_dim[@ceil|round|cumulative][!noblank] (bert_dim 0 = no BERT input)")
    ap.add_argument("--sentences", default=os.path.join(ROOT, "scripts", "synth_sentences_yue.txt"))
    ap.add_argument("--dataset-dir", required=True, help="for the speaker embeddings")
    ap.add_argument("--n-speakers", type=int, default=2)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "base.yaml"))
    ap.add_argument("--flow-decoder", default=os.path.join(ROOT, "pretrained_models", "flow_decoder.pt"))
    ap.add_argument("--hift", default=os.path.join(ROOT, "pretrained_models", "hift.pt"))
    ap.add_argument("--bert-model")
    ap.add_argument("--bert-stats")
    ap.add_argument("--n-timesteps", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--segmentation", choices=["pydips", "clause", "char", "given"], default="pydips",
                    help="how the text is cut into words before g2p: pydips (coarse, as in training; English phrases stay whole), clauses (cut at punctuation only), "
                         "one character per word, or given (the sentences file is already word-segmented: use with --no-text-normalize)")
    ap.add_argument("--no-text-normalize", action="store_true",
                    help="skip the WeTextProcessing stage (numbers, dates, money, symbols -> spoken Cantonese) that runs before the segmentation")
    ap.add_argument("--flat-word-pos", action="store_true", help="give every phone the same word-position id (readings and segmentation unchanged otherwise)")
    ap.add_argument("--match-teacher", help="teacher_render.py manifest.jsonl: synthesise each sentence in the teacher's voice for it (see above)")
    ap.add_argument("--prompts-meta", help="the teacher's prompts/meta.json (with --match-teacher)")
    ap.add_argument("--prompt", action="store_true", help="use each speaker's reference clip as the decoder's zero-shot prompt (see above)")
    ap.add_argument("--prompt-max-sec", type=float, default=6.0)
    ap.add_argument("--prosody-ckpt", help="prosody_cfm_*.pt from scripts/train_prosody_cfm.py: sample the decoder's input features with it instead of using the encoder's mean")
    ap.add_argument("--prosody-cfg-scale", type=float, default=0.0)
    ap.add_argument("--prosody-temperature", type=float, default=0.5)
    ap.add_argument("--prosody-steps", type=int, default=10)
    ap.add_argument("--dp-temperature", type=float, help="flow-matching duration predictor: noise temperature (default: the model's infer_temperature)")
    ap.add_argument("--dp-pause-temperature", type=float, help="flow-matching duration predictor: noise temperature for the pause tokens (punctuation, edge pads) "
                    "while --dp-temperature applies to the others")
    ap.add_argument("--dp-cfg-scale", type=float, help="flow-matching duration predictor: guidance scale (default: the model's infer_cfg_scale)")
    ap.add_argument("--length-scale", type=float, default=1.0, help="multiplies every token's duration after rounding (1.1 = 10%% slower): calibrates the total length of a sampler "
                    "whose speaking rate differs from the teacher's (scripts/length_vs_teacher.py), also stretches the pauses")
    a = ap.parse_args()

    from jyutvoice.text import _symbol_to_id, text_to_sequence
    from jyutvoice.text.bert_align import BertPhoneFeatures, expand_to_phones
    from jyutvoice.text.cantonese.readings import cleaned, default_jyutping
    from jyutvoice.text.symbols import punctuations
    from jyutvoice.utils.utils import intersperse

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    punct_ids = torch.tensor(sorted({_symbol_to_id[p] for p in punctuations if p in _symbol_to_id}), device=device)
    specs = []
    for m in a.model:
        name, rest = m.split("=", 1)
        ckpt, tail = rest.rsplit(":", 1)
        tail, _, flags = tail.partition("!")
        bert_dim, _, rounding = tail.partition("@")
        specs.append((name, ckpt, int(bert_dim), rounding or "ceil", flags != "noblank"))
    sentences = [s.strip() for s in open(a.sentences, encoding="utf-8") if s.strip()]
    if not a.no_text_normalize:  # raw text in, before any segmentation: the normaliser's patterns need the digits and units unseparated
        from jyutvoice.text.text_processor import normalize_text

        sentences = [normalize_text(s, "yue") for s in sentences]
    if a.segmentation == "pydips":
        from pydips import BertModel

        from jyutvoice.text.text_processor import segment_mixed

        ws = BertModel()
        segmented = [" ".join(segment_mixed(s, lambda t: ws.cut(t, mode="coarse"))) for s in sentences]  # English phrases stay whole
    elif a.segmentation == "given":  # the sentences file is already normalised and word-segmented (e.g. by prepare_text on another machine)
        segmented = list(sentences)
    elif a.segmentation == "clause":
        import re

        segmented = [" ".join(t for t in re.split(r"([，。？！、；：…,.?!;:])", s.replace(" ", "")) if t) for s in sentences]
    else:
        segmented = [" ".join(s.replace(" ", "")) for s in sentences]

    bert = None
    if any(d > 0 for _, _, d, _, _ in specs):
        bert = BertPhoneFeatures.from_pretrained(a.bert_model, device=str(device))
        stats = np.load(a.bert_stats)
        bert.mean, bert.std = torch.from_numpy(stats["mean"]), torch.from_numpy(stats["std"])

    features = []  # per sentence: tensors fed to every model
    for text in segmented:
        ids, tones, word_pos, syl_pos, lang, word2ph = text_to_sequence(text, "yue", return_word2ph=True)
        if a.flat_word_pos:
            word_pos = [1 if p != 0 else 0 for p in word_pos]  # 0 stays the id of the pads at both ends
        unit = bert.unit_features([cleaned(text)])[0] if bert is not None else None
        features.append({
            "x": ids, "tone": tones, "word_pos": word_pos, "syllable_pos": syl_pos, "lang": lang, "word2ph": word2ph, "unit": unit,
            "intended": [s for s in default_jyutping(cleaned(text)) if s not in punctuations],
        })  # fmt: skip

    prompt_of = None
    if a.match_teacher:
        metas = json.load(open(a.prompts_meta, encoding="utf-8"))["prompts"]
        rows_t = load_prompt_rows(a.dataset_dir, metas, a.prompt_max_sec)
        speakers = [(rid, e) for rid, e, _, _ in rows_t]
        prompts = {rid: (mel, h) for rid, _, mel, h in rows_t}
        prompt_of = {}
        for line in open(a.match_teacher, encoding="utf-8"):
            r = json.loads(line)
            prompt_of[r["text"]] = r["prompt"]
    else:
        speakers = pick_speakers(a.dataset_dir, a.n_speakers)
        prompts = load_prompts(a.dataset_dir, {s for s, _ in speakers}, a.prompt_max_sec) if a.prompt else {}
    print("speakers:", [s for s, _ in speakers], flush=True)
    os.makedirs(a.out, exist_ok=True)
    results = open(os.path.join(a.out, "results.jsonl"), "w", encoding="utf-8")
    hift = None
    for name, ckpt, bert_dim, rounding, add_blank in specs:
        tts, cfg = build_tts(a.config, ckpt, a.flow_decoder, bert_dim, device)
        if a.prosody_ckpt:
            from jyutvoice.models.prosody_cfm import ProsodyCFM

            pc = torch.load(a.prosody_ckpt, map_location="cpu", weights_only=False)
            pm = ProsodyCFM(**pc["model_config"])
            pm.load_state_dict(pc["ema"])
            tts.prosody_cfm = pm.eval().to(device)
            print(f"prosody model from {a.prosody_ckpt} (epoch {pc['epoch']}, val {pc['val_loss']:.4f}); cfg {a.prosody_cfg_scale}, temperature {a.prosody_temperature}, steps {a.prosody_steps}", flush=True)
        if hasattr(tts.dp, "infer_temperature"):
            if a.dp_temperature is not None:
                tts.dp.infer_temperature = a.dp_temperature
            if a.dp_cfg_scale is not None:
                tts.dp.infer_cfg_scale = a.dp_cfg_scale
        if hift is None:
            hift = cfg["hift"].eval().to(device)
            hift.load_state_dict(torch.load(a.hift, map_location="cpu"))
        t0 = time.time()
        for sp, (sp_id, spk) in enumerate(speakers):
            os.makedirs(os.path.join(a.out, "wavs", name, f"spk{sp}"), exist_ok=True)
            for k, (text, feat) in enumerate(zip(segmented, features)):
                if prompt_of is not None and prompt_of.get(sentences[k]) != sp:
                    continue  # this sentence belongs to another teacher voice
                t = lambda v: torch.tensor([intersperse(v, 0) if add_blank else v], device=device)  # noqa: E731
                x = t(feat["x"])
                kw = dict(
                    x=x, x_lengths=torch.tensor([x.shape[1]], device=device), lang=t(feat["lang"]), tone=t(feat["tone"]), word_pos=t(feat["word_pos"]),
                    syllable_pos=t(feat["syllable_pos"]), spk_embed=spk.to(device), n_timesteps=a.n_timesteps, duration_rounding=rounding,
                )  # fmt: skip
                if a.dp_pause_temperature is not None:  # same tokens as the datamodule's pause mask: punctuation and the pads at both ends (with their blanks)
                    pm = torch.isin(x[0], punct_ids)
                    edge = 3 if add_blank else 1
                    pm[:edge] = True
                    pm[-edge:] = True
                    kw["dp_temperature"], kw["dp_pause_temperature"], kw["dp_pause_mask"] = a.dp_temperature, a.dp_pause_temperature, pm.unsqueeze(0)
                if a.length_scale != 1.0:
                    kw["length_scale"] = a.length_scale
                if a.prosody_ckpt:
                    kw["prosody_kwargs"] = dict(n_steps=a.prosody_steps, cfg_scale=a.prosody_cfg_scale, temperature=a.prosody_temperature)
                if a.prompt:
                    mel_p, h_p = prompts[sp_id]
                    kw["prompt_feat"], kw["prompt_h"] = mel_p.unsqueeze(0).to(device), h_p.unsqueeze(0).to(device)
                if bert_dim > 0:
                    kw["bert"] = expand_to_phones(feat["unit"], feat["word2ph"], add_blank=add_blank).unsqueeze(0).to(device)
                torch.manual_seed(a.seed + 100 * sp + k)
                with torch.no_grad():
                    out = tts.synthesise(**kw)
                    wav, _ = hift.inference(out["mel"])
                path = os.path.join("wavs", name, f"spk{sp}", f"s{k:02d}.wav")
                sf.write(os.path.join(a.out, path), wav.squeeze().cpu().numpy(), 24000)
                results.write(json.dumps({
                    "model": name, "spk": sp, "spk_id": sp_id, "sid": k, "text": sentences[k], "seg_text": text, "intended": feat["intended"],
                    "frames": int(out["mel"].shape[-1]), "dur": float(wav.shape[-1] / 24000), "wav": path,
                }, ensure_ascii=False) + "\n")  # fmt: skip
        results.flush()
        print(f"{name}: {len(speakers) * len(sentences)} utterances in {time.time() - t0:.0f} s", flush=True)
        del tts
        torch.cuda.empty_cache()
    results.close()


if __name__ == "__main__":
    main()
