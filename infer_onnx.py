#!/usr/bin/env python3
"""
ONNX inference for JyutVoice TTS (Cantonese and Cantonese-English code-mixing).

The text encoder, the duration model (flow-matching sampler, or the regression predictor) and ProsodyCFM run in ONNX Runtime (written by scripts/export_onnx.py,
driven by jyutvoice/utils/onnx_synth.py); the CosyVoice2 flow decoder and the HiFT vocoder run in PyTorch. The speaker is the x-vector of a reference clip (CAMPPlus); the
decoder gets no clip prompt (that needs the speech tokenizer and the flow encoder).

The sentence goes through the training front end: text normalisation (numbers, dates, money -> spoken Cantonese; WeTextProcessing), word segmentation (pydips; English phrases
stay whole), g2p. Without pydips (or with already prepared text) use --segmentation given: the text is then used as written, words separated by spaces.

Usage:
    python infer_onnx.py --onnx_dir onnx_models --text "我喺Zoom開會，但係個microphone有問題。" --ref_audio tmp/ref.wav --output output.wav
"""

import argparse

import torch  # before onnxruntime (imported inside load_session): the other order picks up the system libstdc++, which scipy then rejects
import torchaudio
from hyperpyyaml import load_hyperpyyaml

from jyutvoice.utils.onnx_infer import (
    extract_spk_embedding,
    load_flow_decoder,
    prepare_text,
    synthesise_wave,
    to_16k_mono,
)
from jyutvoice.utils.onnx_synth import OnnxAcousticModel, load_session, text_tokens


def main():
    parser = argparse.ArgumentParser(description="JyutVoice TTS ONNX Inference", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--lang", default="yue", choices=["yue", "en"], help="yue: Cantonese, English words inside it are read in English; en: English only")
    parser.add_argument("--phone", default=None, help="Phonetic transcription (for Cantonese, optional)")
    parser.add_argument("--segmentation", default="pydips", choices=["pydips", "given"], help="pydips: cut the Chinese parts into words; given: the text is already word-segmented")
    parser.add_argument("--no_text_normalize", action="store_true", help="skip the WeTextProcessing stage (numbers, dates, money, symbols -> spoken Cantonese)")
    parser.add_argument("--ref_audio", required=True, help="Reference audio file for speaker embedding")
    parser.add_argument("--output", required=True, help="Output audio file path")
    parser.add_argument("--onnx_dir", default="onnx_models", help="Directory written by scripts/export_onnx.py (graphs and manifest.json)")
    parser.add_argument("--config", default="configs/onnx_infer.yaml", help="Configuration file path (for decoder and vocoder)")
    parser.add_argument("--flow_decoder_checkpoint", default="pretrained_models/flow_decoder.pt", help="Path to flow decoder weights")
    parser.add_argument("--campplus", default="pretrained_models/campplus.onnx", help="Path to CAMPPlus speaker embedding model")
    parser.add_argument("--hift", default="pretrained_models/hift.pt", help="Path to HiFT vocoder weights")
    parser.add_argument("--n_timesteps", type=int, default=10, help="Number of flow decoder steps")
    parser.add_argument("--length_scale", type=float, default=1.0, help="Speech pace: durations are multiplied by it (1.1 = 10%% slower, pauses too)")
    parser.add_argument("--rounding", choices=["ceil", "round", "cumulative"], help="how predicted durations become frames (default: the exported model's: cumulative for the flow sampler)")
    parser.add_argument("--dp_temperature", type=float, default=0.3, help="flow duration sampler: noise scale for speech tokens")
    parser.add_argument("--dp_pause_temperature", type=float, default=1.0, help="flow duration sampler: noise scale for pause tokens (punctuation, sentence edges)")
    parser.add_argument("--prosody_temperature", type=float, default=0.5, help="ProsodyCFM noise scale")
    parser.add_argument("--no_prosody", action="store_true", help="skip ProsodyCFM: the decoder gets the encoder's mean")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for ONNX inference if available")
    parser.add_argument("--threads", type=int, default=0, help="ONNX Runtime intra-op threads on the CPU (0: its default)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with open(args.config, "r") as f:
        configs = load_hyperpyyaml(f)

    print(f"Loading ONNX models from {args.onnx_dir}...")
    acoustic = OnnxAcousticModel(args.onnx_dir, use_gpu=args.use_gpu, threads=args.threads)
    m = acoustic.manifest
    print(f"  duration model: {acoustic.dp_kind}; ProsodyCFM: {'yes' if acoustic.has_prosody and not args.no_prosody else 'no'}; exported from {m['source']['checkpoint']}")

    print(f"Loading flow decoder from {args.flow_decoder_checkpoint} and HiFT vocoder from {args.hift}...")
    decoder, spk_embed_affine_layer = load_flow_decoder(configs, args.flow_decoder_checkpoint, device)
    hift = configs["hift"].eval().to(device)
    hift.load_state_dict(torch.load(args.hift, map_location="cpu"))

    print("Extracting speaker embedding from reference audio...")
    spk_model = load_session(args.campplus, threads=1)
    ref_audio, ref_sr = torchaudio.load(args.ref_audio)
    spk_embed = torch.nn.functional.normalize(extract_spk_embedding(spk_model, to_16k_mono(ref_audio, ref_sr)), dim=1)  # the model normalises it first
    with torch.no_grad():
        c = spk_embed_affine_layer(spk_embed.to(device))  # the flow decoder's speaker conditioning

    text = prepare_text(args.text, args.lang, args.segmentation, not args.no_text_normalize)
    print(f"Text: {args.text!r} -> {text!r}")
    tokens = text_tokens(text, lang=args.lang, phone=args.phone, add_blank=m["add_blank"])

    wav, info = synthesise_wave(
        acoustic, decoder, hift, spk_embed, c, tokens, device=device, seed=args.seed, n_timesteps=args.n_timesteps,
        rounding=args.rounding, length_scale=args.length_scale, dp_temperature=args.dp_temperature,
        dp_pause_temperature=args.dp_pause_temperature, prosody_temperature=args.prosody_temperature, use_prosody=not args.no_prosody,
    )  # fmt: skip
    print(f"  {info['frames']} frames ({info['frames'] / 50:.2f} s) from {tokens['x'].shape[1]} tokens")
    t_acoustic, t_vocoder = info["t_acoustic"], info["t_vocoder"]

    torchaudio.save(args.output, wav.cpu(), 24000)
    dur = wav.shape[1] / 24000
    print(f"Saved {args.output}: {dur:.2f} s of audio")
    print(f"Synthesis time: ONNX acoustic model {t_acoustic:.2f} s + flow decoder and vocoder (PyTorch) {t_vocoder:.2f} s (RTF {(t_acoustic + t_vocoder) / dur:.2f})")


if __name__ == "__main__":
    main()
