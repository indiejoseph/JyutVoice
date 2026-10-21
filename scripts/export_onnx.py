#!/usr/bin/env python3
"""Export the acoustic model of JyutVoice to ONNX: the text encoder, the duration model and (optionally) ProsodyCFM.

    python scripts/export_onnx.py --config configs/export_onnx.yaml
    python scripts/export_onnx.py --checkpoint checkpoints/best/jyutvoice_flowdp_epoch019.ckpt --base-config configs/synth_flowdp.yaml \\
        --prosody-ckpt checkpoints/best/prosody_cfm_mix1.pt --output-dir onnx_models_best

Written to the output directory (the graphs have a dynamic batch and sequence length; wrappers in jyutvoice/models/onnx_wrappers.py):

    text_encoder.onnx           token ids, positions, speaker -> encoder_output, mu_x, x_mask
    flow_duration_sampler.onnx  encoder_output, x_mask, spk_embed, noise -> logw      (a flow-matching duration predictor: the Euler loop is in the graph)
    duration_predictor.onnx     encoder_output, x_mask -> logw                        (a regression duration predictor, instead of the sampler)
    prosody_sampler.onnx        mu_y, h_text_y, tau, dur, y_mask, spk_embed, noise -> mu_y_out   (ProsodyCFM, when --prosody-ckpt is given)
    manifest.json               what was exported and from which checkpoints; infer_onnx.py / jyutvoice.utils.onnx_synth.OnnxAcousticModel read it

The flow decoder (CosyVoice2) and the HiFT vocoder stay in PyTorch: infer_onnx.py loads them from pretrained_models/. Every graph is checked against PyTorch on real
sentences of two different lengths (identical noise for the samplers), and the whole chain (rounding, alignment, ProsodyCFM) against `JyutVoiceTTS.synthesise` with the
decoder left out; the export fails when a difference exceeds --atol. A TensorRT engine of the flow decoder's estimator is only built with --tensorrt.
"""
import argparse
import inspect
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from jyutvoice.models.flow_duration_predictor import FlowDurationPredictor  # noqa: E402
from jyutvoice.models.onnx_wrappers import (  # noqa: E402
    DurationPredictorWrapper,
    FlowDurationSamplerWrapper,
    ProsodyCFMSamplerWrapper,
    TextEncoderWrapper,
)
from jyutvoice.utils import onnx_synth as osy  # noqa: E402

# word-segmented, as the front end hands them to g2p (English phrases whole); two lengths so that the dynamic axes are exercised
CHECK_SENTENCES = [
    "我 喺 Zoom 開會 ，",
    "佢 話 個 interview 好 順利 ， 希望 可以 攞到 offer 。",
]
CHECK_SENTENCES.append(" ， ".join([CHECK_SENTENCES[1]] * 5))  # ~450 tokens: far beyond the length the graphs are traced on
SEQ = {0: "batch", 1: "seq_len"}  # (B, T) inputs
FEAT = {0: "batch", 2: "seq_len"}  # (B, C, T) tensors
SPK = {0: "batch"}
FRAMES = {0: "batch", 2: "frames"}  # (B, C, T_mel) tensors


def load_tts(checkpoint, base_config):
    """The model of `base_config` with the weights of `checkpoint`. Only encoder and duration model are needed: the flow decoder of the config stays at its initial weights."""
    from hyperpyyaml import load_hyperpyyaml

    with open(base_config) as f:
        cfg = load_hyperpyyaml(f)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    tts = cfg["tts"]
    res = tts.load_state_dict(state, strict=False)
    wanted = ("encoder.", "dp.")
    missing = [k for k in res.missing_keys if k.startswith(wanted)]
    unexpected = [k for k in res.unexpected_keys if k.startswith(wanted)]
    if missing or unexpected:
        raise SystemExit(f"{checkpoint} does not fit {base_config}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return tts.eval(), cfg


def load_prosody(path):
    from jyutvoice.models.prosody_cfm import ProsodyCFM

    pc = torch.load(path, map_location="cpu", weights_only=False)
    model = ProsodyCFM(**pc["model_config"])
    model.load_state_dict(pc["ema"])
    return model.eval(), pc


def sentence_inputs(text, add_blank=True):
    """Encoder inputs of one segmented sentence, in the encoder's argument order: the same tokens infer_onnx.py builds."""
    tok = {k: torch.from_numpy(v) for k, v in osy.text_tokens(text, "yue", add_blank=add_blank).items()}
    return dict(x=tok["x"], x_lengths=torch.tensor([tok["x"].shape[1]]), lang=tok["lang"], tone=tok["tone"], word_pos=tok["word_pos"], syllable_pos=tok["syllable_pos"])


def export(module, args, path, input_names, output_names, dynamic_axes, opset):
    kw = {}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False  # the TorchScript exporter: newer torch defaults to the dynamo one, which needs onnxscript
    torch.onnx.export(
        module, args, path, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, opset_version=opset,
        do_constant_folding=True, export_params=True, **kw,
    )  # fmt: skip
    import onnx

    onnx.checker.check_model(onnx.load(path))
    print(f"  wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def compare(name, ort_out, torch_out, atol):
    worst = 0.0
    for o, t in zip(ort_out, torch_out):
        worst = max(worst, float(np.abs(o - t.detach().cpu().numpy()).max()))
    status = "ok" if worst <= atol else "TOO LARGE"
    print(f"  {name}: max |onnx - torch| = {worst:.2e} ({status})")
    return worst <= atol


def check_graphs(tts, prosody, paths, kind, n_steps, add_blank, atol):
    """Every graph against PyTorch on the check sentences (two lengths), identical noise for the samplers."""
    ok = True
    sess = {k: osy.load_session(p) for k, p in paths.items()}
    g = torch.Generator().manual_seed(0)
    for text in CHECK_SENTENCES:
        inp = sentence_inputs(text, add_blank)
        T = inp["x"].shape[1]
        print(f"checking {text!r} ({T} tokens)")
        spk = torch.nn.functional.normalize(torch.randn(1, tts.spk_embed_affine_layer.in_features, generator=g), dim=1)
        with torch.no_grad():
            enc_out, mu_x, x_mask = tts.encoder(*inp.values(), spk)
            feed = {k: v.numpy() for k, v in inp.items()}
            feed["spk_embed"] = spk.numpy()
            ok &= compare("text_encoder", sess["text_encoder"].run(None, feed), (enc_out, mu_x, x_mask), atol)

            pm = osy.pause_token_mask(inp["x"][0].numpy(), osy.punctuation_ids(), add_blank)
            temps = torch.from_numpy(osy.token_temperatures(pm, 0.3, 1.0))
            if kind == "flow":
                torch.manual_seed(1)
                ref = tts.dp(enc_out, x_mask, g=spk, n_steps=n_steps, cfg_scale=0.0, temperature=temps)
                torch.manual_seed(1)
                noise = torch.randn(1, 1, T) * temps * x_mask
                run = sess["duration_model"].run(None, {"encoder_output": enc_out.numpy(), "x_mask": x_mask.numpy(), "spk_embed": spk.numpy(), "noise": noise.numpy()})
            else:
                ref = tts.dp(enc_out, x_mask)
                run = sess["duration_model"].run(None, {"encoder_output": enc_out.numpy(), "x_mask": x_mask.numpy()})
            ok &= compare("duration_model", run, [ref], atol)
            if prosody is None:
                continue
            for scale in (1.0, 1.25):  # a length scale other than 1 gives fractional durations: the alignment is cut differently (1.25: exact in binary, so no sum can round across a frame)
                w = osy.round_durations(ref.numpy(), "cumulative" if kind == "flow" else "ceil")
                counts, y_len = osy.frame_counts(w, scale)
                attn = torch.zeros(1, T, y_len)
                attn[0, torch.arange(T).repeat_interleave(torch.from_numpy(counts)), torch.arange(y_len)] = 1.0
                mu_y = torch.matmul(attn.transpose(1, 2), mu_x.transpose(1, 2)).transpose(1, 2)
                h_text_y = torch.matmul(attn.transpose(1, 2), enc_out.transpose(1, 2)).transpose(1, 2)
                y_mask = torch.ones(1, 1, y_len)
                torch.manual_seed(2)
                ref_p = prosody(mu_y, h_text_y, attn.unsqueeze(1), y_mask, g=spk, n_steps=n_steps, cfg_scale=0.0, temperature=0.5)
                torch.manual_seed(2)
                noise = torch.randn(1, prosody.in_channels, y_len) * 0.5 * y_mask
                tau, dur = osy.phone_position(counts)
                run = sess["prosody_model"].run(None, {
                    "mu_y": mu_y.numpy(), "h_text_y": h_text_y.numpy(), "tau": tau, "dur": dur, "y_mask": y_mask.numpy(), "spk_embed": spk.numpy(), "noise": noise.numpy(),
                })  # fmt: skip
                ok &= compare(f"prosody_model (length scale {scale}, {y_len} frames)", run, [ref_p], atol)
    return ok


def check_chain(tts, prosody, onnx_dir, add_blank, atol):
    """`OnnxAcousticModel` against `JyutVoiceTTS.synthesise` (decoder left out: it returns the features the decoder would get) for the same random draws."""
    ok = True
    model = osy.OnnxAcousticModel(onnx_dir)
    decoder, tts.decoder, tts.prosody_cfm = tts.decoder, None, prosody
    infer_steps = getattr(tts.dp, "infer_steps", None)
    if model.dp_kind == "flow":
        tts.dp.infer_steps = model.manifest["duration_model"]["n_steps"]  # synthesise runs the sampler with its own step count
    try:
        g = torch.Generator().manual_seed(3)
        for text in CHECK_SENTENCES:
            for scale in (1.0, 1.25):
                inp = sentence_inputs(text, add_blank)
                spk = torch.nn.functional.normalize(torch.randn(1, tts.spk_embed_affine_layer.in_features, generator=g), dim=1)
                kw = dict(length_scale=scale, duration_rounding=model.default_rounding)
                if model.dp_kind == "flow":
                    pm = osy.pause_token_mask(inp["x"][0].numpy(), osy.punctuation_ids(), add_blank)
                    kw.update(dp_temperature=0.3, dp_pause_temperature=1.0, dp_pause_mask=torch.from_numpy(pm).unsqueeze(0))
                if model.has_prosody:
                    kw["prosody_kwargs"] = dict(n_steps=model.manifest["prosody_model"]["n_steps"], cfg_scale=0.0, temperature=0.5)
                torch.manual_seed(4)
                ref = tts.synthesise(**inp, spk_embed=spk, **kw)["encoder_outputs"]
                torch.manual_seed(4)
                out = model(*(inp[k].numpy() for k in ("x", "lang", "tone", "word_pos", "syllable_pos")), spk.numpy(), length_scale=scale,
                            randn=lambda shape: torch.randn(shape).numpy())  # fmt: skip
                if out.mu_y.shape != tuple(ref.shape):
                    print(f"  chain {text!r} scale {scale}: {out.mu_y.shape[-1]} frames, torch {ref.shape[-1]}")
                    ok = False
                    continue
                ok &= compare(f"chain {text[:12]!r} length scale {scale} ({ref.shape[-1]} frames)", [out.mu_y], [ref], atol)
    finally:
        tts.decoder, tts.prosody_cfm = decoder, None
        if infer_steps is not None:
            tts.dp.infer_steps = infer_steps
    return ok


def build_tensorrt(args):
    from jyutvoice.utils.common import convert_onnx_to_trt

    trt_kwargs = {
        "min_shape": [(2, 80, 4), (2, 1, 4), (2, 80, 4), (2, 80, 4)],
        "opt_shape": [(2, 80, 500), (2, 1, 500), (2, 80, 500), (2, 80, 500)],
        "max_shape": [(2, 80, 3000), (2, 1, 3000), (2, 80, 3000), (2, 80, 3000)],
        "input_names": ["x", "mask", "mu", "cond"],
    }
    convert_onnx_to_trt(args.trt_plan, trt_kwargs, args.trt_onnx, fp16=True)


def export_all(tts, prosody, out_dir, add_blank, n_steps, opset, max_tokens, atol, source):
    """Write the graphs and manifest.json of `tts` (+ `prosody`, a ProsodyCFM or None) to `out_dir` and check them against PyTorch. True when every difference is within `atol`."""
    os.makedirs(out_dir, exist_ok=True)
    kind = "flow" if isinstance(tts.dp, FlowDurationPredictor) else "regression"
    # example inputs: a real sentence through the real encoder and (for the graphs after it) the real chain
    inp = sentence_inputs(CHECK_SENTENCES[1], add_blank)
    spk = torch.nn.functional.normalize(torch.randn(1, tts.spk_embed_affine_layer.in_features, generator=torch.Generator().manual_seed(0)), dim=1)
    with torch.no_grad():
        enc_out, mu_x, x_mask = tts.encoder(*inp.values(), spk)
    T = inp["x"].shape[1]
    names = {"text_encoder": "text_encoder.onnx"}
    print("exporting")
    enc_in = ["x", "x_lengths", "lang", "tone", "word_pos", "syllable_pos", "spk_embed"]
    export(TextEncoderWrapper(tts.encoder, max_tokens).eval(), (*inp.values(), spk), os.path.join(out_dir, names["text_encoder"]), enc_in, list(osy.ENCODER_OUTPUTS),
           {**{k: SEQ for k in enc_in[:1] + enc_in[2:6]}, "x_lengths": SPK, "spk_embed": SPK, "encoder_output": FEAT, "mu_x": FEAT, "x_mask": FEAT}, opset)  # fmt: skip
    with torch.no_grad():
        if kind == "flow":
            names["duration_model"] = "flow_duration_sampler.onnx"
            noise = torch.randn(1, 1, T) * 0.5
            export(FlowDurationSamplerWrapper(tts.dp, n_steps).eval(), (enc_out, x_mask, spk, noise), os.path.join(out_dir, names["duration_model"]),
                   ["encoder_output", "x_mask", "spk_embed", "noise"], ["logw"],
                   {"encoder_output": FEAT, "x_mask": FEAT, "spk_embed": SPK, "noise": FEAT, "logw": FEAT}, opset)  # fmt: skip
            logw = tts.dp(enc_out, x_mask, g=spk, n_steps=n_steps, cfg_scale=0.0, temperature=0.3)
        else:
            names["duration_model"] = "duration_predictor.onnx"
            export(DurationPredictorWrapper(tts.dp).eval(), (enc_out, x_mask), os.path.join(out_dir, names["duration_model"]), ["encoder_output", "x_mask"], ["logw"],
                   {"encoder_output": FEAT, "x_mask": FEAT, "logw": FEAT}, opset)  # fmt: skip
            logw = tts.dp(enc_out, x_mask)
        if prosody is not None:
            names["prosody_model"] = "prosody_sampler.onnx"
            counts, y_len = osy.frame_counts(osy.round_durations(logw.numpy(), "cumulative" if kind == "flow" else "ceil"))
            tau, dur = (torch.from_numpy(v) for v in osy.phone_position(counts))
            ex = (torch.from_numpy(osy.expand_frames(mu_x.numpy(), counts)), torch.from_numpy(osy.expand_frames(enc_out.numpy(), counts)), tau, dur, torch.ones(1, 1, y_len), spk,
                  torch.randn(1, prosody.in_channels, y_len) * 0.5)  # fmt: skip
            pin = ["mu_y", "h_text_y", "tau", "dur", "y_mask", "spk_embed", "noise"]
            export(ProsodyCFMSamplerWrapper(prosody, n_steps).eval(), ex, os.path.join(out_dir, names["prosody_model"]), pin, ["mu_y_out"],
                   {**{k: FRAMES for k in pin if k != "spk_embed"}, "spk_embed": SPK, "mu_y_out": FRAMES}, opset)  # fmt: skip

    paths = {k: os.path.join(out_dir, v) for k, v in names.items()}
    ok = check_graphs(tts, prosody, paths, kind, n_steps, add_blank, atol)

    manifest = {
        "format": 1,
        "text_encoder": names["text_encoder"],
        "duration_model": {"kind": kind, "file": names["duration_model"], "n_steps": n_steps if kind == "flow" else None,
                           "rounding": "cumulative" if kind == "flow" else "ceil"},
        "prosody_model": {"file": names["prosody_model"], "n_steps": n_steps, "residual": bool(prosody.residual)} if prosody is not None else None,
        "add_blank": bool(add_blank),
        "max_tokens": max_tokens,
        "n_vocab": tts.encoder.n_vocab,
        "spk_embed_dim": tts.spk_embed_affine_layer.in_features,
        "source": source,
    }  # fmt: skip
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    ok &= check_chain(tts, prosody, out_dir, add_blank, atol)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "export_onnx.yaml"), help="yaml with the defaults of the options below")
    ap.add_argument("--checkpoint", help="training checkpoint with the encoder and the duration model")
    ap.add_argument("--base-config", help="training config that builds the model (configs/synth_flowdp.yaml for a flow-matching duration predictor, configs/base.yaml for a regression one)")
    ap.add_argument("--prosody-ckpt", help="prosody_cfm_*.pt from scripts/train_prosody_cfm.py (omit for no ProsodyCFM)")
    ap.add_argument("--no-prosody", action="store_true", help="ignore the prosody checkpoint of the config")
    ap.add_argument("--output-dir")
    ap.add_argument("--opset", type=int)
    ap.add_argument("--max-tokens", type=int, help="longest token sequence (blanks included) the text encoder graph accepts")
    ap.add_argument("--steps", type=int, help="Euler steps unrolled into the sampler graphs")
    ap.add_argument("--atol", type=float, help="largest accepted difference to PyTorch")
    ap.add_argument("--tensorrt", action="store_true", help="also build a TensorRT engine of the flow decoder's estimator (--trt-onnx -> --trt-plan)")
    ap.add_argument("--trt-onnx", default="pretrained_models/flow.decoder.estimator.fp32.onnx")
    ap.add_argument("--trt-plan", default="pretrained_models/flow.decoder.estimator.fp16.mygpu.plan")
    a = ap.parse_args()

    from hyperpyyaml import load_hyperpyyaml

    with open(a.config) as f:
        cfg = load_hyperpyyaml(f)
    pick = lambda cli, key: cli if cli is not None else cfg[key]  # noqa: E731
    checkpoint, base_config = pick(a.checkpoint, "checkpoint_path"), pick(a.base_config, "base_config")
    prosody_ckpt = None if a.no_prosody else pick(a.prosody_ckpt, "prosody_checkpoint")
    out_dir, opset, n_steps, atol = pick(a.output_dir, "output_dir"), pick(a.opset, "opset_version"), pick(a.steps, "euler_steps"), pick(a.atol, "atol")
    max_tokens = pick(a.max_tokens, "max_tokens")
    os.makedirs(out_dir, exist_ok=True)

    print(f"model: {checkpoint} with {base_config}")
    tts, base = load_tts(checkpoint, base_config)
    add_blank = base.get("add_blank", True)
    prosody = load_prosody(prosody_ckpt)[0] if prosody_ckpt else None
    print(f"duration model: {'flow' if isinstance(tts.dp, FlowDurationPredictor) else 'regression'}; ProsodyCFM: {prosody_ckpt or 'none'}; add_blank {add_blank}; {n_steps} Euler steps")

    ok = export_all(tts, prosody, out_dir, add_blank, n_steps, opset, max_tokens, atol,
                    {"checkpoint": checkpoint, "base_config": base_config, "prosody_checkpoint": prosody_ckpt})  # fmt: skip
    if not ok:
        raise SystemExit("export finished, but a graph differs from PyTorch by more than --atol (see above)")
    if a.tensorrt:
        build_tensorrt(a)
    print(f"done: {out_dir} (run: python infer_onnx.py --onnx_dir {out_dir} --text ... --ref_audio ... --output ...)")


if __name__ == "__main__":
    main()
