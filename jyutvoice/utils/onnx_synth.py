"""Acoustic model of JyutVoice in ONNX Runtime: text encoder, duration model and ProsodyCFM, with numpy in between (torch-free).

`OnnxAcousticModel` is `JyutVoiceTTS.synthesise` up to the decoder's input: it takes the token ids of a sentence and the speaker's x-vector and returns `mu_y`, the
(1, 80, T_mel) features the frozen flow decoder turns into a mel. The helpers below replay what synthesise does between the networks (rounding the predicted
durations to frames, the alignment `generate_path` builds, the within-phone position ProsodyCFM is conditioned on) so that the output matches PyTorch's for the same
noise (tests/test_onnx_export.py). The graphs are written by scripts/export_onnx.py together with the `manifest.json` this class reads.
"""
import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

ENCODER_OUTPUTS = ("encoder_output", "mu_x", "x_mask")


def _cumsum(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """float32 running total accumulated in double, as `torch.cumsum` does on the CPU: the frame boundaries are `ceil`s of these totals, and a float32 accumulation
    differs from torch's in the last bit (55.000004 against 55.0), which moves a boundary by a frame."""
    return np.cumsum(a.astype(np.float64), axis=axis).astype(np.float32)


def pause_token_mask(x: np.ndarray, punct_ids, add_blank: bool = True) -> np.ndarray:
    """(T,) bool: the tokens the duration sampler treats as pauses: punctuation and the pad tokens at both ends (with the blanks next to them), the same tokens as the
    datamodule's pause mask. `x`: (T,) symbol ids, blanks included."""
    pm = np.isin(x, np.asarray(list(punct_ids)))
    edge = 3 if add_blank else 1
    pm[:edge] = True
    pm[-edge:] = True
    return pm


def punctuation_ids():
    from jyutvoice.text.symbols import punctuations, symbols

    return sorted({symbols.index(p) for p in punctuations if p in symbols})


def token_temperatures(pause_mask: np.ndarray, speech: float, pause: float) -> np.ndarray:
    """(1, 1, T) float32 noise scale of the duration sampler per token."""
    return np.where(pause_mask, pause, speech).astype(np.float32).reshape(1, 1, -1)


def round_durations(logw: np.ndarray, rounding: str = "ceil") -> np.ndarray:
    """Predicted log durations (1, 1, T) -> float32 durations the way `JyutVoiceTTS.synthesise` rounds them before the ceiling it always applies:
    "ceil" leaves them as predicted, "round" rounds to nearest (at least one frame), "cumulative" rounds the running total (`distribute_frames`), so the length of the
    utterance is the sum of the predictions with no per-token bias."""
    w = np.exp(logw.astype(np.float32))
    if rounding == "ceil":
        return w
    if rounding == "round":
        return np.maximum(np.round(w), 1.0).astype(np.float32)
    if rounding == "cumulative":
        total = np.maximum(np.round(w.sum()), 1.0).astype(np.float32)
        scale = total / np.maximum(w.sum(dtype=np.float32), np.float32(1e-6))
        cum = np.round(_cumsum(w * scale))
        dur = cum - np.concatenate([np.zeros_like(cum[..., :1]), cum[..., :-1]], axis=-1)
        return np.maximum(dur, 1.0).astype(np.float32)
    raise ValueError(f"rounding must be 'ceil', 'round' or 'cumulative', got {rounding!r}")


def frame_counts(w: np.ndarray, length_scale: float = 1.0):
    """Durations (1, 1, T) -> (counts (T,) int64, y_len): the frames of every token, as the alignment synthesise builds from them.

    Every duration is rounded up and then scaled by `length_scale` (so a scale other than 1 gives fractional durations); the mel has int(sum) frames (at least 1)
    and token k owns the frames from ceil(cumsum up to k-1) to ceil(cumsum up to k), cut at the end of the mel."""
    w_ceil = np.ceil(w).astype(np.float32) * np.float32(length_scale)
    y_len = max(int(np.float32(w_ceil.astype(np.float64).sum())), 1)
    cum = _cumsum(w_ceil.reshape(-1))
    ends = np.clip(np.ceil(cum), 0, y_len).astype(np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    return ends - starts, y_len


def expand_frames(features: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """(1, C, T) token features -> (1, C, T_mel): every token's column repeated for its frames."""
    return np.repeat(features, counts, axis=-1)


def phone_position(counts: np.ndarray):
    """`ProsodyCFM.compute_fractional_pos` from the frame counts: tau (1, 1, T_mel) float32, the position of every frame inside its token (0 for the first frame, below 1),
    and dur (1, 1, T_mel) float32, the frames of that token (at least 1)."""
    n = int(counts.sum())
    starts = np.cumsum(counts) - counts
    intra = np.arange(n) - np.repeat(starts, counts)
    dur = np.maximum(np.repeat(counts, counts), 1).astype(np.float32)
    tau = np.clip(intra.astype(np.float32) / dur, 0.0, 1.0)
    return tau.reshape(1, 1, -1), dur.reshape(1, 1, -1)


def text_tokens(text: str, lang: str = "yue", phone: Optional[str] = None, add_blank: bool = True) -> dict:
    """A word-segmented, normalised sentence (see jyutvoice.text.text_processor.prepare_text) -> the model's token arrays, each (1, T) int64: x, lang, tone,
    word_pos, syllable_pos, with a blank token between the tokens when the model was trained with `add_blank`."""
    from jyutvoice.text import text_to_sequence

    ids, tones, word_pos, syllable_pos, lang_ids = text_to_sequence(text, lang=lang, phone=phone)

    def arr(v):
        if add_blank:  # the blank token (id 0) between and around the tokens, as jyutvoice.utils.utils.intersperse
            v = [x for t in v for x in (0, t)] + [0]
        return np.array([v], dtype=np.int64)

    return {"x": arr(ids), "lang": arr(lang_ids), "tone": arr(tones), "word_pos": arr(word_pos), "syllable_pos": arr(syllable_pos)}


def load_session(path: str, use_gpu: bool = False, threads: int = 0):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads  # 0: onnxruntime's default (all cores)
    providers = ["CPUExecutionProvider"]
    if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


@dataclass
class AcousticResult:
    mu_y: np.ndarray  # (1, n_feats, T_mel) the flow decoder's input features
    counts: np.ndarray  # (T_text,) frames of every token
    logw: np.ndarray  # (1, 1, T_text) what the duration model predicted
    mu_y_mean: Optional[np.ndarray] = None  # the encoder's mean before ProsodyCFM (None without it)


class OnnxAcousticModel:
    def __init__(self, onnx_dir: str, use_gpu: bool = False, threads: int = 0):
        with open(os.path.join(onnx_dir, "manifest.json"), encoding="utf-8") as f:
            self.manifest = m = json.load(f)
        if m.get("format") != 1:
            raise ValueError(f"{onnx_dir}/manifest.json has format {m.get('format')!r}; this code reads format 1: export again with scripts/export_onnx.py")
        sess = lambda name: load_session(os.path.join(onnx_dir, name), use_gpu, threads)  # noqa: E731
        self.encoder = sess(m["text_encoder"])
        self.dp = sess(m["duration_model"]["file"])
        self.dp_kind = m["duration_model"]["kind"]  # "flow" or "regression"
        self.prosody = sess(m["prosody_model"]["file"]) if m.get("prosody_model") else None
        self.add_blank = m["add_blank"]
        self.max_tokens = m["max_tokens"]
        self.default_rounding = m["duration_model"]["rounding"]
        self._punct_ids = punctuation_ids()
        self._rng = np.random.default_rng()

    @property
    def has_prosody(self) -> bool:
        return self.prosody is not None

    def __call__(
        self,
        x: np.ndarray,
        lang: np.ndarray,
        tone: np.ndarray,
        word_pos: np.ndarray,
        syllable_pos: np.ndarray,
        spk_embed: np.ndarray,
        *,
        rounding: Optional[str] = None,
        length_scale: float = 1.0,
        dp_temperature: float = 0.3,
        dp_pause_temperature: float = 1.0,
        prosody_temperature: float = 0.5,
        use_prosody: bool = True,
        seed: Optional[int] = None,
        randn: Optional[Callable[[tuple], np.ndarray]] = None,
    ) -> AcousticResult:
        """x, lang, tone, word_pos, syllable_pos: (1, T) int64, the blanks between tokens already in; spk_embed (1, spk_dim) float32, L2-normalised.

        `rounding` (default: the exported model's own) and `length_scale` as in `JyutVoiceTTS.synthesise`; `dp_temperature` / `dp_pause_temperature` are the noise scales
        of the flow duration sampler for speech tokens and pause tokens (ignored by the regression predictor); `prosody_temperature` that of ProsodyCFM. The noise comes from
        `randn(shape)` (default: numpy's generator, seeded by `seed`); the duration noise is drawn first."""
        if x.shape[0] != 1:
            raise ValueError("one sentence at a time (batch size 1)")
        if x.shape[1] > self.max_tokens:
            raise ValueError(
                f"{x.shape[1]} tokens: the text encoder graph takes at most {self.max_tokens} (its rotary position tables end there). Split the text at punctuation, "
                "or export again with a larger --max-tokens"
            )
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if randn is None:
            randn = lambda shape: self._rng.standard_normal(shape, dtype=np.float32)  # noqa: E731
        spk_embed = np.ascontiguousarray(spk_embed, dtype=np.float32)
        ints = {
            "x": x, "x_lengths": np.array([x.shape[1]]), "lang": lang, "tone": tone, "word_pos": word_pos, "syllable_pos": syllable_pos,
        }  # fmt: skip
        feed = {k: np.ascontiguousarray(v, dtype=np.int64) for k, v in ints.items()}
        feed["spk_embed"] = spk_embed
        enc_out, mu_x, x_mask = self.encoder.run(list(ENCODER_OUTPUTS), feed)

        if self.dp_kind == "flow":
            temps = token_temperatures(pause_token_mask(x[0], self._punct_ids, self.add_blank), dp_temperature, dp_pause_temperature)
            noise = randn((1, 1, x.shape[1])).astype(np.float32) * temps * x_mask
            logw = self.dp.run(["logw"], {"encoder_output": enc_out, "x_mask": x_mask, "spk_embed": spk_embed, "noise": noise})[0]
        else:
            logw = self.dp.run(["logw"], {"encoder_output": enc_out, "x_mask": x_mask})[0]
        w = round_durations(logw, rounding or self.default_rounding)
        counts, _ = frame_counts(w, length_scale)
        if counts.sum() == 0:
            raise ValueError("the sentence has no frames: empty text?")

        mu_y = expand_frames(mu_x, counts)
        if self.prosody is None or not use_prosody:
            return AcousticResult(mu_y, counts, logw)
        h_text_y = expand_frames(enc_out, counts)
        tau, dur = phone_position(counts)
        y_mask = np.ones((1, 1, mu_y.shape[-1]), dtype=np.float32)
        noise = randn(mu_y.shape).astype(np.float32) * np.float32(prosody_temperature) * y_mask
        out = self.prosody.run(
            ["mu_y_out"], {"mu_y": mu_y, "h_text_y": h_text_y, "tau": tau, "dur": dur, "y_mask": y_mask, "spk_embed": spk_embed, "noise": noise}
        )[0]  # fmt: skip
        return AcousticResult(out, counts, logw, mu_y_mean=mu_y)
