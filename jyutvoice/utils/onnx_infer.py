"""What the ONNX command line (infer_onnx.py) and the API (api/service.py) share: the text front end, the speaker x-vector of a clip, and the step from the acoustic
model's output to a waveform (flow decoder + HiFT vocoder, PyTorch). The acoustic model itself is `jyutvoice.utils.onnx_synth.OnnxAcousticModel`."""
import time
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi

SPK_SAMPLE_RATE = 16000  # CAMPPlus
OUTPUT_SAMPLE_RATE = 24000  # HiFT

_segmenter: Optional[Callable[[str], list]] = None


def get_segmenter() -> Callable[[str], list]:
    """The pydips Cantonese word segmenter (coarse mode, as in training), loaded once."""
    global _segmenter
    if _segmenter is None:
        from pydips import BertModel

        model = BertModel()
        _segmenter = lambda text: model.cut(text, mode="coarse")  # noqa: E731
    return _segmenter


def prepare_text(text: str, lang: str = "yue", segmentation: str = "pydips", normalize: bool = True) -> str:
    """Raw sentence -> the word-segmented text g2p takes (see jyutvoice.text.text_processor): normalised (numbers, dates, money -> spoken Cantonese), then cut into
    words with pydips, English phrases whole. `segmentation="given"` leaves the words as written."""
    from jyutvoice.text.text_processor import has_chinese, normalize_text, segment_mixed

    if normalize:
        text = normalize_text(text, lang)
    if segmentation == "pydips" and has_chinese(text):
        text = " ".join(segment_mixed(text, get_segmenter()))
    return text


def load_flow_decoder(configs: dict, checkpoint_path: str, device) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """The CosyVoice2 flow decoder and its speaker layer (what turns the x-vector into the decoder's conditioning) from flow_decoder.pt; `configs` is configs/onnx_infer.yaml."""
    decoder = configs["decoder"].eval().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    prefix = "decoder."
    decoder.load_state_dict({k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)})
    affine = torch.nn.Linear(configs.get("spk_embed_dim", 192), configs.get("n_feats", 80)).to(device)
    prefix = "spk_embed_affine_layer."
    affine.load_state_dict({k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)})
    return decoder, affine.eval()


def to_16k_mono(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """(channels, samples) at `sample_rate` -> (1, samples) at 16 kHz."""
    waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != SPK_SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SPK_SAMPLE_RATE)(waveform)
    return waveform


def extract_spk_embedding(spk_model, speech: torch.Tensor) -> torch.Tensor:
    """CAMPPlus x-vector (1, 192), not yet normalised, of a 16 kHz mono waveform (1, N); `spk_model` is an onnxruntime session of campplus.onnx."""
    feat = kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=SPK_SAMPLE_RATE)
    spk_feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = spk_model.run(None, {spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()})[0]
    return torch.from_numpy(embedding.reshape(1, -1).astype(np.float32))


def synthesise_wave(
    acoustic, decoder, hift, spk_embed: torch.Tensor, c: torch.Tensor, tokens: Dict[str, np.ndarray], *, device, seed: int, n_timesteps: int = 10, **acoustic_kwargs
) -> Tuple[torch.Tensor, dict]:
    """One utterance: tokens (`onnx_synth.text_tokens`) -> waveform (1, T) at 24 kHz.

    `spk_embed` (1, 192) is the L2-normalised x-vector, `c` the decoder's conditioning from it (the speaker layer applied to `spk_embed`). `seed` seeds the sampling of
    the acoustic model (durations, ProsodyCFM) and the vocoder's source noise (`torch.manual_seed`); the flow decoder's own noise is a buffer that CausalConditionalCFM fills
    with seed 0 when it is built, the same in every run. So the same seed gives the same waveform in any process (with the same number of threads: sums are not associative);
    `acoustic_kwargs` go to `OnnxAcousticModel.__call__` (length_scale, dp_temperature, ...).
    Returns the waveform and {"frames", "t_acoustic", "t_vocoder"} (frames of the mel, seconds spent in each part)."""
    t0 = time.time()
    res = acoustic(tokens["x"], tokens["lang"], tokens["tone"], tokens["word_pos"], tokens["syllable_pos"], spk_embed.numpy(), seed=seed, **acoustic_kwargs)
    t_acoustic = time.time() - t0
    mu_y = torch.from_numpy(res.mu_y).to(device)
    t0 = time.time()
    torch.manual_seed(seed)
    with torch.no_grad():
        mel, _ = decoder(
            mu=mu_y, mask=torch.ones(1, 1, mu_y.shape[-1], device=device), spks=c, cond=torch.zeros_like(mu_y), n_timesteps=n_timesteps, temperature=1.0, streaming=False,
        )  # fmt: skip
        wav, _ = hift.inference(mel)
    return wav, {"frames": int(mu_y.shape[-1]), "t_acoustic": t_acoustic, "t_vocoder": time.time() - t0}
