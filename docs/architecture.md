# Architecture

## The idea

CosyVoice2 makes speech in two stages: an autoregressive LLM writes discrete speech tokens (25 Hz), and a flow-matching decoder plus the HiFT vocoder turn tokens into audio.
The flow decoder first runs a *flow encoder* that maps the tokens to 80-d hidden states at 50 Hz (called `decoder_h` in this repo, and `mu` inside the decoder); the decoder
generates the mel spectrogram conditioned on them.

JyutVoice replaces the slow LLM with a small non-autoregressive model that predicts **`decoder_h` directly from text** and then reuses the frozen decoder and vocoder.
Training targets are the `decoder_h` states that CosyVoice2's own flow encoder produces for real recordings (or for the teacher LLM's tokens), so no mel loss is needed and
the decoder is never touched. This is why training is cheap and why the decoder's voice quality comes for free.

## Inference pipeline

```
 text
  |  jyutvoice.text.text_processor.normalize_text    numbers/dates/money -> spoken Cantonese (WeTextProcessing fork), BEFORE segmentation
  |  pydips (coarse) + script_split                  Chinese words; English phrases stay whole
  |  jyutvoice.text (g2p)                            phones, tones, word_pos, syllable_pos, lang ids; Jyutping + English ARPAbet
  v
 token ids (blank token 0 between all tokens: length 2P+1)
  |  TextEncoder (RoPE transformer)  + speaker x-vector   ->  x_enc (192, T), mu_x (80, T)
  |  FlowDurationPredictor (CFM over log durations)       ->  frames per token   (cumulative rounding, length_scale)
  |  expand tokens to frames (alignment path)             ->  mu_y (80, T_mel) and h_text_y (192, T_mel)
  |  ProsodyCFM (CFM, frame level)                        ->  decoder_h-like features (80, T_mel)   [replaces mu_y]
  v
 CosyVoice2 CausalConditionalCFM (frozen, 10 Euler steps)   spks = affine(x-vector), cond = 0 (no clip prompt in the ONNX path)
  v
 HiFT vocoder (frozen) -> 24 kHz waveform           (mel frame = 480 samples = 50 Hz)
```

The speaker is the 192-d **CAMPPlus x-vector** of a reference clip (`pretrained_models/campplus.onnx`, 16 kHz fbank). It conditions the encoder, the duration sampler and
ProsodyCFM, and (through `spk_embed_affine_layer`, 192 -> 80) the flow decoder. The ONNX path gives the decoder **no clip prompt**; the PyTorch `synthesise(prompt_feat=,
prompt_h=)` can, and it measurably raises speaker similarity, but a prompt needs the CosyVoice speech tokenizer and flow encoder at inference (see `decisions-and-known-issues.md`).

## Components

| part | class / file | size | trained here? | notes |
|---|---|---|---|---|
| text encoder | `models/text_encoder.py::TextEncoder` | 6.9M | yes | Glow-TTS style with RoPE attention (`encoder_type: RoPE Encoder`, 192 channels, 6 layers, 2 heads, FFN 768), prenet, speaker `gin`. Embeddings: `emb` (97 symbols), `lang_emb`, `tone_emb` (11), word/syllable position. Outputs `x_enc` (192-d states, used by the sampler and ProsodyCFM) and `mu_x` (80-d, the "mean" in `decoder_h` space). Optional `bert_dim` projection (off in all shipped models). |
| duration (shipped) | `models/flow_duration_predictor.py::FlowDurationPredictor` | 2.4M | yes | Optimal-transport flow matching over per-token log durations, conv blocks mix neighbours. Sampled with per-token noise: **0.3 for phones/blanks, 1.0 for pause tokens** (punctuation, sentence-edge pads). `infer_cfg_scale` must be 0. |
| duration (older) | `models/duration_predictor.py::DurationPredictor` | 0.35M | yes | Regression (Glow-TTS). Deterministic; with `ceil` rounding the speech is 18-25% too slow and long pauses never appear. Kept for the baseline and ONNX. |
| frame-level prosody | `models/prosody_cfm.py::ProsodyCFM` | 3.6M | yes (separately) | CFM over `decoder_h` at 50 Hz, conditioned on `mu_y`, `h_text_y`, the position inside each phone (`tau`), the phone length (`dur`) and the speaker. Sampled with noise 0.5, no guidance, 10 steps. Optional `residual` (generate `decoder_h - mu_y`) and `use_mark` (next-punctuation features) variants exist but are not shipped. |
| flow decoder | `flow/flow_matching.py::CausalConditionalCFM` (+ `flow/decoder.py`) | - | no (frozen CosyVoice2) | Its `rand_noise` buffer is filled after `set_all_random_seed(0)`, so the decoder noise is **identical in every process**; a request `seed` does not change it. |
| vocoder | `hifigan/generator.py::HiFTGenerator` | - | no (frozen CosyVoice2) | Source noise comes from the global torch RNG, which `seed` does control. |
| speaker model | `pretrained_models/campplus.onnx` | - | no | x-vector, `utils/onnx_infer.py::extract_spk_embedding` |
| speech tokenizer, flow encoder | `pretrained_models/speech_tokenizer_v2.onnx`, `flow_encoder.pt` | - | no | Only for building training data (`decoder_h`), never at inference |

`JyutVoiceTTS` (`models/jyutvoice_tts.py`) wires the trainable parts together: `forward` computes the training losses, `synthesise` is the reference PyTorch inference path.
`models/slp.py` (speech length predictor) is a legacy side path with its own training script (`train_slp.py`); it is not part of the shipped pipeline.

## Training objective (what `JyutVoiceTTS.forward` does)

1. Encode the tokens -> `mu_x`.
2. **Monotonic alignment search (MAS)** between `mu_x` (Gaussian means, unit variance) and the real `decoder_h` of the row. Done in fp32 (bf16 rounding is as large as the score gaps).
   **Pause-aware MAS:** `pause_bonus` rewards giving silent frames (a `silence` column from a `SilenceProbe` on `decoder_h`) to pause tokens, so commas get their real pause
   instead of the neighbouring phones stretching. Bonus 10 sends about 94% of long silences to pause tokens at negligible prior cost.
3. Durations per token = row sums of the alignment path; **duration loss** = the sampler's flow-matching loss on log durations (or log-MSE for the regression predictor).
4. **Prior loss**: how well the expanded `mu_y` (or, with ProsodyCFM attached, the ProsodyCFM flow loss + 0.1 x an anchor MSE that keeps `mu_x` aligned) explains `decoder_h`.
   Rows without audio (teacher rows) carry `prior_weight` 0: they train durations only, because their phones come from g2p while the teacher's own reading may differ.

In the shipped recipe the encoder is trained first (regression durations), then **frozen** while the flow sampler is trained, and ProsodyCFM is trained last on cached encoder
features (`scripts/build_prosody_cache.py`), never end to end. See [training.md](training.md).

## Tensor conventions

- Tokens: `x`, `lang`, `tone`, `word_pos`, `syllable_pos`, all `(1, T)` int64. With `add_blank: true` (all shipped models) token id 0 is interleaved between and around every
  token (`T = 2P + 1`). Removing blanks hurt Cantonese intelligibility (syllable error 8.7% vs 4.0%), so keep them.
- Vocabulary: 97 ids = pad `_`, `SP`, `UNK`, `! ? … , . ' -`, then the union of Cantonese (Jyutping), English (ARPAbet) and Mandarin symbols (`text/symbols.py`).
- Language ids: 0 Cantonese, 1 Mandarin, 2 English (blank tokens also carry 0). Tone ids: 0 pad / punctuation, 1-6 Cantonese tones, 7 English consonant, 8 unstressed, 9 primary, 10 secondary
  stress (`text/english/stress.py`, `N_TONE = 11`; word and syllable position tables have 4 entries each).
- Frames: 24 kHz audio, hop 480, **50 Hz**; `decoder_h`, `mu_y`, mel all `(80, T_mel)`. CosyVoice2 speech tokens are 25 Hz (`token_mel_ratio: 2`).
- ONNX text encoder accepts at most `max_tokens` (1024, about 30 s) tokens: its rotary tables are baked into the graph at export time.

## Code map

```
jyutvoice/
  train.py                      Lightning training entry: python -m jyutvoice.train --config configs/<x>.yaml [--device cpu] [--validate-only]
  models/                       jyutvoice_tts.py (losses + synthesise), text_encoder.py, flow_duration_predictor.py, duration_predictor.py, prosody_cfm.py,
                                onnx_wrappers.py (traceable modules for export), baselightningmodule.py, slp.py (legacy)
  text/                         cleaners.py, text_processor.py (normalise/segment), script_split.py (code-mixing), multilingual.py, symbols.py,
                                cantonese/{g2p,colloquial,readings}.py, english/{g2p,stress,normalization}.py, mandarin/ (unused now), bert_align.py (optional)
  data/                         text_mel_datamodule.py (the datamodule), sd3_build.py + mce_build.py (builder workers), bert_sidefile.py
  flow/ hifigan/ transformer/   the CosyVoice2 decoder, vocoder and shared layers (vendored)
  utils/                        onnx_synth.py (numpy side of ONNX inference), onnx_infer.py (shared by CLI and API), flow_features.py (training-data features),
                                silence_probe.py, model.py (alignment helpers), monotonic_align/
api/                            FastAPI service (see inference-and-api.md)      docker/   its Dockerfile and compose file
configs/                        hyperpyyaml configs; paths inside are machine-specific, edit them
scripts/                        training-data builders, teacher generation, training of ProsodyCFM, export, evaluation and diagnostics (each has a docstring with usage)
tests/                          pytest; see the Tests section of evaluation.md for what is covered and what skips
```

Configs are `hyperpyyaml` files, not Hydra: `!new:` builds objects, `!ref <name>` references values, `!name:` makes partials. `train.py` loads one file and trains
`cfg["tts"]` with `cfg["data"]` and `cfg["trainer"]`. `init_ckpt` (+ `init_ckpt_prefixes`) is a weights-only warm start; `ckpt_path` resumes everything.
