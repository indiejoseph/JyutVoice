# JyutVoice: documentation for agents

Start here. These pages are written for an AI agent (or a new engineer) who has to change, run or evaluate this repository without the history that produced it. They say
what exists, how the parts fit, how to run and check them, and what has already been decided or tried, so that nothing is re-discovered or re-litigated.
`README.md` in the repository root is the user-facing overview; these pages go deeper and are more candid about what does not work.

## What this is, in one minute

JyutVoice is a lightweight, non-autoregressive text-to-speech model for **Cantonese, with English words mixed in** (Mandarin was dropped on purpose). It does not
generate audio itself. It turns text into the *input features* of the frozen CosyVoice2 flow decoder, which turns them into a mel spectrogram that the frozen HiFT
vocoder renders at 24 kHz:

```
text -> normalise -> word-segment -> g2p -> TextEncoder -> duration sampler -> expand to frames -> ProsodyCFM -> frozen CosyVoice2 flow decoder -> frozen HiFT -> wav
                                                  ^ speaker x-vector (CAMPPlus) conditions every learned stage
```

What is trained here: the text encoder (6.9M parameters), a flow-matching duration sampler (2.4M) and ProsodyCFM (3.6M). The decoder and vocoder come from CosyVoice2 and
are never trained. The voice comes from the x-vector of a reference clip. Everything except the decoder and vocoder runs in ONNX Runtime in the deployed path.

## Pages

| page | read it when |
|---|---|
| [architecture.md](architecture.md) | you need the model, the tensors, the code map |
| [text-frontend.md](text-frontend.md) | you touch text normalisation, segmentation, g2p, English, code-mixing, punctuation |
| [training.md](training.md) | you train anything, build data, or need to know what the checkpoints are |
| [inference-and-api.md](inference-and-api.md) | you export ONNX, run `infer_onnx.py`, or run/change the HTTP API and its Docker image |
| [evaluation.md](evaluation.md) | you need to judge a change: which metrics exist, what each can and cannot tell |
| [decisions-and-known-issues.md](decisions-and-known-issues.md) | **before proposing anything**: decisions already taken, negative results, known problems, gotchas |

## Status (state on 2026-10-21; `git log` is the authority for code)

- **Deployed pipeline** (what `configs/export_onnx.yaml`, `infer_onnx.py` and the API serve): English-capable encoder + flow duration sampler + ProsodyCFM, trained
  2026-10-21. Weights are not in git (`*.ckpt *.pt *.pth *.safetensors *.onnx` are ignored); `onnx_models/` is produced by `scripts/export_onnx.py`.
- **`checkpoints/best/`** still holds the earlier *Cantonese-only* best (flow duration sampler + ProsodyCFM "mix1"). The English-capable pipeline is **not** designated best:
  it costs about one point of Cantonese syllable error rate and has the known English problem below.
- **Known problem, parked by the maintainer:** English words inside Cantonese are not always right in sound or stress (`Zoom` -> "drone", `interview` stress). See
  [decisions-and-known-issues.md](decisions-and-known-issues.md). Do not start work on it unless asked.
- **HTTP API + Docker (CPU)** exist and are verified (`api/`, `docker/`), and so does a **web playground** for the API (`playground/`, modelled on the SoulX-Podcast playground; see [inference-and-api.md](inference-and-api.md)).

## Working agreements (the maintainer's standing preferences)

1. **Do not commit, push, delete checkpoints or rewrite history unless asked.** Checkpoints are expensive; deleting one needs explicit confirmation. Look at a file before
   deleting or overwriting it.
2. **Never print or commit secrets** (API keys, tokens, `.env` files, license material). `docker/.env` is git-ignored and holds the API key.
3. **Third-party data and code beyond declared dependencies: ask first.** The MCE corpus (`AlienKevin/mixed_cantonese_and_english_speech`) has no license on its card; it is
   used for internal training only, never redistribute it or models trained on it without checking.
4. **Run installs, builds and tests yourself**; do not hand them back to the user. If an environment blocks a command, say which and ask them to run it.
5. **Intelligibility is measured by ASR, naturalness only by the maintainer's ears.** Never claim a model "sounds better" from ASR scores; say what was measured.
   Report regressions honestly (the English work costs Cantonese accuracy: say so).
6. **Do not use the Cantonese-fine-tuned CosyVoice2 LLM as a teacher for English or code-mixed targets**, the maintainer judged it bad there. The original CosyVoice2 weights are the
   teacher for English/Mandarin if one is ever needed.
7. Match the surrounding code style: comments explain *why* and record measured facts; keep module docstrings current.

## Environment in one paragraph

Python 3.11. Training needs `lightning`, `hyperpyyaml`, `datasets` and a GPU; inference needs only torch, onnxruntime and the packages in `api/requirements.txt`.
`import torch` **before** `onnxruntime`, `datasets` or `pyarrow` or a libstdc++ `CXXABI` error follows; `python -I` (isolated mode) cannot import `jyutvoice` because it drops the working directory from the path. The word
segmenter (`pydips`), the WeTextProcessing fork and the English g2p's NLTK data are needed for the real front end; many tests skip without them. Details and the other
gotchas are in [decisions-and-known-issues.md](decisions-and-known-issues.md).

```bash
python -m pytest tests --deselect tests/test_number_utils.py -q -p no:warnings   # 279 pass, 16 skip on a machine without pydips
```

`tests/test_number_utils.py` has 11 stale expectations ("3 dollars" vs spelled-out words) that fail identically on a clean checkout; they are known and left alone.
