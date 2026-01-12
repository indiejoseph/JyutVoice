# Copilot Instructions for JyutVoice TTS

JyutVoice is a Cantonese-focused multilingual TTS system (yue, zh, en) using Flow Matching (Matcha-TTS style) and transfer learning.

## Architecture & Frameworks
- **Core Library**: `jyutvoice/` contains all model, data, and text logic.
- **Framework**: Built with **PyTorch Lightning**. Models inherit from `BaseLightningClass` in [jyutvoice/models/baselightningmodule.py](jyutvoice/models/baselightningmodule.py).
- **Configuration**: Uses **HyperPyYAML** for object instantiation in `configs/*.yaml`.
    - `!new:class_name` instantiates a class.
    - `!ref <key>` references another value in the config.
- **Model Stages**:
    1. **TextEncoder**: [jyutvoice/models/text_encoder.py](jyutvoice/models/text_encoder.py) (Phonemes, Tones, Language, Positional IDs).
    2. **DurationPredictor**: [jyutvoice/models/duration_predictor.py](jyutvoice/models/duration_predictor.py).
    3. **Decoder (Flow Matching)**: [jyutvoice/flow/flow_matching.py](jyutvoice/flow/flow_matching.py) and [jyutvoice/flow/decoder.py](jyutvoice/flow/decoder.py).
    4. **Vocoder**: HiFi-GAN variant (HiFT) in [jyutvoice/hifigan/generator.py](jyutvoice/hifigan/generator.py).

## Critical Developer Workflows
- **Training**: Run `python -m jyutvoice.train --config configs/base.yaml`.
- **Data Preparation**: Use [scripts/prepare_dataset.py](scripts/prepare_dataset.py) to process raw wave/text pairs into the Hugging Face `datasets` format used by [jyutvoice/data/text_mel_datamodule.py](jyutvoice/data/text_mel_datamodule.py).
- **Inference**: Use [infer.py](infer.py) for waveform synthesis.
- **ONNX Export**: Use [scripts/export_onnx.py](scripts/export_onnx.py) with [configs/export_onnx.yaml](configs/export_onnx.yaml).

## Key Patterns & Conventions
- **Language Codes**: `yue` (Cantonese), `zh` (Mandarin), `en` (English).
- **Multi-Factor Text Input**: Models expect `phonemes`, `tones`, `lang`, `word_pos`, and `syllable_pos`.
- **Multilingual G2P**: [jyutvoice/text/multilingual.py](jyutvoice/text/multilingual.py) handles segmenting and routing to specific G2P modules.
- **Callback/Logger Instantiation**: Nested configs use `_target_` keys (Hydra-style) which are instantiated manually in [jyutvoice/utils/instantiators.py](jyutvoice/utils/instantiators.py).

## Directory Roadmap
- `jyutvoice/text/`: G2P logic for Cantonese, Mandarin, and English.
- `jyutvoice/models/`: Lightning modules and sub-networks.
- `scripts/`: Production utilities (download, prep, export).
- `configs/`: YAML definitions for all model and training parameters.
