# JyutVoice TTS - Cantonese Text-to-Speech with Transfer Learning

**A high-quality Cantonese text-to-speech system with transfer learning for fast training and excellent quality.**

> Working on this repository (human or agent)? Start with [`docs/README.md`](docs/README.md): architecture, text front end, training recipe, inference/API, evaluation, and the decisions and known issues that should not be re-litigated.

## Features

- 🎤 High-quality 24kHz Cantonese speech synthesis
- ⚡ 3-5x faster training (transfer learning with frozen CosyVoice2 decoder)
- 💾 3x less GPU memory (8GB vs 24GB)
- 🌍 Multilingual support (Cantonese, Mandarin, English)
- 🔬 PyTorch Lightning + WandB for easy experimentation
- ✅ Fully tested and production-ready

## Quick Start

### Installation

```bash
git clone https://github.com/indiejoseph/JyutVoice.git
cd JyutVoice
conda create -n jyutvoice python=3.11
conda activate jyutvoice
pip install -r requirements.txt
```

### Download Pretrained Models

```bash
python scripts/download_pretrain_weights.py
```

### Training

```bash
python -m jyutvoice.train
```

Monitor on WandB: `https://wandb.ai/[your-username]/jyutvoice-tts`

### Prepare Your Dataset

**Dataset format (JSON or CSV):**
```json
{"text": "你好世界", "lang": "zh", "audio": "speaker_001/001.wav"}
{"text": "这是测试", "lang": "zh", "audio": "speaker_001/002.wav"}
{"text": "Hello world", "lang": "en", "audio": "speaker_001/003.wav"}
```

**Process dataset (word segmentation + validation):**
```bash
python scripts/prepare_dataset.py --dataset sample_data.json --output processed_dataset/
```

**Update config to use your dataset:**
```yaml
# In configs/base.yaml
data:
  dataset_path: processed_dataset/
  batch_size: 4
```

Supported languages: `zh` (Mandarin), `yue` (Cantonese), `en` (English)

### Inference

```python
import torch
from jyutvoice.models.jyutvoice_tts import JyutVoiceTTS

model = JyutVoiceTTS.load_from_checkpoint("checkpoint.ckpt")
model.eval()

# Generate speech
with torch.no_grad():
    output = model.synthesise(
        x=text_tokens,
        x_lengths=x_lengths,
        lang=lang_tokens,
        tone=tone_tokens,
        word_pos=word_pos_tokens,
        syllable_pos=syllable_pos_tokens,
        n_timesteps=10,
        spk_embed=speaker_embedding,
    )
mel_spec = output['decoder_outputs']  # (1, 80, mel_length)
```

### Best model

The current best configuration is the encoder + flow duration sampler (`checkpoints/best/jyutvoice_flowdp_epoch019.ckpt`) with the frame-level `ProsodyCFM` trained on corpus + teacher rows (`checkpoints/best/prosody_cfm_mix1.pt`, noise 0.5). `JyutVoiceTTS.synthesise(..., use_prosody=True)` samples the decoder input with it; the ProsodyCFM defaults (noise 0.5, no guidance, 10 steps) are the chosen ones. Settings, scores and a render command are in `checkpoints/best/README.md`.

The English-capable successor (encoder `checkpoints_yue_en`, its flow duration sampler and a ProsodyCFM trained on its features; Cantonese + English code-mixing) is what `configs/export_onnx.yaml` exports by default. It reads English words much better (97% of English words recognised against 47% before) at the cost of about one point of Cantonese syllable error rate; its sampler renders about 10% shorter than the teacher (median length ratio 0.90 on 60 held-out sentences), `--length_scale 1.1` calibrates it (0.99).

### ONNX inference

The text encoder, the duration model (flow-matching sampler or regression predictor) and ProsodyCFM run in ONNX Runtime; the CosyVoice2 flow decoder and HiFT stay in PyTorch.

```bash
# 1. export (graphs + manifest.json in onnx_models/; every graph is checked against PyTorch, the whole chain against JyutVoiceTTS.synthesise)
python scripts/export_onnx.py                      # checkpoint, config and ProsodyCFM weights come from configs/export_onnx.yaml
python scripts/export_onnx.py --checkpoint checkpoints/best/jyutvoice_flowdp_epoch019.ckpt \
    --prosody-ckpt checkpoints/best/prosody_cfm_mix1.pt --output-dir onnx_models_best   # the Cantonese-only best

# 2. synthesise (text is normalised, word-segmented with pydips and read by g2p like in training)
python infer_onnx.py --onnx_dir onnx_models --text "我喺Zoom開會，但係個microphone有問題。" --ref_audio ref.wav --output out.wav
```

The flow decoder gets only the x-vector of the reference clip, no clip prompt (that needs the speech tokenizer and the flow encoder), unlike `synthesise(..., prompt_feat=...)`. `--no_prosody` skips ProsodyCFM, `--segmentation given` takes text that is already word-segmented (no pydips needed). The sampler graphs take their random noise as an input and the Euler loops are unrolled in the graph (10 steps, as exported). The text encoder accepts at most `max_tokens` (1024, about 30 s of speech) tokens: longer text must be split at punctuation. The rotary position tables are baked into the graph, which is why the limit is fixed at export time.

### HTTP API (Docker, CPU)

`api/` is a FastAPI service over the ONNX path and `docker/` runs it on the CPU. The reference clip comes in the request as base64 (`prompt_audio`), the endpoints and the request shape follow the SoulX-Podcast API (`GET /`, `GET /health`, `POST /v1/audio/speech` with Bearer auth, a `Prompt-Cache-Id` voice cache and optional streaming).

```bash
cp docker/.env.example docker/.env      # set API_KEY (openssl rand -hex 32)
docker compose -f docker/docker-compose.yml up -d --build   # needs onnx_models/ (see above) and pretrained_models/ next to it
python api/test_client.py --ref ref.wav --text "我喺Zoom開會，但係個microphone有問題。" --output out.wav
```

The weights are mounted into the container, not baked into the image. On a CPU the flow decoder and vocoder dominate: about real time at 10 flow steps. See `api/README.md` for the fields, errors, settings and what differs from the SoulX-Podcast API.

#### Playground

`playground/` is a web page for trying the API (preset or uploaded voice, dialogues, seed, streaming, a copyable `curl` call), modelled on the SoulX-Podcast playground. With the API running:

```bash
docker compose -f playground/docker-compose.yml --env-file docker/.env up -d --build   # http://127.0.0.1:8898
```

It holds the API key itself and has no login, so keep it on localhost or behind one. See `playground/README.md`.

## Inference Examples

Use the provided `infer.py` script for easy text-to-speech synthesis with voice cloning:

### Basic Usage

```bash
# English text
python infer.py --text "Hello world, this is a test." --lang en --ref_audio reference.wav --output output.wav

# Mandarin Chinese
python infer.py --text "你好，欢迎使用这个语音合成系统。" --lang zh --ref_audio reference.wav --output output.wav

# Cantonese
python infer.py --text "你好，歡迎使用呢個語音合成系統。" --lang yue --ref_audio reference.wav --output output.wav

# Multilingual (mixed languages)
python infer.py --text "你好，歡迎使用呢個 Text to speech 語音合成系統。" --lang multilingual --ref_audio reference.wav --output output.wav
```

### Advanced Options

```bash
# Adjust speech speed (0.5 = slower, 2.0 = faster)
python infer.py --text "Hello world" --lang en --ref_audio reference.wav --output output.wav --length_scale 0.8

# Change diffusion steps (more steps = higher quality but slower)
python infer.py --text "Hello world" --lang en --ref_audio reference.wav --output output.wav --n_timesteps 20

# Use custom model paths
python infer.py --text "Hello world" --lang en --ref_audio reference.wav --output output.wav \
    --tts_checkpoint my_model.ckpt \
    --flow_encoder custom_flow_encoder.pt \
    --hift custom_hift.pt
```

### Voice Cloning

The system supports voice cloning using reference audio:

1. **Reference Audio Requirements:**
   - Format: WAV, MP3, or other common audio formats
   - Sample rate: Any (automatically resampled)
   - Duration: 3-10 seconds recommended
   - Content: Clear speech from target speaker

2. **Example with different reference audio:**
   ```bash
   # Use a high-quality reference for better voice cloning
   python infer.py --text "今天天气真好" --lang zh --ref_audio high_quality_sample.wav --output cloned_voice.wav
   ```

### Cantonese with Phonetic Input

For precise Cantonese pronunciation, you can provide phonetic transcription:

```bash
python infer.py --text "呢隻嘢真係好勁呀。" --lang yue --phone "ni1 zek3 je5 zan1 hai6 hou2 kem5 aa3 ." --ref_audio reference.wav --output output.wav
```

### Model Configuration

The inference script uses these default model paths (can be overridden):

- **TTS Checkpoint:** `pretrained_models/epoch=0-step=55872.ckpt`
- **Flow Encoder:** `pretrained_models/flow_encoder.pt`
- **Speech Tokenizer:** `pretrained_models/speech_tokenizer_v2.onnx`
- **Speaker Model:** `pretrained_models/campplus.onnx`
- **HiFi-GAN Vocoder:** `pretrained_models/hift.pt`

### Performance

- **GPU Memory:** ~4GB for inference
- **Latency:** ~1-2 seconds per sentence
- **Output Quality:** 24kHz, high-fidelity speech

## Architecture

```
Text Input → Text Encoder (7.2M) → Duration Predictor → Prior Loss
                                        ↓
                                  Alignment (MAS)
                                        ↓
                    Flow Decoder (71.3M frozen) → Diffusion Loss
                                        ↓
                              Output: 24kHz Speech
```

**Model**: 78.5M total parameters (7.2M trainable)
- **Text Encoder** (trainable): RoPE transformer for linguistic representation
- **Flow Decoder** (frozen): CosyVoice2 for high-quality mel-spectrogram generation
- **Speaker Embedding**: Affine transformation for speaker control

## Configuration

Edit `configs/base.yaml`:

```yaml
# Audio
sample_rate: 24000
n_fft: 1920
hop_length: 480
n_feats: 80

# Training
trainer:
  accelerator: cuda
  devices: 1
  precision: bf16-mixed
  max_epochs: 50

# Data
data:
  dataset_path: tmp/dummy_dataset
  batch_size: 4
  num_workers: 0

# Optimizer
optimizer: !name:torch.optim.AdamW
  lr: 1e-4
```

## Model Setup

### Download Pretrained Models

Run the download script to get all required models:

```bash
python scripts/download_pretrain_weights.py
```

This will download and prepare:

- **campplus.onnx** - Speaker embedding model
- **speech_tokenizer_v2.onnx** - Speech tokenizer
- **flow_encoder.pt** & **flow_decoder.pt** - Extracted from CosyVoice2 flow model
- **hift.pt** - HiFi-GAN vocoder
- **pretrain.pt** - JyutVoice transfer learning checkpoint (generated)

## Transfer Learning

- **Pretrained Decoder**: Frozen weights from CosyVoice2 (flow encoder)
- **Automatic Loading**: `1039 weights` load on model initialization
  - Encoder: 127 weights
  - Decoder: 910 weights (from CosyVoice2)
  - Speaker layer: 2 weights
- **Performance**: 3-5x faster convergence, same quality as training from scratch

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` in config |
| Checkpoint not found | Check `pretrained_models/pretrain.pt` exists |
| Training is slow | Enable `bf16-mixed` precision, increase `num_workers` |
| NaN losses | Reduce learning rate to `5e-5` |

### Known issues

- **English words inside Cantonese are not always right in sound or stress.** In the English-capable model, `Zoom` can sound like "drone" (its `z` and `m` get about one frame, 20 ms, where the aligned data has two to three) and `interview` has the stress in the wrong place. The text front end is correct (g2p gives the dictionary readings) and the recogniser hears both words correctly, so the error is in phone timing and prosody, which the ASR scores cannot see. The English phones are learned from only 27 h of Hong Kong speakers' code-mixed English. Hong Kong speakers do not use a native English accent when they mix English into Cantonese, so more native English speech is not assumed to be the fix. Not worked on; open until it is decided what the target pronunciation should be.

## Project Structure

```
jyutvoice-tts/
├── configs/
│   └── base.yaml                  # Configuration
├── jyutvoice/
│   ├── train.py                   # Training entry point
│   ├── models/
│   │   ├── jyutvoice_tts.py
│   │   ├── text_encoder.py
│   │   └── baselightningmodule.py
│   ├── data/
│   │   └── text_mel_datamodule.py
│   ├── flow/
│   │   ├── flow_matching.py
│   │   └── decoder.py
│   ├── text/
│   │   └── cleaners.py
│   └── utils/
├── api/                         # FastAPI service over the ONNX path (README.md, test_client.py)
├── docker/                      # Dockerfile, docker-compose.yml, .env.example for the CPU API
├── pretrained_models/
│   ├── pretrain.pt              # Transfer learning checkpoint (CosyVoice2 + modules init)
│   ├── campplus.onnx            # Speaker embedding model (download from HF)
│   ├── speech_tokenizer_v2.onnx # Speech tokenizer (download from HF)
│   ├── flow_decoder.pt          # Flow decoder weights
│   └── hift.pt                  # HiFi-GAN vocoder
└── scripts/
    └── prepare_dataset.py
```

## Related Work

- [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice) - Pretrained decoder
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) - Text encoder
- [PyTorch Lightning](https://lightning.ai/) - Training framework

## Disclaimer

This work builds upon code from Matcha-TTS and CosyVoice with modified linguistic features. The content provided above is for academic and research purposes only and is intended to demonstrate technical capabilities in text-to-speech synthesis.

## License

MIT License - see [LICENSE](LICENSE) file

## Citation

```bibtex
@software{jyutvoice2025,
  title={JyutVoice TTS: Cantonese Text-to-Speech with Transfer Learning},
  author={Joseph Cheng},
  year={2025},
  url={https://github.com/indiejoseph/jyutvoice-tts}
}

@inproceedings{mehta2024matcha,
  title={Matcha-{TTS}: A fast {TTS} architecture with conditional flow matching},
  author={Mehta, Shivam and Tu, Ruibo and Beskow, Jonas and Sz{\'e}kely, {\'E}va and Henter, Gustav Eje},
  booktitle={Proc. ICASSP},
  year={2024}
}

@article{du2024cosyvoice,
  title={Cosyvoice: A scalable multilingual zero-shot text-to-speech synthesizer based on supervised semantic tokens},
  author={Du, Zhihao and Chen, Qian and Zhang, Shiliang and Hu, Kai and Lu, Heng and Yang, Yexin and Hu, Hangrui and Zheng, Siqi and Gu, Yue and Ma, Ziyang and others},
  journal={arXiv preprint arXiv:2407.05407},
  year={2024}
}

@article{du2024cosyvoice,
  title={Cosyvoice 2: Scalable streaming speech synthesis with large language models},
  author={Du, Zhihao and Wang, Yuxuan and Chen, Qian and Shi, Xian and Lv, Xiang and Zhao, Tianyu and Gao, Zhifu and Yang, Yexin and Gao, Changfeng and Wang, Hui and others},
  journal={arXiv preprint arXiv:2412.10117},
  year={2024}
}

@article{du2025cosyvoice,
  title={CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training},
  author={Du, Zhihao and Gao, Changfeng and Wang, Yuxuan and Yu, Fan and Zhao, Tianyu and Wang, Hao and Lv, Xiang and Wang, Hui and Shi, Xian and An, Keyu and others},
  journal={arXiv preprint arXiv:2505.17589},
  year={2025}
}

@inproceedings{lyu2025build,
  title={Build LLM-Based Zero-Shot Streaming TTS System with Cosyvoice},
  author={Lyu, Xiang and Wang, Yuxuan and Zhao, Tianyu and Wang, Hao and Liu, Huadai and Du, Zhihao},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--2},
  year={2025},
  organization={IEEE}
}
```

## Support

- � [GitHub Issues](https://github.com/indiejoseph/jyutvoice-tts/issues)
- � [GitHub Discussions](https://github.com/indiejoseph/jyutvoice-tts/discussions)
- 📧 indiejoseph@gmail.com
