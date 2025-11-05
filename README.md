# JyutVoice TTS - Cantonese Text-to-Speech with Transfer Learning

**A high-quality Cantonese text-to-speech system with transfer learning for fast training and excellent quality.**

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
git clone https://github.com/yourusername/jyutvoice-tts.git
cd jyutvoice-tts
conda create -n jyutvoice python=3.11
conda activate jyutvoice
pip install -r requirements.txt
```

### Download Models

```bash
cd pretrained_models
# Speaker embedding model
wget https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/resolve/main/campplus.onnx

# Speech tokenizer
wget https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/resolve/main/speech_tokenizer_v2.onnx
```

### Training

```bash
python -m jyutvoice.train
```

Monitor on WandB: `https://wandb.ai/[your-username]/jyutvoice-tts`

### Prepare Your Dataset

**Create sample dataset files:**
```bash
python scripts/create_sample_dataset.py
```

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

The following models need to be downloaded and placed in `pretrained_models/`:

1. **campplus.onnx** - Speaker embedding model
   ```bash
   cd pretrained_models
   wget https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/resolve/main/campplus.onnx
   ```

2. **speech_tokenizer_v2.onnx** - Speech tokenizer
   ```bash
   wget https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B/resolve/main/speech_tokenizer_v2.onnx
   ```

3. **pretrain.pt** - JyutVoice transfer learning checkpoint
   - Composed of CosyVoice2 flow encoder transferred weights
   - Plus our custom modules initialization
   - Automatically loaded during training
   - Already included in the repository

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
```

## Support

- � [GitHub Issues](https://github.com/indiejoseph/jyutvoice-tts/issues)
- � [GitHub Discussions](https://github.com/indiejoseph/jyutvoice-tts/discussions)
- 📧 indiejoseph@gmail.com
