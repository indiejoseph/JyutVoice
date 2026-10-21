#!/bin/bash
# One-time setup of the SD3 dataset build environment on winrex (run by the user: it downloads packages from PyPI).
# A venv on top of conda's python (torch 2.8 + CUDA, onnxruntime, whisper, transformers, ToJyutping, soundfile are already there) so that the base
# environment, which other services may use, is not touched. Only the 12 packages the build imports and the base env lacks are installed.
set -e
B=$HOME/sd3_build
/home/joseph/miniconda3/bin/python -m venv --system-site-packages $B/env
# huggingface-hub < 1.0: the base env's transformers 4.57 refuses 2.x, which the newest datasets pulls in (found on the first run)
$B/env/bin/pip install "huggingface-hub>=0.34.0,<1.0" "datasets<5" g2p_en gdown hydra-core inflect nltk pycantonese pydips pypinyin wandb wget wordsegment
# the base env's torchvision (0.25, built for torch 2.10) breaks `import transformers` ModernBERT code paths on its torch 2.8 (found on the first run)
$B/env/bin/pip install --no-deps "torchvision==0.23.0"
$B/env/bin/python -c "from transformers import ModernBertModel; import datasets, g2p_en, pycantonese, pydips, pypinyin, wordsegment, ToJyutping, torch, onnxruntime, whisper; print('imports ok', torch.__version__)"
