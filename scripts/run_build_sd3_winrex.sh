#!/bin/bash
# Runs the SD3 dataset build on winrex (plan -> bert on GPU 0 -> shards on a CPU pool -> verify). Resumable: finished stages and shards are skipped.
#   nohup ~/sd3_build/code/scripts/run_build_sd3_winrex.sh [WORKERS=26] > ~/sd3_build/out/run.log 2>&1 &
# GPUs 4-7 and part of the CPU belong to production services: the CPU work runs at nice 10 and only GPU 0 is used.
B=$HOME/sd3_build
OUT=$B/out
WORKERS=${1:-26}   # ~2.1 GB RSS each in the pilot; winrex has ~88 GB available next to the production services
source $B/env/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd $B/code || exit 1
mkdir -p $OUT
echo "started $(date) on $(hostname), $WORKERS workers"

if [ ! -s $OUT/plan/plan.json ]; then
    OMP_NUM_THREADS=1 nice -n 10 python scripts/build_sd3_dataset.py plan --transcripts "$HOME/sd3_asr/out/full_nonnews/shard*.jsonl" \
        --vocals-dir $HOME/sd3_asr/full/vocals --out-dir $OUT --num-shards 240 --workers 24 || exit 1
fi
echo "plan done $(date)"

if [ ! -s $OUT/bert/meta.json ]; then
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 python scripts/build_sd3_dataset.py bert --out-dir $OUT --model $B/models/ModernCantoneseBert-large-ext \
        --device cuda --threads 8 --calibrate-n 4000 --batch-size 128 --dtype fp32 || exit 1
fi
echo "bert done $(date)"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python scripts/build_sd3_dataset.py build --out-dir $OUT --pretrained $B/pretrained_models --workers $WORKERS || exit 1
echo "build done $(date)"

python scripts/build_sd3_dataset.py verify --out-dir $OUT --sample 5
echo "finished $(date)"
