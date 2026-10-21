#!/bin/sh
#PJM -L rscgrp=a-batch
#PJM -L node=1
#PJM -L elapse=4:00:00
#PJM -o /home/pj24001684/ku40000295/jc/projects/JyutVoice/tmp/sd3_dataset/job_%j.out
#PJM -j

# Same pipeline as run_build_sd3.sh on one CPU node of a-batch (120 cores, 453 GB; no GPU needed: the ModernBERT stage takes ~25 min on CPU).
# Every stage skips work that is already on disk, so re-submitting after a time-out (or after run_build_sd3.sh made progress) continues.

source $HOME/venv/jc/bin/activate
export PATH=$HOME/jc/ffmpeg-master-latest-linux64-gpl-shared/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

P=$HOME/jc/projects
JV=$P/JyutVoice
OUT=$JV/tmp/sd3_dataset
CODE=$JV/tmp/sd3_code
MODEL=$P/CantoneseModernBert/result/ModernCantoneseBert-large-ext
WORKERS=${WORKERS:-96}
cd $CODE || exit 1
mkdir -p $OUT
echo "job $PJM_JOBID on $(hostname), $(nproc) cores visible, started $(date)"

if [ ! -s $OUT/plan/plan.json ]; then
    python scripts/build_sd3_dataset.py plan --transcripts "$JV/tmp/asr_sd3/results/full_nonnews/shard*.jsonl" \
        --vocals-dir $P/speech-dataset3/downloads/vocals --out-dir $OUT --num-shards 240 || exit 1
fi
echo "plan done $(date)"

if [ ! -s $OUT/bert/meta.json ]; then
    python scripts/build_sd3_dataset.py bert --out-dir $OUT --model $MODEL --device cpu --calibrate-n 4000 --batch-size 64 --dtype fp32 || exit 1
fi
echo "bert done $(date)"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/build_sd3_dataset.py build --out-dir $OUT --pretrained $JV/pretrained_models --workers $WORKERS || exit 1
echo "build done $(date)"

python scripts/build_sd3_dataset.py verify --out-dir $OUT --sample 5
echo "finished $(date)"
