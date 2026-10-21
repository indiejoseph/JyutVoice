#!/bin/sh
#PJM -L rscgrp=b-batch
#PJM -L gpu=1
#PJM -L elapse=6:00:00
#PJM -o /home/pj24001684/ku40000295/jc/projects/JyutVoice/tmp/sd3_dataset/job_%j.out
#PJM -j

# Builds the speech-dataset3 training shards (see scripts/build_sd3_dataset.py): plan (CPU) -> bert features (GPU) -> shards (CPU pool) -> verify.
# Every stage skips work that is already on disk, so after a time-out simply submit again.
# Result: $OUT/shards/shard_*  (TextMelDataModule dataset_path: "$OUT/shards/shard_*", bert_dim: 1024)
# The code runs from a snapshot ($CODE: jyutvoice/, scripts/build_sd3_dataset.py and a `bert` symlink for the English g2p's tokenizer), not from the
# working copy on genkai, which is older than the local one this was written and tested against.

module load cuda/12.8.1
source $HOME/venv/jc/bin/activate
export PATH=$HOME/jc/ffmpeg-master-latest-linux64-gpl-shared/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

P=$HOME/jc/projects
JV=$P/JyutVoice
OUT=$JV/tmp/sd3_dataset
CODE=$JV/tmp/sd3_code
MODEL=$P/CantoneseModernBert/result/ModernCantoneseBert-large-ext
WORKERS=${WORKERS:-28}
cd $CODE || exit 1
mkdir -p $OUT
echo "job $PJM_JOBID on $(hostname), $(nproc) cores visible, started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -s $OUT/plan/plan.json ]; then
    python scripts/build_sd3_dataset.py plan --transcripts "$JV/tmp/asr_sd3/results/full_nonnews/shard*.jsonl" \
        --vocals-dir $P/speech-dataset3/downloads/vocals --out-dir $OUT --num-shards 240 || exit 1
fi
echo "plan done $(date)"

if [ ! -s $OUT/bert/meta.json ]; then
    python scripts/build_sd3_dataset.py bert --out-dir $OUT --model $MODEL --calibrate-n 4000 --batch-size 128 --dtype fp32 || exit 1
fi
echo "bert done $(date)"

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/build_sd3_dataset.py build --out-dir $OUT --pretrained $JV/pretrained_models --workers $WORKERS || exit 1
echo "build done $(date)"

python scripts/build_sd3_dataset.py verify --out-dir $OUT --sample 5
echo "finished $(date)"
