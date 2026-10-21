#!/bin/sh
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=1:30:00
#PJM -o /home/pj24001684/ku40000295/jc/projects/JyutVoice/tmp/teacher_bench/job_%j.out
#PJM -j

# Full benchmark for scripts/teacher_bench/teacher_bench.py on one MIG slice:
#   100 long Cantonese sentences x {zero_shot, none} prompt modes (generation + rendered wavs),
#   then teacher-forced hidden-state extraction at several batch sizes.
# Run run_smoke.sh first. Results: tmp/teacher_bench/runs/full/{report.json,manifest.tsv,wavs/}

module load cuda/12.8.1
source $HOME/venv/jc/bin/activate

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

P=$HOME/jc/projects
JV=$P/JyutVoice
IN=$JV/tmp/teacher_bench
cd $JV
mkdir -p $IN

[ -s $IN/sentences.jsonl ] || python scripts/teacher_bench/make_text_pool.py \
    --parquet $P/hk_news/data/clustered_summarized/news_summarized_dedup.parquet \
    --out $IN/sentences.jsonl --n 100 --char_counts $IN/train_char_counts.json
[ -s $IN/prompts/meta.json ] || python scripts/teacher_bench/make_prompts.py \
    --dataset $JV/tmp/dataset_processed --out_dir $IN/prompts --k 4

python scripts/teacher_bench/teacher_bench.py \
    --cosyvoice_dir $P/CosyVoice \
    --base_model_dir $P/CosyVoice/pretrained_models/CosyVoice2-0.5B \
    --llm_ckpt $P/tts202501/cosyvoice_exp20250907_prod/results/cosyvoice2-2025090507/best_model/model_acc_0.1813.pt \
    --qwen_dir $P/tts202501/cosyvoice_exp20250907_prod/qwen_pretrained/blank \
    --work_dir $IN/runs/full \
    --sentences $IN/sentences.jsonl --prompts $IN/prompts/meta.json \
    --train_csv $P/tts202501/cosyvoice_exp20250907_prod/dataset20250907.csv \
    --modes gen,hidden --prompt_modes zero_shot,none \
    --n_sentences 100 --n_hidden 512 --hidden_batch_sizes 8,16,32,64
