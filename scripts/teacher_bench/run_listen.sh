#!/bin/sh
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=0:50:00
#PJM -o /home/pj24001684/ku40000295/jc/projects/JyutVoice/tmp/teacher_bench/job_listen_%j.out
#PJM -j

# Listening set from the fine-tuned teacher only (fp16), rendered, on the FILTERED pools (no code-mixing, numbers converted to
# Chinese numerals, only the punctuation 、，。？！ the teacher was trained on, no leaked tags):
#   listen_len : 48 pieces of 8-110 characters       listen_long : 32 long sentences of 20-140 characters
# Both with and without a speaker prompt. Wavs: tmp/teacher_bench/runs/listen_*/wavs/{zero_shot,none}/<id>_p<k>.wav,
# texts and stats in each manifest.tsv (the `raw` text before filtering is in the sentences*.jsonl files).

module load cuda/12.8.1
source $HOME/venv/jc/bin/activate

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

P=$HOME/jc/projects
JV=$P/JyutVoice
IN=$JV/tmp/teacher_bench
cd $JV

FT_LLM=$P/tts202501/cosyvoice_exp20250907_prod/results/cosyvoice2-2025090507/best_model/model_acc_0.1813.pt
FT_QWEN=$P/tts202501/cosyvoice_exp20250907_prod/qwen_pretrained/blank
BASE=$P/CosyVoice/pretrained_models/CosyVoice2-0.5B
COMMON="--cosyvoice_dir $P/CosyVoice --base_model_dir $BASE --prompts $IN/prompts/meta.json \
    --train_csv $P/tts202501/cosyvoice_exp20250907_prod/dataset20250907.csv --modes gen --prompt_modes zero_shot,none \
    --max_ratio 8 --llm_ckpt $FT_LLM --qwen_dir $FT_QWEN"

echo "######## listen_len"
python scripts/teacher_bench/teacher_bench.py $COMMON --sentences $IN/sentences_len.jsonl --n_sentences 48 --work_dir $IN/runs/listen_len
echo "######## listen_long"
python scripts/teacher_bench/teacher_bench.py $COMMON --sentences $IN/sentences.jsonl --n_sentences 32 --work_dir $IN/runs/listen_long
