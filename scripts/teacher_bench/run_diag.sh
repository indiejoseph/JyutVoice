#!/bin/sh
#PJM -L rscgrp=b-batch-mig
#PJM -L gpu=1
#PJM -L elapse=0:50:00
#PJM -o /home/pj24001684/ku40000295/jc/projects/JyutVoice/tmp/teacher_bench/job_diag_%j.out
#PJM -j

# Diagnostic for the "teacher never emits end-of-speech" smoke result. 48 pieces of 8-110 characters, both prompt modes,
# generation capped at 8 speech tokens per text token (typical is ~4-6); wavs are rendered for config A only. Three configs:
#   A  fine-tuned LLM, fp16        B  fine-tuned LLM, fp32 (rules fp16 in/out)
#   C  upstream CosyVoice2-0.5B llm.pt with the stock vocab/tokenizer (control: does the base model terminate on long text?)
# Read the per-length eos_rate in each runs/diag_*/report.json (summarised at the end of this log).

module load cuda/12.8.1
source $HOME/venv/jc/bin/activate

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

P=$HOME/jc/projects
JV=$P/JyutVoice
IN=$JV/tmp/teacher_bench
cd $JV
mkdir -p $IN

[ -s $IN/sentences_len.jsonl ] || python scripts/teacher_bench/make_text_pool.py \
    --parquet $P/hk_news/data/clustered_summarized/news_summarized_dedup.parquet \
    --out $IN/sentences_len.jsonl --n 48 --bins 8-16,16-24,24-32,32-48,48-72,72-110 --clauses --char_counts $IN/train_char_counts.json
[ -s $IN/prompts/meta.json ] || python scripts/teacher_bench/make_prompts.py \
    --dataset $JV/tmp/dataset_processed --out_dir $IN/prompts --k 4

FT_LLM=$P/tts202501/cosyvoice_exp20250907_prod/results/cosyvoice2-2025090507/best_model/model_acc_0.1813.pt
FT_QWEN=$P/tts202501/cosyvoice_exp20250907_prod/qwen_pretrained/blank
BASE=$P/CosyVoice/pretrained_models/CosyVoice2-0.5B
COMMON="--cosyvoice_dir $P/CosyVoice --base_model_dir $BASE --sentences $IN/sentences_len.jsonl --prompts $IN/prompts/meta.json \
    --train_csv $P/tts202501/cosyvoice_exp20250907_prod/dataset20250907.csv \
    --modes gen --prompt_modes zero_shot,none --n_sentences 48 --max_ratio 8"

echo "######## A: fine-tuned, fp16"
python scripts/teacher_bench/teacher_bench.py $COMMON --llm_ckpt $FT_LLM --qwen_dir $FT_QWEN --work_dir $IN/runs/diag_A_ft_fp16
echo "######## B: fine-tuned, fp32"
python scripts/teacher_bench/teacher_bench.py $COMMON --llm_ckpt $FT_LLM --qwen_dir $FT_QWEN --no_fp16 --no_render --work_dir $IN/runs/diag_B_ft_fp32
echo "######## C: upstream base llm.pt, fp16 (control)"
python scripts/teacher_bench/teacher_bench.py $COMMON --llm_ckpt $BASE/llm.pt --qwen_dir $BASE/CosyVoice-BlankEN --no_render --work_dir $IN/runs/diag_C_base_fp16

echo "######## SUMMARY (eos_rate / median speech tokens per char, by length bin)"
python - <<EOF
import json, glob, os
for d in sorted(glob.glob("$IN/runs/diag_*")):
    p = os.path.join(d, "report.json")
    if not os.path.exists(p):
        print(os.path.basename(d), ": no report (run failed)"); continue
    gen = json.load(open(p))["gen"]
    for mode, r in gen.items():
        cells = "  ".join(f"{b}:{v['eos_rate']:.2f}/{v['speech_tokens_per_char_p50']:.1f}" for b, v in r["by_bin"].items())
        print(f"{os.path.basename(d):22s} {mode:9s} {cells}")
EOF
