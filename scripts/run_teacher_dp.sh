#!/usr/bin/env bash
# Duration-predictor distillation from the teacher rows (configs/teacher_dp.yaml) with a given pause bonus on one GPU.
#   nohup scripts/run_teacher_dp.sh <gpu> <pause_bonus> <tag> > ~/teacher/logs/dp_<tag>.log 2>&1 &
# Writes configs/teacher_dp_<tag>.yaml (the base config with that bonus and checkpoints_teacher_dp_<tag>) and trains from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
gpu=$1; bonus=$2; tag=$3
cfg="configs/teacher_dp_${tag}.yaml"
sed -e "s#^pause_bonus: [0-9.]*#pause_bonus: ${bonus}#" -e "s#dirpath: checkpoints_teacher_dp\$#dirpath: checkpoints_teacher_dp_${tag}#" configs/teacher_dp.yaml > "$cfg"
grep -n "^pause_bonus:\|dirpath:" "$cfg"
CUDA_VISIBLE_DEVICES=$gpu HF_HUB_OFFLINE=1 exec "${PYTHON:-python}" -m jyutvoice.train --config "$cfg"
