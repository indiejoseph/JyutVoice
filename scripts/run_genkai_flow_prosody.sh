#!/bin/sh
#PJM -L rscgrp=c-batch
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -j

set -e

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate matcha-tts

cd $HOME/jc/projects/JyutVoice/

export PYTHONUNBUFFERED=1
echo "Starting Flow-Matching Duration & Prosody training on Genkai..."
python -m jyutvoice.train --config configs/genkai_flow_prosody.yaml
