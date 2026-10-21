#!/usr/bin/env bash
# Adds the prosody noise-0.7 player (tmp/match1_prosody_t07) to the listening page tmp/listen_prosody, right after the noise-0.5 one, and rebuilds index.html.
#   bash scripts/add_noise07_to_page.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/add_variant_to_page.py --page-dir tmp/listen_prosody --src tmp/match1_prosody_t07 --model student_prosody --name prosody_noise0.7 --after prosody_noise0.5
python3 scripts/make_listening_page.py --synth-dir tmp/listen_prosody
