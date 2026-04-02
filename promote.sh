#!/bin/bash

# =============================================================================
# Promote MLflow runs to production
# Usage:
#   bash deploy/promote.sh                        # promote both MF and SASRec
#   bash deploy/promote.sh --model mf             # promote MF only
#   bash deploy/promote.sh --model sasrec         # promote SASRec only
#   bash deploy/promote.sh --show                 # view current prod config
# =============================================================================

# --- Set these after reviewing MLflow UI ---
MF_RUN_ID="52b31fe21445410bbee0ea92d60760f7"
SASREC_RUN_ID="4eb32b70066343b2926ce49aa0c4eba1"

# =============================================================================

MODEL_FLAG=${1:-"both"}   # default: promote both

if [[ "$1" == "--show" ]]; then
    python src/deploy/promote.py --show
    exit 0
fi

if [[ "$MODEL_FLAG" == "--model" && "$2" == "mf" ]]; then
    python src/deploy/promote.py \
        --model mf \
        --run_id $MF_RUN_ID

elif [[ "$MODEL_FLAG" == "--model" && "$2" == "sasrec" ]]; then
    python src/deploy/promote.py \
        --model sasrec \
        --run_id $SASREC_RUN_ID

else
    # promote both
    python src/deploy/promote.py \
        --model mf     --run_id $MF_RUN_ID \
        --model sasrec --run_id $SASREC_RUN_ID
fi