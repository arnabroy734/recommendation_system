#!/bin/bash

# ── Data ──────────────────────────────────────────────
DATA_DIR="data/ml-32m"
START="2018-01-01"
END="2018-06-30"

# ── Model ─────────────────────────────────────────────
LOSS="bpr"           # mse or bpr
DIM=64
EPOCHS=20
LR=0.005
REG1=0.01
REG2=0.01
N_NEG=4              # negatives per positive (BPR only)

# ── Cold start filters ────────────────────────────────
MIN_RATING=3.0
MIN_U_RAT=5
MIN_I_RAT=5

# ── Evaluation ────────────────────────────────────────
EVAL_MONTHS=5
EVAL_K="5 10 20 100 300 500"
MIN_EVAL_RATINGS="5"
TOP_N_RECS=600

# ── Output ────────────────────────────────────────────
LOG_ROOT="train_logs/MF"

# ── Run ───────────────────────────────────────────────
python src/training/matrix_factorisation.py \
    --data_dir        $DATA_DIR \
    --start           $START \
    --end             $END \
    --loss            $LOSS \
    --dim             $DIM \
    --epochs          $EPOCHS \
    --lr              $LR \
    --reg1            $REG1 \
    --reg2            $REG2 \
    --n_neg           $N_NEG \
    --min_rating      $MIN_RATING \
    --min_u_rat       $MIN_U_RAT \
    --min_i_rat       $MIN_I_RAT \
    --eval_months     $EVAL_MONTHS \
    --eval_k          $EVAL_K \
    --min_eval_ratings $MIN_EVAL_RATINGS \
    --top_n_recs      $TOP_N_RECS \
    --log_root        $LOG_ROOT