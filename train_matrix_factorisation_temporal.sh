#!/bin/bash

# ── Data ──────────────────────────────────────────────
DATA_DIR="data/ml-32m"
START="2018-01-01"
END="2018-06-30"

# ── Model ─────────────────────────────────────────────
DIM=64
EPOCHS=20
LR=0.005
REG1=0.01
REG2=0.001
N_NEG=4
BETA=0.4
ALPHA_INIT=0.001

# ── Cold start filters ────────────────────────────────
MIN_RATING=3.0
MIN_U_RAT=10
MIN_I_RAT=10

# ── Evaluation ────────────────────────────────────────
EVAL_MONTHS=5
EVAL_K="5 10 20"
MIN_EVAL_RATINGS="10"
TOP_N_RECS=50

# ── Output ────────────────────────────────────────────
LOG_ROOT="train_logs/MF_temporal"

LOSS="mse"

# ── Run ───────────────────────────────────────────────
python src/training/matrix_factorisation_temporal.py \
    --data_dir          $DATA_DIR \
    --start             $START \
    --end               $END \
    --dim               $DIM \
    --epochs            $EPOCHS \
    --lr                $LR \
    --reg1              $REG1 \
    --reg2              $REG2 \
    --n_neg             $N_NEG \
    --beta              $BETA \
    --alpha_init        $ALPHA_INIT \
    --min_rating        $MIN_RATING \
    --min_u_rat         $MIN_U_RAT \
    --min_i_rat         $MIN_I_RAT \
    --eval_months       $EVAL_MONTHS \
    --eval_k            $EVAL_K \
    --min_eval_ratings  $MIN_EVAL_RATINGS \
    --top_n_recs        $TOP_N_RECS \
    --log_root          $LOG_ROOT \
    --loss              $LOSS