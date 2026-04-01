#!/bin/bash

# ── Data ──────────────────────────────────────────────
DATA_DIR="data/ml-32m"
START="2016-06-30"
END="2018-06-30"

# ── Model ─────────────────────────────────────────────
DIM=64
N_HEADS=2
N_LAYERS=2
MAX_LEN=300
LOSS="bpr"

# ── Training ──────────────────────────────────────────
EPOCHS=30
LR=0.001
BATCH_SIZE=128
N_NEG=4

# ── Cold start filters ────────────────────────────────
MIN_RATING=3.0
MIN_U_RAT=10
MIN_I_RAT=10

# ── Evaluation ────────────────────────────────────────
EVAL_MONTHS=5
EVAL_K="5 10 20 100 300 500"
MIN_EVAL_RATINGS="10"
TOP_N_RECS=600

# ── Output ────────────────────────────────────────────
LOG_ROOT="train_logs/SASREC"

# ── Run ───────────────────────────────────────────────
python src/training/sasrec.py \
    --data_dir          $DATA_DIR \
    --start             $START \
    --end               $END \
    --dim               $DIM \
    --n_heads           $N_HEADS \
    --n_layers          $N_LAYERS \
    --max_len           $MAX_LEN \
    --loss              $LOSS \
    --epochs            $EPOCHS \
    --lr                $LR \
    --batch_size        $BATCH_SIZE \
    --n_neg             $N_NEG \
    --min_rating        $MIN_RATING \
    --min_u_rat         $MIN_U_RAT \
    --min_i_rat         $MIN_I_RAT \
    --eval_months       $EVAL_MONTHS \
    --eval_k            $EVAL_K \
    --min_eval_ratings  $MIN_EVAL_RATINGS \
    --top_n_recs        $TOP_N_RECS \
    --log_root          $LOG_ROOT \
    # --use_genre