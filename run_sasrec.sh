#!/bin/bash

# =============================================================================
# Train SASRec model
# Usage:
#   bash run_sasrec.sh              # base SASRec
#   bash run_sasrec.sh --use_genre  # SASRec with genre fusion
# =============================================================================

# --- Training window ---
START="2016-06-30"
END="2018-06-30"

# --- Eval window ---
EVAL_START="2018-07-01"
EVAL_END="2018-07-30"

# --- Model hyperparams ---
LOSS="bpr"          # bce or bpr
DIM=64
EPOCHS=30
LR=0.001
MAX_LEN=200
BATCH_SIZE=128
N_NEG=4
N_HEADS=2
N_LAYERS=2

# --- Data filters ---
MIN_RATING=3.0
MIN_U_RAT=5
MIN_I_RAT=5

# --- Eval settings ---
EVAL_K="5 10 15 20 50 100"
TOP_N_RECS=100
MIN_EVAL_RATINGS="5"

# --- Genre ---
N_GENRES=12

# =============================================================================

python src/training/sasrec.py \
    --start            $START \
    --end              $END \
    --eval_start       $EVAL_START \
    --eval_end         $EVAL_END \
    --loss             $LOSS \
    --dim              $DIM \
    --epochs           $EPOCHS \
    --lr               $LR \
    --max_len          $MAX_LEN \
    --batch_size       $BATCH_SIZE \
    --n_neg            $N_NEG \
    --n_heads          $N_HEADS \
    --n_layers         $N_LAYERS \
    --min_rating       $MIN_RATING \
    --min_u_rat        $MIN_U_RAT \
    --min_i_rat        $MIN_I_RAT \
    --eval_k           $EVAL_K \
    --top_n_recs       $TOP_N_RECS \
    --min_eval_ratings $MIN_EVAL_RATINGS \
    --n_genres         $N_GENRES \
    "$@"                # pass --use_genre if provided on command line