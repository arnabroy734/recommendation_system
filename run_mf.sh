#!/bin/bash

# =============================================================================
# Train Matrix Factorisation model
# Usage: bash run_mf.sh
# =============================================================================

# --- Training window ---
START="2016-06-30"
END="2018-06-30"

# --- Eval window ---
EVAL_START="2018-07-01"
EVAL_END="2018-07-30"

# --- Model hyperparams ---
LOSS="bpr"          # bpr or mse
DIM=64
EPOCHS=20
LR=0.005
REG1=0.01
REG2=0.001
N_NEG=4

# --- Data filters ---
MIN_RATING=3.0
MIN_U_RAT=5
MIN_I_RAT=5

# --- Eval settings ---
EVAL_K="5 10 15 20 50 100"
TOP_N_RECS=100
MIN_EVAL_RATINGS="5"

# =============================================================================

python src/training/matrix_factorisation.py \
    --start            $START \
    --end              $END \
    --eval_start       $EVAL_START \
    --eval_end         $EVAL_END \
    --loss             $LOSS \
    --dim              $DIM \
    --epochs           $EPOCHS \
    --lr               $LR \
    --reg1             $REG1 \
    --reg2             $REG2 \
    --n_neg            $N_NEG \
    --min_rating       $MIN_RATING \
    --min_u_rat        $MIN_U_RAT \
    --min_i_rat        $MIN_I_RAT \
    --eval_k           $EVAL_K \
    --top_n_recs       $TOP_N_RECS \
    --min_eval_ratings $MIN_EVAL_RATINGS