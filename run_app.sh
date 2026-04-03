#!/bin/bash
# scripts/serve.sh
# ─────────────────────────────────────────────────────────────────
# Usage:
#   ./scripts/serve.sh                  # GPU, defaults
#   ./scripts/serve.sh --cpu            # force CPU
#   ./scripts/serve.sh --port 8001      # custom port
#   ./scripts/serve.sh --cpu --port 8001
# ─────────────────────────────────────────────────────────────────

set -e

# ── defaults ──────────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT=8001
FORCE_CPU=0
WORKERS=1
LOG_LEVEL="info"

export FORCE_CPU=1
export CACHE_ENABLED=1
# ── parse args ────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --cpu)        FORCE_CPU=1;      shift ;;
    --port)       PORT="$2";        shift 2 ;;
    --workers)    WORKERS="$2";     shift 2 ;;
    --log-level)  LOG_LEVEL="$2";   shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── required env check ────────────────────────────────────────────
if [[ -z "$SASREC_RUN_ID" ]]; then
  echo "ERROR: SASREC_RUN_ID is not set."
  echo "  export SASREC_RUN_ID=<your_mlflow_run_id>"
  exit 1
fi

# ── optional env with defaults ────────────────────────────────────
export CANDIDATES_DB="${CANDIDATES_DB:-data/candidates.db}"
export CANDIDATE_CACHE_SIZE="${CANDIDATE_CACHE_SIZE:-5000}"
export CANDIDATE_CACHE_TTL="${CANDIDATE_CACHE_TTL:-3600}"
export OUTPUT_CACHE_SIZE="${OUTPUT_CACHE_SIZE:-10000}"
export OUTPUT_CACHE_TTL="${OUTPUT_CACHE_TTL:-600}"
export FORCE_CPU="$FORCE_CPU"

# ── print config ──────────────────────────────────────────────────
echo "────────────────────────────────────────"
echo "  RecSys Serving API"
echo "────────────────────────────────────────"
echo "  Host            : $HOST:$PORT"
echo "  Device          : $([ "$FORCE_CPU" = "1" ] && echo "CPU (forced)" || echo "GPU (if available)")"
echo "  SASREC_RUN_ID   : $SASREC_RUN_ID"
echo "  CANDIDATES_DB   : $CANDIDATES_DB"
echo "  Workers         : $WORKERS"
echo "  Log level       : $LOG_LEVEL"
echo "  Cache           : $([ "${CACHE_ENABLED:-1}" = "1" ] && echo "ON" || echo "OFF")"
echo "────────────────────────────────────────"

# inside the script, add this debug line before uvicorn
# this line initializes conda for the current shell session
source "$(conda info --base)/etc/profile.d/conda.sh"

# now activate works
conda activate base



# ── launch ────────────────────────────────────────────────────────
uvicorn src.serving.app:app \
  --host "$HOST" \
  --port "$PORT" \
  --workers "$WORKERS" \
  --log-level "$LOG_LEVEL"