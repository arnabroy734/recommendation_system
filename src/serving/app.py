"""
src/serving/app.py
-------------------
FastAPI serving layer for the recommendation system.

Endpoints
---------
  GET /health
  GET /recommendations/{user_id}   →  MF candidates → SASRec re-rank → top-N
  GET /next-item/{user_id}         →  same pipeline, top-1
  GET /cache/stats                 →  L1 / L2 hit-miss counters
  GET /cache/clear                 →  reset both caches (for benchmarking)

Startup args (env vars)
-----------------------
  SASREC_RUN_ID     MLflow run_id for SASRec (required)
  CANDIDATES_DB     path to candidates.db     (default: data/candidates.db)
  FORCE_CPU         set to "1" to disable CUDA (default: 0)
  CANDIDATE_CACHE_SIZE   (default: 5000)
  CANDIDATE_CACHE_TTL    (default: 3600)
  OUTPUT_CACHE_SIZE      (default: 10000)
  OUTPUT_CACHE_TTL       (default: 600)

Usage
-----
  uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.serving.utils import (
    COLD_START_KEY,
    build_caches,
    build_sequence,
    get_candidates_cached,
    get_device,
    load_candidates_db,
    load_movielens_db,
    load_sasrec_artifacts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Global state  (populated at startup)
# ─────────────────────────────────────────────────────────────────────────────

state: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# SASRec inference helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sasrec_score(
    model:        torch.nn.Module,
    seq:          list[int],
    item_indices: np.ndarray,
    device:       torch.device,
) -> np.ndarray:
    """
    Score candidate items using the last position of SASRec's output.

    seq is chronological [i1, i2, ..., i_most_recent], no padding.
    We feed it as-is and take h[:, -1, :] as the user state vector,
    then dot with item embeddings of candidates.
    """
    if len(seq) == 0:
        return np.zeros(len(item_indices), dtype=np.float32)

    # (1, seq_len)  — no padding, last token is most recent
    seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)

    # forward → (1, L, dim)
    h, _ = model(seq_tensor)

    # user state = last position (most recent item's contextual repr)
    user_vec = h[:, -1, :]   # (1, dim)

    # candidate item embeddings  →  (n_candidates, dim)
    item_tensor = torch.tensor(item_indices, dtype=torch.long, device=device)
    item_embs   = model.item_emb(item_tensor)   # (n_candidates, dim)

    # scores  →  (n_candidates,)
    scores = (item_embs @ user_vec.squeeze(0)).cpu().numpy()

    return scores

def rerank_candidates(
    candidates:  list[dict],
    seq:         list[int],
    encoder,
    model:       torch.nn.Module,
    device:      torch.device,
    top_n:       int,
) -> list[dict]:
    """
    Re-rank MF candidates with SASRec scores.

    Pipeline
    --------
    1. Map candidate movie_ids → item indices (skip unknowns)
    2. Score with SASRec
    3. Sort descending, take top_n
    4. Return [{movie_id, sasrec_score, rank}]
    """
    # ── map movie_id → item_idx, keep only known items ────────────────────────
    known, movie_ids, item_indices = [], [], []
    for c in candidates:
        mid = c["movie_id"]
        if mid in encoder.item_id:
            known.append(mid)
            movie_ids.append(mid)
            item_indices.append(encoder.item_id[mid])

    if not known:
        # no overlap between candidates and encoder — return as-is
        return [
            {"movie_id": c["movie_id"], "sasrec_score": 0.0, "rank": i + 1}
            for i, c in enumerate(candidates[:top_n])
        ]

    item_indices_np = np.array(item_indices)

    # ── SASRec scores ─────────────────────────────────────────────────────────
    scores = sasrec_score(model, seq, item_indices_np, device)

    # ── sort + top_n ──────────────────────────────────────────────────────────
    top_idx = np.argsort(scores)[::-1][:top_n]

    return [
        {
            "movie_id":     int(movie_ids[i]),
            "sasrec_score": round(float(scores[i]), 6),
            "rank":         rank + 1,
        }
        for rank, i in enumerate(top_idx)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan  (replaces deprecated @app.on_event)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── required env ──────────────────────────────────────────────────────────
    sasrec_run_id = os.environ.get("SASREC_RUN_ID")
    if not sasrec_run_id:
        raise RuntimeError("SASREC_RUN_ID env var not set.")

    candidates_db_path = os.environ.get("CANDIDATES_DB", "data/candidates.db")
    force_cpu          = os.environ.get("FORCE_CPU", "0") == "1"

    # ── device ────────────────────────────────────────────────────────────────
    device           = get_device(force_cpu=force_cpu)
    state["device"]  = device

    # ── databases ─────────────────────────────────────────────────────────────
    state["db"]  = load_movielens_db()
    state["cdb"] = load_candidates_db(db_path=candidates_db_path)

    # ── SASRec artifacts ──────────────────────────────────────────────────────
    artifacts        = load_sasrec_artifacts(sasrec_run_id, device)
    state["model"]   = artifacts["model"]
    state["encoder"] = artifacts["encoder"]
    state["args"]    = artifacts["args"]

    # ── caches ────────────────────────────────────────────────────────────────
    cache_enabled = os.environ.get("CACHE_ENABLED", "1") == "1"

    state["caches"] = build_caches(
        candidate_maxsize = int(os.environ.get("CANDIDATE_CACHE_SIZE", 5000)),
        candidate_ttl     = int(os.environ.get("CANDIDATE_CACHE_TTL", 3600)),
        output_maxsize    = int(os.environ.get("OUTPUT_CACHE_SIZE",   10000)),
        output_ttl        = int(os.environ.get("OUTPUT_CACHE_TTL",    600)),
        enabled           = cache_enabled,
    )

    logger.info("✓ All artifacts loaded. Serving ready.")
    yield

    # ── shutdown ──────────────────────────────────────────────────────────────
    state["cdb"].close()
    logger.info("CandidatesDB connection closed.")


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RecSys Serving API",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class RecItem(BaseModel):
    movie_id:     int
    sasrec_score: float
    rank:         int


class RecsResponse(BaseModel):
    user_id:     int
    as_of_date:  str
    cached:      bool
    latency_ms:  float
    results:     list[RecItem]


class NextItemResponse(BaseModel):
    user_id:     int
    as_of_date:  str
    cached:      bool
    latency_ms:  float
    movie_id:    int
    sasrec_score: float


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline  (shared by both rec endpoints)
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    user_id:    int,
    as_of_date: str,
    top_n:      int,
) -> tuple[list[dict], bool]:

    caches  = state["caches"]
    enabled = caches["enabled"]
    model   = state["model"]
    encoder = state["encoder"]
    args    = state["args"]
    device  = state["device"]
    db      = state["db"]
    cdb     = state["cdb"]

    # ── L1 output cache ───────────────────────────────────────────────
    output_key = (user_id, as_of_date, top_n)
    if enabled and output_key in caches["output_cache"]:
        caches["stats"]["output"]["hit"] += 1
        return caches["output_cache"][output_key], True

    if enabled:
        caches["stats"]["output"]["miss"] += 1

    # ── L2 candidate fetch ────────────────────────────────────────────
    candidates = get_candidates_cached(
        user_id=user_id,
        cdb=cdb,
        caches=caches,
        top_n=top_n,
    )

    # ── build sequence ────────────────────────────────────────────────
    seq = build_sequence(
        db=db,
        encoder=encoder,
        user_id=user_id,
        as_of_date=as_of_date,
        max_seq_len=args["max_len"],
    )

    # ── SASRec re-rank ────────────────────────────────────────────────
    results = rerank_candidates(
        candidates=candidates,
        seq=seq,
        encoder=encoder,
        model=model,
        device=device,
        top_n=top_n,
    )

    # ── store in L1 only if cache enabled ─────────────────────────────
    if enabled:
        caches["output_cache"][output_key] = results

    return results, False

# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(state.get("device", "not loaded")),
    }


@app.get("/recommendations/{user_id}", response_model=RecsResponse)
def get_recommendations(
    user_id:    int,
    as_of_date: Optional[str] = Query(
        default=None,
        description="Simulation date (YYYY-MM-DD). Defaults to today."
    ),
    top_n:      int = Query(default=10, ge=1, le=500),
):
    """
    Return top-N recommendations for a user.

    - Known user  →  personal MF candidates re-ranked by SASRec
    - New user    →  global popular candidates re-ranked by SASRec
                     (empty seq → scores are 0, popular order preserved)
    """
    as_of = as_of_date or str(date.today())

    t0 = time.perf_counter()
    try:
        results, from_cache = run_pipeline(user_id, as_of, top_n)
    except Exception as e:
        logger.exception(f"Pipeline error for user_id={user_id}")
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = (time.perf_counter() - t0) * 1000

    return RecsResponse(
        user_id    = user_id,
        as_of_date = as_of,
        cached     = from_cache,
        latency_ms = round(latency_ms, 3),
        results    = results,
    )


@app.get("/next-item/{user_id}", response_model=NextItemResponse)
def get_next_item(
    user_id:    int,
    as_of_date: Optional[str] = Query(
        default=None,
        description="Simulation date (YYYY-MM-DD). Defaults to today."
    ),
    top_n:      int = Query(
        default=500,
        ge=1, le=500,
        description="Candidate pool size before taking top-1."
    ),
):
    """
    Return the single most likely next item for a user.
    Runs the full pipeline with top_n candidates, returns rank-1.
    """
    as_of = as_of_date or str(date.today())

    t0 = time.perf_counter()
    try:
        results, from_cache = run_pipeline(user_id, as_of, top_n)
    except Exception as e:
        logger.exception(f"Pipeline error for user_id={user_id}")
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = (time.perf_counter() - t0) * 1000

    if not results:
        raise HTTPException(status_code=404, detail="No candidates found.")

    top = results[0]
    return NextItemResponse(
        user_id      = user_id,
        as_of_date   = as_of,
        cached       = from_cache,
        latency_ms   = round(latency_ms, 3),
        movie_id     = top["movie_id"],
        sasrec_score = top["sasrec_score"],
    )


@app.get("/cache/stats")
def cache_stats():
    """Hit/miss counters for L1 (output) and L2 (candidate) caches."""
    caches = state["caches"]
    stats  = caches["stats"]

    def hit_rate(s):
        total = s["hit"] + s["miss"]
        return round(s["hit"] / total * 100, 1) if total else 0.0

    return {
        "L2_candidate_cache": {
            **stats["candidate"],
            "hit_rate_pct":    hit_rate(stats["candidate"]),
            "current_size":    len(caches["candidate_cache"]),
            "maxsize":         caches["candidate_cache"].maxsize,
        },
        "L1_output_cache": {
            **stats["output"],
            "hit_rate_pct":    hit_rate(stats["output"]),
            "current_size":    len(caches["output_cache"]),
            "maxsize":         caches["output_cache"].maxsize,
        },
    }


@app.get("/cache/clear")
def cache_clear():
    """
    Clear both caches and reset hit/miss counters.
    Call this between benchmark runs to ensure a cold start.
    """
    caches = state["caches"]
    caches["candidate_cache"].clear()
    caches["output_cache"].clear()
    caches["stats"]["candidate"] = {"hit": 0, "miss": 0}
    caches["stats"]["output"]    = {"hit": 0, "miss": 0}
    logger.info("Both caches cleared.")
    return {"status": "cleared"}