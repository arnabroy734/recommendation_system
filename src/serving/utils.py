"""
src/serving/utils.py
---------------------
Helpers for loading all inference artifacts and setting up the two-level
cache used by app.py.

Artifacts loaded
----------------
  - MovieLensDB          (ratings + metadata, for user history)
  - CandidatesDB         (pre-computed MF candidates, for candidate pool)
  - SASRec model         (PyTorch, for re-ranking)
  - SASRec encoder       (item_id mapping)
  - SASRec args          (max_seq_len, hidden_units, etc.)

Cache
-----
  L2  candidate_cache   key = user_id  |  "__global_popular__" for cold-start
  L1  output_cache      key = (user_id, as_of_date, top_n)
"""

import logging
import pickle
from pathlib import Path

import mlflow
import torch
from cachetools import TTLCache

from src.data.db_simulator import MovieLensDB
from src.data.candidates_db import CandidatesDB
from src.training.matrix_factorisation import Encoder

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

COLD_START_KEY = "__global_popular__"


# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device(force_cpu: bool = False) -> torch.device:
    """
    Returns CUDA device if available and not force_cpu, else CPU.
    Logged at startup so benchmarks know which device was used.
    """
    if force_cpu:
        device = torch.device("cpu")
        logger.info("Device: CPU (forced)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Device: CUDA — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Device: CPU (CUDA not available)")
    return device


# ─────────────────────────────────────────────────────────────────────────────
# MLflow artifact path resolver
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_artifact_dir(run_id: str) -> Path:
    """
    MLflow artifact_uri already ends in /artifacts.
    If artifacts were logged with artifact_path='artifacts',
    files live one level deeper (artifacts/artifacts/).
    Probes both layouts and returns whichever contains args.pkl.
    """
    client       = mlflow.tracking.MlflowClient()
    artifact_uri = client.get_run(run_id).info.artifact_uri
    base_dir     = Path(artifact_uri.replace("file://", ""))

    nested = base_dir / "artifacts"
    if (nested / "args.pkl").exists():
        logger.info(f"Artifact layout: nested  →  {nested}")
        return nested

    logger.info(f"Artifact layout: flat  →  {base_dir}")
    return base_dir


# ─────────────────────────────────────────────────────────────────────────────
# MovieLensDB
# ─────────────────────────────────────────────────────────────────────────────

def load_movielens_db() -> MovieLensDB:
    """Load and return a fully initialised MovieLensDB instance."""
    db = MovieLensDB()
    db.load_data()
    logger.info("MovieLensDB loaded")
    return db


# ─────────────────────────────────────────────────────────────────────────────
# CandidatesDB
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates_db(db_path: str = "data/candidates.db") -> CandidatesDB:
    """
    Open and return a connected CandidatesDB instance.
    Caller is responsible for closing (or use as context manager).
    """
    cdb = CandidatesDB(db_path=db_path)
    cdb.connect()
    logger.info(f"CandidatesDB connected  →  {db_path}")
    return cdb


# ─────────────────────────────────────────────────────────────────────────────
# SASRec artifacts
# ─────────────────────────────────────────────────────────────────────────────

def load_sasrec_artifacts(run_id: str, device: torch.device) -> dict:
    """
    Load SASRec model, encoder and args from an MLflow run.

    Returns
    -------
    {
        "model"   : SASRec (torch.nn.Module, eval mode, on device),
        "encoder" : encoder object  (item_id ↔ idx mappings),
        "args"    : argparse.Namespace  (max_seq_len, hidden_units, …),
    }
    """
    artifact_dir = _resolve_artifact_dir(run_id)

    # ── args ──────────────────────────────────────────────────────────────────
    args_path = artifact_dir / "args.pkl"
    with open(args_path, "rb") as f:
        args = pickle.load(f)
    logger.info(f"SASRec args loaded  →  max_seq_len={args['max_len']}")

    # ── encoder ───────────────────────────────────────────────────────────────
    enc_path = artifact_dir / "encoder.pkl"
    with open(enc_path, "rb") as f:
        encoder = pickle.load(f)
    logger.info(f"SASRec encoder loaded  →  {encoder.n_items} items")

    # ── model ─────────────────────────────────────────────────────────────────
    from src.training.sasrec_architecture import SASRec  # local import to avoid circular deps

    model = SASRec(
        n_items=encoder.n_items,
        dim=args['dim'],
        max_len=args['max_len'],
        n_heads=args['n_heads'],
        n_layers=args['n_layers']
    ).to(device)

    ckpt_path = artifact_dir / "model.pt"
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"SASRec model loaded  →  {ckpt_path.name}  on {device}")

    return {
        "model":   model,
        "encoder": encoder,
        "args":    args,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sequence builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sequence(
    db: MovieLensDB,
    encoder,
    user_id: int,
    as_of_date: str,
    max_seq_len: int,
) -> list[int]:
    """
    Fetch user's watch history up to as_of_date, map to item indices,
    truncate to max_seq_len (most recent), NO padding.

    Returns list of item indices (internal encoder space), empty if cold-start.
    """
    history = db.get_user_history(
        user_id=user_id,
        as_of_date=as_of_date,
        limit=max_seq_len,           # fetch exactly what we need, ordered DESC
    )

    if not history:
        logger.debug(f"user_id={user_id} — empty history at {as_of_date}")
        return []

    # history is DESC (most recent first) → reverse to chronological
    history_asc = list(reversed(history))

    seq = [
        encoder.item_id[r["movie_id"]]
        for r in history_asc
        if r["movie_id"] in encoder.item_id
    ]

    # truncate to max_seq_len (most recent)
    seq = seq[-max_seq_len:]

    logger.debug(f"user_id={user_id}  seq_len={len(seq)}  as_of={as_of_date}")
    return seq


# ─────────────────────────────────────────────────────────────────────────────
# Two-level cache factory
# ─────────────────────────────────────────────────────────────────────────────

def build_caches(
    candidate_maxsize: int = 5000,
    candidate_ttl:     int = 3600,
    output_maxsize:    int = 10000,
    output_ttl:        int = 600,
    enabled:           bool = True,       # ← ADD THIS
) -> dict:
    caches = {
        "candidate_cache": TTLCache(maxsize=candidate_maxsize, ttl=candidate_ttl),
        "output_cache":    TTLCache(maxsize=output_maxsize,    ttl=output_ttl),
        "enabled": enabled,               # ← ADD THIS
        "stats": {
            "candidate": {"hit": 0, "miss": 0},
            "output":    {"hit": 0, "miss": 0},
        },
    }
    logger.info(
        f"Caches initialised  →  enabled={enabled}  |  "
        f"L2 candidates (max={candidate_maxsize}, ttl={candidate_ttl}s)  |  "
        f"L1 output (max={output_maxsize}, ttl={output_ttl}s)"
    )
    return caches

# ─────────────────────────────────────────────────────────────────────────────
# Candidate fetch with two-level cache + cold-start sentinel
# ─────────────────────────────────────────────────────────────────────────────
def get_candidates_cached(
    user_id:  int,
    cdb:      CandidatesDB,
    caches:   dict,
    top_n:    int = 500,
) -> list[dict]:
    cache   = caches["candidate_cache"]
    stats   = caches["stats"]["candidate"]
    enabled = caches["enabled"]             # ← ADD

    # ── cache disabled — go straight to DB ───────────────────────────
    if not enabled:                         # ← ADD
        if cdb.user_exists(user_id):
            return cdb.get_candidates(user_id, top_n=top_n)
        return cdb.get_global_candidates(top_n=top_n)

    # ── known user path ───────────────────────────────────────────────
    if user_id in cache:
        stats["hit"] += 1
        return cache[user_id]

    if cdb.user_exists(user_id):
        stats["miss"] += 1
        candidates     = cdb.get_candidates(user_id, top_n=top_n)
        cache[user_id] = candidates
        return candidates

    # ── cold-start path ───────────────────────────────────────────────
    if COLD_START_KEY in cache:
        stats["hit"] += 1
        return cache[COLD_START_KEY]

    stats["miss"] += 1
    candidates            = cdb.get_global_candidates(top_n=top_n)
    cache[COLD_START_KEY] = candidates
    return candidates