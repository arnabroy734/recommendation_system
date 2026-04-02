"""
deploy/generate_candidates.py
-----------------------------
Generates top-N MF candidates per user and saves to data/candidates.db.

Reads the promoted MF run from prod_config.json, loads artifacts from
MLflow, scores all unseen items per user using:
    score(u, i) = P[u] @ Q[i] + b_i[i]

Cold-start users (not in encoder) receive global top-N popular items
from the training window.

Usage:
    python deploy/generate_candidates.py
    python deploy/generate_candidates.py --top_n 300
"""

import argparse
import json
import logging
import pickle
import time
from pathlib import Path

import mlflow
import numpy as np

from src.data.db_simulator import MovieLensDB
from src.data.candidates_db import CandidatesDB
from src.training.matrix_factorisation import Encoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

PROD_CONFIG_PATH = Path("prod_config.json")


# ---------------------------------------------------------------------------
# Config + artifact loading
# ---------------------------------------------------------------------------

def load_prod_config() -> dict:
    if not PROD_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "prod_config.json not found. "
            "Run `python deploy/promote.py --model mf --run_id <id>` first."
        )
    with open(PROD_CONFIG_PATH) as f:
        return json.load(f)

def _resolve_artifact_dir(run_id: str) -> Path:
    """
    MLflow artifact_uri already ends in /artifacts.
    If artifacts were logged with artifact_path='artifacts',
    the real files live one level deeper (artifacts/artifacts/).
    This probes both and returns whichever actually contains encoder.pkl.
    """
    client       = mlflow.tracking.MlflowClient()
    artifact_uri = client.get_run(run_id).info.artifact_uri
    base_dir     = Path(artifact_uri.replace("file://", ""))

    # Probe: some runs log into a nested 'artifacts/' sub-folder
    nested = base_dir / "artifacts"
    if (nested / "encoder.pkl").exists():
        return nested          # MF case  →  .../artifacts/artifacts/

    return base_dir            # SASRec / correct case  →  .../artifacts/


def load_mf_artifacts(run_id: str) -> dict:
    artifact_dir = _resolve_artifact_dir(run_id)
    logger.info(f"Loading MF artifacts from {artifact_dir}")

    def load_npy(name): return np.load(artifact_dir / name)
    def load_pkl(name):
        with open(artifact_dir / name, "rb") as f:
            return pickle.load(f)

    return {
        "encoder":         load_pkl("encoder.pkl"),
        "user_embeddings": load_npy("user_embeddings.npy"),
        "item_embeddings": load_npy("item_embeddings.npy"),
        "item_bias":       load_npy("item_bias.npy"),
    }


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def compute_global_popular(
    db: MovieLensDB,
    enc,
    train_start: str,
    train_end: str,
    top_n: int,
) -> list:
    """
    Top-N most rated items within the training window.
    Only includes items present in the encoder.
    Returns list of movie_ids sorted by popularity desc.
    """
    df      = db.get_ratings_by_daterange(train_start, train_end)
    counts  = df[df["movieId"].isin(enc.item_id)]["movieId"].value_counts()
    popular = counts.head(top_n).index.tolist()
    logger.info(f"Global popular fallback: {len(popular)} items")
    return popular


def generate_candidates_for_user(
    u_idx: int,
    seen_item_indices: set,
    all_item_indices: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    b_i: np.ndarray,
    top_n: int,
) -> tuple:
    """
    Score unseen items for a single user:
        score(u, i) = P[u] @ Q[i] + b_i[i]

    b_u excluded — constant offset per user, doesn't affect item ranking.
    Returns (top_item_indices, top_scores) sorted desc.
    """
    unseen_mask    = ~np.isin(all_item_indices, list(seen_item_indices))
    unseen_indices = all_item_indices[unseen_mask]

    if len(unseen_indices) == 0:
        return np.array([]), np.array([])

    scores  = Q[unseen_indices] @ P[u_idx] + b_i[unseen_indices]
    top_idx = np.argsort(scores)[::-1][:top_n]

    return unseen_indices[top_idx], scores[top_idx]


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate MF candidates and save to data/candidates.db"
    )
    p.add_argument(
        "--top_n", type=int, default=500,
        help="Top-N candidates per user (default: 500)"
    )
    p.add_argument(
        "--batch_size", type=int, default=1000,
        help="Users per SQLite insert batch (default: 1000)"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── prod config ──────────────────────────────────────────────────────────
    config    = load_prod_config()
    mf_config = config.get("mf")
    if not mf_config:
        raise KeyError("No MF model in prod_config.json. Promote one first.")

    run_id      = mf_config["run_id"]
    train_end   = mf_config["train_end"]
    train_start = mf_config["params"].get("start", "2000-01-01")

    logger.info(f"MF run_id    : {run_id}")
    logger.info(f"run_name     : {mf_config['run_name']}")
    logger.info(f"train window : {train_start} → {train_end}")
    logger.info(f"top_n        : {args.top_n}")

    # ── load artifacts ───────────────────────────────────────────────────────
    artifacts = load_mf_artifacts(run_id)
    enc       = artifacts["encoder"]
    P         = artifacts["user_embeddings"]   # (n_users, dim)
    Q         = artifacts["item_embeddings"]   # (n_items, dim)
    b_i       = artifacts["item_bias"]         # (n_items,)

    logger.info(
        f"Embeddings: {enc.n_users:,} users | "
        f"{enc.n_items:,} items | dim={P.shape[1]}"
    )

    # ── seen_per_user from training window ───────────────────────────────────
    db = MovieLensDB()
    db.load_data()

    logger.info("Building seen_per_user ...")
    train_df      = db.get_ratings_by_daterange(train_start, train_end)
    seen_per_user = {}
    for uid, grp in train_df.groupby("userId"):
        if uid in enc.user_id:
            u = enc.user_id[uid]
            seen_per_user[u] = set(
                enc.item_id[m] for m in grp["movieId"] if m in enc.item_id
            )

    all_item_indices = np.arange(enc.n_items)

    # ── global popular fallback ──────────────────────────────────────────────
    popular_movie_ids = compute_global_popular(
        db, enc, train_start, train_end, args.top_n
    )
    global_rows = [(mid, rank + 1) for rank, mid in enumerate(popular_movie_ids)]

    # ── generate + write to CandidatesDB ────────────────────────────────────
    logger.info(f"Generating candidates for {enc.n_users:,} users ...")
    t0     = time.time()
    n_cold = 0

    with CandidatesDB() as cdb:
        cdb.init_tables()
        cdb.insert_global_candidates(global_rows)

        batch = []
        for u_idx, uid_orig in enc.id_user.items():
            seen = seen_per_user.get(u_idx, set())
            top_item_indices, top_scores = generate_candidates_for_user(
                u_idx, seen, all_item_indices, P, Q, b_i, args.top_n
            )

            if len(top_item_indices) == 0:
                # cold start — fall back to global popular
                n_cold += 1
                for rank, mid in enumerate(popular_movie_ids, start=1):
                    batch.append((int(uid_orig), int(mid), None, rank))
            else:
                for rank, (i_idx, score) in enumerate(
                    zip(top_item_indices, top_scores), start=1
                ):
                    movie_id = enc.id_item[i_idx]
                    batch.append((
                        int(uid_orig), int(movie_id),
                        round(float(score), 6), rank
                    ))

            if len(batch) >= args.batch_size * args.top_n:
                cdb.insert_candidates(batch)
                batch = []

        if batch:
            cdb.insert_candidates(batch)

    # ── summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    db_path = CandidatesDB().db_path

    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(f"Cold-start users (global popular fallback): {n_cold:,}")
    logger.info(f"Candidates saved → {db_path.resolve()}")
    logger.info(f"DB size: {db_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()