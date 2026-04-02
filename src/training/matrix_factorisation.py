"""
matrix_factorisation.py
-----------------------
Matrix Factorisation training with SGD for MSE and BPR losses.
Integrated with MLflow tracking via ExperimentTracker.

Usage examples:
    # BPR
    python matrix_factorisation.py --start 2018-01-01 --end 2018-06-30 \
        --eval_start 2018-07-01 --eval_end 2018-12-31 --loss bpr

    # MSE
    python matrix_factorisation.py --start 2018-01-01 --end 2018-06-30 \
        --eval_start 2018-07-01 --eval_end 2018-12-31 --loss mse \
        --dim 128 --lr 0.005 --reg1 0.01 --reg2 0.01 --epochs 20 --eval_k 5 10 20
"""

import argparse
import logging
import os
import pickle
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit, prange

# sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.db_simulator import MovieLensDB
from src.tracking.tracker import ExperimentTracker
from src.tracking.config import EXPERIMENT_NAMES
from src.artifacts.local_store import LocalArtifactStore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder:
    """Two-way mapping between original IDs and contiguous matrix indices."""

    def __init__(self):
        self.user_id  = {}
        self.id_user  = {}
        self.item_id  = {}
        self.id_item  = {}

    def encode(self, df: pd.DataFrame):
        max_u = max(self.id_user.keys(), default=-1)
        max_i = max(self.id_item.keys(), default=-1)

        for uid, mid in df[["userId", "movieId"]].itertuples(index=False):
            if uid not in self.user_id:
                max_u += 1
                self.user_id[uid]   = max_u
                self.id_user[max_u] = uid
            if mid not in self.item_id:
                max_i += 1
                self.item_id[mid]   = max_i
                self.id_item[max_i] = mid

    @property
    def n_users(self):
        return len(self.user_id)

    @property
    def n_items(self):
        return len(self.item_id)


# ---------------------------------------------------------------------------
# MF Model
# ---------------------------------------------------------------------------

class MFModel:
    """Matrix Factorisation model: stores embeddings and biases."""

    def __init__(self, n_users: int, n_items: int, dim: int, mu: float):
        self.dim  = dim
        self.mu   = mu
        scale     = 0.01
        self.P    = np.random.normal(0, scale, (n_users, dim)).astype(np.float32)
        self.Q    = np.random.normal(0, scale, (n_items, dim)).astype(np.float32)
        self.b_u  = np.zeros(n_users, dtype=np.float32)
        self.b_i  = np.zeros(n_items, dtype=np.float32)

    def predict(self, u: int, i: int) -> float:
        return self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]

    def predict_batch(self, u: int, item_indices: np.ndarray) -> np.ndarray:
        return self.mu + self.b_u[u] + self.b_i[item_indices] + self.Q[item_indices] @ self.P[u]


# ---------------------------------------------------------------------------
# Data loading & filtering
# ---------------------------------------------------------------------------

def load_training_data(
    db: MovieLensDB,
    start: str,
    end: str,
    min_rating: float = 3.0,
    min_user_ratings: int = 10,
    min_item_ratings: int = 10,
) -> pd.DataFrame:
    logger.info(f"Loading data from {start} to {end} ...")
    df = db.get_ratings_by_daterange(start, end).copy()

    logger.info(f"Raw interactions: {len(df):,}")
    df = df[df["rating"] >= min_rating].copy()
    logger.info(f"After rating >= {min_rating}: {len(df):,}")

    prev_len  = -1
    iteration = 0
    while prev_len != len(df):
        prev_len    = len(df)
        iteration  += 1
        user_counts = df["userId"].value_counts()
        item_counts = df["movieId"].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        valid_items = item_counts[item_counts >= min_item_ratings].index
        df = df[df["userId"].isin(valid_users) & df["movieId"].isin(valid_items)]

    logger.info(
        f"After co-filtering (iters={iteration}): {len(df):,} interactions | "
        f"{df['userId'].nunique():,} users | {df['movieId'].nunique():,} items"
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Negative sampler for BPR
# ---------------------------------------------------------------------------

class NegativeSampler:
    def __init__(self, df: pd.DataFrame, enc: Encoder, n_neg: int = 4):
        self.n_neg     = n_neg
        self.enc       = enc
        self.all_items = np.array(list(enc.id_item.keys()))

        counts  = df["movieId"].value_counts()
        weights = np.array([
            counts.get(enc.id_item[i], 0) for i in self.all_items
        ], dtype=np.float64)
        weights        = weights / weights.sum()
        self.weights   = weights

        logger.info("Building per-user positive sets for negative sampling ...")
        self.user_pos = {}
        for uid_orig, grp in df.groupby("userId"):
            u = enc.user_id[uid_orig]
            self.user_pos[u] = set(enc.item_id[m] for m in grp["movieId"].values)

    def sample(self, u: int) -> np.ndarray:
        pos        = self.user_pos.get(u, set())
        negs       = []
        candidates = np.random.choice(self.all_items, size=self.n_neg * 10, p=self.weights)
        for c in candidates:
            if c not in pos:
                negs.append(c)
            if len(negs) == self.n_neg:
                break
        while len(negs) < self.n_neg:
            c = np.random.choice(self.all_items)
            if c not in pos:
                negs.append(c)
        return np.array(negs)


# ---------------------------------------------------------------------------
# SGD training (numba)
# ---------------------------------------------------------------------------

@njit(parallel=True)
def train_one_epoch_mse(P, Q, b_u, b_i, mu, interactions, lr, reg1, reg2):
    N          = interactions.shape[0]
    total_loss = 0.0
    for idx in prange(N):
        u    = int(interactions[idx, 0])
        i    = int(interactions[idx, 1])
        r    = interactions[idx, 2]
        pred = mu + b_u[u] + b_i[i]
        for d in range(P.shape[1]):
            pred += P[u, d] * Q[i, d]
        e           = r - pred
        total_loss += e * e
        for d in range(P.shape[1]):
            pu_d      = P[u, d]
            P[u, d]  += lr * (e * Q[i, d] - reg1 * pu_d)
            Q[i, d]  += lr * (e * pu_d    - reg1 * Q[i, d])
        b_u[u] += lr * (e - reg2 * b_u[u])
        b_i[i] += lr * (e - reg2 * b_i[i])
    return total_loss / N


@njit(parallel=True)
def train_one_epoch_bpr(P, Q, b_u, b_i, mu, interactions, neg_samples, lr, reg1, reg2):
    N          = interactions.shape[0]
    n_neg      = neg_samples.shape[1]
    total_loss = 0.0
    for idx in prange(N):
        u = int(interactions[idx, 0])
        i = int(interactions[idx, 1])
        for n in range(n_neg):
            j   = int(neg_samples[idx, n])
            rui = mu + b_u[u] + b_i[i]
            ruj = mu + b_u[u] + b_i[j]
            for d in range(P.shape[1]):
                rui += P[u, d] * Q[i, d]
                ruj += P[u, d] * Q[j, d]
            diff = rui - ruj
            if diff > 30.0:
                sig = 1.0
            elif diff < -30.0:
                sig = 0.0
            else:
                sig = 1.0 / (1.0 + np.exp(-diff))
            theta       = 1.0 - sig
            total_loss += -np.log(sig + 1e-10)
            for d in range(P.shape[1]):
                pu_d      = P[u, d]
                P[u, d]  += lr * (theta * (Q[i, d] - Q[j, d]) - reg1 * pu_d)
                Q[i, d]  += lr * (theta * pu_d                 - reg1 * Q[i, d])
                Q[j, d]  -= lr * (theta * pu_d                 + reg1 * Q[j, d])
            b_i[i] += lr * (theta - reg2 * b_i[i])
            b_i[j] -= lr * (theta + reg2 * b_i[j])
    return total_loss / max(N * n_neg, 1)


# ---------------------------------------------------------------------------
# Evaluation — single window, Recall@K and NDCG@K
# ---------------------------------------------------------------------------

def evaluate(
    model: MFModel,
    enc: Encoder,
    db: MovieLensDB,
    eval_start: str,
    eval_end: str,
    train_df: pd.DataFrame,
    min_rating: float,
    min_eval_ratings: list,
    eval_k: list,
    top_n_recs: int,
) -> dict:
    """
    Evaluate on a single date window [eval_start, eval_end].
    Returns flat dict of metrics:
        {'recall@5_min5': 0.32, 'ndcg@10_min5': 0.28, ...}
    """
    logger.info(f"Evaluating on window: {eval_start} → {eval_end}")

    eval_raw = db.get_ratings_by_daterange(eval_start, eval_end)
    eval_raw = eval_raw[eval_raw["rating"] >= min_rating]
    eval_raw = eval_raw[eval_raw["userId"].isin(enc.user_id)]
    eval_raw = eval_raw[eval_raw["movieId"].isin(enc.item_id)]

    if eval_raw.empty:
        logger.warning("No eval data found in the given window.")
        return {}

    # build seen set from training data
    seen_per_user = {}
    for uid_orig, grp in train_df.groupby("userId"):
        u = enc.user_id[uid_orig]
        seen_per_user[u] = set(enc.item_id[m] for m in grp["movieId"] if m in enc.item_id)

    all_item_indices = np.array(list(enc.id_item.keys()))
    user_eval_counts = eval_raw.groupby("userId")["movieId"].count()

    # ground truth
    gt_per_user = {}
    for uid_orig, grp in eval_raw.groupby("userId"):
        gt_per_user[uid_orig] = set(grp["movieId"].values)

    # generate top-N recs
    recs = {}
    for uid_orig in user_eval_counts.index:
        u      = enc.user_id[uid_orig]
        seen   = seen_per_user.get(u, set())
        unseen = all_item_indices[~np.isin(all_item_indices, list(seen))]
        if len(unseen) == 0:
            continue
        scores         = model.predict_batch(u, unseen)
        top_idx        = np.argsort(scores)[::-1][:top_n_recs]
        recs[uid_orig] = [enc.id_item[unseen[i]] for i in top_idx]

    # compute metrics per (min_eval_ratings, K)
    metrics = {}
    for min_er in min_eval_ratings:
        qualifying = user_eval_counts[user_eval_counts >= min_er].index
        if len(qualifying) == 0:
            logger.info(f"  min_eval_ratings={min_er}: no qualifying users")
            continue

        for k in eval_k:
            recalls, ndcgs = [], []
            for uid_orig in qualifying:
                gt        = gt_per_user.get(uid_orig, set())
                user_recs = recs.get(uid_orig, [])
                if not gt or not user_recs:
                    continue
                top_k = set(user_recs[:k])
                hits  = len(top_k & gt)
                recalls.append(hits / min(len(gt), k))

                dcg  = 0.0
                idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), k)))
                for rank_0, item in enumerate(user_recs[:k]):
                    if item in gt:
                        dcg += 1.0 / np.log2(rank_0 + 2)
                ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

            mean_recall = float(np.mean(recalls)) if recalls else 0.0
            mean_ndcg   = float(np.mean(ndcgs))   if ndcgs   else 0.0

            metrics[f"recall_at_{k}_min{min_er}"] = mean_recall
            metrics[f"ndcg_at_{k}_min{min_er}"]   = mean_ndcg
            metrics[f"n_users_min{min_er}"]        = len(recalls)

            logger.info(
                f"  min_ratings={min_er:2d} | K={k:2d} | "
                f"Recall@{k}={mean_recall:.4f} | NDCG@{k}={mean_ndcg:.4f} | "
                f"users={len(recalls)}"
            )

    return metrics


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------

def save_artifacts(tmpdir: Path, model: MFModel, enc: Encoder, args, losses: dict):
    np.save(tmpdir / "user_embeddings.npy", model.P)
    np.save(tmpdir / "item_embeddings.npy", model.Q)
    np.save(tmpdir / "user_bias.npy",       model.b_u)
    np.save(tmpdir / "item_bias.npy",       model.b_i)
    np.save(tmpdir / "mu.npy",              np.array([model.mu]))

    with open(tmpdir / "encoder.pkl", "wb") as f:
        pickle.dump(enc, f)
    with open(tmpdir / "args.pkl", "wb") as f:
        pickle.dump(vars(args), f)

    _plot_losses(losses, tmpdir, args)
    logger.info(f"Artifacts prepared in {tmpdir}")


def _plot_losses(losses: dict, out_dir: Path, args):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, values in losses.items():
        ax.plot(range(1, len(values) + 1), values, marker="o", markersize=3, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"MF | loss={args.loss} | dim={args.dim} | "
        f"lr={args.lr} | reg1={args.reg1} | reg2={args.reg2}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train MF with SGD (MSE or BPR)")

    p.add_argument("--start",             type=str,   required=True)
    p.add_argument("--end",               type=str,   required=True)
    p.add_argument("--loss",              type=str,   default="bpr",        choices=["mse", "bpr"])
    p.add_argument("--dim",               type=int,   default=64)
    p.add_argument("--epochs",            type=int,   default=20)
    p.add_argument("--lr",                type=float, default=0.005)
    p.add_argument("--reg1",              type=float, default=0.01)
    p.add_argument("--reg2",              type=float, default=0.001)
    p.add_argument("--n_neg",             type=int,   default=4)
    p.add_argument("--min_rating",        type=float, default=3.0)
    p.add_argument("--min_u_rat",         type=int,   default=10)
    p.add_argument("--min_i_rat",         type=int,   default=10)
    p.add_argument("--eval_start",        type=str,   required=True)
    p.add_argument("--eval_end",          type=str,   required=True)
    p.add_argument("--eval_k",            type=int,   nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--top_n_recs",        type=int,   default=50)
    p.add_argument("--min_eval_ratings",  type=int,   nargs="+", default=[5])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args     = parse_args()
    run_name = f"mf_{args.loss}_dim{args.dim}"

    store  = LocalArtifactStore()
    params = {
        "start":            args.start,
        "end":              args.end,
        "eval_start":       args.eval_start,
        "eval_end":         args.eval_end,
        "loss":             args.loss,
        "dim":              args.dim,
        "epochs":           args.epochs,
        "lr":               args.lr,
        "reg1":             args.reg1,
        "reg2":             args.reg2,
        "n_neg":            args.n_neg,
        "min_rating":       args.min_rating,
        "min_u_rat":        args.min_u_rat,
        "min_i_rat":        args.min_i_rat,
        "eval_k":           str(args.eval_k),
        "min_eval_ratings": str(args.min_eval_ratings),
        "top_n_recs":       args.top_n_recs,
    }

    with ExperimentTracker(
        experiment_name=EXPERIMENT_NAMES["mf"],
        run_name=run_name,
        store=store,
        tags={"model_type": "matrix_factorisation"}
    ) as tracker:

        tracker.log_params(params)
        logger.info(f"MLflow run started | run_id={tracker.run_id} | run_name={run_name}")

        # ---- load DB ----
        db = MovieLensDB()
        db.load_data()

        # ---- training data ----
        train_df = load_training_data(
            db, args.start, args.end,
            min_rating=args.min_rating,
            min_user_ratings=args.min_u_rat,
            min_item_ratings=args.min_i_rat,
        )

        # ---- encoder ----
        enc = Encoder()
        enc.encode(train_df)
        logger.info(f"Encoded {enc.n_users:,} users and {enc.n_items:,} items")
        tracker.set_tag("n_users", str(enc.n_users))
        tracker.set_tag("n_items", str(enc.n_items))

        # ---- model ----
        mu    = train_df["rating"].mean()
        model = MFModel(enc.n_users, enc.n_items, args.dim, mu)
        logger.info(f"Model initialised | mu={mu:.4f}")

        # ---- build interaction arrays ----
        if args.loss == "mse":
            interactions          = train_df[["userId", "movieId", "rating"]].copy()
            interactions["u_idx"] = interactions["userId"].map(enc.user_id)
            interactions["i_idx"] = interactions["movieId"].map(enc.item_id)
            inter_arr             = interactions[["u_idx", "i_idx", "rating"]].values.astype(np.float32)
        else:
            interactions          = train_df[["userId", "movieId"]].copy()
            interactions["u_idx"] = interactions["userId"].map(enc.user_id)
            interactions["i_idx"] = interactions["movieId"].map(enc.item_id)
            inter_arr             = interactions[["u_idx", "i_idx"]].values.astype(np.int32)
            sampler               = NegativeSampler(train_df, enc, n_neg=args.n_neg)

        # ---- training loop ----
        train_losses = []
        t0           = time.time()

        for epoch in range(1, args.epochs + 1):
            t_ep = time.time()

            if args.loss == "bpr":
                neg_samples = np.array(
                    [sampler.sample(int(inter_arr[idx, 0])) for idx in range(len(inter_arr))],
                    dtype=np.int32
                )
                np.random.shuffle(inter_arr)
                loss = train_one_epoch_bpr(
                    model.P, model.Q, model.b_u, model.b_i, np.float32(model.mu),
                    inter_arr, neg_samples,
                    np.float32(args.lr), np.float32(args.reg1), np.float32(args.reg2)
                )
            else:
                np.random.shuffle(inter_arr)
                loss = train_one_epoch_mse(
                    model.P, model.Q, model.b_u, model.b_i, np.float32(model.mu),
                    inter_arr,
                    np.float32(args.lr), np.float32(args.reg1), np.float32(args.reg2)
                )

            train_losses.append(float(loss))
            tracker.log_metric("train_loss", float(loss), step=epoch)
            logger.info(f"Epoch {epoch:3d}/{args.epochs} | loss={loss:.6f} | time={time.time()-t_ep:.1f}s")

        tracker.set_tag("training_time_sec", str(round(time.time() - t0, 1)))

        # ---- evaluation ----
        eval_metrics = evaluate(
            model=model, enc=enc, db=db,
            eval_start=args.eval_start, eval_end=args.eval_end,
            train_df=train_df, min_rating=args.min_rating,
            min_eval_ratings=args.min_eval_ratings,
            eval_k=args.eval_k, top_n_recs=args.top_n_recs,
        )
        if eval_metrics:
            tracker.log_metrics(eval_metrics, step=args.epochs)

        # ---- artifacts ----
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            save_artifacts(tmpdir, model, enc, args, {"train_loss": train_losses})
            tracker.log_artifacts(str(tmpdir), artifact_path="artifacts")

        logger.info(f"Run complete | run_id={tracker.run_id}")


if __name__ == "__main__":
    main()