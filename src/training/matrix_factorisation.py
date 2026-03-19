"""
matrix_factorisation.py
--------
Matrix Factorisation training with SGD for MSE and BPR losses.

Usage examples:
    # BPR
    python train.py --start 2018-01-01 --end 2018-06-30 --loss bpr

    # MSE
    python train.py --start 2018-01-01 --end 2018-06-30 --loss mse \
        --dim 64 --lr 0.005 --reg1 0.01 --reg2 0.01 --epochs 20 --eval_k 5 10 20

Math reference:
    MSE: e_ui = r_ui - r_hat_ui
         p_u += lr * (e_ui * q_i  - reg1 * p_u)
         q_i += lr * (e_ui * p_u  - reg1 * q_i)
         b_u += lr * (e_ui        - reg2 * b_u)
         b_i += lr * (e_ui        - reg2 * b_i)

    BPR: theta_uij = 1 - sigmoid(r_hat_ui - r_hat_uj)
         p_u += lr * (theta * (q_i - q_j) - reg1 * p_u)
         q_i += lr * (theta * p_u          - reg1 * q_i)
         q_j -= lr * (theta * p_u          + reg1 * q_j)   [note: minus]
         b_i += lr * (theta                - reg2 * b_i)
         b_j -= lr * (theta                + reg2 * b_j)
         b_u cancels in BPR, no update
"""

import argparse
import logging
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from numba import njit, prange

# ---------------------------------------------------------------------------
# Add project root to path so db_simulator is importable
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.db_simulator import MovieLensDB

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
        self.user_id  = {}   # original userId  -> matrix index
        self.id_user  = {}   # matrix index     -> original userId
        self.item_id  = {}   # original movieId -> matrix index
        self.id_item  = {}   # matrix index     -> original movieId

    def encode(self, df: pd.DataFrame):
        """
        Build mappings from a dataframe with columns userId and movieId.
        Incremental — safe to call multiple times.
        """
        max_u = max(self.id_user.keys(), default=-1)
        max_i = max(self.id_item.keys(), default=-1)

        for uid, mid in df[["userId", "movieId"]].itertuples(index=False):
            if uid not in self.user_id:
                max_u += 1
                self.user_id[uid]  = max_u
                self.id_user[max_u] = uid
            if mid not in self.item_id:
                max_i += 1
                self.item_id[mid]  = max_i
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
        self.P    = np.random.normal(0, scale, (n_users, dim)).astype(np.float32)  # user embeddings
        self.Q    = np.random.normal(0, scale, (n_items, dim)).astype(np.float32)  # item embeddings
        self.b_u  = np.zeros(n_users, dtype=np.float32)                            # user biases
        self.b_i  = np.zeros(n_items, dtype=np.float32)                            # item biases

    def predict(self, u: int, i: int) -> float:
        return self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]

    def predict_batch(self, u: int, item_indices: np.ndarray) -> np.ndarray:
        """Score all items in item_indices for user u."""
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
    """
    Pull ratings from db within [start, end], apply filters:
      - rating >= min_rating
      - users with >= min_user_ratings interactions
      - items with >= min_item_ratings interactions
    Returns cleaned dataframe with columns: userId, movieId, rating, timestamp
    """
    logger.info(f"Loading data from {start} to {end} ...")
    df = db.get_ratings_by_daterange(start, end).copy()
    df = df.rename(columns={"movieId": "movieId"})

    logger.info(f"Raw interactions: {len(df):,}")

    # filter by minimum rating
    df = df[df["rating"] >= min_rating].copy()
    logger.info(f"After rating >= {min_rating}: {len(df):,}")

    # iterative co-filtering until stable
    prev_len = -1
    iteration = 0
    while prev_len != len(df):
        prev_len = len(df)
        iteration += 1
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
    """
    Samples negatives for BPR from movies the user has NOT rated,
    weighted by global popularity within the training window.
    """

    def __init__(self, df: pd.DataFrame, enc: Encoder, n_neg: int = 4):
        self.n_neg   = n_neg
        self.enc     = enc
        self.all_items = np.array(list(enc.id_item.keys()))  # matrix indices

        # popularity weights (within window)
        counts = df["movieId"].value_counts()
        weights = np.array([
            counts.get(enc.id_item[i], 0) for i in self.all_items
        ], dtype=np.float64)
        weights = weights / weights.sum()
        self.weights = weights

        # build per-user positive set for fast exclusion
        logger.info("Building per-user positive sets for negative sampling ...")
        self.user_pos = {}
        for uid_orig, grp in df.groupby("userId"):
            u = enc.user_id[uid_orig]
            self.user_pos[u] = set(enc.item_id[m] for m in grp["movieId"].values)

    def sample(self, u: int) -> np.ndarray:
        """Return n_neg negative item indices for user u."""
        pos = self.user_pos.get(u, set())
        negs = []
        # sample with replacement until we have n_neg distinct negatives
        candidates = np.random.choice(self.all_items, size=self.n_neg * 10, p=self.weights)
        for c in candidates:
            if c not in pos:
                negs.append(c)
            if len(negs) == self.n_neg:
                break
        # fallback: uniform if not enough candidates
        while len(negs) < self.n_neg:
            c = np.random.choice(self.all_items)
            if c not in pos:
                negs.append(c)
        return np.array(negs)


# ---------------------------------------------------------------------------
# SGD training
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@njit(parallel=True)
def train_one_epoch_mse(P, Q, b_u, b_i, mu, interactions, lr, reg1, reg2):
    """
    interactions: float32 array of shape (N, 3) — [u_idx, i_idx, rating]
    """
    N  = interactions.shape[0]
    total_loss = 0.0

    for idx in prange(N):
        u   = int(interactions[idx, 0])
        i   = int(interactions[idx, 1])
        r   = interactions[idx, 2]

        pred = mu + b_u[u] + b_i[i]
        for d in range(P.shape[1]):
            pred += P[u, d] * Q[i, d]

        e = r - pred
        total_loss += e * e

        for d in range(P.shape[1]):
            pu_d       = P[u, d]
            P[u, d]   += lr * (e * Q[i, d] - reg1 * pu_d)
            Q[i, d]   += lr * (e * pu_d    - reg1 * Q[i, d])
        b_u[u] += lr * (e - reg2 * b_u[u])
        b_i[i] += lr * (e - reg2 * b_i[i])

    return total_loss / N

@njit(parallel=True)
def train_one_epoch_bpr(P, Q, b_u, b_i, mu, interactions, neg_samples, lr, reg1, reg2):
    """
    interactions: int32 array of shape (N, 2)  — [u_idx, i_idx]
    neg_samples:  int32 array of shape (N, n_neg) — pre-sampled negatives
    """
    N          = interactions.shape[0]
    n_neg      = neg_samples.shape[1]
    total_loss = 0.0

    for idx in prange(N):
        u = int(interactions[idx, 0])
        i = int(interactions[idx, 1])

        for n in range(n_neg):
            j = int(neg_samples[idx, n])

            rui = mu + b_u[u] + b_i[i]
            ruj = mu + b_u[u] + b_i[j]
            for d in range(P.shape[1]):
                rui += P[u, d] * Q[i, d]
                ruj += P[u, d] * Q[j, d]

            diff  = rui - ruj
            # safe sigmoid
            if diff > 30.0:
                sig = 1.0
            elif diff < -30.0:
                sig = 0.0
            else:
                sig = 1.0 / (1.0 + np.exp(-diff))

            theta       = 1.0 - sig
            total_loss += -np.log(sig + 1e-10)

            for d in range(P.shape[1]):
                pu_d       = P[u, d]
                P[u, d]   += lr * (theta * (Q[i, d] - Q[j, d]) - reg1 * pu_d)
                Q[i, d]   += lr * (theta * pu_d                 - reg1 * Q[i, d])
                Q[j, d]   -= lr * (theta * pu_d                 + reg1 * Q[j, d])
            b_i[i] += lr * (theta - reg2 * b_i[i])
            b_i[j] -= lr * (theta + reg2 * b_i[j])

    return total_loss / max(N * n_neg, 1)


# ---------------------------------------------------------------------------
# Validation MSE (for monitoring during training)
# ---------------------------------------------------------------------------

def compute_val_mse(model: MFModel, enc: Encoder, val_df: pd.DataFrame) -> float:
    errors = []
    for uid, mid, r in val_df[["userId", "movieId", "rating"]].itertuples(index=False):
        if uid not in enc.user_id or mid not in enc.item_id:
            continue
        u    = enc.user_id[uid]
        i    = enc.item_id[mid]
        pred = model.predict(u, i)
        errors.append((r - pred) ** 2)
    return float(np.mean(errors)) if errors else float("nan")


# ---------------------------------------------------------------------------
# Saving utilities
# ---------------------------------------------------------------------------

def save_run(log_dir: Path, model: MFModel, enc: Encoder, args, losses: dict):
    """Save model, encoder, embeddings, and loss curves."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # embeddings as .npy
    np.save(log_dir / "user_embeddings.npy", model.P)
    np.save(log_dir / "item_embeddings.npy", model.Q)
    np.save(log_dir / "user_bias.npy",       model.b_u)
    np.save(log_dir / "item_bias.npy",       model.b_i)
    np.save(log_dir / "mu.npy",              np.array([model.mu]))
    logger.info(f"Saved embeddings → {log_dir}")

    # encoder
    with open(log_dir / "encoder.pkl", "wb") as f:
        pickle.dump(enc, f)

    # full model
    with open(log_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    # args
    with open(log_dir / "args.pkl", "wb") as f:
        pickle.dump(vars(args), f)

    # loss curves
    _plot_losses(losses, log_dir, args)
    logger.info(f"Training run saved → {log_dir}")


def _plot_losses(losses: dict, log_dir: Path, args):
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
    fig.savefig(log_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    logger.info(f"Loss curve saved → {log_dir / 'loss_curve.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train MF with SGD (MSE or BPR)")

    # data
    p.add_argument("--data_dir",   type=str,   default="data/ml-32m",  help="MovieLens data directory")
    p.add_argument("--start",      type=str,   required=True,          help="Training window start YYYY-MM-DD")
    p.add_argument("--end",        type=str,   required=True,          help="Training window end   YYYY-MM-DD")

    # model
    p.add_argument("--loss",       type=str,   default="bpr",          choices=["mse", "bpr"])
    p.add_argument("--dim",        type=int,   default=64,             help="Latent dimension")
    p.add_argument("--epochs",     type=int,   default=20,             help="Number of SGD epochs")
    p.add_argument("--lr",         type=float, default=0.005,          help="Learning rate")
    p.add_argument("--reg1",       type=float, default=0.01,           help="Regularisation for embeddings")
    p.add_argument("--reg2",       type=float, default=0.001,          help="Regularisation for biases")
    p.add_argument("--n_neg",      type=int,   default=4,              help="Negative samples per positive (BPR)")

    # cold start filters
    p.add_argument("--min_rating", type=float, default=3.0,            help="Minimum rating to count as positive")
    p.add_argument("--min_u_rat",  type=int,   default=10,             help="Min ratings per user in window")
    p.add_argument("--min_i_rat",  type=int,   default=10,             help="Min ratings per item in window")

    # eval
    p.add_argument("--eval_months",type=int,   default=5,              help="Number of monthly eval windows after end")
    p.add_argument("--eval_k",     type=int,   nargs="+", default=[5, 10, 15, 20], help="K values for Recall@K and NDCG@K")
    p.add_argument("--top_n_recs", type=int,   default=50,             help="Top-N recommendations to save per user")
    p.add_argument("--min_eval_ratings", type=int, nargs="+", default=[5], help="Min ratings in eval window to qualify user")

    # output
    p.add_argument("--log_root",   type=str,   default="train_logs/MF", help="Root folder for training logs")

    return p.parse_args()


def main():
    args    = parse_args()
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_root) / f"{args.start}_{args.end}_{run_tag}_{args.loss}_dim{args.dim}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # file handler for this run
    fh = logging.FileHandler(log_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    logger.info(f"Run: {run_tag} | loss={args.loss} | dim={args.dim} | "
                f"lr={args.lr} | reg1={args.reg1} | reg2={args.reg2}")
    logger.info(f"Log dir: {log_dir}")

    # ---- load DB ----
    db = MovieLensDB(data_dir=args.data_dir)
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

    # ---- model ----
    mu    = train_df["rating"].mean()
    model = MFModel(enc.n_users, enc.n_items, args.dim, mu)
    logger.info(f"Model initialised | mu={mu:.4f}")

    # ---- build interaction arrays ----
    if args.loss == "mse":
        interactions = train_df[["userId", "movieId", "rating"]].copy()
        interactions["u_idx"] = interactions["userId"].map(enc.user_id)
        interactions["i_idx"] = interactions["movieId"].map(enc.item_id)
        inter_arr = interactions[["u_idx", "i_idx", "rating"]].values.astype(np.float32)
    else:
        interactions = train_df[["userId", "movieId"]].copy()
        interactions["u_idx"] = interactions["userId"].map(enc.user_id)
        interactions["i_idx"] = interactions["movieId"].map(enc.item_id)
        inter_arr = interactions[["u_idx", "i_idx"]].values.astype(np.int32)
        sampler   = NegativeSampler(train_df, enc, n_neg=args.n_neg)

    # ---- training loop ----
    train_losses = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        # inside main() training loop, BPR branch:
        if args.loss == "bpr":
            # pre-sample all negatives for this epoch (outside numba)
            neg_samples = np.array(
                [sampler.sample(int(inter_arr[idx, 0])) for idx in range(len(inter_arr))],
                dtype=np.int32
            )  # shape (N, n_neg)

            np.random.shuffle(inter_arr)  # shuffle positives each epoch

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

        train_losses.append(loss)
        elapsed = time.time() - t_ep
        logger.info(f"Epoch {epoch:3d}/{args.epochs} | loss={loss:.6f} | time={elapsed:.1f}s")

    total_time = time.time() - t0
    logger.info(f"Training complete in {total_time:.1f}s")

    # ---- save model, embeddings, encoder ----
    losses = {"train_loss": train_losses}
    save_run(log_dir, model, enc, args, losses)

    # ---- multi-month evaluation ----
    logger.info("=" * 60)
    logger.info("Starting multi-month evaluation ...")
    logger.info("=" * 60)

    end_dt = pd.Timestamp(args.end)

    # accumulate seen movies per user across training + all prior eval windows
    seen_per_user = {}
    for uid_orig, grp in train_df.groupby("userId"):
        u = enc.user_id[uid_orig]
        seen_per_user[u] = set(enc.item_id[m] for m in grp["movieId"] if m in enc.item_id)

    all_item_indices = np.array(list(enc.id_item.keys()))

    summary_rows = []

    for m in range(1, args.eval_months + 1):
        window_start = end_dt + relativedelta(months=m - 1)
        window_end   = end_dt + relativedelta(months=m)
        ws_str       = window_start.strftime("%Y-%m-%d")
        we_str       = window_end.strftime("%Y-%m-%d")

        logger.info(f"\nEval window {m}: {ws_str} → {we_str}")

        # pull eval data — only known users and items
        eval_raw = db.get_ratings_by_daterange(ws_str, we_str)
        eval_raw = eval_raw[eval_raw["rating"] >= args.min_rating]
        eval_raw = eval_raw[eval_raw["userId"].isin(enc.user_id)]
        eval_raw = eval_raw[eval_raw["movieId"].isin(enc.item_id)]

        if eval_raw.empty:
            logger.info(f"  No eval data for window {m}, skipping.")
            continue

        # ---- generate top-N recommendations for all qualifying users ----
        # qualify = user has >= min_eval_ratings in this window (check separately per threshold)
        user_eval_counts = eval_raw.groupby("userId")["movieId"].count()

        recs_rows = []

        for uid_orig, count in user_eval_counts.items():
            u = enc.user_id[uid_orig]

            # items to score = all items minus seen so far
            seen     = seen_per_user.get(u, set())
            unseen   = all_item_indices[~np.isin(all_item_indices, list(seen))]

            if len(unseen) == 0:
                continue

            scores   = model.predict_batch(u, unseen)
            top_idx  = np.argsort(scores)[::-1][:args.top_n_recs]
            top_items = unseen[top_idx]
            top_scores = scores[top_idx]

            for rank, (item_idx, score) in enumerate(zip(top_items, top_scores), start=1):
                recs_rows.append({
                    "userId":       uid_orig,
                    "movieId":      enc.id_item[item_idx],
                    "score":        round(float(score), 6),
                    "rank":         rank,
                    "eval_window":  m,
                    "window_start": ws_str,
                    "window_end":   we_str,
                    "n_eval_ratings": int(count),
                })

        if not recs_rows:
            logger.info(f"  No recommendations generated for window {m}.")
            continue

        recs_df = pd.DataFrame(recs_rows)
        recs_path = log_dir / f"recs_window_{m:02d}.csv"
        recs_df.to_csv(recs_path, index=False)
        logger.info(f"  Saved {len(recs_df):,} recommendation rows → {recs_path.name}")

        # ---- compute Recall@K and NDCG@K per min_eval_ratings threshold ----
        # ground truth for this window = movies user rated (that are known)
        gt_per_user = {}
        for uid_orig, grp in eval_raw.groupby("userId"):
            gt_per_user[uid_orig] = set(grp["movieId"].values)

        for min_er in args.min_eval_ratings:
            qualifying = user_eval_counts[user_eval_counts >= min_er].index

            if len(qualifying) == 0:
                logger.info(f"  min_eval_ratings={min_er}: no qualifying users")
                continue

            for k in args.eval_k:
                recalls, ndcgs = [], []

                for uid_orig in qualifying:
                    gt = gt_per_user.get(uid_orig, set())
                    if not gt:
                        continue

                    user_recs = recs_df[recs_df["userId"] == uid_orig].sort_values("rank")
                    top_k_items = set(user_recs.head(k)["movieId"].values)

                    # Recall@K
                    hits    = len(top_k_items & gt)
                    recall  = hits / min(len(gt), k)
                    recalls.append(recall)

                    # NDCG@K
                    dcg  = 0.0
                    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), k)))
                    for rank_0, item in enumerate(user_recs.head(k)["movieId"].values):
                        if item in gt:
                            dcg += 1.0 / np.log2(rank_0 + 2)
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    ndcgs.append(ndcg)

                mean_recall = float(np.mean(recalls))
                mean_ndcg   = float(np.mean(ndcgs))

                logger.info(
                    f"  Window {m} | min_ratings={min_er:2d} | K={k:2d} | "
                    f"Recall@{k}={mean_recall:.4f} | NDCG@{k}={mean_ndcg:.4f} | "
                    f"users={len(recalls)}"
                )

                summary_rows.append({
                    "eval_window":       m,
                    "window_start":      ws_str,
                    "window_end":        we_str,
                    "min_eval_ratings":  min_er,
                    "K":                 k,
                    "n_users":           len(recalls),
                    "recall":            round(mean_recall, 6),
                    "ndcg":              round(mean_ndcg, 6),
                })

        # ---- update seen movies with this window's interactions ----
        for uid_orig, grp in eval_raw.groupby("userId"):
            if uid_orig in enc.user_id:
                u = enc.user_id[uid_orig]
                if u not in seen_per_user:
                    seen_per_user[u] = set()
                seen_per_user[u].update(
                    enc.item_id[m_id] for m_id in grp["movieId"] if m_id in enc.item_id
                )

    # ---- save summary metrics ----
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = log_dir / "eval_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"\nEval summary saved → {summary_path}")
        _plot_eval_summary(summary_df, log_dir, args)

    logger.info(f"\nAll done. Everything saved in: {log_dir}")


def _plot_eval_summary(summary_df: pd.DataFrame, log_dir: Path, args):
    """Plot Recall@K and NDCG@K across eval windows for each K and min_eval_ratings."""
    for metric in ["recall", "ndcg"]:
        fig, axes = plt.subplots(
            1, len(args.min_eval_ratings),
            figsize=(6 * len(args.min_eval_ratings), 5),
            sharey=False
        )
        if len(args.min_eval_ratings) == 1:
            axes = [axes]

        for ax, min_er in zip(axes, args.min_eval_ratings):
            sub = summary_df[summary_df["min_eval_ratings"] == min_er]
            for k in args.eval_k:
                ksub = sub[sub["K"] == k].sort_values("eval_window")
                ax.plot(
                    ksub["eval_window"], ksub[metric],
                    marker="o", markersize=4, label=f"K={k}"
                )
            ax.set_title(f"min_ratings={min_er}")
            ax.set_xlabel("Eval window (months after training end)")
            ax.set_ylabel(f"{metric.upper()}@K")
            ax.legend()
            ax.grid(alpha=0.3)

        fig.suptitle(
            f"{metric.upper()}@K over time | loss={args.loss} | dim={args.dim}",
            fontsize=13
        )
        plt.tight_layout()
        path = log_dir / f"{metric}_over_time.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → {path}")


if __name__ == "__main__":
    main()