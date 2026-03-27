"""
matrix_factorisation_temporal.py
---------------------------------
Temporal Matrix Factorisation (TimeSVD++ style) with BPR loss.

Model:
    r_hat_ui(t) = mu + b_u + b_i(t) + p_u(t).T @ q_i

    Item bias:
        b_i(t) = b_i + b_i_bin[bin(t)]
        bin(t) = min(t_days // 30, n_bins - 1)

    User embedding (per dimension k):
        p_u(t)[k] = p_uk + alpha_uk * dev_u(t)
        dev_u(t)  = sign(t - t_u) * |( t - t_u ) / T|^beta
        t_u       = mean rating timestamp of user u in training window (days)
        T         = training window length in days
        beta      = 0.4 (command line, default)

    Item embedding q_i: static (no temporal term)
    b_u: kept but cancels in BPR, no update

BPR loss for triple (u, i, j, t):
    r_hat_uij(t) = (b_i - b_j) + (b_i_bin[m] - b_j_bin[m]) + p_u(t).T @ (q_i - q_j)
    theta        = 1 - sigmoid(r_hat_uij(t))

    Updates (vectorised):
        dq        = q_i - q_j                         (compute once)
        p_u_t     = p_u + dev_u(t) * alpha_u          (compute once)

        b_i       += lr * (theta  - reg2 * b_i)
        b_j       -= lr * (theta  + reg2 * b_j)
        b_i_bin[m]+= lr * (theta  - reg2 * b_i_bin[m])
        b_j_bin[m]-= lr * (theta  + reg2 * b_j_bin[m])
        p_u       += lr * (theta * dq - reg1 * p_u)
        alpha_u   += lr * (theta * dev_u(t) * dq - reg1 * alpha_u)
        q_i       += lr * (theta * p_u_t  - reg1 * q_i)
        q_j       -= lr * (theta * p_u_t  + reg1 * q_j)

Serving (base parameters only):
    r_hat_ui = mu + b_u + b_i + p_u.T @ q_i

Usage (from project root):
    python src/training/matrix_factorisation_temporal.py \
        --start 2018-01-01 --end 2018-06-30 \
        --dim 64 --lr 0.005 --reg1 0.01 --reg2 0.01 \
        --epochs 20 --beta 0.4 --eval_k 5 10 20
"""

import argparse
import logging
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.db_simulator import MovieLensDB
from src.training.matrix_factorisation import Encoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Temporal MF Model
# ══════════════════════════════════════════════════════════════════════════

class TemporalMFModel:
    """
    Stores all learnable parameters for the temporal MF model.

    Base parameters (used at serving time):
        P       : (n_users, dim)   base user embeddings
        Q       : (n_items, dim)   item embeddings (static)
        b_i     : (n_items,)       base item biases
        b_u     : (n_users,)       user biases (cancels in BPR, kept for completeness)

    Temporal parameters (training only):
        Alpha   : (n_users, dim)   per-dimension drift coefficients
        B_bin   : (n_items, n_bins) monthly item bias deviations
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        dim: int,
        n_bins: int,
        mu: float,
        alpha_init: float = 0.001,
    ):
        self.dim    = dim
        self.n_bins = n_bins
        self.mu     = mu

        scale = 0.01

        # base parameters
        self.P   = np.random.normal(0, scale, (n_users, dim)).astype(np.float32)
        self.Q   = np.random.normal(0, scale, (n_items, dim)).astype(np.float32)
        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)

        # temporal parameters
        # Alpha: small uniform init as agreed
        self.Alpha = np.random.uniform(
            -alpha_init, alpha_init, (n_users, dim)
        ).astype(np.float32)
        # B_bin: zero init — no prior drift
        self.B_bin = np.zeros((n_items, n_bins), dtype=np.float32)

    def predict_base(self, u: int, i: int) -> float:
        """Base prediction — used at serving time."""
        return float(self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i])

    def predict_base_batch(self, u: int, item_indices: np.ndarray) -> np.ndarray:
        """Batch base prediction for all items in item_indices."""
        return (
            self.mu
            + self.b_u[u]
            + self.b_i[item_indices]
            + self.Q[item_indices] @ self.P[u]
        )

    def predict_temporal(self, u: int, i: int, dev: float, bin_t: int) -> float:
        """Full temporal prediction — used during training."""
        p_u_t = self.P[u] + dev * self.Alpha[u]
        return float(
            self.mu
            + self.b_u[u]
            + self.b_i[i]
            + self.B_bin[i, bin_t]
            + p_u_t @ self.Q[i]
        )


# ══════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════

def load_training_data(
    db: MovieLensDB,
    start: str,
    end: str,
    min_rating: float = 3.0,
    min_user_ratings: int = 10,
    min_item_ratings: int = 10,
) -> pd.DataFrame:
    """
    Pull ratings from db within [start, end].
    Applies rating filter and iterative co-filtering.
    Returns dataframe with: userId, movieId, rating, timestamp
    """
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


# ══════════════════════════════════════════════════════════════════════════
# Temporal helpers
# ══════════════════════════════════════════════════════════════════════════

def compute_temporal_features(
    train_df: pd.DataFrame,
    enc: Encoder,
    start: str,
    end: str,
) -> tuple:
    """
    Compute per-interaction temporal features and per-user t_u.

    Returns:
        t_days_arr : float32 array (N,) — days since window start per interaction
        t_u_arr    : float32 array (n_users,) — mean t_days per user
        T          : float — training window length in days
        n_bins     : int — number of monthly bins
    """
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    T        = float((end_ts - start_ts).days)
    n_bins   = max(1, int(np.ceil(T / 30)))

    logger.info(f"  Training window: {T:.0f} days | n_bins: {n_bins}")

    # per-interaction t_days
    t_days = (train_df["timestamp"] - start_ts).dt.total_seconds() / 86400.0
    t_days = t_days.astype(np.float32).values

    # per-user t_u = mean t_days (after co-filtering, so only qualifying interactions)
    train_df = train_df.copy()
    train_df["t_days"] = t_days
    train_df["u_idx"]  = train_df["userId"].map(enc.user_id)

    t_u_series = train_df.groupby("u_idx")["t_days"].mean()
    t_u_arr    = np.zeros(enc.n_users, dtype=np.float32)
    for u_idx, val in t_u_series.items():
        t_u_arr[int(u_idx)] = float(val)

    return t_days, t_u_arr, T, n_bins


def compute_dev(t: float, t_u: float, T: float, beta: float) -> float:
    """
    dev_u(t) = sign(t - t_u) * |(t - t_u) / T|^beta
    Normalised by T so dev is always in [-1, +1].
    """
    diff = t - t_u
    if diff == 0.0:
        return 0.0
    return float(np.sign(diff) * (abs(diff) / T) ** beta)


def compute_bin(t: float, n_bins: int) -> int:
    """Map t_days to monthly bin index."""
    return min(int(t // 30), n_bins - 1)


# ══════════════════════════════════════════════════════════════════════════
# Negative sampler (identical to matrix_factorisation.py)
# ══════════════════════════════════════════════════════════════════════════

class NegativeSampler:
    """
    Samples negatives for BPR from movies the user has NOT rated,
    weighted by popularity within the training window.
    """

    def __init__(self, df: pd.DataFrame, enc: Encoder, n_neg: int = 4):
        self.n_neg     = n_neg
        self.enc       = enc
        self.all_items = np.array(list(enc.id_item.keys()))

        counts  = df["movieId"].value_counts()
        weights = np.array(
            [counts.get(enc.id_item[i], 0) for i in self.all_items],
            dtype=np.float64,
        )
        self.weights = weights / weights.sum()

        logger.info("Building per-user positive sets for negative sampling ...")
        self.user_pos = {}
        for uid_orig, grp in df.groupby("userId"):
            u = enc.user_id[uid_orig]
            self.user_pos[u] = set(enc.item_id[m] for m in grp["movieId"].values)

    def sample(self, u: int) -> np.ndarray:
        pos        = self.user_pos.get(u, set())
        negs       = []
        candidates = np.random.choice(
            self.all_items, size=self.n_neg * 10, p=self.weights
        )
        for c in candidates:
            if c not in pos:
                negs.append(c)
            if len(negs) == self.n_neg:
                break
        while len(negs) < self.n_neg:
            c = np.random.choice(self.all_items)
            if c not in pos:
                negs.append(c)
        return np.array(negs, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════
# Training — one SGD epoch (BPR temporal)
# NOTE: written for readability and Numba extensibility.
#       All inner operations use numpy scalar/vector ops — no Python objects
#       inside the hot loop. When adding @njit later:
#         - replace self.model.* with plain numpy arrays passed as arguments
#         - replace sampler.sample() with pre-sampled neg_samples array
#         - replace compute_dev/compute_bin with inline math
# ══════════════════════════════════════════════════════════════════════════

def train_one_epoch_bpr_temporal(
    model: TemporalMFModel,
    interactions: np.ndarray,
    neg_samples: np.ndarray,
    t_u_arr: np.ndarray,
    T: float,
    beta: float,
    lr: float,
    reg1: float,
    reg2: float,
) -> float:
    """
    One SGD epoch for temporal BPR.

    Args:
        interactions : int32 (N, 2) — [u_idx, i_idx]
        neg_samples  : int32 (N, n_neg) — pre-sampled negatives
        t_u_arr      : float32 (n_users,) — mean timestamp per user
        T            : training window length in days
        beta         : dev exponent
        lr, reg1, reg2 : hyperparameters

    NOTE: t_days is stored in interactions[:,2] as float32.
          interactions is float32 (N, 3) — [u_idx, i_idx, t_days]
    """
    N          = interactions.shape[0]
    n_neg      = neg_samples.shape[1]
    total_loss = 0.0

    # shuffle interaction order each epoch
    perm = np.random.permutation(N)

    for idx in perm:
        u     = int(interactions[idx, 0])
        i     = int(interactions[idx, 1])
        t     = float(interactions[idx, 2])
        t_u   = float(t_u_arr[u])
        bin_t = compute_bin(t, model.n_bins)
        dev   = compute_dev(t, t_u, T, beta)

        # temporal user embedding and precomputed reusable terms
        p_u_t = model.P[u] + dev * model.Alpha[u]   # shape (dim,)

        for n in range(n_neg):
            j = int(neg_samples[idx, n])

            # ── score difference ──────────────────────────────────────────
            dq       = model.Q[i] - model.Q[j]          # (dim,) reuse below
            b_diff   = (
                (model.b_i[i] - model.b_i[j])
                + (model.B_bin[i, bin_t] - model.B_bin[j, bin_t])
            )
            r_uij    = b_diff + float(p_u_t @ dq)

            # ── error signal ──────────────────────────────────────────────
            sig      = 1.0 / (1.0 + np.exp(-np.clip(r_uij, -30.0, 30.0)))
            theta    = 1.0 - sig
            total_loss += -np.log(sig + 1e-10)

            # ── base item biases ──────────────────────────────────────────
            model.b_i[i] += lr * (theta  - reg2 * model.b_i[i])
            model.b_i[j] -= lr * (theta  + reg2 * model.b_i[j])

            # ── monthly item bias deviations ──────────────────────────────
            model.B_bin[i, bin_t] += lr * (theta  - reg2 * model.B_bin[i, bin_t])
            model.B_bin[j, bin_t] -= lr * (theta  + reg2 * model.B_bin[j, bin_t])

            # ── base user embedding ───────────────────────────────────────
            model.P[u] += lr * (theta * dq - reg1 * model.P[u])

            # ── drift coefficient ─────────────────────────────────────────
            model.Alpha[u] += lr * (theta * dev * dq - reg1 * model.Alpha[u])

            # ── item embeddings (use p_u_t computed before updates) ───────
            model.Q[i] += lr * (theta * p_u_t - reg1 * model.Q[i])
            model.Q[j] -= lr * (theta * p_u_t + reg1 * model.Q[j])

    return total_loss / max(N * n_neg, 1)


# ══════════════════════════════════════════════════════════════════════════
# Saving
# ══════════════════════════════════════════════════════════════════════════

def save_run(
    log_dir: Path,
    model: TemporalMFModel,
    enc: Encoder,
    args,
    losses: dict,
):
    log_dir.mkdir(parents=True, exist_ok=True)

    # base parameters (used at serving)
    np.save(log_dir / "user_embeddings.npy", model.P)
    np.save(log_dir / "item_embeddings.npy", model.Q)
    np.save(log_dir / "user_bias.npy",       model.b_u)
    np.save(log_dir / "item_bias.npy",       model.b_i)
    np.save(log_dir / "mu.npy",              np.array([model.mu]))

    # temporal parameters (saved for future analysis)
    np.save(log_dir / "alpha_embeddings.npy", model.Alpha)
    np.save(log_dir / "item_bin_bias.npy",    model.B_bin)

    logger.info(f"Saved embeddings → {log_dir}")

    # encoder and args
    with open(log_dir / "encoder.pkl", "wb") as f:
        pickle.dump(enc, f)
    with open(log_dir / "args.pkl", "wb") as f:
        pickle.dump(vars(args), f)

    # loss plot
    _plot_losses(losses, log_dir, args)
    logger.info(f"Training run saved → {log_dir}")


def _plot_losses(losses: dict, log_dir: Path, args):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, values in losses.items():
        ax.plot(range(1, len(values) + 1), values,
                marker="o", markersize=3, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BPR Loss")
    ax.set_title(
        f"Temporal MF | dim={args.dim} | beta={args.beta} | "
        f"lr={args.lr} | reg1={args.reg1} | reg2={args.reg2}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(log_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    logger.info(f"Loss curve saved → {log_dir / 'loss_curve.png'}")


def _plot_eval_summary(summary_df: pd.DataFrame, log_dir: Path, args):
    for metric in ["recall", "ndcg"]:
        fig, axes = plt.subplots(
            1, len(args.min_eval_ratings),
            figsize=(6 * len(args.min_eval_ratings), 5),
            sharey=False,
        )
        if len(args.min_eval_ratings) == 1:
            axes = [axes]

        for ax, min_er in zip(axes, args.min_eval_ratings):
            sub = summary_df[summary_df["min_eval_ratings"] == min_er]
            for k in args.eval_k:
                ksub = sub[sub["K"] == k].sort_values("eval_window")
                ax.plot(
                    ksub["eval_window"], ksub[metric],
                    marker="o", markersize=4, label=f"K={k}",
                )
            ax.set_title(f"min_ratings={min_er}")
            ax.set_xlabel("Eval window (months after training end)")
            ax.set_ylabel(f"{metric.upper()}@K")
            ax.legend()
            ax.grid(alpha=0.3)

        fig.suptitle(
            f"{metric.upper()}@K over time | Temporal MF | dim={args.dim}",
            fontsize=13,
        )
        plt.tight_layout()
        path = log_dir / f"{metric}_over_time.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → {path}")


# ══════════════════════════════════════════════════════════════════════════
# Evaluation helpers (identical logic to matrix_factorisation.py)
# ══════════════════════════════════════════════════════════════════════════

def run_eval_windows(
    model: TemporalMFModel,
    enc: Encoder,
    db: MovieLensDB,
    train_df: pd.DataFrame,
    args,
    log_dir: Path,
):
    end_dt           = pd.Timestamp(args.end)
    all_item_indices = np.array(list(enc.id_item.keys()))
    summary_rows     = []

    # seen movies per user — start from training window
    seen_per_user = {}
    for uid_orig, grp in train_df.groupby("userId"):
        u = enc.user_id[uid_orig]
        seen_per_user[u] = set(
            enc.item_id[m] for m in grp["movieId"] if m in enc.item_id
        )

    for m in range(1, args.eval_months + 1):
        ws = (end_dt + relativedelta(months=m - 1)).strftime("%Y-%m-%d")
        we = (end_dt + relativedelta(months=m)).strftime("%Y-%m-%d")

        logger.info(f"\nEval window {m}: {ws} → {we}")

        eval_raw = db.get_ratings_by_daterange(ws, we)
        eval_raw = eval_raw[eval_raw["rating"] >= args.min_rating]
        eval_raw = eval_raw[eval_raw["userId"].isin(enc.user_id)]
        eval_raw = eval_raw[eval_raw["movieId"].isin(enc.item_id)]

        if eval_raw.empty:
            logger.info(f"  No eval data for window {m}, skipping.")
            continue

        user_eval_counts = eval_raw.groupby("userId")["movieId"].count()
        gt_per_user = {
            uid: set(grp["movieId"].values)
            for uid, grp in eval_raw.groupby("userId")
        }

        # ── generate top-N recs using BASE parameters only ────────────────
        recs_rows = []
        for uid_orig, count in user_eval_counts.items():
            u      = enc.user_id[uid_orig]
            seen   = seen_per_user.get(u, set())
            unseen = all_item_indices[~np.isin(all_item_indices, list(seen))]

            if len(unseen) == 0:
                continue

            # base prediction — no temporal terms at serving time
            scores     = model.predict_base_batch(u, unseen)
            top_idx    = np.argsort(scores)[::-1][:args.top_n_recs]
            top_items  = unseen[top_idx]
            top_scores = scores[top_idx]

            for rank, (item_idx, score) in enumerate(
                zip(top_items, top_scores), start=1
            ):
                recs_rows.append({
                    "userId":          uid_orig,
                    "movieId":         enc.id_item[item_idx],
                    "score":           round(float(score), 6),
                    "rank":            rank,
                    "eval_window":     m,
                    "window_start":    ws,
                    "window_end":      we,
                    "n_eval_ratings":  int(count),
                })

        if not recs_rows:
            logger.info(f"  No recs generated for window {m}.")
            continue

        recs_df   = pd.DataFrame(recs_rows)
        recs_path = log_dir / f"recs_window_{m:02d}.csv"
        recs_df.to_csv(recs_path, index=False)
        logger.info(f"  Saved {len(recs_df):,} rows → {recs_path.name}")

        # ── metrics ───────────────────────────────────────────────────────
        for min_er in args.min_eval_ratings:
            qualifying = user_eval_counts[user_eval_counts >= min_er].index
            if len(qualifying) == 0:
                continue

            for k in args.eval_k:
                recalls, ndcgs = [], []

                for uid_orig in qualifying:
                    gt        = gt_per_user.get(uid_orig, set())
                    if not gt:
                        continue
                    user_recs = (
                        recs_df[recs_df["userId"] == uid_orig]
                        .sort_values("rank")
                    )
                    top_k_items = list(user_recs.head(k)["movieId"].values)

                    # Recall@K
                    hits   = len(set(top_k_items) & gt)
                    recall = hits / min(len(gt), k)
                    recalls.append(recall)

                    # NDCG@K
                    idcg = sum(
                        1.0 / np.log2(r + 2)
                        for r in range(min(len(gt), k))
                    )
                    dcg = sum(
                        1.0 / np.log2(r + 2)
                        for r, item in enumerate(top_k_items)
                        if item in gt
                    )
                    ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

                mean_recall = float(np.mean(recalls))
                mean_ndcg   = float(np.mean(ndcgs))

                logger.info(
                    f"  Window {m} | min_ratings={min_er:2d} | K={k:2d} | "
                    f"Recall@{k}={mean_recall:.4f} | NDCG@{k}={mean_ndcg:.4f} | "
                    f"users={len(recalls)}"
                )
                summary_rows.append({
                    "eval_window":      m,
                    "window_start":     ws,
                    "window_end":       we,
                    "min_eval_ratings": min_er,
                    "K":                k,
                    "n_users":          len(recalls),
                    "recall":           round(mean_recall, 6),
                    "ndcg":             round(mean_ndcg, 6),
                })

        # update seen with this window's interactions
        for uid_orig, grp in eval_raw.groupby("userId"):
            if uid_orig in enc.user_id:
                u = enc.user_id[uid_orig]
                if u not in seen_per_user:
                    seen_per_user[u] = set()
                seen_per_user[u].update(
                    enc.item_id[mid]
                    for mid in grp["movieId"]
                    if mid in enc.item_id
                )

    return summary_rows


# ══════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Temporal MF (TimeSVD++ style) with BPR loss"
    )

    # data
    p.add_argument("--data_dir",   type=str,   default="data/ml-32m")
    p.add_argument("--start",      type=str,   required=True,
                   help="Training window start YYYY-MM-DD")
    p.add_argument("--end",        type=str,   required=True,
                   help="Training window end   YYYY-MM-DD")

    # model
    p.add_argument("--dim",        type=int,   default=64)
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--lr",         type=float, default=0.005)
    p.add_argument("--reg1",       type=float, default=0.01,
                   help="Regularisation for embeddings (P, Q, Alpha)")
    p.add_argument("--reg2",       type=float, default=0.001,
                   help="Regularisation for biases (b_i, B_bin)")
    p.add_argument("--n_neg",      type=int,   default=4)
    p.add_argument("--beta",       type=float, default=0.4,
                   help="Exponent for dev_u(t) = sign(dt)*|dt/T|^beta")
    p.add_argument("--alpha_init", type=float, default=0.001,
                   help="Uniform init range [-a, +a] for Alpha")

    # cold start filters
    p.add_argument("--min_rating", type=float, default=3.0)
    p.add_argument("--min_u_rat",  type=int,   default=10)
    p.add_argument("--min_i_rat",  type=int,   default=10)

    # eval
    p.add_argument("--eval_months",        type=int,   default=5)
    p.add_argument("--eval_k",             type=int,   nargs="+",
                   default=[5, 10, 15, 20])
    p.add_argument("--min_eval_ratings",   type=int,   nargs="+",
                   default=[5, 10])
    p.add_argument("--top_n_recs",         type=int,   default=50)

    # output
    p.add_argument("--log_root",   type=str,
                   default="train_logs/MF_temporal")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    args    = parse_args()
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = (
        Path(args.log_root)
        / f"{args.start}_{args.end}_{run_tag}_temporal_dim{args.dim}"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    # file log handler
    fh = logging.FileHandler(log_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    logger.info(
        f"Temporal MF | dim={args.dim} | beta={args.beta} | "
        f"lr={args.lr} | reg1={args.reg1} | reg2={args.reg2}"
    )
    logger.info(f"Log dir: {log_dir}")

    # ── load DB ───────────────────────────────────────────────────────────
    db = MovieLensDB(data_dir=args.data_dir)
    db.load_data()

    # ── training data ─────────────────────────────────────────────────────
    train_df = load_training_data(
        db, args.start, args.end,
        min_rating=args.min_rating,
        min_user_ratings=args.min_u_rat,
        min_item_ratings=args.min_i_rat,
    )

    # ── encoder ───────────────────────────────────────────────────────────
    enc = Encoder()
    enc.encode(train_df)
    logger.info(f"Encoded {enc.n_users:,} users | {enc.n_items:,} items")

    # ── temporal features ─────────────────────────────────────────────────
    t_days_arr, t_u_arr, T, n_bins = compute_temporal_features(
        train_df, enc, args.start, args.end
    )
    logger.info(f"n_bins={n_bins} | T={T:.0f} days")

    # ── model ─────────────────────────────────────────────────────────────
    mu    = float(train_df["rating"].mean())
    model = TemporalMFModel(
        n_users    = enc.n_users,
        n_items    = enc.n_items,
        dim        = args.dim,
        n_bins     = n_bins,
        mu         = mu,
        alpha_init = args.alpha_init,
    )
    logger.info(f"Model initialised | mu={mu:.4f} | n_bins={n_bins}")

    # ── interaction array ─────────────────────────────────────────────────
    # shape (N, 3): [u_idx, i_idx, t_days]
    df_copy          = train_df.copy()
    df_copy["u_idx"] = df_copy["userId"].map(enc.user_id)
    df_copy["i_idx"] = df_copy["movieId"].map(enc.item_id)
    df_copy["t_days"] = t_days_arr

    inter_arr = df_copy[["u_idx", "i_idx", "t_days"]].values.astype(np.float32)

    # ── negative sampler ──────────────────────────────────────────────────
    sampler = NegativeSampler(train_df, enc, n_neg=args.n_neg)

    # ── training loop ─────────────────────────────────────────────────────
    train_losses = []
    t0           = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()

        # pre-sample all negatives for this epoch
        neg_samples = np.array(
            [sampler.sample(int(inter_arr[idx, 0])) for idx in range(len(inter_arr))],
            dtype=np.int32,
        )   # shape (N, n_neg)

        loss = train_one_epoch_bpr_temporal(
            model      = model,
            interactions = inter_arr,
            neg_samples  = neg_samples,
            t_u_arr    = t_u_arr,
            T          = T,
            beta       = args.beta,
            lr         = args.lr,
            reg1       = args.reg1,
            reg2       = args.reg2,
        )

        train_losses.append(loss)
        elapsed = time.time() - t_ep
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | loss={loss:.6f} | "
            f"time={elapsed:.1f}s"
        )

    logger.info(f"Training complete in {time.time() - t0:.1f}s")

    # ── save ──────────────────────────────────────────────────────────────
    save_run(log_dir, model, enc, args, {"train_loss": train_losses})

    # ── evaluation ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting multi-month evaluation (base parameters) ...")
    logger.info("=" * 60)

    summary_rows = run_eval_windows(model, enc, db, train_df, args, log_dir)

    if summary_rows:
        summary_df   = pd.DataFrame(summary_rows)
        summary_path = log_dir / "eval_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Eval summary saved → {summary_path}")
        _plot_eval_summary(summary_df, log_dir, args)

    logger.info(f"\nAll done. Everything saved in: {log_dir}")


if __name__ == "__main__":
    main()