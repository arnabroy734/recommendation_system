"""
sasrec_eval_candidates.py
--------------------------
Evaluate SASRec re-ranking quality on top-N MF candidates.

Two evaluations:
  Eval 1 — Batch NDCG/Recall per eval window
            SASRec re-ranks MF top-N candidates, measured against ground truth
  Eval 2 — Sequential next-item prediction (aggregated across eval2_months)
            SASRec sees train_seq + partial July/Aug/... seq, predicts next item
            within MF candidate pool, measures HitRate@5 and @10

Location : src/eval/sasrec_eval_candidates.py

Usage:
    python src/eval/sasrec_eval_candidates.py \
        --start 2018-01-01 --end 2018-06-30

    python src/eval/sasrec_eval_candidates.py \
        --start 2018-01-01 --end 2018-06-30 \
        --top_n 300 --eval_k 5 10 --eval2_months 2 \
        --min_rating 3.0
"""

import argparse
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.db_simulator import MovieLensDB
from src.training.sasrec_architecture import SASRec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Run directory finders
# ─────────────────────────────────────────────────────────────────────────────

def find_run_dir(root: Path, start: str, end: str, label: str) -> Path:
    prefix    = f"{start}_{end}_"
    matches   = sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name.startswith(prefix)],
        key=lambda d: d.name,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(
            f"No {label} run found in '{root}' matching '{prefix}*'.\n"
            f"Available: {[d.name for d in root.iterdir() if d.is_dir()]}"
        )
    chosen = matches[0]
    if len(matches) > 1:
        logger.warning(f"[{label}] Multiple runs match — using most recent: {chosen.name}")
    logger.info(f"[{label}] Run dir: {chosen}")
    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_mf_recs(mf_run_dir: Path, top_n: int) -> dict:
    """
    Load all recs_window_XX.csv files.
    Returns dict: {window_idx (int) -> DataFrame with columns userId,movieId,rank,...}
    Only keeps rank <= top_n rows.
    """
    recs = {}
    for path in sorted(mf_run_dir.glob("recs_window_*.csv")):
        win_idx = int(path.stem.split("_")[-1])
        df      = pd.read_csv(path)
        df      = df[df["rank"] <= top_n].copy()
        recs[win_idx] = df
        logger.info(
            f"  Loaded {path.name}: {df['userId'].nunique():,} users | "
            f"{len(df):,} rows (top_n={top_n})"
        )
    if not recs:
        raise FileNotFoundError(f"No recs_window_*.csv found in {mf_run_dir}")
    return recs

def load_sasrec_model(sasrec_run_dir: Path, args_dict: dict, device: torch.device):
    """Load SASRec model and encoder. Architecture params come from CLI args."""
    with open(sasrec_run_dir / "encoder.pkl", "rb") as f:
        enc = pickle.load(f)

    model = SASRec(
        n_items  = enc.n_items,
        dim      = args_dict["dim"],
        max_len  = args_dict["max_len"],
        n_heads  = args_dict["n_heads"],
        n_layers = args_dict["n_layers"],
        dropout  = args_dict["dropout"],
    ).to(device)

    state = torch.load(sasrec_run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    logger.info(
        f"SASRec loaded | dim={args_dict['dim']} | "
        f"layers={args_dict['n_layers']} | max_len={args_dict['max_len']} | "
        f"n_items={enc.n_items:,}"
    )
    return model, enc


def load_training_sequences(db: MovieLensDB, enc, start: str, end: str,
                            min_rating: float) -> dict:
    """
    Build per-user training sequence from [start, end].
    Returns dict: {userId_orig -> list of movieIds sorted by timestamp}
    Only includes items known to SASRec encoder.
    """
    df  = db.get_ratings_by_daterange(start, end)
    df  = df[df["rating"] >= min_rating].copy()
    df  = df[df["userId"].isin(enc.user_id)]
    df  = df[df["movieId"].isin(enc.item_id)]
    df  = df.sort_values(["userId", "timestamp"])

    seqs = {}
    for uid, grp in df.groupby("userId"):
        seqs[uid] = list(grp["movieId"].values)
    logger.info(f"Training sequences built for {len(seqs):,} users")
    return seqs


# ─────────────────────────────────────────────────────────────────────────────
# SASRec inference helper
# ─────────────────────────────────────────────────────────────────────────────

def get_user_hidden(model, enc, seq_orig_ids: list, max_len: int,
                    device: torch.device) -> np.ndarray:
    """
    Given a list of original movieIds (the user's sequence),
    run SASRec forward and return the last-position hidden state (dim,).
    """
    items   = [enc.item_id[m] + 1 for m in seq_orig_ids if m in enc.item_id]
    items   = items[-max_len:]
    pad_len = max_len - len(items)
    seq     = [0] * pad_len + items
    seq_t   = torch.LongTensor(seq).unsqueeze(0).to(device)

    with torch.no_grad():
        h, _ = model(seq_t)                    # (1, L, dim)
    return h.cpu().numpy()[0, -1]              # (dim,)


def score_candidates(model, enc, h_u: np.ndarray,
                     candidate_movie_ids: list) -> dict:
    """
    Score a list of candidate movieIds using h_u · item_emb.
    Returns dict {movieId -> score}.
    Skips candidates not in encoder vocab.
    """
    emb_w  = model.item_emb.weight.detach().cpu().numpy()  # (n_items+1, dim)
    scores = {}
    for mid in candidate_movie_ids:
        if mid not in enc.item_id:
            continue
        idx         = enc.item_id[mid] + 1    # 1-indexed
        scores[mid] = float(emb_w[idx] @ h_u)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(recommended: list, ground_truth: set, k: int) -> float:
    hits = len(set(recommended[:k]) & ground_truth)
    return hits / min(len(ground_truth), k)


def ndcg_at_k(recommended: list, ground_truth: set, k: int) -> float:
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(ground_truth), k)))
    if idcg == 0:
        return 0.0
    dcg = sum(
        1.0 / np.log2(r + 2)
        for r, item in enumerate(recommended[:k])
        if item in ground_truth
    )
    return dcg / idcg


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation 1 — Batch NDCG / Recall per window
# ─────────────────────────────────────────────────────────────────────────────

def eval1_batch(
    model, enc, db,
    train_seqs: dict,
    mf_recs:    dict,
    args,
    device:     torch.device,
) -> pd.DataFrame:
    """
    Per eval window:
      - For each user, get MF top-N candidates
      - Run SASRec on training sequence → re-rank candidates
      - Compare top-K re-ranked list against July/Aug/... ground truth
      - Report Recall@K and NDCG@K
    """
    logger.info("=" * 60)
    logger.info("EVAL 1 — Batch NDCG / Recall")
    logger.info("=" * 60)

    max_len = args["max_len"]
    rows    = []

    for win_idx, recs_df in sorted(mf_recs.items()):
        ws = recs_df["window_start"].iloc[0]
        we = recs_df["window_end"].iloc[0]
        logger.info(f"\nWindow {win_idx}: {ws} → {we}")

        # ── filter users by min_eval_ratings ──────────────────────────
        min_er   = args["min_eval_ratings"]
        user_er  = recs_df.groupby("userId")["n_eval_ratings"].first()
        qualifying = user_er[user_er >= min_er].index
        recs_df  = recs_df[recs_df["userId"].isin(qualifying)].copy()
        logger.info(
            f"  After n_eval_ratings >= {min_er} filter: "
            f"{recs_df['userId'].nunique():,} users"
        )

        # ground truth for this window
        gt_df = db.get_ratings_by_daterange(ws, we)
        gt_df = gt_df[gt_df["rating"] >= args["min_rating"]]
        gt_df = gt_df[gt_df["userId"].isin(enc.user_id)]
        gt_per_user = {
            uid: set(grp["movieId"].values)
            for uid, grp in gt_df.groupby("userId")
        }
        # After building gt_per_user, add this filter
        if args.get("min_eval_ratings", 1) > 1:
            gt_per_user = {
                uid: gt
                for uid, gt in gt_per_user.items()
                if len(gt) >= args["min_eval_ratings"]
            }

        # users present in both recs and ground truth
        users_in_window = set(recs_df["userId"].unique()) & set(gt_per_user.keys())
        logger.info(f"  Users with recs AND ground truth: {len(users_in_window):,}")

        # group candidates by user
        cands_per_user = {
            uid: list(grp["movieId"].values)
            for uid, grp in recs_df.groupby("userId")
        }

        per_k = {k: {"recalls": [], "ndcgs": []} for k in args["eval_k"]}
        skipped = 0

        for uid in users_in_window:
            seq = train_seqs.get(uid, [])
            if not seq:
                skipped += 1
                continue

            candidates = cands_per_user.get(uid, [])
            if not candidates:
                skipped += 1
                continue

            # SASRec re-rank
            h_u    = get_user_hidden(model, enc, seq, max_len, device)
            scores = score_candidates(model, enc, h_u, candidates)

            if not scores:
                skipped += 1
                continue

            ranked = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)
            gt     = gt_per_user[uid]

            for k in args["eval_k"]:
                per_k[k]["recalls"].append(recall_at_k(ranked, gt, k))
                per_k[k]["ndcgs"].append(ndcg_at_k(ranked, gt, k))

        logger.info(f"  Skipped users (no seq / no candidates): {skipped}")

        for k in args["eval_k"]:
            r = np.mean(per_k[k]["recalls"]) if per_k[k]["recalls"] else 0.0
            n = np.mean(per_k[k]["ndcgs"])   if per_k[k]["ndcgs"]   else 0.0
            n_users = len(per_k[k]["recalls"])
            logger.info(
                f"  K={k:2d} | Recall={r:.4f} | NDCG={n:.4f} | users={n_users:,}"
            )
            rows.append({
                "eval":         "batch",
                "eval_window":  win_idx,
                "window_start": ws,
                "window_end":   we,
                "K":            k,
                "n_users":      n_users,
                "recall":       round(r, 6),
                "ndcg":         round(n, 6),
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation 2 — Sequential next-item prediction (aggregated)
# ─────────────────────────────────────────────────────────────────────────────

def eval2_sequential(
    model, enc, db,
    train_seqs:   dict,
    mf_recs:      dict,
    args,
    device:       torch.device,
) -> pd.DataFrame:
    """
    Sliding next-item prediction, reported PER MONTH and aggregated.
    Aug sequences include July interactions (cumulative).
    """
    logger.info("=" * 60)
    logger.info("EVAL 2 — Sequential Next-Item Prediction (per month)")
    logger.info("=" * 60)

    max_len      = args["max_len"]
    eval2_months = args["eval2_months"]

    # union MF pool across all windows per user (fixed retriever pool)
    all_recs = pd.concat(list(mf_recs.values()), ignore_index=True)
    all_recs = all_recs[all_recs["rank"] <= args["top_n"]]

    # ── filter users by min_eval_ratings ──────────────────────────────
    min_er    = args["min_eval_ratings"]
    user_er   = all_recs.groupby("userId")["n_eval_ratings"].max()  # max across windows
    qualifying = user_er[user_er >= min_er].index
    all_recs  = all_recs[all_recs["userId"].isin(qualifying)].copy()
    logger.info(
        f"  Eval2: After n_eval_ratings >= {min_er} filter: "
        f"{all_recs['userId'].nunique():,} users"
    )

    cands_per_user = {
        uid: list(grp["movieId"].unique())
        for uid, grp in all_recs.groupby("userId")
    }

    sorted_windows = sorted(mf_recs.keys())[:eval2_months]

    # stores per-window interactions for cumulative sequence building
    window_seqs = {}

    rows = []

    # accumulators for final aggregate
    agg_hits          = {k: [] for k in args["eval_k"]}
    agg_random_hits  = {k: [] for k in args["eval_k"]}
    total_skipped_vocab  = 0
    total_skipped_nopool = 0

    for win_idx in sorted_windows:
        recs_df = mf_recs[win_idx]
        ws      = recs_df["window_start"].iloc[0]
        we      = recs_df["window_end"].iloc[0]
        logger.info(f"\nWindow {win_idx}: {ws} → {we}")

        

        # load this window's interactions in timestamp order
        win_df = db.get_ratings_by_daterange(ws, we)
        win_df = win_df[win_df["rating"] >= args["min_rating"]]
        win_df = win_df[win_df["userId"].isin(enc.user_id)]
        win_df = win_df.sort_values(["userId", "timestamp"])

        win_items_per_user = {}
        for uid, grp in win_df.groupby("userId"):
            win_items_per_user[uid] = list(grp["movieId"].values)

        window_seqs[win_idx] = win_items_per_user

        users_in_window = set(recs_df["userId"].unique())

        # per-month accumulators
        month_hits           = {k: [] for k in args["eval_k"]}
        month_random_hits = {k: [] for k in args["eval_k"]}
        win_skipped_vocab    = 0
        win_skipped_nopool   = 0
        win_predictions      = 0

        for uid in users_in_window:
            candidates = cands_per_user.get(uid, [])
            if not candidates:
                win_skipped_nopool += 1
                continue

            # cumulative sequence = train_seq + all PREVIOUS window items
            cumulative_seq = list(train_seqs.get(uid, []))
            for prev_win in sorted_windows:
                if prev_win >= win_idx:
                    break
                prev_items = window_seqs.get(prev_win, {}).get(uid, [])
                cumulative_seq += [m for m in prev_items if m in enc.item_id]

            current_items = win_items_per_user.get(uid, [])
            if len(current_items) < 2:
                continue

            for i in range(len(current_items)):
                target = current_items[i]

                if target not in enc.item_id:
                    win_skipped_vocab += 1
                    continue

                if target not in set(candidates):
                    win_skipped_nopool += 1
                    continue

                prefix_items = current_items[:i]
                valid_prefix = [m for m in prefix_items if m in enc.item_id]
                input_seq    = cumulative_seq + valid_prefix

                if not input_seq:
                    continue

                h_u    = get_user_hidden(model, enc, input_seq, max_len, device)
                scores = score_candidates(model, enc, h_u, candidates)

                if not scores:
                    continue

                # ── SASRec ranked ────────────────────────────────────────────
                ranked = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)

                # ── Random baseline — shuffle same candidate pool ────────────
                scored_keys    = list(scores.keys())   # same items SASRec scored
                random_ranked  = scored_keys.copy()
                np.random.shuffle(random_ranked)

                for k in args["eval_k"]:
                    # SASRec
                    hit = 1 if target in ranked[:k] else 0
                    month_hits[k].append(hit)
                    agg_hits[k].append(hit)

                    # Random
                    random_hit = 1 if target in random_ranked[:k] else 0
                    month_random_hits[k].append(random_hit)
                    agg_random_hits[k].append(random_hit)

                win_predictions += 1

        total_skipped_vocab  += win_skipped_vocab
        total_skipped_nopool += win_skipped_nopool

        logger.info(
            f"  Predictions: {win_predictions:,} | "
            f"skipped (vocab): {win_skipped_vocab:,} | "
            f"skipped (not in pool): {win_skipped_nopool:,}"
        )

        # per-month rows
        for k in args["eval_k"]:
            hitrate        = float(np.mean(month_hits[k]))        if month_hits[k]        else 0.0
            random_hitrate = float(np.mean(month_random_hits[k])) if month_random_hits[k] else 0.0
            lift           = (hitrate / random_hitrate) if random_hitrate > 0 else float("inf")

            logger.info(
                f"  HitRate@{k:2d} | SASRec={hitrate:.4f} | "
                f"Random={random_hitrate:.4f} | "
                f"Lift={lift:.2f}x  ({len(month_hits[k]):,} predictions)"
            )
            rows.append({
                "eval":              "sequential",
                "eval_window":       win_idx,
                "window_start":      ws,
                "window_end":        we,
                "K":                 k,
                "n_predictions":     len(month_hits[k]),
                "hitrate_sasrec":    round(hitrate, 6),
                "hitrate_random":    round(random_hitrate, 6),
                "lift":              round(lift, 4),
                "skipped_vocab":     win_skipped_vocab,
                "skipped_not_pool":  win_skipped_nopool,
                "is_aggregate":      False,
            })

    # aggregate row across all months
    logger.info(f"\n{'─'*40}")
    logger.info(f"AGGREGATE (windows 1–{eval2_months})")
    logger.info(
        f"  Total skipped (vocab):   {total_skipped_vocab:,}\n"
        f"  Total skipped (no pool): {total_skipped_nopool:,}"
    )
    for k in args["eval_k"]:
        hitrate        = float(np.mean(agg_hits[k]))        if agg_hits[k]        else 0.0
        random_hitrate = float(np.mean(agg_random_hits[k])) if agg_random_hits[k] else 0.0
        lift           = (hitrate / random_hitrate) if random_hitrate > 0 else float("inf")

        logger.info(
            f"  HitRate@{k:2d} | SASRec={hitrate:.4f} | "
            f"Random={random_hitrate:.4f} | "
            f"Lift={lift:.2f}x  ({len(agg_hits[k]):,} predictions)"
        )
        rows.append({
            "eval":              "sequential",
            "eval_window":       f"agg_1-{eval2_months}",
            "window_start":      "aggregate",
            "window_end":        "aggregate",
            "K":                 k,
            "n_predictions":     len(agg_hits[k]),
            "hitrate_sasrec":    round(hitrate, 6),
            "hitrate_random":    round(random_hitrate, 6),
            "lift":              round(lift, 4),
            "skipped_vocab":     total_skipped_vocab,
            "skipped_not_pool":  total_skipped_nopool,
            "is_aggregate":      True,
        })

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_eval1(summary_df: pd.DataFrame, out_dir: Path, eval_k: list):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    for metric in ["recall", "ndcg"]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for k in eval_k:
            sub = summary_df[summary_df["K"] == k].sort_values("eval_window")
            ax.plot(sub["eval_window"], sub[metric],
                    marker="o", markersize=4, label=f"K={k}")
        ax.set_xlabel("Eval window")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Eval 1 — SASRec re-rank {metric.upper()}@K per window")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        path = out_dir / f"eval1_{metric}_over_windows.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Plot → {path}")


def plot_eval2(summary_df: pd.DataFrame, out_dir: Path, eval_k: list):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    monthly = summary_df[summary_df["is_aggregate"] == False].copy()
    monthly["eval_window"] = monthly["eval_window"].astype(int)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["steelblue", "tomato", "seagreen", "orange"]

    for i, k in enumerate(eval_k):
        sub = monthly[monthly["K"] == k].sort_values("eval_window")
        c   = colors[i % len(colors)]
        ax.plot(sub["eval_window"], sub["hitrate_sasrec"],
                marker="o", markersize=5, color=c,
                label=f"SASRec @{k}", linewidth=2)
        ax.plot(sub["eval_window"], sub["hitrate_random"],
                marker="s", markersize=4, color=c,
                label=f"Random @{k}", linewidth=1.5,
                linestyle="--", alpha=0.7)

    ax.set_xlabel("Eval window (month)")
    ax.set_ylabel("HitRate")
    ax.set_title("Eval 2 — SASRec vs Random HitRate@K per month")
    ax.legend(ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "eval2_hitrate_per_month.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Plot → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate SASRec re-ranking quality on MF candidates."
    )
    p.add_argument("--start",        type=str,   required=True,
                   help="Training window start YYYY-MM-DD")
    p.add_argument("--end",          type=str,   required=True,
                   help="Training window end   YYYY-MM-DD")
    p.add_argument("--top_n",        type=int,   default=500,
                   help="Top-N MF candidates to use per user")
    p.add_argument("--eval_k",       type=int,   nargs="+", default=[5, 10],
                   help="K values for all metrics")
    p.add_argument("--eval2_months", type=int,   default=2,
                   help="Number of months to use for Eval 2 sequential prediction")
    p.add_argument("--min_rating",   type=float, default=3.0)
    p.add_argument("--mf_root",      type=str,   default="train_logs/MF")
    p.add_argument("--sasrec_root",  type=str,   default="train_logs/SASREC")
    p.add_argument("--data_dir",     type=str,   default="data/ml-32m")
    p.add_argument("--out_root",     type=str,
                   default="eval_results/SASREC_ON_CANDIDATES")
    p.add_argument("--min_eval_ratings", type=int, default=10,
               help="Min ground truth ratings per user to include in eval")
    # ── SASRec architecture (must match the trained model) ──
    p.add_argument("--dim",      type=int,   default=64)
    p.add_argument("--max_len",  type=int,   default=300)
    p.add_argument("--n_heads",  type=int,   default=2)
    p.add_argument("--n_layers", type=int,   default=2)
    p.add_argument("--dropout",  type=float, default=0.2)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── output directory ─────────────────────────────────────────────
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_root)
        / f"{args.start}_{args.end}_{run_tag}_top{args.top_n}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    # add file handler
    fh = logging.FileHandler(out_dir / "eval.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    # ── load DB ──────────────────────────────────────────────────────
    db = MovieLensDB(data_dir=args.data_dir)
    db.load_data()

    # ── find run dirs ────────────────────────────────────────────────
    mf_run_dir     = find_run_dir(Path(args.mf_root),     args.start, args.end, "MF")
    sasrec_run_dir = find_run_dir(Path(args.sasrec_root), args.start, args.end, "SASREC")

    # ── load MF recs ─────────────────────────────────────────────────
    logger.info("Loading MF recs ...")
    mf_recs = load_mf_recs(mf_run_dir, args.top_n)

    # ── load SASRec model + encoder ──────────────────────────────────
    # ── load SASRec model + encoder ──────────────────────────────────────────────
    logger.info("Loading SASRec model ...")
    model, enc = load_sasrec_model(sasrec_run_dir, vars(args), device)

    # eval_args dict — now reads from CLI args directly, no sasrec_args needed
    eval_args = {
        "max_len":      args.max_len,
        "eval_k":       args.eval_k,
        "top_n":        args.top_n,
        "min_rating":   args.min_rating,
        "eval2_months": args.eval2_months,
        "min_eval_ratings": args.min_eval_ratings
    }

    # ── build training sequences ─────────────────────────────────────
    logger.info("Building training sequences ...")
    train_seqs = load_training_sequences(
        db, enc, args.start, args.end, args.min_rating
    )

    # pack args as dict for passing around
    # eval_args = {
    #     "max_len":      sasrec_args["max_len"],
    #     "eval_k":       args.eval_k,
    #     "top_n":        args.top_n,
    #     "min_rating":   args.min_rating,
    #     "eval2_months": args.eval2_months,
    # }

    # ── Eval 1 ───────────────────────────────────────────────────────
    eval1_df = eval1_batch(
        model, enc, db, train_seqs, mf_recs, eval_args, device
    )
    eval1_path = out_dir / "eval1_batch_summary.csv"
    eval1_df.to_csv(eval1_path, index=False)
    logger.info(f"\nEval 1 summary → {eval1_path}")
    plot_eval1(eval1_df, out_dir, args.eval_k)

    # ── Eval 2 ───────────────────────────────────────────────────────
    eval2_df = eval2_sequential(
        model, enc, db, train_seqs, mf_recs, eval_args, device
    )
    eval2_path = out_dir / "eval2_sequential_summary.csv"
    eval2_df.to_csv(eval2_path, index=False)
    logger.info(f"\nEval 2 summary → {eval2_path}")
    plot_eval2(eval2_df, out_dir, args.eval_k)

    logger.info(f"\nAll done → {out_dir}")


if __name__ == "__main__":
    main()