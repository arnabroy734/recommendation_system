"""
mf_sasrec_comparison.py
-----------------------
Side-by-side analysis of MF and SASRec recommendation quality.

Analyses:
  1. Length-wise NDCG@10 — users bucketed by training-window rating count
     [20-50, 50-100, 100-150, 150+], NDCG@10 from eval window 1,
     grouped bar chart MF vs SASRec per bucket.

  2. Genre-shift users — find top-3 users with highest JSD between
     first-half and second-half genre distributions in training window.
     For each user produce two separate figures:
       - MF:    watch history | top-20 recs coloured by genre
       - SASRec: watch history | top-20 recs coloured by genre

Usage (from project root):
    python src/analysis/mf_sasrec_comparison.py \
        --start 2018-01-01 \
        --end   2018-06-30 \
        --data_dir data/ml-32m
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.db_simulator import MovieLensDB
from src.training.matrix_factorisation import Encoder   

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────
FONT_SIZE  = 14
TOP_N_RECS = 20
BRACKETS   = [(5, 20), (20, 50), (50, 100), (100, 150), (150, int(1e9))]
# BRACKETS   = [(10, 10**10)]

BRACKET_LABELS = [" < 20","20-50", "50-100", "100-150", "150+"]
# BRACKET_LABELS = ["All"]

GENRE_COLORS = {
    "Action":    "#E24B4A",
    "Romance":   "#FF69B4",
    "Sci-Fi":    "#378ADD",
    "Thriller":  "#BA7517",
    "Horror":    "#1D9E75",
    "Comedy":    "#FF7F00",
    "Drama":     "#8A2BE2",
    "Adventure": "#A65628",
    "Animation": "#F781BF",
    "Crime":     "#4DAF4A",
    "Other":     "#AAAAAA",
}

ALL_GENRES = list(GENRE_COLORS.keys())


# ══════════════════════════════════════════════════════════════════════════
# Run folder helpers
# ══════════════════════════════════════════════════════════════════════════

def find_latest_bpr_run(log_root: Path, start: str, end: str) -> Path:
    """
    Find the latest BPR run folder matching {start}_{end} pattern.
    Folder name format: {start}_{end}_{timestamp}_bpr_dim{dim}
    """
    pattern = f"{start}_{end}"
    matches = [
        d for d in log_root.iterdir()
        if d.is_dir()
        and d.name.startswith(pattern)
        and "bpr" in d.name.lower()
        and (d / "recs_window_01.csv").exists()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No BPR run matching '{pattern}' with recs_window_01.csv in {log_root}"
        )
    # sort by folder name — timestamp is embedded so lexicographic = chronological
    latest = sorted(matches)[-1]
    logger.info(f"  Found run: {latest.name}")
    return latest


def load_recs(run_dir: Path) -> pd.DataFrame:
    """Load recs_window_01.csv and keep top-20 per user."""
    recs_df = pd.read_csv(run_dir / "recs_window_01.csv")
    recs_df = (
        recs_df.sort_values("rank")
               .groupby("userId")
               .head(TOP_N_RECS)
               .reset_index(drop=True)
    )
    return recs_df


def load_encoder(run_dir: Path) -> Encoder:
    with open(run_dir / "encoder.pkl", "rb") as f:
        return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════
# Genre helpers
# ══════════════════════════════════════════════════════════════════════════

def build_movie_genre_lookup(db: MovieLensDB) -> dict:
    lookup = {}
    for _, row in db.movies_df.iterrows():
        genres = row["genres"]
        if isinstance(genres, list):
            lookup[row["movieId"]] = [
                g for g in genres
                if g not in ("(no genres listed)", "Unknown")
            ]
        else:
            lookup[row["movieId"]] = []
    return lookup


def primary_genre(movie_id: int, lookup: dict,
                  target: str = None) -> str:
    genres = lookup.get(movie_id, [])
    if not genres:
        return "Other"
    if target and target in genres:
        return target
    return genres[0]


def genre_distribution(movie_ids, lookup: dict) -> dict:
    """Return normalised genre distribution over a list of movie ids."""
    counts = {}
    for mid in movie_ids:
        for g in lookup.get(mid, []):
            if g not in ("(no genres listed)", "Unknown"):
                counts[g] = counts.get(g, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {}
    return {g: c / total for g, c in counts.items()}


def dist_to_vector(dist: dict, genres: list) -> np.ndarray:
    """Convert genre distribution dict to fixed-length numpy vector."""
    v = np.array([dist.get(g, 0.0) for g in genres], dtype=np.float64)
    s = v.sum()
    return v / s if s > 0 else v


# ══════════════════════════════════════════════════════════════════════════
# NDCG@K helper
# ══════════════════════════════════════════════════════════════════════════

def ndcg_at_k(recommended: list, ground_truth: set, k: int) -> float:
    top_k = recommended[:k]
    idcg  = sum(1.0 / np.log2(r + 2) for r in range(min(len(ground_truth), k)))
    if idcg == 0:
        return 0.0
    dcg = sum(
        1.0 / np.log2(r + 2)
        for r, item in enumerate(top_k)
        if item in ground_truth
    )
    return dcg / idcg


# ══════════════════════════════════════════════════════════════════════════
# Analysis 1 — Length-wise NDCG@10
# ══════════════════════════════════════════════════════════════════════════

def compute_length_ndcg(
    recs_df,
    enc,
    train_df,
    db,
    start,
    end,
    k=10,
    min_eval_ratings=10,
):

    ws = recs_df["window_start"].iloc[0]
    we = recs_df["window_end"].iloc[0]

    eval_raw = db.get_ratings_by_daterange(ws, we)

    # ✅ match training
    eval_raw = eval_raw[eval_raw["rating"] >= 3.0]
    eval_raw = eval_raw[eval_raw["userId"].isin(enc.user_id)]
    eval_raw = eval_raw[eval_raw["movieId"].isin(enc.item_id)]

    # ✅ user filtering
    user_eval_counts = eval_raw.groupby("userId")["movieId"].count()
    qualifying_users = user_eval_counts[user_eval_counts >= min_eval_ratings].index

    # GT
    gt_per_user = {
        uid: set(grp["movieId"].values)
        for uid, grp in eval_raw.groupby("userId")
    }

    # training counts
    train_counts = train_df.groupby("userId").size().to_dict()

    rows = []

    for uid, grp in recs_df.groupby("userId"):

        if uid not in qualifying_users:
            continue

        if uid not in gt_per_user:
            continue

        gt = gt_per_user[uid]

        rec_ids = grp.sort_values("rank")["movieId"].tolist()

        # NDCG
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), k)))
        if idcg == 0:
            continue

        dcg = sum(
            1.0 / np.log2(r + 2)
            for r, item in enumerate(rec_ids[:k])
            if item in gt
        )

        rows.append({
            "userId": uid,
            "training_count": train_counts.get(uid, 0),
            "ndcg": dcg / idcg,
        })

    return pd.DataFrame(rows)


# def compute_length_ndcg(
#     recs_df: pd.DataFrame,
#     enc: Encoder,
#     train_df: pd.DataFrame,
#     db: MovieLensDB,
#     start: str,
#     end: str,
#     k: int = 10,
# ) -> pd.DataFrame:
#     """
#     For each user compute:
#       - training_count : number of ratings in training window
#       - ndcg@k         : from recs_window_01
#     Returns dataframe with columns: userId, training_count, ndcg
#     """
#     # ground truth = ratings in eval window (window_start, window_end from recs)
#     ws = recs_df["window_start"].iloc[0]
#     we = recs_df["window_end"].iloc[0]
#     logger.info(f"  Ground truth window: {ws} → {we}")

#     eval_raw = db.get_ratings_by_daterange(ws, we)
#     eval_raw = eval_raw[eval_raw["userId"].isin(enc.user_id)]
#     eval_raw = eval_raw[eval_raw["movieId"].isin(enc.item_id)]
#     gt_per_user = {
#         uid: set(grp["movieId"].values)
#         for uid, grp in eval_raw.groupby("userId")
#     }

#     # training counts per user (in training window only)
#     train_counts = train_df.groupby("userId").size().to_dict()

#     rows = []
#     for uid, grp in recs_df.groupby("userId"):
#         if uid not in gt_per_user:
#             continue
#         gt      = gt_per_user[uid]
#         rec_ids = grp.sort_values("rank")["movieId"].tolist()
#         score   = ndcg_at_k(rec_ids, gt, k)
#         rows.append({
#             "userId":         uid,
#             "training_count": train_counts.get(uid, 0),
#             "ndcg":           score,
#         })

#     return pd.DataFrame(rows)


def plot_length_ndcg_comparison(
    mf_df: pd.DataFrame,
    sasr_df: pd.DataFrame,
    out_path: Path,
    start: str,
    end: str,
    k: int = 10,
):
    """Grouped bar chart: MF vs SASRec NDCG@K per rating-count bucket."""

    def bucket_mean(df):
        means = []
        counts = []
        for lo, hi in BRACKETS:
            sub = df[(df["training_count"] >= lo) & (df["training_count"] < hi)]
            means.append(sub["ndcg"].mean() if len(sub) > 0 else 0.0)
            counts.append(len(sub))
        return means, counts

    mf_means,   mf_counts   = bucket_mean(mf_df)
    sasr_means, sasr_counts = bucket_mean(sasr_df)

    x     = np.arange(len(BRACKETS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_mf   = ax.bar(x - width/2, mf_means,   width,
                        label="MF (BPR)",  color="#378ADD",
                        edgecolor="black", linewidth=0.6)
    bars_sasr = ax.bar(x + width/2, sasr_means, width,
                        label="SASRec",    color="#E24B4A",
                        edgecolor="black", linewidth=0.6)

    # annotate bars with value + user count
    for bars, means, cnts in [
        (bars_mf,   mf_means,   mf_counts),
        (bars_sasr, sasr_means, sasr_counts),
    ]:
        for bar, val, cnt in zip(bars, means, cnts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}\n(n={cnt})",
                ha="center", va="bottom",
                fontsize=FONT_SIZE - 3, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lbl}\nratings" for lbl in BRACKET_LABELS],
        fontsize=FONT_SIZE - 1,
    )
    ax.set_ylabel(f"Mean NDCG@{k}", fontsize=FONT_SIZE)
    ax.set_title(
        f"NDCG@{k} by user activity level — MF vs SASRec\n"
        f"Training window: {start} → {end}",
        fontsize=FONT_SIZE + 1,
    )
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(mf_means), max(sasr_means)) * 1.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════
# Analysis 2 — Genre-shift user detection
# ══════════════════════════════════════════════════════════════════════════

def find_genre_shift_users(
    train_df: pd.DataFrame,
    enc: Encoder,
    lookup: dict,
    n_users: int = 3,
    min_ratings: int = 20,
) -> list:
    """
    Split each user's training history into first and second half
    chronologically. Compute JSD between genre distributions.
    Return top-n_users with highest JSD — these are genre shifters.

    Returns list of dicts:
        {userId, jsd, first_half_dist, second_half_dist, history}
    """
    all_genres = sorted({
        g for mid in lookup for g in lookup[mid]
    })

    results = []
    encoded_users = set(enc.user_id.keys())

    for uid, grp in train_df.groupby("userId"):
        if uid not in encoded_users:
            continue
        grp = grp.sort_values("timestamp")
        if len(grp) < min_ratings:
            continue

        mid_point  = len(grp) // 2
        first_ids  = grp.iloc[:mid_point]["movieId"].tolist()
        second_ids = grp.iloc[mid_point:]["movieId"].tolist()

        first_dist  = genre_distribution(first_ids,  lookup)
        second_dist = genre_distribution(second_ids, lookup)

        if not first_dist or not second_dist:
            continue

        p = dist_to_vector(first_dist,  all_genres)
        q = dist_to_vector(second_dist, all_genres)

        # add small epsilon to avoid zero-division in JSD
        p = p + 1e-9;  p /= p.sum()
        q = q + 1e-9;  q /= q.sum()

        jsd = float(jensenshannon(p, q))

        results.append({
            "userId":          uid,
            "jsd":             jsd,
            "first_dist":      first_dist,
            "second_dist":     second_dist,
            "history":         grp["movieId"].tolist(),
            "n_ratings":       len(grp),
        })

    results.sort(key=lambda x: x["jsd"], reverse=True)
    top = results[:n_users]
    for r in top:
        logger.info(
            f"  Genre-shift user {r['userId']} | "
            f"JSD={r['jsd']:.4f} | n_ratings={r['n_ratings']}"
        )
    return top


# ══════════════════════════════════════════════════════════════════════════
# Analysis 2 — Case study plot (one model)
# ══════════════════════════════════════════════════════════════════════════

def plot_genre_shift_case_study(
    user_info: dict,
    recs_df: pd.DataFrame,
    db: MovieLensDB,
    lookup: dict,
    model_name: str,
    start: str,
    end: str,
    out_path: Path,
):
    """
    Three-panel figure:
      Left   : first-half genre distribution (bar)
      Middle : second-half genre distribution (bar)
      Right  : top-20 recommendations coloured by genre (horizontal bars)
    """
    uid      = user_info["userId"]
    jsd      = user_info["jsd"]
    user_recs = recs_df[recs_df["userId"] == uid].sort_values("rank").head(TOP_N_RECS)

    if user_recs.empty:
        logger.warning(f"  No recs for user {uid} in {model_name}, skipping.")
        return

    # ── genre distributions ───────────────────────────────────────────────
    def top_genres_bar(dist, ax, title):
        top = sorted(dist, key=dist.get, reverse=True)[:10]
        vals   = [dist[g] for g in top]
        colors = [GENRE_COLORS.get(g, GENRE_COLORS["Other"]) for g in top]
        bars   = ax.bar(top, vals, color=colors,
                        edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f"{v:.2f}",
                ha="center", va="bottom",
                fontsize=FONT_SIZE - 4,
            )
        ax.set_title(title, fontsize=FONT_SIZE - 1)
        ax.set_ylabel("Genre fraction", fontsize=FONT_SIZE - 2)
        ax.tick_params(axis="x", rotation=35, labelsize=FONT_SIZE - 4)
        ax.tick_params(axis="y", labelsize=FONT_SIZE - 4)
        ax.grid(axis="y", alpha=0.3)

    # ── rec list ──────────────────────────────────────────────────────────
    rec_rows = []
    for _, row in user_recs.iterrows():
        mid    = row["movieId"]
        title  = db.movies_df[db.movies_df["movieId"] == mid]["title"].values
        title  = title[0][:38] if len(title) else str(mid)
        pg     = primary_genre(mid, lookup)
        rec_rows.append({
            "rank":   int(row["rank"]),
            "title":  title,
            "genre":  pg,
        })
    rec_df = pd.DataFrame(rec_rows)

    # ── figure ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    top_genres_bar(
        user_info["first_dist"], axes[0],
        f"First-half genre dist\n(user {uid})"
    )
    top_genres_bar(
        user_info["second_dist"], axes[1],
        f"Second-half genre dist\n(JSD = {jsd:.4f})"
    )

    # recs panel
    ax   = axes[2]
    y    = np.arange(len(rec_df))
    cols = [GENRE_COLORS.get(r["genre"], GENRE_COLORS["Other"])
            for _, r in rec_df.iterrows()]
    ax.barh(y, np.ones(len(rec_df)), color=cols,
            edgecolor="black", linewidth=0.4, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r['rank']:2d}. {r['title']}" for _, r in rec_df.iterrows()],
        fontsize=FONT_SIZE - 4,
    )
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_title(
        f"Top-{TOP_N_RECS} recommendations\n(colour = primary genre)",
        fontsize=FONT_SIZE - 1,
    )

    # legend
    seen = rec_df["genre"].unique()
    handles = [
        plt.Rectangle((0, 0), 1, 1,
                       color=GENRE_COLORS.get(g, GENRE_COLORS["Other"]))
        for g in seen
    ]
    ax.legend(handles, seen, fontsize=FONT_SIZE - 4,
              loc="lower right", title="Genre",
              title_fontsize=FONT_SIZE - 3)

    fig.suptitle(
        f"Genre-shift analysis — {model_name} | user {uid}\n"
        f"Training: {start} → {end} | n_ratings={user_info['n_ratings']}",
        fontsize=FONT_SIZE + 1, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="MF vs SASRec comparison analysis"
    )
    p.add_argument("--start",      type=str, required=True,
                   help="Training window start YYYY-MM-DD")
    p.add_argument("--end",        type=str, required=True,
                   help="Training window end   YYYY-MM-DD")
    p.add_argument("--data_dir",   type=str, default="data/ml-32m")
    p.add_argument("--mf_root",    type=str, default="train_logs/MF",
                   help="Root folder for MF runs")
    p.add_argument("--sasr_root",  type=str, default="train_logs/SASREC",
                   help="Root folder for SASRec runs")
    p.add_argument("--out_root",   type=str, default="analysis/comparison",
                   help="Output root folder")
    p.add_argument("--ndcg_k",     type=int, default=10)
    p.add_argument("--n_shift",    type=int, default=3,
                   help="Number of genre-shift users to analyse")
    p.add_argument("--min_ratings",type=int, default=100,
                   help="Min training ratings for genre-shift detection")
    return p.parse_args()


def main():
    args     = parse_args()
    out_root = Path(args.out_root) / f"{args.start}_{args.end}"
    out_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output dir: {out_root}")

    # ── find run folders ──────────────────────────────────────────────────
    logger.info("Finding MF run ...")
    mf_run   = find_latest_bpr_run(Path(args.mf_root),   args.start, args.end)
    logger.info("Finding SASRec run ...")
    sasr_run = find_latest_bpr_run(Path(args.sasr_root), args.start, args.end)

    # ── load encoders and recs ────────────────────────────────────────────
    mf_enc   = load_encoder(mf_run)
    sasr_enc = load_encoder(sasr_run)
    mf_recs  = load_recs(mf_run)
    sasr_recs = load_recs(sasr_run)

    logger.info(
        f"MF   recs: {mf_recs['userId'].nunique():,} users | "
        f"{len(mf_recs):,} rows"
    )
    logger.info(
        f"SASRec recs: {sasr_recs['userId'].nunique():,} users | "
        f"{len(sasr_recs):,} rows"
    )

    # ── load DB ───────────────────────────────────────────────────────────
    logger.info("Loading MovieLens DB ...")
    db = MovieLensDB(data_dir=args.data_dir)
    db.load_data()

    lookup   = build_movie_genre_lookup(db)
    train_df = db.get_ratings_by_daterange(args.start, args.end)
    train_df = train_df[train_df["rating"] >= 3.0]
    logger.info(f"Training interactions loaded: {len(train_df):,}")

    # ── Analysis 1: length-wise NDCG ─────────────────────────────────────
    logger.info("\n── Analysis 1: Length-wise NDCG@10 ─────────────────────────")

    mf_ndcg = compute_length_ndcg(
        mf_recs, mf_enc, train_df, db,
        args.start, args.end, k=args.ndcg_k,
    )
    sasr_ndcg = compute_length_ndcg(
        sasr_recs, sasr_enc, train_df, db,
        args.start, args.end, k=args.ndcg_k,
    )

    plot_length_ndcg_comparison(
        mf_ndcg, sasr_ndcg,
        out_root / "length_wise_ndcg.png",
        args.start, args.end, k=args.ndcg_k,
    )

    # ── Analysis 2: genre-shift users ────────────────────────────────────
    logger.info("\n── Analysis 2: Genre-shift users ───────────────────────────")

    # use MF encoder's users as reference pool — both models share same users
    shift_users = find_genre_shift_users(
        train_df    = train_df,
        enc         = mf_enc,
        lookup      = lookup,
        n_users     = args.n_shift,
        min_ratings = args.min_ratings,
    )

    if not shift_users:
        logger.warning("No genre-shift users found — try lowering --min_ratings")
    else:
        for user_info in shift_users:
            uid = user_info["userId"]

            # MF plot
            plot_genre_shift_case_study(
                user_info  = user_info,
                recs_df    = mf_recs,
                db         = db,
                lookup     = lookup,
                model_name = "MF (BPR)",
                start      = args.start,
                end        = args.end,
                out_path   = out_root / f"genre_shift_user{uid}_mf.png",
            )

            # SASRec plot
            plot_genre_shift_case_study(
                user_info  = user_info,
                recs_df    = sasr_recs,
                db         = db,
                lookup     = lookup,
                model_name = "SASRec",
                start      = args.start,
                end        = args.end,
                out_path   = out_root / f"genre_shift_user{uid}_sasrec.png",
            )

    logger.info(f"\nAll done. Outputs in: {out_root}")


if __name__ == "__main__":
    main()