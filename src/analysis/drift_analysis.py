"""
drift_analysis.py
-----------------
Temporal drift analysis: user preference shift + item popularity shift.
Both metrics are measured as CUMULATIVE drift from month_1 baseline.

Usage:
    python src/analysis/drift_analysis.py \
        --start    2018-01-01 \
        --data_dir data/ml-32m \
        --log_root train_logs/MF \
        --min_ratings 10 \
        --top_n_movies 200

What it does:
    1. Finds the run folder in log_root whose name starts with --start
    2. Parses training end date from folder name
    3. Builds 6 training months + 5 eval months
    4. Metric 1 — Item popularity drift:
           Jaccard distance between top-N movies of month_i vs month_1 (baseline)
           1 - Jaccard(top_N_month_i, top_N_month_1)
           Starts at 0, rises as the popular item landscape rotates
    5. Metric 2 — User preference drift:
           Per-user JSD between genre distribution of month_i vs month_1
           Averaged across users active in BOTH month_1 and month_i
           Starts at 0, rises as individual tastes shift
    6. Plot 1 — Item popularity drift curve
    7. Plot 2 — User preference drift curve
    8. Plot 3 — Combined, both curves, vertical freeze line
    Saves all to analysis/drift/<run_folder_name>/
"""

import argparse
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import jensenshannon
from dateutil.relativedelta import relativedelta

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.db_simulator import MovieLensDB

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         14,
    "axes.titlesize":    16,
    "axes.labelsize":    14,
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "legend.fontsize":   13,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

PALETTE = {
    "item":      "#3498DB",
    "user":      "#E67E22",
    "freeze":    "#E74C3C",
    "train_bg":  "#EAF3DE",
    "eval_bg":   "#FAECE7",
}

EVAL_MONTHS = 5


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_run_dir(log_root: Path, start: str) -> Path:
    matches = [d for d in sorted(log_root.iterdir())
               if d.is_dir() and d.name.startswith(start)]
    if not matches:
        raise FileNotFoundError(
            f"No run folder starting with '{start}' found in {log_root}"
        )
    print(f"  Found run folder: {matches[0].name}")
    return matches[0]


def parse_train_end(run_dir: Path) -> str:
    m = re.match(r"\d{4}-\d{2}-\d{2}_(\d{4}-\d{2}-\d{2})", run_dir.name)
    if not m:
        raise ValueError(f"Cannot parse training end from: {run_dir.name}")
    return m.group(1)


def month_windows(start: str, train_end: str, n_eval: int):
    """
    Returns list of (w_start, w_end, label, is_eval).
    Covers training months + n_eval eval months.
    """
    windows = []
    cur     = pd.Timestamp(start).replace(day=1)
    freeze  = pd.Timestamp(train_end)

    while True:
        w_start = cur
        w_end   = cur + relativedelta(months=1) - pd.Timedelta(days=1)
        label   = cur.strftime("%Y-%m")
        is_eval = w_start > freeze
        windows.append((w_start, w_end, label, is_eval))
        cur = cur + relativedelta(months=1)

        eval_count = sum(1 for _, _, _, ie in windows if ie)
        if eval_count >= n_eval:
            break

    return windows


def save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def add_freeze_line(ax, freeze_label: str, labels: list):
    """Vertical dashed line between last training month and first eval month."""
    freeze_x = None
    for i, lbl in enumerate(labels):
        if lbl > freeze_label:
            freeze_x = i - 0.5
            break
    if freeze_x is not None:
        ax.axvline(freeze_x, color=PALETTE["freeze"],
                   linestyle="--", linewidth=2.0, zorder=5)
    return freeze_x


def shade_regions(ax, freeze_x, n):
    if freeze_x is not None:
        ax.axvspan(-0.5,      freeze_x, color=PALETTE["train_bg"], alpha=0.4, zorder=0)
        ax.axvspan(freeze_x,  n - 0.5,  color=PALETTE["eval_bg"],  alpha=0.4, zorder=0)


# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — Item popularity drift (Jaccard distance from baseline)
# ─────────────────────────────────────────────────────────────────────────────

def compute_item_drift(db, windows, min_ratings, top_n):
    """
    For each month, get top_n movies by rating count.
    Measure Jaccard distance between month_i top-N and month_1 top-N (baseline).
    Returns (labels, drift_scores) where drift_scores[0] = 0.0 by definition.
    """
    print(f"\n── Item popularity per month (top {top_n} movies) ───────────────────")

    # compute top-N per month
    monthly_top = []
    for w_start, w_end, label, _ in windows:
        df     = db.get_ratings_by_daterange(
            w_start.strftime("%Y-%m-%d"),
            w_end.strftime("%Y-%m-%d")
        )
        counts = df.groupby("movieId").size()
        counts = counts[counts >= min_ratings]
        top_n_set = set(counts.nlargest(top_n).index.tolist())
        monthly_top.append((label, top_n_set))
        print(f"    {label}: {len(counts):,} qualifying movies, top-{top_n} captured")

    # baseline = month_1
    baseline_label, baseline_set = monthly_top[0]
    print(f"\n  Baseline = {baseline_label} ({len(baseline_set)} movies)")

    labels = []
    scores = []

    for label, top_set in monthly_top:
        if len(top_set) == 0 or len(baseline_set) == 0:
            scores.append(np.nan)
        else:
            intersection = len(top_set & baseline_set)
            union        = len(top_set | baseline_set)
            jaccard_sim  = intersection / union
            jaccard_dist = 1.0 - jaccard_sim   # drift score: 0 = identical, 1 = no overlap
            scores.append(jaccard_dist)
        labels.append(label)
        print(f"    {label} vs baseline: Jaccard dist = {scores[-1]:.4f}")

    return labels, scores


# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 — User preference drift (per-user JSD from baseline)
# ─────────────────────────────────────────────────────────────────────────────

def get_user_genre_dists(db, w_start, w_end, movies_df, all_genres, min_ratings):
    """
    Returns dict: userId -> normalised genre distribution array
    for users with >= min_ratings in the given window.
    """
    df = db.get_ratings_by_daterange(
        w_start.strftime("%Y-%m-%d"),
        w_end.strftime("%Y-%m-%d")
    )
    user_counts  = df.groupby("userId").size()
    active_users = user_counts[user_counts >= min_ratings].index
    df           = df[df["userId"].isin(active_users)]

    if df.empty:
        return {}

    merged = df.merge(movies_df[["movieId", "genres"]], on="movieId", how="left")
    merged = merged.dropna(subset=["genres"])
    merged = merged.explode("genres")
    merged = merged[merged["genres"] != "(no genres listed)"]

    user_dists = {}
    for uid, grp in merged.groupby("userId"):
        counts = grp["genres"].value_counts()
        dist   = np.array([counts.get(g, 0) for g in all_genres], dtype=np.float64)
        total  = dist.sum()
        if total > 0:
            user_dists[uid] = dist / total

    return user_dists


def compute_user_drift(db, windows, movies_df, min_ratings):
    """
    For each month, compute per-user genre distribution.
    Measure average JSD between month_i distribution and month_1 (baseline) distribution.
    Only users active in BOTH month_1 AND month_i are included.
    Returns (labels, drift_scores) where drift_scores[0] = 0.0 by definition.
    """
    all_genres = sorted(set(
        g for genres in movies_df["genres"].dropna()
        for g in genres if g != "(no genres listed)"
    ))

    print(f"\n── User genre preference per month ──────────────────────────────────")

    # compute per-user genre dist for all months
    monthly_dists = []
    for w_start, w_end, label, _ in windows:
        dists = get_user_genre_dists(db, w_start, w_end, movies_df, all_genres, min_ratings)
        monthly_dists.append((label, dists))
        print(f"    {label}: {len(dists):,} active users")

    # baseline = month_1
    baseline_label, baseline_dists = monthly_dists[0]
    baseline_users = set(baseline_dists.keys())
    print(f"\n  Baseline = {baseline_label} ({len(baseline_users)} users)")

    labels = []
    scores = []

    for label, month_dists in monthly_dists:
        # only users active in both baseline and this month
        common_users = baseline_users & set(month_dists.keys())

        if len(common_users) < 10:
            scores.append(np.nan)
            labels.append(label)
            print(f"    {label}: not enough common users ({len(common_users)}), skipping")
            continue

        jsds = []
        for uid in common_users:
            d1  = baseline_dists[uid]
            d2  = month_dists[uid]
            jsd = jensenshannon(d1, d2, base=2) ** 2
            jsds.append(jsd)

        mean_jsd = float(np.mean(jsds))
        scores.append(mean_jsd)
        labels.append(label)
        print(f"    {label} vs baseline: mean JSD = {mean_jsd:.6f} "
              f"(n={len(common_users)} common users)")

    return labels, scores


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _build_legend_handles(color, metric_label):
    train_patch = mpatches.Patch(color=PALETTE["train_bg"], alpha=0.7, label="training window")
    eval_patch  = mpatches.Patch(color=PALETTE["eval_bg"],  alpha=0.7, label="eval window")
    return [
        plt.Line2D([0], [0], color=color, marker="o", linewidth=2.5, label=metric_label),
        plt.Line2D([0], [0], color=PALETTE["freeze"], linestyle="--",
                   linewidth=2.0, label="model frozen (embedding fixed)"),
        train_patch, eval_patch
    ]


def _annotate(ax, x, scores):
    for xi, v in zip(x, scores):
        if not np.isnan(v):
            offset = max([s for s in scores if not np.isnan(s)]) * 0.02
            ax.text(xi, v + offset, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=10)


def plot_item_drift(labels, scores, freeze_label, out_dir):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        "Item Popularity Drift — Cumulative Jaccard Distance from Baseline (Month 1)",
        fontsize=16, fontweight="bold"
    )

    x        = np.arange(len(labels))
    freeze_x = add_freeze_line(ax, freeze_label, labels)
    shade_regions(ax, freeze_x, len(labels))

    ax.plot(x, scores, color=PALETTE["item"],
            marker="o", markersize=8, linewidth=2.5, zorder=4)
    _annotate(ax, x, scores)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Jaccard Distance (0 = identical to baseline)")
    ax.set_ylim(-0.02, max(s for s in scores if not np.isnan(s)) * 1.2)
    ax.legend(handles=_build_legend_handles(PALETTE["item"], "Jaccard distance"), fontsize=12)

    fig.text(0.5, -0.02,
             "Rising curve = the set of popular movies is rotating away from what the model was trained on.",
             ha="center", fontsize=12, style="italic", color="#555")
    plt.tight_layout()
    save_fig(fig, out_dir, "item_popularity_drift.png")


def plot_user_drift(labels, scores, freeze_label, out_dir):
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        "User Preference Drift — Avg Per-User JSD from Baseline (Month 1)",
        fontsize=16, fontweight="bold"
    )

    x        = np.arange(len(labels))
    freeze_x = add_freeze_line(ax, freeze_label, labels)
    shade_regions(ax, freeze_x, len(labels))

    ax.plot(x, scores, color=PALETTE["user"],
            marker="o", markersize=8, linewidth=2.5, zorder=4)
    _annotate(ax, x, scores)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Mean per-user JSD (0 = no change from baseline)")
    ax.set_ylim(-0.002, max(s for s in scores if not np.isnan(s)) * 1.2)
    ax.legend(handles=_build_legend_handles(PALETTE["user"], "mean JSD"), fontsize=12)

    fig.text(0.5, -0.02,
             "Rising curve = individual users are watching different genres compared to when the model was trained.",
             ha="center", fontsize=12, style="italic", color="#555")
    plt.tight_layout()
    save_fig(fig, out_dir, "user_preference_drift.png")


def plot_combined(item_labels, item_scores,
                  user_labels, user_scores,
                  freeze_label, out_dir):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=False)
    fig.suptitle(
        "Temporal Drift Analysis — Cumulative Drift from Training Baseline",
        fontsize=17, fontweight="bold"
    )

    # ── top: item drift ───────────────────────────────────────────────────────
    x1       = np.arange(len(item_labels))
    freeze_x1 = add_freeze_line(ax1, freeze_label, item_labels)
    shade_regions(ax1, freeze_x1, len(item_labels))
    ax1.plot(x1, item_scores, color=PALETTE["item"],
             marker="o", markersize=8, linewidth=2.5, zorder=4)
    _annotate(ax1, x1, item_scores)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(item_labels, rotation=30, ha="right", fontsize=11)
    ax1.set_ylabel("Jaccard Distance")
    ax1.set_ylim(-0.02, max(s for s in item_scores if not np.isnan(s)) * 1.2)
    ax1.set_title("Item Popularity Drift (vs Month 1 baseline)", fontweight="bold")
    ax1.legend(handles=_build_legend_handles(PALETTE["item"], "Jaccard distance"), fontsize=11)

    # ── bottom: user drift ────────────────────────────────────────────────────
    x2        = np.arange(len(user_labels))
    freeze_x2 = add_freeze_line(ax2, freeze_label, user_labels)
    shade_regions(ax2, freeze_x2, len(user_labels))
    ax2.plot(x2, user_scores, color=PALETTE["user"],
             marker="o", markersize=8, linewidth=2.5, zorder=4)
    _annotate(ax2, x2, user_scores)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(user_labels, rotation=30, ha="right", fontsize=11)
    ax2.set_ylabel("Mean per-user JSD")
    ax2.set_ylim(-0.002, max(s for s in user_scores if not np.isnan(s)) * 1.2)
    ax2.set_title("User Genre Preference Drift (vs Month 1 baseline)", fontweight="bold")
    ax2.legend(handles=_build_legend_handles(PALETTE["user"], "mean JSD"), fontsize=11)

    fig.text(
        0.5, -0.01,
        "Both curves start at 0 (month 1 = baseline).  "
        "Green = training window  ·  Red = eval window  ·  "
        "Dashed line = embeddings frozen.",
        ha="center", fontsize=12, style="italic", color="#555"
    )
    plt.tight_layout()
    save_fig(fig, out_dir, "drift_combined.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Temporal drift analysis")
    p.add_argument("--start",        type=str, required=True,           help="Training start date YYYY-MM-DD")
    p.add_argument("--data_dir",     type=str, default="data/ml-32m",   help="MovieLens data directory")
    p.add_argument("--log_root",     type=str, default="train_logs/MF", help="Root MF run folder")
    p.add_argument("--min_ratings",  type=int, default=10,              help="Min ratings per user/movie per month")
    p.add_argument("--top_n_movies", type=int, default=200,             help="Top-N movies for Jaccard popularity drift")
    return p.parse_args()


def main():
    args = parse_args()

    # ── find run ──────────────────────────────────────────────────────────────
    log_root = Path(args.log_root)
    print(f"\nSearching for run starting with {args.start} in {log_root} ...")
    run_dir   = find_run_dir(log_root, args.start)
    train_end = parse_train_end(run_dir)
    print(f"  Training window: {args.start} → {train_end}")

    # ── output ────────────────────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir      = project_root / "analysis" / "drift" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {out_dir}\n")

    # ── month windows ─────────────────────────────────────────────────────────
    windows = month_windows(args.start, train_end, EVAL_MONTHS)
    print(f"  Months: {len(windows)} total")
    for w_start, w_end, label, is_eval in windows:
        tag = "EVAL " if is_eval else "TRAIN"
        print(f"    [{tag}] {label}: {w_start.date()} → {w_end.date()}")

    freeze_label = max(lbl for _, _, lbl, ie in windows if not ie)
    print(f"\n  Freeze line after: {freeze_label}")

    # ── load data ─────────────────────────────────────────────────────────────
    print("\nLoading raw data ...")
    db = MovieLensDB(data_dir=args.data_dir)
    db.load_data()
    movies_df = db.movies_df.copy()

    # ── compute metrics ───────────────────────────────────────────────────────
    item_labels, item_scores = compute_item_drift(
        db, windows, args.min_ratings, args.top_n_movies
    )
    user_labels, user_scores = compute_user_drift(
        db, windows, movies_df, args.min_ratings
    )

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\n── Saving plots ─────────────────────────────────────────────────────")
    plot_item_drift(item_labels, item_scores, freeze_label, out_dir)
    plot_user_drift(user_labels, user_scores, freeze_label, out_dir)
    plot_combined(item_labels, item_scores,
                  user_labels, user_scores,
                  freeze_label, out_dir)

    print(f"\n✅ All drift plots saved to: {out_dir}")


if __name__ == "__main__":
    main()