"""
bias_analysis.py
----------------
Bias term analysis for a trained MF model run.

Usage:
    python src/analysis/bias_analysis.py \
        --run_dir  train_logs/MF/2018-01-01_2018-06-30_20240815_bpr_dim64 \
        --data_dir data/ml-32m \
        --top_n    20

Outputs (saved to analysis/bias/<run_folder_name>/):
    1. user_bias_vs_raw.png     — Top/Bottom N users: model bias vs actual avg rating
    2. item_bias_vs_raw.png     — Top/Bottom N movies: model bias vs actual avg rating
    3. genre_bias_vs_raw.png    — Avg model bias per genre vs avg raw rating per genre (side by side)
    4. item_bias_vs_popularity.png — Scatter: item bias vs rating count (log scale)
"""

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── project root on path ──────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.db_simulator import MovieLensDB
from src.training.matrix_factorisation import Encoder

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          14,
    "axes.titlesize":     16,
    "axes.labelsize":     14,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "legend.fontsize":    13,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#F8F9FA",
    "axes.grid":          True,
    "grid.color":         "white",
    "grid.linewidth":     1.2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

PALETTE = {
    "pos":      "#2ECC71",   # high bias  — green
    "neg":      "#E74C3C",   # low bias   — red
    "model":    "#3498DB",   # model bars — blue
    "raw":      "#E67E22",   # raw bars   — orange
    "scatter":  "#9B59B6",   # scatter    — purple
    "mean":     "#E74C3C",   # mean line  — red
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_window_from_folder(run_dir: Path):
    """Extract start and end dates from folder name like 2018-01-01_2018-06-30_..."""
    name = run_dir.name
    m = re.match(r"(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})", name)
    if not m:
        raise ValueError(f"Cannot parse date window from folder name: {name}")
    return m.group(1), m.group(2)


def load_artifacts(run_dir: Path):
    """Load embeddings, biases, mu, and encoder from run directory."""
    b_u     = np.load(run_dir / "user_bias.npy")
    b_i     = np.load(run_dir / "item_bias.npy")
    mu_arr  = np.load(run_dir / "mu.npy")
    mu      = float(mu_arr[0])
    with open(run_dir / "encoder.pkl", "rb") as f:
        enc = pickle.load(f)
    return b_u, b_i, mu, enc


def wrap_title(title: str, maxlen: int = 35) -> str:
    """Truncate long movie titles for display."""
    return title if len(title) <= maxlen else title[:maxlen - 1] + "…"


def save_fig(fig, path: Path, name: str):
    path.mkdir(parents=True, exist_ok=True)
    fpath = path / name
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — User bias vs raw average rating
# ─────────────────────────────────────────────────────────────────────────────

def plot_user_bias_vs_raw(b_u, enc, train_df, mu, top_n, out_dir, window_str):
    print("Building user bias vs raw plot...")

    # ── model top/bottom users ────────────────────────────────────────────────
    top_idx    = np.argsort(b_u)[::-1][:top_n]
    bottom_idx = np.argsort(b_u)[:top_n]

    def build_rows(indices, label):
        rows = []
        for idx in indices:
            uid   = enc.id_user[idx]
            bias  = float(b_u[idx])
            raw   = train_df[train_df["userId"] == uid]["rating"]
            avg_r = float(raw.mean()) if len(raw) > 0 else np.nan
            n     = len(raw)
            rows.append({"uid": uid, "bias": bias, "avg_rating": avg_r, "n": n, "group": label})
        return rows

    rows   = build_rows(top_idx, "high") + build_rows(bottom_idx, "low")
    df_plt = pd.DataFrame(rows)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"User Bias Analysis  |  Training window: {window_str}",
        fontsize=18, fontweight="bold", y=1.01
    )

    def bar_panel(ax, sub, col, ylabel, title, color, show_labels=True):
        labels = [f"u{r['uid']}" for _, r in sub.iterrows()]
        vals   = sub[col].values
        bars   = ax.bar(labels if show_labels else [""] * len(vals),
                        vals, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.axhline(mu if col == "avg_rating" else 0,
                   color=PALETTE["mean"], linestyle="--", linewidth=1.5,
                   label="global mean" if col == "avg_rating" else "zero bias")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=11)
        ax.legend(fontsize=11)
        # annotate
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(vals) - min(vals)) * 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9)

    high = df_plt[df_plt["group"] == "high"].reset_index(drop=True)
    low  = df_plt[df_plt["group"] == "low"].reset_index(drop=True)

    bar_panel(axes[0, 0], high, "bias",       "Model Bias (b_u)",    f"Top {top_n} Users — Model Bias",         PALETTE["pos"], show_labels=True)
    bar_panel(axes[0, 1], high, "avg_rating", "Avg Rating (raw)",    f"Top {top_n} Users — Raw Avg Rating",      PALETTE["pos"], show_labels=False)
    bar_panel(axes[1, 0], low,  "bias",       "Model Bias (b_u)",    f"Bottom {top_n} Users — Model Bias",       PALETTE["neg"], show_labels=True)
    bar_panel(axes[1, 1], low,  "avg_rating", "Avg Rating (raw)",    f"Bottom {top_n} Users — Raw Avg Rating",   PALETTE["neg"], show_labels=False)

    # ── annotation explaining what to look for ────────────────────────────────
    fig.text(0.5, -0.01,
             "Validation: users with high model bias (top-left) should show higher raw avg ratings (top-right), and vice versa.",
             ha="center", fontsize=13, style="italic", color="#555555")

    plt.tight_layout()
    save_fig(fig, out_dir, "user_bias_vs_raw.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Item bias vs raw average rating
# ─────────────────────────────────────────────────────────────────────────────

def plot_item_bias_vs_raw(b_i, enc, train_df, movies_df, mu, top_n, out_dir, window_str):
    print("Building item bias vs raw plot...")

    top_idx    = np.argsort(b_i)[::-1][:top_n]
    bottom_idx = np.argsort(b_i)[:top_n]

    def build_rows(indices, label):
        rows = []
        for idx in indices:
            mid   = enc.id_item[idx]
            bias  = float(b_i[idx])
            raw   = train_df[train_df["movieId"] == mid]["rating"]
            avg_r = float(raw.mean()) if len(raw) > 0 else np.nan
            n     = len(raw)
            title_row = movies_df[movies_df["movieId"] == mid]
            title = wrap_title(title_row.iloc[0]["title"]) if not title_row.empty else str(mid)
            rows.append({"mid": mid, "title": title, "bias": bias,
                         "avg_rating": avg_r, "n": n, "group": label})
        return rows

    rows   = build_rows(top_idx, "high") + build_rows(bottom_idx, "low")
    df_plt = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(
        f"Movie Bias Analysis  |  Training window: {window_str}",
        fontsize=18, fontweight="bold", y=1.01
    )

    def bar_panel(ax, sub, col, ylabel, title, color, show_labels = True):
        labels = sub["title"].tolist()
        vals   = sub[col].values
        bars   = ax.barh(labels if show_labels else [""] * len(vals),
                         vals, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ref    = mu if col == "avg_rating" else 0
        ax.axvline(ref, color=PALETTE["mean"], linestyle="--", linewidth=1.5,
                   label="global mean" if col == "avg_rating" else "zero bias")
        ax.set_xlabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=11)
        for bar, v in zip(bars, vals):
            ax.text(v + (max(vals) - min(vals)) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.2f}", va="center", fontsize=9)

    high = df_plt[df_plt["group"] == "high"].reset_index(drop=True)
    low  = df_plt[df_plt["group"] == "low"].reset_index(drop=True)

    bar_panel(axes[0, 0], high, "bias",       "Model Bias (b_i)",  f"Top {top_n} Movies — Model Bias",        PALETTE["pos"], True)
    bar_panel(axes[0, 1], high, "avg_rating", "Avg Rating (raw)",  f"Top {top_n} Movies — Raw Avg Rating",     PALETTE["pos"], True)
    bar_panel(axes[1, 0], low,  "bias",       "Model Bias (b_i)",  f"Bottom {top_n} Movies — Model Bias",      PALETTE["neg"], True)
    bar_panel(axes[1, 1], low,  "avg_rating", "Avg Rating (raw)",  f"Bottom {top_n} Movies — Raw Avg Rating",  PALETTE["neg"], True)

    fig.text(0.5, -0.01,
             "Validation: movies with high model bias (left) should show higher raw avg ratings (right), and vice versa.",
             ha="center", fontsize=13, style="italic", color="#555555")

    plt.tight_layout()
    save_fig(fig, out_dir, "item_bias_vs_raw.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Genre bias (model) vs Genre avg rating (raw) — side by side
# ─────────────────────────────────────────────────────────────────────────────

def plot_genre_bias_vs_raw(b_i, enc, train_df, movies_df, out_dir, window_str):
    print("Building genre bias vs raw plot...")

    # ── build per-item dataframe with bias ────────────────────────────────────
    item_rows = []
    for idx, mid in enc.id_item.items():
        item_rows.append({"movieId": mid, "model_bias": float(b_i[idx])})
    item_df = pd.DataFrame(item_rows)

    # merge with movies to get genres
    merged = item_df.merge(movies_df[["movieId", "genres"]], on="movieId", how="left")
    merged = merged.dropna(subset=["genres"])

    # explode genres
    merged = merged.explode("genres")
    merged = merged[merged["genres"] != "(no genres listed)"]

    # model avg bias per genre
    genre_bias = (merged.groupby("genres")["model_bias"]
                        .agg(["mean", "sem", "count"])
                        .rename(columns={"mean": "avg_bias", "sem": "sem_bias", "count": "n_items"})
                        .reset_index())

    # ── raw avg rating per genre (training window) ────────────────────────────
    raw_merged = train_df.merge(movies_df[["movieId", "genres"]], on="movieId", how="left")
    raw_merged = raw_merged.dropna(subset=["genres"])
    raw_merged = raw_merged.explode("genres")
    raw_merged = raw_merged[raw_merged["genres"] != "(no genres listed)"]

    genre_raw = (raw_merged.groupby("genres")["rating"]
                           .agg(["mean", "sem"])
                           .rename(columns={"mean": "avg_rating", "sem": "sem_rating"})
                           .reset_index())

    # ── merge and sort by model bias ─────────────────────────────────────────
    genre_df = genre_bias.merge(genre_raw, on="genres", how="inner")
    genre_df = genre_df.sort_values("avg_bias", ascending=True).reset_index(drop=True)

    genres   = genre_df["genres"].tolist()
    y_pos    = np.arange(len(genres))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    fig.suptitle(
        f"Genre Analysis  |  Model Bias vs Raw Avg Rating  |  Window: {window_str}",
        fontsize=18, fontweight="bold"
    )

    # — left: model avg bias per genre —
    bars1 = ax1.barh(y_pos, genre_df["avg_bias"],
                     xerr=genre_df["sem_bias"],
                     color=[PALETTE["pos"] if v >= 0 else PALETTE["neg"]
                            for v in genre_df["avg_bias"]],
                     edgecolor="white", linewidth=0.8,
                     error_kw={"elinewidth": 1.2, "capsize": 3}, zorder=3)
    ax1.axvline(0, color=PALETTE["mean"], linestyle="--", linewidth=1.5, label="zero bias")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(genres, fontsize=13)
    ax1.set_xlabel("Avg Model Bias (b_i)", fontsize=14)
    ax1.set_title("Model: Avg Item Bias per Genre", fontweight="bold")
    ax1.legend()

    # annotate n_items
    for i, (v, n) in enumerate(zip(genre_df["avg_bias"], genre_df["n_items"])):
        ax1.text(v + (0.002 if v >= 0 else -0.002),
                 i, f"n={n}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=9, color="#333")

    # — right: raw avg rating per genre —
    global_mean = train_df["rating"].mean()
    bars2 = ax2.barh(y_pos, genre_df["avg_rating"],
                     xerr=genre_df["sem_rating"],
                     color=[PALETTE["pos"] if v >= global_mean else PALETTE["neg"]
                            for v in genre_df["avg_rating"]],
                     edgecolor="white", linewidth=0.8,
                     error_kw={"elinewidth": 1.2, "capsize": 3}, zorder=3)
    ax2.axvline(global_mean, color=PALETTE["mean"], linestyle="--", linewidth=1.5,
                label=f"global mean ({global_mean:.2f})")
    ax2.set_xlabel("Avg Raw Rating", fontsize=14)
    ax2.set_title("Raw Data: Avg Rating per Genre", fontweight="bold")
    ax2.legend()

    for i, v in enumerate(genre_df["avg_rating"]):
        ax2.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=10, color="#333")

    fig.text(0.5, -0.01,
             "Genres sorted by model bias (ascending). "
             "Green = above zero/mean, Red = below zero/mean. Error bars = ±1 SEM.",
             ha="center", fontsize=13, style="italic", color="#555555")

    plt.tight_layout()
    save_fig(fig, out_dir, "genre_bias_vs_raw.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Item bias vs popularity (rating count)
# ─────────────────────────────────────────────────────────────────────────────

def plot_item_bias_vs_popularity(b_i, enc, train_df, movies_df, out_dir, window_str):
    print("Building item bias vs popularity scatter...")

    rows = []
    for idx, mid in enc.id_item.items():
        raw   = train_df[train_df["movieId"] == mid]["rating"]
        count = len(raw)
        rows.append({"movieId": mid, "bias": float(b_i[idx]), "count": count})

    df_sc = pd.DataFrame(rows)
    df_sc = df_sc[df_sc["count"] > 0]

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        f"Item Bias vs Popularity  |  Training window: {window_str}",
        fontsize=18, fontweight="bold"
    )

    sc = ax.scatter(
        df_sc["count"], df_sc["bias"],
        c=df_sc["bias"], cmap="RdYlGn",
        alpha=0.45, s=18, linewidths=0, zorder=3
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Item Bias (b_i)", fontsize=13)

    ax.axhline(0, color=PALETTE["mean"], linestyle="--", linewidth=1.5, label="zero bias")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Ratings in Training Window (log scale)", fontsize=14)
    ax.set_ylabel("Item Bias (b_i)", fontsize=14)
    ax.set_title("Does popularity correlate with positive bias?", fontweight="bold")
    ax.legend()

    # ── annotate top 5 and bottom 5 outliers ─────────────────────────────────
    top5    = df_sc.nlargest(5, "bias")
    bottom5 = df_sc.nsmallest(5, "bias")
    for _, row in pd.concat([top5, bottom5]).iterrows():
        title_row = movies_df[movies_df["movieId"] == row["movieId"]]
        label     = wrap_title(title_row.iloc[0]["title"], 25) if not title_row.empty else str(int(row["movieId"]))
        ax.annotate(label,
                    xy=(row["count"], row["bias"]),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=9, color="#222",
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8))

    corr = df_sc["count"].corr(df_sc["bias"])
    ax.text(0.02, 0.97, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes, fontsize=13,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc"))

    plt.tight_layout()
    save_fig(fig, out_dir, "item_bias_vs_popularity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Bias term analysis for a trained MF run.")
    p.add_argument("--data_dir", type=str, default="data/ml-32m", help="MovieLens data directory")
    p.add_argument("--top_n",    type=int, default=10, help="Top/bottom N users and movies to show")
    p.add_argument("--log_root", type=str, default="train_logs/MF", help="Root folder containing all MF run subfolders")
    return p.parse_args()


def main():
    args    = parse_args()
    # run_dir = Path(args.run_dir)

    args     = parse_args()
    log_root = Path(args.log_root)
    run_dirs = [d for d in sorted(log_root.iterdir()) if d.is_dir()]

    if not run_dirs:
        raise FileNotFoundError(f"No subfolders found in: {log_root}")
    

    print(f"Found {len(run_dirs)} run(s) to process: {[d.name for d in run_dirs]}")
    
    # ── load raw data ─────────────────────────────────────────────────────────
    print("Loading raw data via MovieLensDB...")
    db = MovieLensDB(data_dir=args.data_dir)
    db.load_data()
    for run_dir in run_dirs:
        print(f"\n{'='*60}")
        print(f"Processing: {run_dir.name}")
        print(f"{'='*60}")
        # ── parse window ──────────────────────────────────────────────────────────
        start, end = parse_window_from_folder(run_dir)
        window_str = f"{start}  →  {end}"
        print(f"\n Run:    {run_dir.name}")
        print(f" Window: {window_str}")

        # ── output directory ──────────────────────────────────────────────────────
        project_root = Path(__file__).resolve().parent.parent.parent
        out_dir      = project_root / "analysis" / "bias" / run_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f" Output: {out_dir}\n")

        # ── load artifacts ────────────────────────────────────────────────────────
        print("Loading model artifacts...")
        b_u, b_i, mu, enc = load_artifacts(run_dir)
        print(f"  Users: {len(b_u):,}  |  Items: {len(b_i):,}  |  mu: {mu:.4f}")

        

        train_df  = db.get_ratings_by_daterange(start, end)
        movies_df = db.movies_df.copy()
        print(f"  Raw ratings in window: {len(train_df):,}")

        # ── run all plots ─────────────────────────────────────────────────────────
        # print("\n── Plot 1: User bias vs raw avg rating ──────────────────────────────")
        # plot_user_bias_vs_raw(b_u, enc, train_df, mu, args.top_n, out_dir, window_str)

        print("\n── Plot 2: Item bias vs raw avg rating ──────────────────────────────")
        plot_item_bias_vs_raw(b_i, enc, train_df, movies_df, mu, args.top_n, out_dir, window_str)

        # print("\n── Plot 3: Genre bias vs raw avg rating ─────────────────────────────")
        # plot_genre_bias_vs_raw(b_i, enc, train_df, movies_df, out_dir, window_str)

        # print("\n── Plot 4: Item bias vs popularity ──────────────────────────────────")
        # plot_item_bias_vs_popularity(b_i, enc, train_df, movies_df, out_dir, window_str)

        print(f"\n✅ All plots saved to: {out_dir}")


if __name__ == "__main__":
    main()