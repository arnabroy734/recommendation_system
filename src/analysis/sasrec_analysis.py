"""
sasrec_analysis.py
------------------
Analysis script for a trained SASRec model.

Usage:
    python src/analysis/sasrec_analysis.py \
        --start      2018-01-01 \
        --end        2018-06-30 \
        --data_dir   data/ml-32m \
        --log_root   train_logs/SASREC \
        --loss       bpr \
        --dim        64 \
        --n_heads    2 \
        --n_layers   2 \
        --max_len    50 \
        --top_n      20

Analyses:
    1. Attention heatmap + bar chart — last layer, 3 users, one figure per user
    2. Genre order experiment — Action + Comedy, 3 sets, one figure per set
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.db_simulator import MovieLensDB
from src.training.sasrec_architecture import SASRec
from src.training.matrix_factorisation import Encoder

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         15,
    "axes.titlesize":    17,
    "axes.labelsize":    15,
    "xtick.labelsize":   13,
    "ytick.labelsize":   13,
    "legend.fontsize":   14,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    1.2,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── genre mapping ─────────────────────────────────────────────────────────────
ACTION_GENRES = {"Action"}
COMEDY_GENRES = {"Comedy", "Drama", "Romance", "Thriller"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_run_dir(log_root: Path, start: str, end: str, loss: str) -> Path:
    pattern = f"{start}_{end}"
    matches = [
        d for d in sorted(log_root.iterdir())
        if d.is_dir()
        and d.name.startswith(pattern)
        and loss in d.name
    ]
    if not matches:
        raise FileNotFoundError(
            f"No run folder matching '{pattern}' with loss='{loss}' in {log_root}"
        )
    print(f"  Found run folder: {matches[0].name}")
    return matches[0]


def load_model(run_dir: Path, enc, args, device) -> SASRec:
    model = SASRec(
        n_items  = enc.n_items,
        dim      = args.dim,
        max_len  = args.max_len,
        n_heads  = args.n_heads,
        n_layers = args.n_layers,
    ).to(device)
    state = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"  Model loaded from {run_dir / 'model.pt'}")
    return model


def save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def build_seq(item_ids: list, enc, max_len: int, device) -> torch.Tensor:
    """Build left-padded sequence tensor from list of original movieIds."""
    idxs = [enc.item_id[m] + 1 for m in item_ids if m in enc.item_id]
    idxs = idxs[-max_len:]
    pad  = [0] * (max_len - len(idxs))
    seq  = pad + idxs
    return torch.LongTensor(seq).unsqueeze(0).to(device)


def get_top_n_recs(model, seq_tensor, enc, seen_set, top_n, device):
    """Run model on seq_tensor, return top_n unseen original movieIds."""
    with torch.no_grad():
        h, _ = model(seq_tensor)
        h_t  = h[0, -1, :].cpu().numpy()

    emb_w            = model.item_emb.weight.detach().cpu().numpy()
    all_item_indices = np.array(list(enc.id_item.keys()))
    unseen           = all_item_indices[~np.isin(all_item_indices, list(seen_set))]
    item_vecs        = emb_w[unseen + 1]
    scores           = item_vecs @ h_t
    top_idx          = np.argsort(scores)[::-1][:top_n]
    return [enc.id_item[i] for i in unseen[top_idx]]


def classify_rec_genre(movie_id, movies_df) -> str:
    """Classify a recommended movie as Action, Comedy, or Other."""
    row = movies_df[movies_df["movieId"] == movie_id]
    if row.empty:
        return "Other"
    genres = set(row.iloc[0]["genres"])
    if genres & ACTION_GENRES:
        return "Action"
    if genres & COMEDY_GENRES:
        return "Comedy"
    return "Other"


def get_last_layer_attn(model, seq_tensor):
    """Return last layer attention averaged across heads: (L, L) numpy."""
    with torch.no_grad():
        _, attn_scores = model(seq_tensor)
    last = attn_scores[-1]          # (1, n_heads, L, L)
    avg  = last.mean(dim=1)         # (1, L, L)
    return avg[0].cpu().numpy()     # (L, L)


def shorten(title: str, n: int = 18) -> str:
    return title[:n] + "…" if len(title) > n else title


def get_title(mid, movies_df) -> str:
    row = movies_df[movies_df["movieId"] == mid]
    return shorten(row.iloc[0]["title"]) if not row.empty else str(mid)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1 & 2 — Attention per user
# ─────────────────────────────────────────────────────────────────────────────

def plot_attention_user(model, uid, train_df, movies_df, enc, args,
                        out_dir, top_k_bar=30, device="cpu"):

    # ── build sequence ───────────────────────────────────────────────
    hist  = train_df[train_df["userId"] == uid].sort_values("timestamp")
    items = hist["movieId"].tolist()
    seq   = build_seq(items, enc, args.max_len, device)

    # ── attention ────────────────────────────────────────────────────
    attn     = get_last_layer_attn(model, seq)   # (L, L)
    real_len = min(len(items), args.max_len)
    pad_len  = args.max_len - real_len

    real_attn  = attn[pad_len:, pad_len:]
    real_items = items[-real_len:]
    labels     = [get_title(m, movies_df) for m in real_items]

    # ── figure ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    ax  = fig.add_subplot(111)

    fig.suptitle(
        f"Attention Analysis — User {uid} | {real_len} items",
        fontsize=16, fontweight="bold"
    )

    # ── top-k attention (last row) ───────────────────────────────────
    last_row = real_attn[-1, :]
    top_k    = min(top_k_bar, real_len)

    top_pos  = np.argsort(last_row)[::-1][:top_k]
    top_vals = last_row[top_pos]
    top_lbls = [labels[p] for p in top_pos]

    # sort for clean descending bars
    order = np.argsort(top_vals)[::-1]
    top_vals = top_vals[order]
    top_lbls = [top_lbls[i] for i in order]
    top_pos  = top_pos[order]

    # color by recency
    bar_colors = [
        plt.cm.Blues(0.3 + 0.7 * (p / max(real_len - 1, 1)))
        for p in top_pos
    ]

    # ── bar plot ─────────────────────────────────────────────────────
    x = np.arange(top_k)
    bars = ax.bar(x, top_vals, color=bar_colors,
                  edgecolor="black", linewidth=0.6)

    # ── labels ───────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(top_lbls, rotation=60, ha="right", fontsize=9)

    ax.set_ylabel("Attention weight", fontsize=12)
    ax.set_title("Top attended items (last position)", fontsize=13)

    # ── FIX: correct text placement ──────────────────────────────────
    for i, (bar, v) in enumerate(zip(bars, top_vals)):
        ax.text(
            i,
            v + top_vals.max() * 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    # ── remove grid (your issue) ─────────────────────────────────────
    ax.grid(False)

    # ── legend ───────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=plt.cm.Blues(0.3),
               markersize=10, label="Older"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=plt.cm.Blues(1.0),
               markersize=10, label="Recent"),
    ]
    ax.legend(handles=legend_handles, fontsize=10)

    plt.tight_layout()
    save_fig(fig, out_dir, f"attention_user_{uid}.png")



def run_attention_analysis(model, enc, train_df, movies_df, out_dir, args,
                           n_users=3, top_k_bar=30, device="cpu"):
    print("\n── Analysis 1 & 2: Attention heatmap + bar chart ───────────────────")

    user_counts = train_df.groupby("userId").size()
    active      = user_counts[(user_counts >= 30) & (user_counts <=60) ].index.tolist()
    active      = [u for u in active if u in enc.user_id]

    if not active:
        print("  No active users found, skipping.")
        return

    np.random.seed(42)
    chosen = np.random.choice(active, min(n_users, len(active)), replace=False)
    print(f"  Chosen users: {chosen.tolist()}")

    for uid in chosen:
        print(f"  Processing user {uid} ...")
        plot_attention_user(
            model, uid, train_df, movies_df, enc, args,
            out_dir, top_k_bar=top_k_bar, device=device
        )


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3 — Genre order experiment
# ─────────────────────────────────────────────────────────────────────────────

def get_top_genre_movies(genre_set, movies_df, enc, train_df, n=50):
    """Top-n most popular movies whose genres overlap with genre_set."""
    rating_counts = train_df.groupby("movieId").size()
    genre_mids    = movies_df[
        movies_df["genres"].apply(lambda gs: bool(set(gs) & genre_set))
    ]["movieId"].tolist()
    genre_mids = [m for m in genre_mids if m in enc.item_id]
    counts     = rating_counts.reindex(genre_mids, fill_value=0)
    return counts.nlargest(n).index.tolist()


def run_genre_order_experiment(model, enc, movies_df, train_df, out_dir, args,
                               top_n=20, device="cpu"):
    print("\n── Analysis 3: Genre order experiment ──────────────────────────────")

    top_action = get_top_genre_movies(ACTION_GENRES, movies_df, enc, train_df, 50)
    top_comedy = get_top_genre_movies(COMEDY_GENRES, movies_df, enc, train_df, 50)
    print(f"  Action pool: {len(top_action)} | Comedy pool: {len(top_comedy)}")

    np.random.seed(42)

    genre_labels = ["Action", "Comedy", "Other"]
    bar_colors   = {"Action": "#000000", "Comedy": "#FF0080", "Other": "#AAAAAA"}

    for set_idx in range(1, 4):
        action_sample = list(np.random.choice(top_action, 20, replace=False))
        comedy_sample = list(np.random.choice(top_comedy, 10, replace=False))

        orderings = {
            "Action → Comedy\n(20 Action then 10 Comedy)":
                action_sample + comedy_sample,
            "Comedy → Action\n(10 Comedy then 20 Action)":
                comedy_sample + action_sample,
            "Interleaved\n(Action, Comedy alternating)":
                [x for pair in zip(action_sample[:10], comedy_sample)
                 for x in pair] + action_sample[10:],
        }

        fig, axes = plt.subplots(1, 3, figsize=(22, 8), sharey=True)
        fig.suptitle(
            f"Genre Order Experiment — Set {set_idx}\n"
            f"Same 30 Movies (20 Action + 10 Comedy), "
            f"3 Different Orderings → Top-{top_n} Recommendations",
            fontsize=17, fontweight="bold"
        )

        for ax, (order_name, seq_items) in zip(axes, orderings.items()):
            seen_set   = set(
                enc.item_id[m] for m in seq_items if m in enc.item_id
            )
            seq_tensor = build_seq(seq_items, enc, args.max_len, device)
            rec_ids    = get_top_n_recs(
                model, seq_tensor, enc, seen_set, top_n, device
            )

            counts = {"Action": 0, "Comedy": 0, "Other": 0}
            for mid in rec_ids:
                counts[classify_rec_genre(mid, movies_df)] += 1

            values = [counts[g] for g in genre_labels]
            colors = [bar_colors[g] for g in genre_labels]

            bars = ax.bar(
                genre_labels, values,
                color=colors, edgecolor="white",
                linewidth=0.8, zorder=3, width=0.5
            )

            # count annotation above bar
            for bar, v in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    str(v), ha="center", va="bottom",
                    fontsize=15, fontweight="bold"
                )

            # percentage inside bar
            for bar, v in zip(bars, values):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() / 2,
                        f"{v / top_n * 100:.0f}%",
                        ha="center", va="center",
                        fontsize=13,
                        color="white" if v > 3 else "#333",
                        fontweight="bold"
                    )

            ax.set_title(order_name, fontsize=14, fontweight="bold", pad=14)
            ax.set_ylabel("# Recommendations", fontsize=14)
            ax.set_ylim(0, top_n + 4)
            ax.set_xticklabels(genre_labels, fontsize=14)

        fig.text(
            0.5, -0.03,
            "Action genre includes: Action, Thriller, Sci-Fi.  "
            "MF would produce identical results across all 3 orderings — "
            "SASRec is sensitive to the order of interactions.",
            ha="center", fontsize=13, style="italic", color="#555"
        )

        plt.tight_layout()
        save_fig(fig, out_dir, f"genre_order_set{set_idx}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SASRec analysis script")
    p.add_argument("--start",        type=str, required=True)
    p.add_argument("--end",          type=str, required=True)
    p.add_argument("--data_dir",     type=str, default="data/ml-32m")
    p.add_argument("--log_root",     type=str, default="train_logs/SASREC")
    p.add_argument("--loss",         type=str, choices=["bce", "bpr"],
                   default="bpr")
    p.add_argument("--dim",          type=int, default=64)
    p.add_argument("--n_heads",      type=int, default=2)
    p.add_argument("--n_layers",     type=int, default=2)
    p.add_argument("--max_len",      type=int, default=50)
    p.add_argument("--top_n",        type=int, default=20)
    p.add_argument("--n_attn_users", type=int, default=3)
    p.add_argument("--top_k_bar",    type=int, default=30)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    log_root = Path(args.log_root)
    run_dir  = find_run_dir(log_root, args.start, args.end, args.loss)

    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir      = project_root / "analysis" / "SASREC" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {out_dir}")

    with open(run_dir / "encoder.pkl", "rb") as f:
        enc = pickle.load(f)
    print(f"  Encoder: {enc.n_users:,} users | {enc.n_items:,} items")

    print("\nLoading raw data ...")
    db        = MovieLensDB(data_dir=args.data_dir)
    db.load_data()
    movies_df = db.movies_df.copy()
    train_df  = db.get_ratings_by_daterange(args.start, args.end)
    print(f"  Ratings in window: {len(train_df):,}")

    print("\nLoading model ...")
    model = load_model(run_dir, enc, args, device)

    run_attention_analysis(
        model, enc, train_df, movies_df, out_dir, args,
        n_users=args.n_attn_users,
        top_k_bar=args.top_k_bar,
        device=device
    )

    run_genre_order_experiment(
        model, enc, movies_df, train_df, out_dir, args,
        top_n=args.top_n,
        device=device
    )

    print(f"\n✅ All analyses saved to: {out_dir}")


if __name__ == "__main__":
    main()