"""
dim_reduction.py
----------------
Embedding visualisation for all MF training runs under train_logs/MF.

Per run produces:
  Idea 3 — Action sub-category UMAP
      action_subcategory_umap.png
      UMAP of Action movies coloured by sub-category (Action|Thriller,
      Action|Comedy, ..., pure Action). Min 30 movies per sub-category.

  Idea 4 — Genre purity analysis
      genre_purity.png
      For every encoded movie find its top-10 nearest neighbours in
      embedding space. Genre purity = fraction of neighbours sharing
      the same genre. Bar chart per genre.

  Idea U2 — Radar chart: query user vs similar / dissimilar users
      user_radar_<userId>.png  (4 charts, one per query user)
      Genre fingerprint radar comparing query user with 2 similar
      and 2 dissimilar users.

Usage (from project root):
    python src/analysis/dim_reduction.py
    python src/analysis/dim_reduction.py --log_root train_logs/MF \
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
from sklearn.preprocessing import normalize
import umap

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.db_simulator import MovieLensDB
from src.training.matrix_factorisation import Encoder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────
GENRES = ["Action", "Romance", "Sci-Fi", "Thriller", "Horror", "Comedy",
          "Drama", "Adventure", "Animation", "Crime"]

# contrastive palette — one per Action sub-category (up to 10)
SUBCAT_COLORS = [
    "#E24B4A", "#378ADD", "#1D9E75", "#BA7517", "#8A2BE2",
    "#FF7F00", "#A65628", "#F781BF", "#999999", "#4DAF4A",
]

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
}

FONT_SIZE     = 14
MIN_SUBCAT    = 30      # minimum movies per Action sub-category
N_NEIGHBOURS  = 10      # for genre purity
N_POWER_USERS = 50      # sampled for U1
N_RADAR_USERS = 4       # query users for U2


# ══════════════════════════════════════════════════════════════════════════
# Loading helpers
# ══════════════════════════════════════════════════════════════════════════

def load_run(run_dir: Path):
    enc_path  = run_dir / "encoder.pkl"
    item_path = run_dir / "item_embeddings.npy"
    user_path = run_dir / "user_embeddings.npy"
    missing   = [p for p in [enc_path, item_path, user_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing: {[p.name for p in missing]}")
    with open(enc_path, "rb") as f:
        enc = pickle.load(f)
    item_emb = np.load(item_path).astype(np.float32)
    user_emb = np.load(user_path).astype(np.float32)
    logger.info(
        f"  items={item_emb.shape[0]:,} users={user_emb.shape[0]:,} "
        f"dim={item_emb.shape[1]}"
    )
    return enc, item_emb, user_emb


def parse_run_meta(run_name: str) -> dict:
    parts = run_name.split("_")
    try:
        return {
            "start": parts[0], "end": parts[1],
            "loss":  parts[4] if len(parts) > 4 else "?",
            "dim":   parts[5].replace("dim", "") if len(parts) > 5 else "?",
        }
    except Exception:
        return {"start": "?", "end": "?", "loss": "?", "dim": "?"}


def meta_str(meta: dict) -> str:
    return (
        f"{meta['loss'].upper()} | dim={meta['dim']} | "
        f"{meta['start']} → {meta['end']}"
    )


# ══════════════════════════════════════════════════════════════════════════
# Idea 3 — Action sub-category UMAP
# ══════════════════════════════════════════════════════════════════════════

FIVE_GENRES = ["Action", "Romance", "Sci-Fi", "Thriller", "Horror"]
FIVE_COLORS = {
    "Action":   "#000000",
    "Romance":  "#FF0080",
    "Sci-Fi":   "#0080FF",
    "Thriller": "#FF9500",
    "Horror":   "#15FD00",
}

def build_genre_movie_map(db: MovieLensDB, enc) -> dict:
    genre_map = {}
    for genre in FIVE_GENRES:
        movies_df = db.get_movies_by_genres([genre], match="any")
        indices   = [
            enc.item_id[mid]
            for mid in movies_df["movieId"]
            if mid in enc.item_id
        ]
        genre_map[genre] = indices
        logger.info(f"    {genre}: {len(indices):,} movies")
    return genre_map


def build_genre_label_array(genre_map: dict):
    label_of = {}
    for genre in FIVE_GENRES:
        for idx in genre_map[genre]:
            if idx not in label_of:      # first genre wins
                label_of[idx] = genre
    indices = np.array(list(label_of.keys()))
    labels  = np.array([label_of[i] for i in indices])
    return indices, labels


def plot_five_genre_umap(item_indices, item_labels, item_emb,
                          meta, out_path, args):
    embs    = item_emb[item_indices]
    normed  = normalize(embs, norm="l2")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=args.n_neighbors,
        min_dist=args.min_dist, metric="cosine",
        random_state=42, low_memory=False,
    )
    logger.info(f"  Fitting UMAP on {len(item_indices):,} items ...")
    coords = reducer.fit_transform(normed)

    fig, ax = plt.subplots(figsize=(11, 8))
    for genre in FIVE_GENRES:
        mask = item_labels == genre
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=FIVE_COLORS[genre],
            label=f"{genre} ({mask.sum()})",
            s=30, alpha=0.8, linewidths=0,
        )
    ax.set_title(
        f"Item embedding UMAP — 5 genres\n{meta_str(meta)}",
        fontsize=FONT_SIZE + 1,
    )
    ax.set_xlabel("UMAP-1", fontsize=FONT_SIZE)
    ax.set_ylabel("UMAP-2", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.legend(fontsize=FONT_SIZE - 1, markerscale=2, loc="best")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {out_path.name}")

# ══════════════════════════════════════════════════════════════════════════
# Idea 4 — Genre purity
# ══════════════════════════════════════════════════════════════════════════

def build_item_genre_labels(db: MovieLensDB, enc) -> np.ndarray:
    """
    Array of length n_items: entry i = primary genre label or None.
    """
    labels = np.full(len(enc.item_id), None, dtype=object)
    for genre in GENRES:
        genre_df = db.get_movies_by_genres([genre], match="any")
        for mid in genre_df["movieId"]:
            if mid in enc.item_id:
                idx = enc.item_id[mid]
                if labels[idx] is None:
                    labels[idx] = genre
    return labels


def compute_genre_purity(item_emb: np.ndarray, labels: np.ndarray) -> dict:
    """
    For each labelled movie find its top-N_NEIGHBOURS nearest neighbours.
    Purity = fraction of neighbours sharing the same label.
    Returns dict: genre -> mean purity.
    """
    normed           = normalize(item_emb, norm="l2")
    labelled_mask    = np.array([x is not None for x in labels])
    labelled_indices = np.where(labelled_mask)[0]

    if len(labelled_indices) == 0:
        return {}

    labelled_embs   = normed[labelled_indices]
    labelled_labels = labels[labelled_indices]

    logger.info(
        f"  Computing purity for {len(labelled_indices):,} labelled items ..."
    )

    chunk             = 500
    purity_per_genre  = {g: [] for g in GENRES}

    for start in range(0, len(labelled_indices), chunk):
        end  = min(start + chunk, len(labelled_indices))
        sims = labelled_embs[start:end] @ labelled_embs.T   # (chunk, N)

        for local_i in range(end - start):
            global_i          = start + local_i
            row               = sims[local_i].copy()
            row[global_i]     = -np.inf                     # exclude self
            top_k             = np.argsort(row)[::-1][:N_NEIGHBOURS]
            own_label         = labelled_labels[global_i]
            purity            = float(np.mean(labelled_labels[top_k] == own_label))
            if own_label in purity_per_genre:
                purity_per_genre[own_label].append(purity)

    return {
        g: float(np.mean(v))
        for g, v in purity_per_genre.items() if v
    }


def plot_genre_purity(purity: dict, meta: dict, out_path: Path):
    genres = list(purity.keys())
    values = [purity[g] for g in genres]
    colors = [GENRE_COLORS.get(g, "#888888") for g in genres]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars    = ax.bar(genres, values, color=colors, edgecolor="black", linewidth=0.7)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom",
            fontsize=FONT_SIZE - 1, fontweight="bold",
        )

    ax.set_ylim(0, min(1.0, max(values) * 1.25))
    ax.set_title(
        f"Genre purity (top-{N_NEIGHBOURS} neighbours)\n{meta_str(meta)}",
        fontsize=FONT_SIZE + 1,
    )
    ax.set_xlabel("Genre", fontsize=FONT_SIZE)
    ax.set_ylabel(f"Mean purity @ top-{N_NEIGHBOURS}", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════
# User helpers shared by U1 and U2
# ══════════════════════════════════════════════════════════════════════════

def get_genre_fingerprint(uid_orig, db: MovieLensDB, enc) -> dict:
    """Rating count per genre (all GENRES) for one user."""
    uid_ratings = db.ratings_df[db.ratings_df["userId"] == uid_orig]
    uid_ratings = uid_ratings[uid_ratings["movieId"].isin(enc.item_id)]
    counts      = {g: 0 for g in GENRES}
    for mid in uid_ratings["movieId"]:
        movie = db.movies_df[db.movies_df["movieId"] == mid]
        if movie.empty:
            continue
        for g in movie.iloc[0]["genres"]:
            if g in counts:
                counts[g] += 1
    return counts


def select_power_users(db: MovieLensDB, enc, n: int, seed: int = 42) -> list:
    """Sample n power users (>=50 ratings) from encoded users."""
    counts = (
        db.ratings_df[db.ratings_df["userId"].isin(enc.user_id)]
        .groupby("userId").size()
    )
    power = [enc.user_id[u] for u in counts[counts >= 50].index if u in enc.user_id]
    rng   = np.random.default_rng(seed)
    if len(power) >= n:
        return list(rng.choice(power, size=n, replace=False))
    all_u = list(enc.id_user.keys())
    return list(rng.choice(all_u, size=min(n, len(all_u)), replace=False))


def find_similar_dissimilar(u_idx: int, user_emb: np.ndarray,
                             candidate_indices: list, top_k: int = 2):
    normed    = normalize(user_emb, norm="l2")
    u_vec     = normed[u_idx]
    cand      = np.array(candidate_indices)
    sims      = normed[cand] @ u_vec

    self_pos  = np.where(cand == u_idx)[0]

    sim_sims  = sims.copy()
    dissim_sims = sims.copy()

    if len(self_pos):
        sim_sims[self_pos[0]]    = -np.inf   # exclude self from similar
        dissim_sims[self_pos[0]] = +np.inf   # exclude self from dissimilar

    top_sim    = cand[np.argsort(sim_sims)[::-1][:top_k]]
    top_dissim = cand[np.argsort(dissim_sims)[:top_k]]
    return list(top_sim), list(top_dissim)


# ══════════════════════════════════════════════════════════════════════════
# Idea U1 — overlap distributions
# ══════════════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════════════
# Idea U2 — Radar chart
# ══════════════════════════════════════════════════════════════════════════

def _radar(ax, values, angles, color, linestyle, label):
    vals = list(values) + [values[0]]
    angs = list(angles) + [angles[0]]
    ax.plot(angs, vals, color=color, linestyle=linestyle, linewidth=2, label=label)
    ax.fill(angs, vals, color=color, alpha=0.12)


def plot_radar(query_u_idx, power_users, user_emb, db, enc, meta, out_path):
    uid      = enc.id_user[query_u_idx]
    query_fp = get_genre_fingerprint(uid, db, enc)

    sim_idx, dissim_idx = find_similar_dissimilar(
        query_u_idx, user_emb, power_users
    )

    def norm_fp(fp):
        total = max(sum(fp.values()), 1)
        return [fp.get(g, 0) / total for g in GENRES]

    N      = len(GENRES)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_thetagrids(np.degrees(angles), labels=GENRES, fontsize=FONT_SIZE - 1)

    _radar(ax, norm_fp(query_fp), angles,
           color="#222222", linestyle="-", label=f"Query user {uid}")

    sim_colors    = ["#378ADD", "#85B7EB"]
    dissim_colors = ["#E24B4A", "#F09595"]

    for k, s_idx in enumerate(sim_idx):
        s_uid = enc.id_user[s_idx]
        s_fp  = get_genre_fingerprint(s_uid, db, enc)
        _radar(ax, norm_fp(s_fp), angles,
               color=sim_colors[k], linestyle="--",
               label=f"Similar user {s_uid}")

    for k, d_idx in enumerate(dissim_idx):
        d_uid = enc.id_user[d_idx]
        d_fp  = get_genre_fingerprint(d_uid, db, enc)
        _radar(ax, norm_fp(d_fp), angles,
               color=dissim_colors[k], linestyle=":",
               label=f"Dissimilar user {d_uid}")

    ax.set_title(
        f"Genre fingerprint radar — user {uid}\n{meta_str(meta)}",
        fontsize=FONT_SIZE + 1, pad=20,
    )
    ax.legend(fontsize=FONT_SIZE - 2, loc="upper right",
              bbox_to_anchor=(1.35, 1.1))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════
# Per-run pipeline
# ══════════════════════════════════════════════════════════════════════════

def process_run(run_dir: Path, db: MovieLensDB, out_root: Path, args):
    run_name = run_dir.name
    logger.info(f"\n{'='*60}\nProcessing: {run_name}\n{'='*60}")

    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        enc, item_emb, user_emb = load_run(run_dir)
    except FileNotFoundError as e:
        logger.warning(f"  Skipping: {e}")
        return

    meta = parse_run_meta(run_name)

    # ── Idea 3 ────────────────────────────────────────────────────────────
    logger.info("  [Idea 3] 5-genre UMAP ...")
    genre_map               = build_genre_movie_map(db, enc)
    item_indices, item_labels = build_genre_label_array(genre_map)
    if len(item_indices) >= 10:
        plot_five_genre_umap(item_indices, item_labels, item_emb, meta,
                            out_dir / "five_genre_umap.png", args)
    else:
        logger.warning("  Not enough labelled items — skipping Idea 3.")

    # ── Idea 4 ────────────────────────────────────────────────────────────
    logger.info("  [Idea 4] Genre purity ...")
    item_labels = build_item_genre_labels(db, enc)
    purity      = compute_genre_purity(item_emb, item_labels)
    if purity:
        plot_genre_purity(purity, meta, out_dir / "genre_purity.png")
    else:
        logger.warning("  Empty purity — skipping Idea 4.")

    # ── Shared user setup ─────────────────────────────────────────────────
    logger.info("  Selecting power users ...")
    power_users = select_power_users(db, enc, n=N_POWER_USERS)

    # ── Idea U2 ───────────────────────────────────────────────────────────
    logger.info("  [Idea U2] Radar charts ...")
    step        = max(1, len(power_users) // N_RADAR_USERS)
    radar_users = power_users[::step][:N_RADAR_USERS]
    for q_idx in radar_users:
        uid = enc.id_user[q_idx]
        plot_radar(q_idx, power_users, user_emb, db, enc, meta,
                   out_dir / f"user_radar_{uid}.png")

    logger.info(f"  Done → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="MF embedding visualisation")
    p.add_argument("--log_root",    type=str,   default="train_logs/MF")
    p.add_argument("--data_dir",    type=str,   default="data/ml-32m")
    p.add_argument("--out_root",    type=str,   default="analysis/dim_reduction")
    p.add_argument("--n_neighbors", type=int,   default=15)
    p.add_argument("--min_dist",    type=float, default=0.1)
    return p.parse_args()


def main():
    args     = parse_args()
    log_root = Path(args.log_root)
    out_root = Path(args.out_root)

    if not log_root.exists():
        raise FileNotFoundError(f"log_root not found: {log_root}")

    run_dirs = sorted([
        d for d in log_root.iterdir()
        if d.is_dir() and (d / "encoder.pkl").exists()
    ])

    if not run_dirs:
        logger.error(f"No valid run folders in {log_root}")
        return

    logger.info(f"Found {len(run_dirs)} run(s)")
    db = MovieLensDB(data_dir=args.data_dir)
    db.load_data()

    for run_dir in run_dirs:
        try:
            process_run(run_dir, db, out_root, args)
        except Exception as e:
            logger.error(f"Failed on {run_dir.name}: {e}", exc_info=True)

    logger.info(f"\nAll done. Outputs in: {out_root}")


if __name__ == "__main__":
    main()