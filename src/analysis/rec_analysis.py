"""
rec_analysis.py
---------------
Qualitative analysis of recommendations from trained MF models.

For each run folder under train_logs/MF/:
  Uses recs_window_01.csv + encoder.pkl + training data to produce:

  Q1 — Recommendation case studies
       For each of 5 genres find the user with the highest fraction
       of that genre in their watch history (among users in recs CSV).
       Side-by-side plot:
         Left  : watch history genre distribution (top-10 genres, bar chart)
         Right : top-20 recommended movies with their genres (horizontal bars)
       A summary table of selected users is also saved as CSV.

  Q2 — Popularity bias audit
       Using top-20 recs per user, compute what fraction of each
       user's recommendations are "popular" (top-500 most rated movies
       in the training window). Histogram across all users.

  Q4 — Genre consistency heatmap
       For the Q1-selected users, compute genre distribution of their
       top-20 recommendations. Heatmap: rows = selected users (labelled
       by dominant genre), columns = genres.

Outputs saved under:
    analysis/recs/<run_folder_name>/

Usage (from project root):
    python src/analysis/rec_analysis.py
    python src/analysis/rec_analysis.py --log_root train_logs/MF \
                                         --data_dir data/ml-32m
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.db_simulator import MovieLensDB
from src.training.matrix_factorisation import Encoder  # noqa: F401 — needed for pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────
FIVE_GENRES = ["Action", "Romance", "Sci-Fi", "Thriller", "Horror"]
FONT_SIZE   = 14
TOP_N_RECS  = 20
TOP_N_HIST  = 10     # top genres to show in watch history
POP_TOP_N   = 500    # top-N movies considered "popular"

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


# ══════════════════════════════════════════════════════════════════════════
# Loading helpers
# ══════════════════════════════════════════════════════════════════════════

def load_run_artefacts(run_dir: Path):
    """Load encoder and recs_window_01.csv from a run folder."""
    enc_path  = run_dir / "encoder.pkl"
    recs_path = run_dir / "recs_window_01.csv"
    args_path = run_dir / "args.pkl"

    missing = [p for p in [enc_path, recs_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {run_dir}: {[p.name for p in missing]}"
        )

    with open(enc_path, "rb") as f:
        enc = pickle.load(f)

    recs_df = pd.read_csv(recs_path)
    # keep top-20 per user only
    recs_df = (
        recs_df.sort_values("rank")
               .groupby("userId")
               .head(TOP_N_RECS)
               .reset_index(drop=True)
    )

    train_args = None
    if args_path.exists():
        with open(args_path, "rb") as f:
            train_args = pickle.load(f)

    logger.info(
        f"  Recs: {recs_df['userId'].nunique():,} users | "
        f"{len(recs_df):,} rows"
    )
    return enc, recs_df, train_args


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
# Genre helpers
# ══════════════════════════════════════════════════════════════════════════

def get_movie_genres(movie_id, db: MovieLensDB) -> list:
    """Return list of genres for a movie, or ['Unknown']."""
    row = db.movies_df[db.movies_df["movieId"] == movie_id]
    if row.empty:
        return ["Unknown"]
    genres = row.iloc[0]["genres"]
    if isinstance(genres, list):
        return genres
    return ["Unknown"]


def build_movie_genre_lookup(db: MovieLensDB) -> dict:
    """Build movieId -> list of genres dict once for fast lookup."""
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


def get_recs_genre_distribution(user_recs: pd.DataFrame,
                                 db: MovieLensDB,
                                 target_genre: str = None) -> dict:
    counts = {}
    total  = 0
    for mid in user_recs["movieId"]:
        genres = get_movie_genres(mid, db)
        genres = [g for g in genres if g not in ("(no genres listed)", "Unknown")]
        if not genres:
            continue
        # prioritise target genre if present
        if target_genre and target_genre in genres:
            primary = target_genre
        else:
            primary = genres[0]
        counts[primary] = counts.get(primary, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {g: c / total for g, c in counts.items()}


# ══════════════════════════════════════════════════════════════════════════
# Q1 — User selection
# ══════════════════════════════════════════════════════════════════════════

def select_genre_heavy_users(recs_df, db, enc, movie_genre_lookup) -> dict:
    user_ids = recs_df["userId"].unique().tolist()
    logger.info(f"  Computing genre fractions for {len(user_ids):,} users ...")

    # pull all ratings for encoded users in one shot
    all_ratings = db.ratings_df[
        db.ratings_df["userId"].isin(user_ids) &
        db.ratings_df["movieId"].isin(enc.item_id)
    ][["userId", "movieId"]].copy()

    # explode genres vectorised — no per-row Python loop
    all_ratings["genres"] = all_ratings["movieId"].map(movie_genre_lookup)
    all_ratings = all_ratings.explode("genres")
    all_ratings = all_ratings.dropna(subset=["genres"])
    all_ratings = all_ratings[all_ratings["genres"] != ""]

    # genre counts per user
    genre_counts = (
        all_ratings.groupby(["userId", "genres"])
                   .size()
                   .reset_index(name="count")
    )
    total_per_user = genre_counts.groupby("userId")["count"].sum()
    genre_counts["fraction"] = (
        genre_counts["count"] /
        genre_counts["userId"].map(total_per_user)
    )

    # for each target genre pick user with highest fraction
    selected = {}
    for genre in FIVE_GENRES:
        sub = genre_counts[genre_counts["genres"] == genre]
        if sub.empty:
            continue
        best_row  = sub.loc[sub["fraction"].idxmax()]
        uid       = best_row["userId"]
        all_counts_for_user = (
            genre_counts[genre_counts["userId"] == uid]
            .set_index("genres")["count"]
            .to_dict()
        )
        selected[genre] = {
            "userId":        uid,
            "fraction":      best_row["fraction"],
            "genre_counts":  all_counts_for_user,
            "total_ratings": int(total_per_user[uid]),
        }
        logger.info(
            f"    {genre}: userId={uid} | "
            f"fraction={best_row['fraction']:.3f} | "
            f"total={int(total_per_user[uid])}"
        )

    return selected

# ══════════════════════════════════════════════════════════════════════════
# Q1 — Plots
# ══════════════════════════════════════════════════════════════════════════

def plot_case_study(genre: str, user_info: dict,
                    user_recs: pd.DataFrame,
                    db: MovieLensDB, meta: dict, out_path: Path):
    """
    Side-by-side:
      Left  : watch history genre bar chart (top-10 genres)
      Right : top-20 recommended movies with primary genre colour
    """
    # ── left: watch history ───────────────────────────────────────────────
    counts     = user_info["genre_counts"]
    top_genres = sorted(counts, key=counts.get, reverse=True)[:TOP_N_HIST]
    top_counts = [counts[g] for g in top_genres]
    bar_colors = [GENRE_COLORS.get(g, GENRE_COLORS["Other"]) for g in top_genres]

    # ── right: top-20 recs ────────────────────────────────────────────────
    rec_rows = []
    for _, row in user_recs.sort_values("rank").head(TOP_N_RECS).iterrows():
        mid    = row["movieId"]
        title  = db.movies_df[
            db.movies_df["movieId"] == mid
        ]["title"].values
        title  = title[0] if len(title) else f"Movie {mid}"
        genres = get_movie_genres(mid, db)
        # primary_genre = genres[0] if genres else "Unknown"
        if genre in genres:          # genre = the target genre (e.g. "Thriller")
            primary_genre = genre
        else:
            primary_genre = genres[0] if genres else "Unknown"
        rec_rows.append({
            "rank":          int(row["rank"]),
            "title":         title[:40],        # truncate long titles
            "primary_genre": primary_genre,
        })
    rec_df = pd.DataFrame(rec_rows)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # left plot
    ax = axes[0]
    bars = ax.bar(top_genres, top_counts, color=bar_colors,
                  edgecolor="black", linewidth=0.6)
    for bar, val in zip(bars, top_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(top_counts) * 0.01,
            str(val),
            ha="center", va="bottom",
            fontsize=FONT_SIZE - 2, fontweight="bold",
        )
    ax.set_title(
        f"Watch history — user {user_info['userId']}\n"
        f"({genre}-heavy | {genre} fraction = {user_info['fraction']:.2%})",
        fontsize=FONT_SIZE,
    )
    ax.set_xlabel("Genre", fontsize=FONT_SIZE)
    ax.set_ylabel("Rating count", fontsize=FONT_SIZE)
    ax.tick_params(axis="x", rotation=30, labelsize=FONT_SIZE - 2)
    ax.tick_params(axis="y", labelsize=FONT_SIZE - 2)
    ax.grid(axis="y", alpha=0.3)

    # right plot — horizontal bar chart of top-20 recs
    ax2     = axes[1]
    labels  = [f"{r['rank']:2d}. {r['title']}" for _, r in rec_df.iterrows()]
    y_pos   = np.arange(len(labels))
    colors2 = [
        GENRE_COLORS.get(r["primary_genre"], GENRE_COLORS["Other"])
        for _, r in rec_df.iterrows()
    ]

    ax2.barh(y_pos, np.ones(len(labels)), color=colors2,
             edgecolor="black", linewidth=0.4, height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=FONT_SIZE - 3)
    ax2.invert_yaxis()
    ax2.set_xticks([])
    ax2.set_title(
        f"Top-{TOP_N_RECS} recommendations — user {user_info['userId']}\n"
        f"(bar colour = primary genre of movie)",
        fontsize=FONT_SIZE,
    )

    # genre legend for right plot
    seen_genres = rec_df["primary_genre"].unique()
    handles     = [
        plt.Rectangle((0, 0), 1, 1,
                       color=GENRE_COLORS.get(g, GENRE_COLORS["Other"]))
        for g in seen_genres
    ]
    ax2.legend(handles, seen_genres, fontsize=FONT_SIZE - 3,
               loc="lower right", title="Genre", title_fontsize=FONT_SIZE - 2)

    fig.suptitle(
        f"Q1 Case study — {genre}-heavy user\n{meta_str(meta)}",
        fontsize=FONT_SIZE + 1, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {out_path.name}")


def save_selected_users_table(selected: dict, out_path: Path):
    """Save a summary CSV of the selected Q1 users."""
    rows = []
    for genre, info in selected.items():
        rows.append({
            "target_genre":  genre,
            "userId":        info["userId"],
            "genre_fraction": round(info["fraction"], 4),
            "total_ratings": info["total_ratings"],
            "genre_count":   info["genre_counts"].get(genre, 0),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"  Saved selected users table → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════
# Q2 — Popularity bias
# ══════════════════════════════════════════════════════════════════════════

def build_popular_set(db: MovieLensDB, train_start: str,
                      train_end: str) -> set:
    """
    Return set of top-POP_TOP_N movie IDs by rating count
    within the training window.
    """
    train_df = db.get_ratings_by_daterange(train_start, train_end)
    counts   = train_df["movieId"].value_counts()
    top_ids  = set(counts.head(POP_TOP_N).index.tolist())
    logger.info(
        f"  Popular set: top-{POP_TOP_N} movies from "
        f"{train_start} → {train_end}"
    )
    return top_ids


def compute_popularity_fractions(recs_df: pd.DataFrame,
                                  popular_set: set) -> pd.Series:
    """
    For each user compute fraction of top-20 recs that are popular.
    Returns Series: userId -> fraction.
    """
    def frac(grp):
        return grp["movieId"].isin(popular_set).mean()

    return (
        recs_df.sort_values("rank")
               .groupby("userId")
               .head(TOP_N_RECS)
               .groupby("userId")
               .apply(frac)
    )


def plot_popularity_bias(pop_fracs: pd.Series, meta: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(0, 1, 21)
    ax.hist(pop_fracs.values, bins=bins, color="#378ADD",
            edgecolor="white", alpha=0.85)

    mean_frac = pop_fracs.mean()
    ax.axvline(mean_frac, color="#E24B4A", linestyle="--", lw=2,
               label=f"Mean = {mean_frac:.3f}")

    ax.set_title(
        f"Popularity bias — fraction of top-{TOP_N_RECS} recs\n"
        f"that are top-{POP_TOP_N} popular movies\n{meta_str(meta)}",
        fontsize=FONT_SIZE + 1,
    )
    ax.set_xlabel(
        f"Fraction of top-{TOP_N_RECS} recs in top-{POP_TOP_N} popular",
        fontsize=FONT_SIZE,
    )
    ax.set_ylabel("Number of users", fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE - 1)
    ax.legend(fontsize=FONT_SIZE)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════
# Q4 — Genre consistency heatmap
# ══════════════════════════════════════════════════════════════════════════

def build_rec_genre_matrix(selected: dict, recs_df: pd.DataFrame,
                            db: MovieLensDB) -> pd.DataFrame:
    """
    Build a DataFrame: rows = selected users (labelled by dominant genre),
    columns = all genres seen in recs, values = fraction of top-20 recs
    in that genre.
    """
    # collect all genres across all recs for column ordering
    all_genres = set()
    for info in selected.values():
        uid      = info["userId"]
        user_recs = recs_df[recs_df["userId"] == uid]
        dist      = get_recs_genre_distribution(user_recs, db)
        all_genres.update(dist.keys())

    # sort columns: FIVE_GENRES first, then rest alphabetically
    five_present = [g for g in FIVE_GENRES if g in all_genres]
    rest         = sorted(all_genres - set(FIVE_GENRES))
    col_order    = five_present + rest

    rows = {}
    for dominant_genre, info in selected.items():
        uid       = info["userId"]
        user_recs = recs_df[recs_df["userId"] == uid]
        dist      = get_recs_genre_distribution(user_recs, db, target_genre=dominant_genre)
        row_label = f"{dominant_genre}\n(user {uid})"
        rows[row_label] = {g: dist.get(g, 0.0) for g in col_order}

    return pd.DataFrame(rows, index=col_order).T


def plot_genre_consistency(matrix: pd.DataFrame, meta: dict, out_path: Path):
    fig, ax = plt.subplots(
        figsize=(max(12, len(matrix.columns) * 0.9), 6)
    )

    im = ax.imshow(
        matrix.values,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0, vmax=matrix.values.max(),
    )
    plt.colorbar(im, ax=ax, label="Fraction of top-20 recs")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right",
                       fontsize=FONT_SIZE - 2)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=FONT_SIZE - 1)

    # annotate cells
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.values[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center",
                        fontsize=FONT_SIZE - 4,
                        color="black" if val < 0.5 * matrix.values.max()
                        else "white")

    ax.set_title(
        f"Q4 Genre consistency — recommendation genre distribution\n"
        f"for Q1-selected users\n{meta_str(meta)}",
        fontsize=FONT_SIZE + 1,
    )
    ax.set_xlabel("Recommended genre", fontsize=FONT_SIZE)
    ax.set_ylabel("User (dominant genre)", fontsize=FONT_SIZE)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved → {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════
# Per-run pipeline
# ══════════════════════════════════════════════════════════════════════════

def process_run(run_dir: Path, db: MovieLensDB, out_root: Path):
    run_name = run_dir.name
    logger.info(f"\n{'='*60}\nProcessing: {run_name}\n{'='*60}")

    out_dir = out_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────
    try:
        enc, recs_df, train_args = load_run_artefacts(run_dir)
    except FileNotFoundError as e:
        logger.warning(f"  Skipping: {e}")
        return

    meta = parse_run_meta(run_name)

    # resolve training window
    train_start = train_args.get("start") if train_args else meta["start"]
    train_end   = train_args.get("end")   if train_args else meta["end"]

    # ── Q1: select genre-heavy users ─────────────────────────────────────
    logger.info("  [Q1] Selecting genre-heavy users ...")
    movie_genre_lookup = build_movie_genre_lookup(db)
    selected = select_genre_heavy_users(recs_df, db, enc, movie_genre_lookup)

    save_selected_users_table(selected, out_dir / "selected_users.csv")

    for genre, info in selected.items():
        uid       = info["userId"]
        user_recs = recs_df[recs_df["userId"] == uid]
        if user_recs.empty:
            logger.warning(f"  No recs found for user {uid}, skipping Q1 {genre}.")
            continue
        plot_case_study(
            genre=genre,
            user_info=info,
            user_recs=user_recs,
            db=db,
            meta=meta,
            out_path=out_dir / f"q1_case_study_{genre.lower().replace('-','')}.png",
        )

    # ── Q2: popularity bias ───────────────────────────────────────────────
    logger.info("  [Q2] Popularity bias audit ...")
    popular_set = build_popular_set(db, train_start, train_end)
    pop_fracs   = compute_popularity_fractions(recs_df, popular_set)
    plot_popularity_bias(pop_fracs, meta, out_dir / "q2_popularity_bias.png")

    # ── Q4: genre consistency heatmap ─────────────────────────────────────
    logger.info("  [Q4] Genre consistency heatmap ...")
    matrix = build_rec_genre_matrix(selected, recs_df, db)
    if not matrix.empty:
        plot_genre_consistency(matrix, meta, out_dir / "q4_genre_consistency.png")
    else:
        logger.warning("  Empty genre matrix — skipping Q4.")

    logger.info(f"  Done → {out_dir}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Qualitative recommendation analysis"
    )
    p.add_argument("--log_root", type=str, default="train_logs/MF",
                   help="Root folder with run subdirectories")
    p.add_argument("--data_dir", type=str, default="data/ml-32m",
                   help="MovieLens data directory")
    p.add_argument("--out_root", type=str, default="analysis/recs",
                   help="Output root folder")
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
                       and (d / "recs_window_01.csv").exists()
    ])

    if not run_dirs:
        logger.error(f"No valid run folders in {log_root}")
        return

    logger.info(f"Found {len(run_dirs)} run(s)")

    db = MovieLensDB(data_dir=args.data_dir)
    db.load_data()

    for run_dir in run_dirs:
        try:
            process_run(run_dir, db, out_root)
        except Exception as e:
            logger.error(f"Failed on {run_dir.name}: {e}", exc_info=True)

    logger.info(f"\nAll done. Outputs in: {out_root}")


if __name__ == "__main__":
    main()