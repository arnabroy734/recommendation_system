import argparse
import os
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from db_simulator import MovieLensDB

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------

def save_plot(fig, plots_dir: str, filename: str):
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Saved → {path}")


# -----------------------------------------------------------------------------
# 1. Distribution Analysis
# -----------------------------------------------------------------------------

def plot_distributions(ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                       year_start: int, year_end: int, plots_dir: str):
    """
    Plot rating distribution, ratings per user, ratings per movie
    for a given year interval.
    """
    logger.info(f"Plotting distributions for {year_start} - {year_end}...")

    mask = (
        ratings_df["timestamp"].dt.year >= year_start
    ) & (
        ratings_df["timestamp"].dt.year <= year_end
    )
    df = ratings_df[mask]

    if df.empty:
        logger.warning(f"No data found between {year_start} and {year_end}. Skipping.")
        return

    # --- Rating distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    df["rating"].value_counts().sort_index().plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_title(f"Rating Distribution ({year_start}-{year_end})")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    save_plot(fig, plots_dir, f"rating_distribution_{year_start}_{year_end}.png")

    # --- Ratings per user ---
    ratings_per_user = df.groupby("userId").size()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratings_per_user, bins=50, color="steelblue", edgecolor="black", log=True)
    ax.set_title(f"Ratings per User ({year_start}-{year_end})")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Number of Users (log scale)")
    ax.axvline(ratings_per_user.median(), color="red", linestyle="--",
               label=f"Median: {ratings_per_user.median():.0f}")
    ax.legend()
    save_plot(fig, plots_dir, f"ratings_per_user_{year_start}_{year_end}.png")

    # --- Ratings per movie ---
    ratings_per_movie = df.groupby("movieId").size()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratings_per_movie, bins=50, color="darkorange", edgecolor="black", log=True)
    ax.set_title(f"Ratings per Movie ({year_start}-{year_end})")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Number of Movies (log scale)")
    ax.axvline(ratings_per_movie.median(), color="red", linestyle="--",
               label=f"Median: {ratings_per_movie.median():.0f}")
    ax.legend()
    save_plot(fig, plots_dir, f"ratings_per_movie_{year_start}_{year_end}.png")

    logger.info("Distribution plots done.")


# -----------------------------------------------------------------------------
# 2. Temporal Trends
# -----------------------------------------------------------------------------

def plot_temporal_trends(ratings_df: pd.DataFrame, plots_dir: str):
    """
    Plot ratings volume over time and new users over time (yearly).
    """
    logger.info("Plotting temporal trends...")

    ratings_df = ratings_df.copy()
    ratings_df["year"] = ratings_df["timestamp"].dt.year

    # --- Ratings volume per year ---
    volume = ratings_df.groupby("year").size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(volume["year"], volume["count"], marker="o", color="steelblue", linewidth=2)
    ax.fill_between(volume["year"], volume["count"], alpha=0.2, color="steelblue")
    ax.set_title("Ratings Volume Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Ratings")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save_plot(fig, plots_dir, "ratings_volume_over_time.png")

    # --- New users per year ---
    first_seen = ratings_df.groupby("userId")["year"].min().reset_index(name="first_year")
    new_users = first_seen.groupby("first_year").size().reset_index(name="new_users")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(new_users["first_year"], new_users["new_users"], color="seagreen", edgecolor="black")
    ax.set_title("New Users Over Time (Year of First Rating)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of New Users")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    save_plot(fig, plots_dir, "new_users_over_time.png")

    logger.info("Temporal trend plots done.")


# -----------------------------------------------------------------------------
# 3. User Genre Preference Over Time
# -----------------------------------------------------------------------------

def plot_user_genre_preference(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    plots_dir: str
):
    """
    Pick a random power user (100+ ratings) from the entire dataset.
    Find their personal min and max rating dates, divide into 4 equal
    time intervals, and plot liked genre preference (rating >= 4.0)
    for each interval to show how taste changed over time.
    Layout: 2 x 2
    """
    logger.info("Plotting user genre preference over lifetime...")
 
    # Find all power users from entire dataset
    user_counts = ratings_df.groupby("userId").size()
    power_users = user_counts[user_counts >= 100].index.tolist()
 
    if not power_users:
        logger.warning("No power users (100+ ratings) found. Lowering threshold to 30.")
        power_users = user_counts[user_counts >= 30].index.tolist()
 
    if not power_users:
        logger.warning("No qualifying users found. Skipping.")
        return
 
    # Random power user
    user_id = random.choice(power_users)
    logger.info(f"Selected power user: {user_id} ({user_counts[user_id]} total ratings)")
 
    # Get all ratings for this user
    user_df = ratings_df[ratings_df["userId"] == user_id].copy()
 
    # Find personal time range
    t_min = user_df["timestamp"].min()
    t_max = user_df["timestamp"].max()
    total_duration = t_max - t_min
    interval = total_duration / 4
 
    logger.info(f"User active period: {t_min.date()} to {t_max.date()}")
    logger.info(f"Interval duration: {interval.days} days each")
 
    # Build 4 equal intervals
    intervals = []
    for i in range(4):
        i_start = t_min + i * interval
        i_end   = t_min + (i + 1) * interval
        label   = f"{i_start.strftime('%b %Y')} - {i_end.strftime('%b %Y')}"
        intervals.append((i_start, i_end, label))
 
    # Merge genres and explode
    user_df = user_df.merge(movies_df[["movieId", "genres"]], on="movieId", how="left")
    user_df = user_df.explode("genres")
    user_df = user_df[user_df["genres"] != "(no genres listed)"]
 
    # Keep only liked ratings
    liked = user_df[user_df["rating"] >= 4.0]
 
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
 
    fig.suptitle(
        f"User {user_id} — Liked Genre Preference Over Lifetime\n"
        f"({t_min.strftime('%b %Y')} to {t_max.strftime('%b %Y')}, "
        f"{user_counts[user_id]} total ratings)",
        fontsize=14,
        y=1.02
    )
 
    for ax, (i_start, i_end, label) in zip(axes, intervals):
        interval_liked = (
            liked[liked["timestamp"].between(i_start, i_end)]["genres"]
            .value_counts()
            .head(10)
            .sort_values(ascending=True)  # largest bar appears on top
        )
 
        ax.set_title(label, fontsize=11, fontweight="bold")
 
        if interval_liked.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
 
        interval_liked.plot(kind="barh", ax=ax, color="seagreen", edgecolor="black")
        ax.set_xlabel("Count")
        ax.set_ylabel("Genre")
 
    plt.tight_layout()
    save_plot(fig, plots_dir, f"user_{user_id}_genre_preference_lifetime.png")
    logger.info(f"User genre preference lifetime plot saved for user {user_id}.")
 
# -----------------------------------------------------------------------------
# Movie Lifecycle Analysis
# -----------------------------------------------------------------------------
def plot_movie_lifecycle(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    plots_dir: str,
    min_ratings: int = 500
):
    """
    Pick a random movie with at least min_ratings total ratings.
    Plot year-wise count of how many people watched (rated) that movie
    across its entire lifespan in the dataset.
 
    Args:
        ratings_df  : ratings DataFrame with timestamp already as datetime
        movies_df   : movies DataFrame with genres as list
        plots_dir   : folder to save the plot
        min_ratings : minimum total ratings to qualify as a good movie
    """
    logger.info("Plotting movie lifecycle...")
 
    # Find qualifying movies
    movie_counts = ratings_df.groupby("movieId").size()
    qualifying   = movie_counts[movie_counts >= min_ratings].index.tolist()
 
    if not qualifying:
        logger.warning(f"No movies with {min_ratings}+ ratings found. Lowering to 100.")
        qualifying = movie_counts[movie_counts >= 100].index.tolist()
 
    if not qualifying:
        logger.warning("No qualifying movies found. Skipping.")
        return
 
    # Random movie selection
    movie_id = random.choice(qualifying)
    movie_info = movies_df[movies_df["movieId"] == movie_id].iloc[0]
    movie_title  = movie_info["title"]
    movie_genres = movie_info["genres"]
    if isinstance(movie_genres, list):
        movie_genres = ", ".join(movie_genres)
 
    total_ratings = movie_counts[movie_id]
    logger.info(f"Selected movie: '{movie_title}' (id={movie_id}, {total_ratings} total ratings)")
 
    # Get all ratings for this movie
    movie_df = ratings_df[ratings_df["movieId"] == movie_id].copy()
    movie_df["year"] = movie_df["timestamp"].dt.year
 
    # Year-wise rating count
    yearly_counts = movie_df.groupby("year").size().reset_index(name="rating_count")
    yearly_counts = yearly_counts.sort_values("year")
 
    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
 
    ax.bar(
        yearly_counts["year"],
        yearly_counts["rating_count"],
        color="steelblue",
        edgecolor="black",
        width=0.6
    )
 
    # Annotate peak year
    peak_idx  = yearly_counts["rating_count"].idxmax()
    peak_year = yearly_counts.loc[peak_idx, "year"]
    peak_val  = yearly_counts.loc[peak_idx, "rating_count"]
    ax.annotate(
        f"Peak: {peak_year}",
        xy=(peak_year, peak_val),
        xytext=(peak_year, peak_val + peak_val * 0.05),
        ha="center",
        fontsize=9,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red")
    )
 
    ax.set_title(
        f"Movie Lifecycle: {movie_title}\n"
        f"Genres: {movie_genres} | Total Ratings: {total_ratings}",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Ratings")
    ax.set_xticks(yearly_counts["year"])
    ax.tick_params(axis="x", rotation=45)
 
    plt.tight_layout()
    safe_title = movie_title.replace("/", "-").replace(" ", "_")[:40]
    save_plot(fig, plots_dir, f"movie_lifecycle_{movie_id}_{safe_title}.png")
    logger.info(f"Movie lifecycle plot saved for '{movie_title}'.")

# -----------------------------------------------------------------------------
# 4. Genre Popularity Over Time
# -----------------------------------------------------------------------------

def plot_genre_popularity(ratings_df: pd.DataFrame, movies_df: pd.DataFrame,
                          plots_dir: str):
    """
    Plot top 10 genre popularity (rating count) per year.
    """
    logger.info("Plotting genre popularity over time...")

    df = ratings_df.copy()
    df["year"] = df["timestamp"].dt.year
    df = df.merge(movies_df[["movieId", "genres"]], on="movieId", how="left")
    df = df.explode("genres")
    df = df[df["genres"] != "(no genres listed)"]

    genre_year = df.groupby(["year", "genres"]).size().reset_index(name="count")
    pivot      = genre_year.pivot(index="year", columns="genres", values="count").fillna(0)

    top_genres = pivot.sum().nlargest(10).index.tolist()
    pivot      = pivot[top_genres]

    fig, ax = plt.subplots(figsize=(14, 7))
    for genre in top_genres:
        ax.plot(pivot.index, pivot[genre], marker="o", linewidth=2, label=genre)

    ax.set_title("Genre Popularity Over Time (Top 10 Genres)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Ratings")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    save_plot(fig, plots_dir, "genre_popularity_over_time.png")
    logger.info("Genre popularity plot done.")


# -----------------------------------------------------------------------------
# 5. Sparsity Matrix Visualization
# -----------------------------------------------------------------------------

def plot_sparsity_matrix(ratings_df: pd.DataFrame, time_start: str,
                         time_end: str, plots_dir: str):
    """
    Visualize interaction matrix sparsity for a sample of
    1000 users x 1000 movies within a time period.
    """
    logger.info(f"Plotting sparsity matrix for {time_start} to {time_end}...")

    mask = (
        ratings_df["timestamp"] >= pd.Timestamp(time_start)
    ) & (
        ratings_df["timestamp"] <= pd.Timestamp(time_end)
    )
    df = ratings_df[mask]

    if df.empty:
        logger.warning("No data in this period. Skipping sparsity plot.")
        return

    users  = df["userId"].unique()
    movies = df["movieId"].unique()

    sampled_users  = np.random.choice(users,  size=min(1000, len(users)),  replace=False)
    sampled_movies = np.random.choice(movies, size=min(1000, len(movies)), replace=False)

    df_sample = df[
        df["userId"].isin(sampled_users) & df["movieId"].isin(sampled_movies)
    ]

    user_idx  = {u: i for i, u in enumerate(sampled_users)}
    movie_idx = {m: i for i, m in enumerate(sampled_movies)}

    matrix = np.zeros((len(sampled_users), len(sampled_movies)), dtype=np.uint8)
    for _, row in df_sample.iterrows():
        matrix[user_idx[row["userId"]], movie_idx[row["movieId"]]] = 1

    sparsity = 1 - matrix.sum() / matrix.size
    logger.info(f"Sampled matrix sparsity: {sparsity:.4f}")

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spy(matrix, markersize=0.3, color="steelblue")
    ax.set_title(
        f"Interaction Matrix Sparsity ({time_start} to {time_end})\n"
        f"Sample: 1000 users x 1000 movies | Sparsity: {sparsity:.2%}"
    )
    ax.set_xlabel("Movies (sampled)")
    ax.set_ylabel("Users (sampled)")
    save_plot(fig, plots_dir, f"sparsity_matrix_{time_start}_{time_end}.png")
    logger.info("Sparsity matrix plot done.")


# -----------------------------------------------------------------------------
# 6. Training Window Analysis
# -----------------------------------------------------------------------------

def plot_training_window_analysis(ratings_df: pd.DataFrame,
                                  window_sizes: list,
                                  min_ratings_list: list,
                                  plots_dir: str):
    """
    For each combination of window size (days) and min_ratings threshold,
    compute how many users qualify for training.
    Uses the most recent date in dataset as reference point.
    """
    logger.info("Plotting training window analysis...")

    reference_date = ratings_df["timestamp"].max()
    logger.info(f"Reference date (most recent in dataset): {reference_date.date()}")

    results = []
    for window_days in window_sizes:
        window_start = reference_date - pd.Timedelta(days=window_days)
        df_window    = ratings_df[ratings_df["timestamp"] >= window_start]
        user_counts  = df_window.groupby("userId").size()
        total_users  = ratings_df["userId"].nunique()

        for min_r in min_ratings_list:
            qualifying = (user_counts >= min_r).sum()
            results.append({
                "window_days":      window_days,
                "min_ratings":      min_r,
                "qualifying_users": qualifying,
                "qualifying_pct":   qualifying / total_users * 100
            })

    results_df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for min_r in min_ratings_list:
        subset = results_df[results_df["min_ratings"] == min_r]
        axes[0].plot(subset["window_days"], subset["qualifying_users"],
                     marker="o", linewidth=2, label=f"min_ratings={min_r}")
        axes[1].plot(subset["window_days"], subset["qualifying_pct"],
                     marker="o", linewidth=2, label=f"min_ratings={min_r}")

    axes[0].set_title("Qualifying Users vs Window Size")
    axes[0].set_xlabel("Window Size (days)")
    axes[0].set_ylabel("Number of Qualifying Users")
    axes[0].legend()
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    axes[1].set_title("% Qualifying Users vs Window Size")
    axes[1].set_xlabel("Window Size (days)")
    axes[1].set_ylabel("% of Total Users")
    axes[1].legend()
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.suptitle(
        "Training Window Analysis\n"
        "How window size and activity threshold affect qualifying users",
        fontsize=13
    )
    plt.tight_layout()
    save_plot(fig, plots_dir, "training_window_analysis.png")

    csv_path = os.path.join(plots_dir, "training_window_analysis.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"Training window results CSV saved → {csv_path}")
    logger.info("Training window analysis done.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MovieLens EDA Script")
    parser.add_argument("--plots_dir",    type=str, required=True,
                        help="Folder to save all plots")
    parser.add_argument("--data_dir",     type=str, default="data/ml-32m",
                        help="Path to MovieLens data folder")
    parser.add_argument("--year_start",   type=int, default=1995,
                        help="Start year for distribution plots")
    parser.add_argument("--year_end",     type=int, default=2000,
                        help="End year for distribution plots")
    # parser.add_argument("--time_start",   type=str, default="1995-01-01",
    #                     help="Start date for user genre and sparsity plots (YYYY-MM-DD)")
    # parser.add_argument("--time_end",     type=str, default="2000-12-31",
    #                     help="End date for user genre and sparsity plots (YYYY-MM-DD)")
    parser.add_argument("--window_sizes", type=int, nargs="+", default=[30, 90, 180, 365],
                        help="Window sizes in days for training window analysis")
    parser.add_argument("--min_ratings",  type=int, nargs="+", default=[5, 10, 20, 50],
                        help="Min ratings thresholds for training window analysis")
    args = parser.parse_args()

    # Load data
    logger.info("Loading data...")
    db = MovieLensDB(data_dir=args.data_dir)
    db.load_data()
    ratings_df = db.ratings_df
    movies_df  = db.movies_df
    logger.info("Data loaded successfully.")

    logger.info("=" * 60)
    logger.info("Starting EDA...")
    logger.info("=" * 60)

    # plot_distributions(
    #     ratings_df, movies_df,
    #     args.year_start, args.year_end,
    #     args.plots_dir
    # )

    # plot_temporal_trends(ratings_df, args.plots_dir)

    # plot_user_genre_preference(
    #     ratings_df, movies_df,
    #     args.plots_dir
    # )

    plot_movie_lifecycle(
        ratings_df, movies_df, args.plots_dir
    )

    # plot_genre_popularity(ratings_df, movies_df, args.plots_dir)

    # plot_sparsity_matrix(
    #     ratings_df,
    #     f"01-01-{args.year_start}", f"31-12-{args.year_end}",
    #     args.plots_dir
    # )

    # plot_training_window_analysis(
    #     ratings_df,
    #     args.window_sizes,
    #     args.min_ratings,
    #     args.plots_dir
    # )

    logger.info("=" * 60)
    logger.info(f"All EDA plots saved to: {args.plots_dir}")
    logger.info("=" * 60)