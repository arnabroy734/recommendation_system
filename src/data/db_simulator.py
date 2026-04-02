import pandas as pd
import os
from datetime import datetime
import numpy as np

# -------------------------------------------------------------------------
# Genre Vocabulary (fixed ordering — never change after first use)
# -------------------------------------------------------------------------
GENRE_VOCAB = [
    "Action", "Thriller", "Sci-Fi", "Horror", "Romance",
    "Drama", "Adventure", "Documentary", "Crime", "Comedy",
    "Mystery", "Children"
]

class MovieLensDB:
    """
    A database simulator for MovieLens dataset.
    Abstracts CSV files as a database with query-like operations.
    """

    def __init__(self, data_dir: str = "data/ml-32m"):
        self.data_dir = data_dir
        self.ratings_df = None
        self.movies_df = None
        self.links_df = None
        self._loaded = False
        self._genre_index = {}       # ← ADD: movieId -> set of genres
        self._genre_vocab = GENRE_VOCAB  # ← ADD

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------

    def load_data(self):
        """Load all CSVs into memory. Call once before any other operation."""

        print("Loading ratings...")
        self.ratings_df = pd.read_csv(os.path.join(self.data_dir, "ratings.csv"))
        self.ratings_df["timestamp"] = pd.to_datetime(
            self.ratings_df["timestamp"], unit="s"
        )

        print("Loading movies...")
        self.movies_df = pd.read_csv(os.path.join(self.data_dir, "movies.csv"))
        self.movies_df["genres"] = self.movies_df["genres"].str.split("|")
        self._genre_index = {
            row["movieId"]: set(row["genres"])
            for _, row in self.movies_df.iterrows()
        }

        print("Loading links...")
        self.links_df = pd.read_csv(os.path.join(self.data_dir, "links.csv"))

        self._loaded = True
        print("Data loaded successfully.")

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Data not loaded. Call load_data() first.")

    # -------------------------------------------------------------------------
    # User Operations
    # -------------------------------------------------------------------------

    def get_user_ids(self) -> list:
        """Return all unique user IDs."""
        self._check_loaded()
        return self.ratings_df["userId"].unique().tolist()

    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        """Return all ratings by a specific user."""
        self._check_loaded()
        return self.ratings_df[self.ratings_df["userId"] == user_id].copy()

    def get_user_history(self, user_id: int) -> list:
        """Return list of movieIds already rated by a user. 
        Use this to exclude seen items from recommendations."""
        self._check_loaded()
        return self.ratings_df[self.ratings_df["userId"] == user_id]["movieId"].tolist()

    def get_active_users(
        self,
        window_start: str,
        window_end: str,
        min_ratings: int = 10
    ) -> list:
        """
        Return user IDs with at least min_ratings within the given date window.

        Args:
            window_start : start date string e.g. "2023-01-01"
            window_end   : end date string e.g. "2023-06-30"
            min_ratings  : minimum number of ratings to qualify

        Returns:
            list of qualifying user IDs
        """
        self._check_loaded()
        mask = (
            self.ratings_df["timestamp"] >= pd.Timestamp(window_start)
        ) & (
            self.ratings_df["timestamp"] <= pd.Timestamp(window_end)
        )
        windowed = self.ratings_df[mask]
        counts = windowed.groupby("userId").size()
        return counts[counts >= min_ratings].index.tolist()

    # -------------------------------------------------------------------------
    # Movie Operations
    # -------------------------------------------------------------------------

    def get_movie_ids(self) -> list:
        """Return all unique movie IDs."""
        self._check_loaded()
        return self.movies_df["movieId"].unique().tolist()

    def get_movie_by_id(self, movie_id: int) -> dict:
        """Return movie details for a single movie ID."""
        self._check_loaded()
        row = self.movies_df[self.movies_df["movieId"] == movie_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()

    def get_movies_by_ids(self, movie_ids: list) -> pd.DataFrame:
        """Batch fetch movie details for a list of movie IDs."""
        self._check_loaded()
        return self.movies_df[self.movies_df["movieId"].isin(movie_ids)].copy()

    def get_movies_by_genres(self, genres: list, match: str = "any") -> pd.DataFrame:
        """
        Filter movies by genre(s).

        Args:
            genres : list of genre strings e.g. ["Drama", "War"]
            match  : "any" → movie has at least one of the genres (OR logic)
                     "all" → movie has all of the genres (AND logic)

        Returns:
            filtered DataFrame of movies
        """
        self._check_loaded()
        genres_set = set(genres)

        if match == "any":
            mask = self.movies_df["genres"].apply(
                lambda g: bool(genres_set.intersection(g))
            )
        elif match == "all":
            mask = self.movies_df["genres"].apply(
                lambda g: genres_set.issubset(set(g))
            )
        else:
            raise ValueError("match must be 'any' or 'all'")

        return self.movies_df[mask].copy()

    def get_popular_movies(self, top_n: int = 100) -> pd.DataFrame:
        """
        Return top N movies by total rating count.
        Useful as cold start fallback and popularity baseline.
        """
        self._check_loaded()
        counts = self.ratings_df.groupby("movieId").size().reset_index(name="rating_count")
        top_ids = counts.nlargest(top_n, "rating_count")["movieId"]
        result = self.movies_df[self.movies_df["movieId"].isin(top_ids)].copy()
        result = result.merge(counts, on="movieId")
        return result.sort_values("rating_count", ascending=False)

    def get_active_movies(
        self,
        window_start: str,
        window_end: str,
        min_ratings: int = 10
    ) -> list:
        """
        Return movie IDs with at least min_ratings within the given date window.

        Args:
            window_start : start date string e.g. "2023-01-01"
            window_end   : end date string e.g. "2023-06-30"
            min_ratings  : minimum number of ratings to qualify

        Returns:
            list of qualifying movie IDs
        """
        self._check_loaded()
        mask = (
            self.ratings_df["timestamp"] >= pd.Timestamp(window_start)
        ) & (
            self.ratings_df["timestamp"] <= pd.Timestamp(window_end)
        )
        windowed = self.ratings_df[mask]
        counts = windowed.groupby("movieId").size()
        return counts[counts >= min_ratings].index.tolist()

    # -------------------------------------------------------------------------
    # Rating Operations
    # -------------------------------------------------------------------------

    def get_ratings_by_daterange(self, start: str, end: str) -> pd.DataFrame:
        """Return all ratings within a date range."""
        self._check_loaded()
        mask = (
            self.ratings_df["timestamp"] >= pd.Timestamp(start)
        ) & (
            self.ratings_df["timestamp"] <= pd.Timestamp(end)
        )
        return self.ratings_df[mask].copy()

    def get_ratings_by_user(self, user_id: int) -> pd.DataFrame:
        """Return all ratings for a specific user."""
        self._check_loaded()
        return self.ratings_df[self.ratings_df["userId"] == user_id].copy()

    def get_ratings_by_movie(self, movie_id: int) -> pd.DataFrame:
        """Return all ratings for a specific movie."""
        self._check_loaded()
        return self.ratings_df[self.ratings_df["movieId"] == movie_id].copy()

    def get_ratings_snapshot(self, as_of_date: str) -> pd.DataFrame:
        """
        Return all ratings up to a given date.
        Simulates what the system knew at time T.
        Useful for simulating retraining at different points in time.
        """
        self._check_loaded()
        return self.ratings_df[
            self.ratings_df["timestamp"] <= pd.Timestamp(as_of_date)
        ].copy()

    # -------------------------------------------------------------------------
    # Date Operations
    # -------------------------------------------------------------------------

    def get_date_range(self) -> dict:
        """Return the min and max dates in the ratings dataset."""
        self._check_loaded()
        return {
            "start": self.ratings_df["timestamp"].min(),
            "end": self.ratings_df["timestamp"].max()
        }

    # -------------------------------------------------------------------------
    # Stats Operations
    # -------------------------------------------------------------------------

    def get_dataset_stats(self) -> dict:
        """Return high level dataset statistics."""
        self._check_loaded()
        n_users = self.ratings_df["userId"].nunique()
        n_movies = self.ratings_df["movieId"].nunique()
        n_ratings = len(self.ratings_df)
        sparsity = 1 - (n_ratings / (n_users * n_movies))
        return {
            "n_users": n_users,
            "n_movies": n_movies,
            "n_ratings": n_ratings,
            "sparsity": round(sparsity, 4),
            "date_range": self.get_date_range()
        }

    def get_rating_distribution(self) -> pd.Series:
        """Return distribution of rating values."""
        self._check_loaded()
        return self.ratings_df["rating"].value_counts().sort_index()

    def get_user_activity_stats(self) -> dict:
        """Return statistics about ratings per user."""
        self._check_loaded()
        counts = self.ratings_df.groupby("userId").size()
        return {
            "mean": round(counts.mean(), 2),
            "median": counts.median(),
            "min": counts.min(),
            "max": counts.max(),
            "std": round(counts.std(), 2)
        }

    def get_item_popularity_stats(self) -> dict:
        """
        Return statistics about ratings per movie.
        Useful for understanding long tail distribution.
        """
        self._check_loaded()
        counts = self.ratings_df.groupby("movieId").size()
        return {
            "mean": round(counts.mean(), 2),
            "median": counts.median(),
            "min": counts.min(),
            "max": counts.max(),
            "std": round(counts.std(), 2)
        }

    # -------------------------------------------------------------------------
    # Genre Operations
    # -------------------------------------------------------------------------

    @property
    def genre_vocab(self) -> list:
        """Return the ordered genre vocabulary list.
        Index position = position in one-hot vector."""
        return self._genre_vocab

    def get_genre_vector(self, movie_id: int) -> np.ndarray:
        """
        Return a 12-dim float32 one-hot vector for a given movie.

        Vector ordering follows GENRE_VOCAB:
            [Action, Thriller, Sci-Fi, Horror, Romance,
            Drama, Adventure, Documentary, Crime, Comedy, Mystery, Children]

        Returns a zero vector for unknown movies (cold-start safe).

        Args:
            movie_id : original MovieLens movieId

        Returns:
            np.ndarray of shape (12,) with dtype float32
        """
        self._check_loaded()
        vec = np.zeros(len(self._genre_vocab), dtype=np.float32)
        genres = self._genre_index.get(movie_id, set())
        for idx, genre in enumerate(self._genre_vocab):
            if genre in genres:
                vec[idx] = 1.0
        return vec

    def get_genre_vectors_batch(self, movie_ids: list) -> np.ndarray:
        """
        Return genre one-hot vectors for a list of movie IDs.

        Use this in SASRec data loading to vectorize an entire
        user sequence in one call.

        Args:
            movie_ids : list of original MovieLens movieIds, length N

        Returns:
            np.ndarray of shape (N, 12) with dtype float32
            Rows preserve the order of movie_ids.
            Unknown movies produce zero rows (cold-start safe).
        """
        self._check_loaded()
        return np.stack(
            [self.get_genre_vector(mid) for mid in movie_ids],
            axis=0
        )

    def get_user_history(
        self,
        user_id: int,
        as_of_date: str,
        limit: int = 200,
    ) -> list[dict]:
        """
        Fetch a user's interaction history up to as_of_date,
        ordered by timestamp DESC, limited to `limit` most recent.

        Returns list of dicts: {movie_id, rating, timestamp}
        """
        df = self.ratings_df[
            (self.ratings_df["userId"] == user_id) &
            (self.ratings_df["timestamp"] <= pd.Timestamp(as_of_date))
        ].sort_values("timestamp", ascending=False).head(limit)

        return df[["movieId", "rating", "timestamp"]].rename(
            columns={"movieId": "movie_id"}
        ).to_dict(orient="records")

# -----------------------------------------------------------------------------
# Quick sanity check
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    db = MovieLensDB(data_dir="data/ml-32m")
    db.load_data()

    print("\n--- Dataset Stats ---")
    print(db.get_dataset_stats())

    print("\n--- Date Range ---")
    print(db.get_date_range())

    print("\n--- Rating Distribution ---")
    print(db.get_rating_distribution())

    print("\n--- User Activity Stats ---")
    print(db.get_user_activity_stats())

    print("\n--- Item Popularity Stats ---")
    print(db.get_item_popularity_stats())

    print("\n--- Active Users (2015-01-01 to 2015-06-30, min 10 ratings) ---")
    active_users = db.get_active_users("2015-01-01", "2015-06-30", min_ratings=10)
    print(f"Qualifying users: {len(active_users)}")

    print("\n--- Active Movies (2015-01-01 to 2015-06-30, min 10 ratings) ---")
    active_movies = db.get_active_movies("2015-01-01", "2015-06-30", min_ratings=10)
    print(f"Qualifying movies: {len(active_movies)}")

    print("\n--- Movies by Genre (Drama OR War) ---")
    movies = db.get_movies_by_genres(["Drama", "War"], match="any")
    print(f"Movies found: {len(movies)}")

    print("\n--- Popular Movies (top 5) ---")
    print(db.get_popular_movies(top_n=5)[["movieId", "title", "rating_count"]])

    print("\n--- Genre Vector (movieId=1) ---")
    print(db.genre_vocab)
    print(db.get_genre_vector(1))

    print("\n--- Batch Genre Vectors (movieIds=[1, 2, 3]) ---")
    print(db.get_genre_vectors_batch([1, 2, 3]))  # shape (3, 12)