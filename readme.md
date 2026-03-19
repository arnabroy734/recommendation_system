# Movie Recommendation System

A recommendation system built on the [MovieLens 32M](https://grouplens.org/datasets/movielens/) dataset.

---

## Data

The project uses the **MovieLens 32M** dataset, which contains ~32 million ratings by ~200,000 users on ~87,000 movies. It comes as three CSV files:

- `ratings.csv` — userId, movieId, rating (0.5–5.0), timestamp
- `movies.csv` — movieId, title, genres (pipe-separated)

---

## DB Simulator

`src/data/db_simulator.py` wraps the raw CSVs in a `MovieLensDB` class that mimics a database with query-like methods. This avoids scattering pandas logic across the codebase.

Key capabilities:
- **User queries** — get ratings, watch history, active users in a time window
- **Movie queries** — fetch by ID, genre, or popularity
- **Rating queries** — filter by date range or snapshot up to a point in time (useful for backtesting)
- **Stats** — dataset sparsity, user activity distribution, item popularity

```python
from src.data.db_simulator import MovieLensDB

db = MovieLensDB(data_dir="data/ml-32m")
db.load_data()

db.get_active_users("2022-01-01", "2022-12-31", min_ratings=10)
db.get_popular_movies(top_n=50)
db.get_ratings_snapshot(as_of_date="2021-06-30")
```

---

## Download Data

Run the download script from the project root:

```bash
bash data_download.sh
```

This downloads and unzips the MovieLens 32M dataset into `data/ml-32m/`.

---

## Run EDA

```bash
cd src/data
python eda.py --plots_dir ../../outputs/eda --data_dir ../../data/ml-32m
```

**Optional args:**

| Argument | Default | Description |
|---|---|---|
| `--year_start` | 1995 | Start year for distribution plots |
| `--year_end` | 2000 | End year for distribution plots |
| `--window_sizes` | 30 90 180 365 | Training window sizes in days |
| `--min_ratings` | 5 10 20 50 | Activity thresholds for window analysis |

Plots are saved to `outputs/eda/`.