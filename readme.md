# Movie Recommendation System

A two-stage recommendation system built on the [MovieLens 32M](https://grouplens.org/datasets/movielens/) dataset, combining **Matrix Factorisation** (retrieval) and **SASRec** (sequential re-ranking). Experiments are tracked with **MLflow**.

***

## Table of Contents

- [Data](#data)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [DB Simulator](#db-simulator)
- [Experiment Tracking](#experiment-tracking)
- [Stage 1 — Matrix Factorisation](#stage-1--matrix-factorisation)
- [Stage 2 — SASRec](#stage-2--sasrec-sequential-recommendation)
- [Two-Stage Pipeline](#two-stage-pipeline)
- [Run Naming Convention](#run-naming-convention)

***

## Data

The project uses the **MovieLens 32M** dataset — ~32 million ratings by ~200,000 users on ~87,000 movies.

| File | Columns |
|---|---|
| `ratings.csv` | `userId`, `movieId`, `rating` (0.5–5.0), `timestamp` |
| `movies.csv` | `movieId`, `title`, `genres` (pipe-separated) |
| `links.csv` | `movieId` → IMDb/TMDb ID |

### Download

```bash
bash data_download.sh
```

Downloads and unzips the dataset into `data/ml-32m/`.

***

## Project Structure

```
recommendation_system/
├── data/
│   └── ml-32m/                          ← MovieLens dataset
├── src/
│   ├── data/
│   │   └── db_simulator.py              ← MovieLensDB wrapper
│   ├── training/
│   │   ├── matrix_factorisation.py      ← MF with MSE/BPR loss (SGD)
│   │   ├── sasrec_architecture.py       ← SASRec + SASRecWithGenre definitions
│   │   └── sasrec.py                    ← SASRec training + evaluation
│   ├── tracking/
│   │   ├── tracker.py                   ← ExperimentTracker (MLflow wrapper)
│   │   └── config.py                    ← Experiment names
│   └── artifacts/
│       └── local_store.py               ← LocalArtifactStore
├── mlruns/                              ← MLflow artifact store (auto-created, gitignored)
├── run_mf.sh                            ← Shell script to train MF
├── run_sasrec.sh                        ← Shell script to train SASRec
├── data_download.sh
├── pyproject.toml
└── README.md
```

***

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install the project as an editable package

```bash
pip install -e .
```

This registers `src` as a package so all `src.*` imports work from anywhere — no `sys.path` hacks needed.

### 3. Download data

```bash
bash data_download.sh
```

***

## DB Simulator

`src/data/db_simulator.py` wraps the raw CSVs in a `MovieLensDB` class with query-like methods.

```python
from src.data.db_simulator import MovieLensDB

db = MovieLensDB()
db.load_data()

db.get_active_users("2018-01-01", "2018-06-30", min_ratings=10)
db.get_popular_movies(top_n=50)
db.get_ratings_by_daterange("2018-01-01", "2018-06-30")

# Genre one-hot (12-dim) for a single movie
db.get_genre_vector(movie_id=1)              # np.ndarray (12,)

# Batch genre vectors for a sequence (0 → zero vector for padding)
db.get_genre_vectors_batch([1, 2, 0, 3])    # np.ndarray (4, 12)
```

**Genre vocabulary (fixed order, index 0–11):**
```
Action, Thriller, Sci-Fi, Horror, Romance, Drama,
Adventure, Documentary, Crime, Comedy, Mystery, Children
```

***

## Experiment Tracking

All training runs are tracked with **MLflow** via `ExperimentTracker`.

```bash
# Launch the MLflow UI (separate terminal)
mlflow ui
# → http://localhost:5000
```

Each run logs:

| What | Detail |
|---|---|
| **Params** | All hyperparameters |
| **Metrics** | `train_loss` per epoch + eval metrics at final epoch |
| **Tags** | `model_type`, `n_users`, `n_items`, `training_time_sec` |
| **Artifacts** | Model weights, encoder, args, loss curve |

> `mlruns/` is gitignored — never commit it.

***

## Stage 1 — Matrix Factorisation

`src/training/matrix_factorisation.py` — SGD-based MF with MSE or BPR loss. Supports user/item bias terms.

```bash
# Recommended
bash run_mf.sh

# Custom config
python src/training/matrix_factorisation.py \
    --start 2016-01-01 --end 2018-06-30 \
    --eval_start 2018-07-01 --eval_end 2018-12-31 \
    --loss bpr --dim 64 --lr 0.01 --epochs 20
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--start` | required | Training window start `YYYY-MM-DD` |
| `--end` | required | Training window end `YYYY-MM-DD` |
| `--eval_start` | required | Eval window start `YYYY-MM-DD` |
| `--eval_end` | required | Eval window end `YYYY-MM-DD` |
| `--loss` | `bpr` | `bpr` or `mse` |
| `--dim` | `64` | Embedding dimension |
| `--lr` | `0.01` | Learning rate |
| `--reg1` | `0.01` | Regularisation for P, Q |
| `--reg2` | `0.01` | Regularisation for biases |
| `--epochs` | `20` | Training epochs |
| `--min_rating` | `3.0` | Minimum rating to treat as positive |
| `--min_u_rat` | `10` | Min ratings per user (co-filtering) |
| `--min_i_rat` | `10` | Min ratings per item (co-filtering) |
| `--eval_k` | `5 10 20` | K values for Recall / NDCG |
| `--top_n_recs` | `50` | Top-N recommendations per user |
| `--min_eval_ratings` | `5` | Min GT ratings to qualify a user |

### MLflow Run Name
```
mf_bpr_dim64
mf_mse_dim128
```

### Artifacts (stored in MLflow)
```
encoder.pkl          ← user/item index mappings
args.pkl             ← all hyperparameters
user_embeddings.npy
item_embeddings.npy
user_bias.npy
item_bias.npy
mu.npy               ← global mean
loss_curve.png
```

### Eval Metrics (logged to MLflow)
```
recall_at_5_min5      ndcg_at_5_min5
recall_at_10_min5     ndcg_at_10_min5
n_users_min5
```

***

## Stage 2 — SASRec (Sequential Recommendation)

`src/training/sasrec.py` — Self-Attentive Sequential Recommendation. Two variants:

- **SASRec** — item ID embeddings only
- **SASRecWithGenre** — item ID + 12-dim genre one-hot fused via a learned linear projection

```bash
# Base SASRec
bash run_sasrec.sh

# With genre fusion
bash run_sasrec.sh --use_genre

# Custom config
python src/training/sasrec.py \
    --start 2016-01-01 --end 2018-06-30 \
    --eval_start 2018-07-01 --eval_end 2018-12-31 \
    --loss bce --dim 64 --n_layers 2 --n_heads 2 \
    --max_len 200 --epochs 30 --lr 0.001 --use_genre
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--start` | required | Training window start `YYYY-MM-DD` |
| `--end` | required | Training window end `YYYY-MM-DD` |
| `--eval_start` | required | Eval window start `YYYY-MM-DD` |
| `--eval_end` | required | Eval window end `YYYY-MM-DD` |
| `--loss` | `bce` | `bce` or `bpr` |
| `--dim` | `64` | Embedding dimension |
| `--n_layers` | `2` | Transformer encoder layers |
| `--n_heads` | `2` | Attention heads (`dim % n_heads == 0`) |
| `--max_len` | `200` | Max sequence length (left-padded) |
| `--lr` | `0.001` | Learning rate (100-step linear warmup) |
| `--epochs` | `30` | Training epochs |
| `--batch_size` | `256` | Batch size |
| `--n_neg` | `4` | Negative samples per positive |
| `--min_rating` | `3.0` | Minimum rating to treat as positive |
| `--min_u_rat` | `10` | Min ratings per user (co-filtering) |
| `--min_i_rat` | `10` | Min ratings per item (co-filtering) |
| `--eval_k` | `5 10 15 20` | K values for Recall / NDCG / HitRate |
| `--top_n_recs` | `50` | Top-N recs for Recall / NDCG eval |
| `--min_eval_ratings` | `10` | Min GT ratings to qualify a user |
| `--use_genre` | `False` | Enable 12-dim genre fusion |
| `--n_genres` | `12` | Genre vocabulary size |

### MLflow Run Name
```
sasrec_bce_dim64
sasrec_bpr_dim64_genre
```

### Artifacts (stored in MLflow)
```
model.pt         ← trained model weights (PyTorch state_dict)
encoder.pkl      ← user/item index mappings
args.pkl         ← all hyperparameters
loss_curve.png
```

### Eval Metrics (logged to MLflow)
```
recall_at_5_min10      ndcg_at_5_min10
recall_at_10_min10     ndcg_at_10_min10
n_users_min10

hit_rate_at_5          ← sequential next-item prediction
hit_rate_at_10
```

#### HitRate@K — Sequential Next-Item

For each user, the model slides forward through their eval interactions:

```
Train history:  [i1, ..., i50]
Eval window:    [i51, i52, ..., i61]

Step 1: seq=[i1...i50]   → predict top-K → hit if i51 ∈ top-K
Step 2: seq=[i1...i51]   → predict top-K → hit if i52 ∈ top-K
...

hit_rate@K = total hits / total prediction steps
```

Items not present in the encoder (unseen during training) are skipped.

***

## Two-Stage Pipeline

```
Stage 1 — MF Retrieval
  Scores all items via P_u · Q_i + biases
  Produces top-N candidates per user

         ↓  top-500 candidate pool

Stage 2 — SASRec Re-ranking
  Builds user sequence from training history
  Runs SASRec forward pass → h_u (last hidden state)
  Scores candidates: h_u · item_emb[i]
  Re-ranks pool by sequential relevance
```

> **Known limitation:** MF candidate pool coverage is ~19% for a 6-month training window — 81% of ground truth items are never retrieved by MF. SASRec re-ranking achieves 6–7× lift over random within the pool but is bounded by retrieval coverage. Replacing MF with a broader retriever is the recommended next step.

***

## Run Naming Convention

MLflow run names follow the pattern:

```
{model}_{loss}_dim{d}[_genre]

mf_bpr_dim64
mf_mse_dim128
sasrec_bce_dim64
sasrec_bpr_dim64_genre
```

When multiple runs exist for the same config, the MLflow UI sorts by start time — the most recent run is at the top.