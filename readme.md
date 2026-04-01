# Movie Recommendation System

A two-stage recommendation system built on the [MovieLens 32M](https://grouplens.org/datasets/movielens/) dataset, combining Matrix Factorisation (retrieval) and SASRec (sequential re-ranking).

---

## Data

The project uses the **MovieLens 32M** dataset — ~32 million ratings by ~200,000 users on ~87,000 movies. Three CSV files:

- `ratings.csv` — userId, movieId, rating (0.5–5.0), timestamp
- `movies.csv` — movieId, title, genres (pipe-separated)
- `links.csv` — movieId to IMDb/TMDb ID mappings

### Download

```bash
bash data_download.sh
```

Downloads and unzips the dataset into `data/ml-32m/`.

---

## Project Structure

```
├── data/
│   └── ml-32m/                         ← MovieLens dataset
├── src/
│   ├── data/
│   │   ├── db_simulator.py             ← MovieLensDB wrapper
│   │   └── eda.py                      ← Exploratory data analysis
│   ├── training/
│   │   ├── matrix_factorisation.py     ← MF with MSE/BPR loss
│   │   ├── sasrec_architecture.py      ← SASRec + SASRecWithGenre model definitions
│   │   └── sasrec.py                   ← SASRec training + evaluation
│   ├── eval/
│   │   └── sasrec_eval_candidates.py   ← SASRec re-ranking eval on MF candidate pool
│   └── analysis/
│       ├── rec_analysis.py             ← MF recommendation qualitative analysis
│       ├── bias_analysis.py            ← MF bias term analysis
│       ├── drift_analysis.py           ← Temporal drift analysis
│       ├── dim_reduction.py            ← Embedding dimensionality reduction (PCA/UMAP)
│       ├── sasrec_analysis.py          ← SASRec attention + genre order analysis
│       └── mf_sasrec_compare.py        ← MF vs SASRec side-by-side comparison
├── train_logs/
│   ├── MF/                             ← MF training runs
│   └── SASREC/                         ← SASRec training runs
├── eval_results/
│   └── SASREC_ON_CANDIDATES/           ← Re-ranking evaluation results
└── analysis/                           ← Analysis outputs (plots, CSVs)
```

---

## DB Simulator

`src/data/db_simulator.py` wraps the raw CSVs in a `MovieLensDB` class with query-like methods.

**Key capabilities:**
- **User queries** — ratings, watch history, active users in a time window
- **Movie queries** — fetch by ID, genre, or popularity
- **Rating queries** — filter by date range or snapshot up to a point in time
- **Genre vectors** — one-hot genre encoding for SASRec genre fusion
- **Stats** — dataset sparsity, user activity, item popularity

```python
from src.data.db_simulator import MovieLensDB

db = MovieLensDB(data_dir="data/ml-32m")
db.load_data()

db.get_active_users("2018-01-01", "2018-06-30", min_ratings=10)
db.get_popular_movies(top_n=50)
db.get_ratings_snapshot(as_of_date="2018-06-30")

# Genre one-hot (12-dim) for a single movie
db.get_genre_vector(movie_id=1)           # np.ndarray shape (12,)

# Batch genre vectors for a sequence
db.get_genre_vectors_batch([1, 2, 3])     # np.ndarray shape (3, 12)
```

**Genre vocabulary (fixed order, index 0–11):**
```
Action, Thriller, Sci-Fi, Horror, Romance, Drama,
Adventure, Documentary, Crime, Comedy, Mystery, Children
```

---

## EDA

```bash
python src/data/eda.py \
    --plots_dir outputs/eda \
    --data_dir data/ml-32m
```

| Argument | Default | Description |
|---|---|---|
| `--year_start` | 1995 | Start year for distribution plots |
| `--year_end` | 2000 | End year for distribution plots |
| `--window_sizes` | 30 90 180 365 | Training window sizes in days |
| `--min_ratings` | 5 10 20 50 | Activity thresholds for window analysis |

Plots saved to `outputs/eda/`.

---

## Stage 1 — Matrix Factorisation

`src/training/matrix_factorisation.py` — SGD-based MF with MSE or BPR loss. Supports user/item bias terms.

```bash
# BPR (recommended)
python src/training/matrix_factorisation.py \
    --start 2018-01-01 --end 2018-06-30 --loss bpr

# Full config
python src/training/matrix_factorisation.py \
    --start 2018-01-01 --end 2018-06-30 --loss mse \
    --dim 64 --lr 0.005 --epochs 20 --eval_k 5 10 20
```

| Argument | Default | Description |
|---|---|---|
| `--start` | required | Training window start `YYYY-MM-DD` |
| `--end` | required | Training window end `YYYY-MM-DD` |
| `--loss` | `bpr` | Loss function: `bpr` or `mse` |
| `--dim` | `64` | Embedding dimension |
| `--lr` | `0.01` | Learning rate |
| `--reg1` | `0.01` | Regularisation for P, Q |
| `--reg2` | `0.01` | Regularisation for biases |
| `--epochs` | `20` | Training epochs |
| `--min_user_ratings` | `5` | Min ratings per user (co-filter) |
| `--min_item_ratings` | `5` | Min ratings per item (co-filter) |
| `--min_rating` | `3.0` | Minimum rating to treat as positive |
| `--eval_k` | `5 10 20` | K values for Recall/NDCG evaluation |
| `--eval_months` | `5` | Months to evaluate after training end |

**Output — `train_logs/MF/{start}_{end}_{timestamp}_{loss}_dim{d}/`:**
```
model.pkl              ← trained P, Q, biases, mu
encoder.pkl            ← user/item index mappings
recs_window_01.csv     ← top-N recommendations for eval month 1
...
recs_window_05.csv
eval_summary.csv       ← Recall@K, NDCG@K per window
loss_curve.png
```

---

## Stage 2 — SASRec (Sequential Recommendation)

`src/training/sasrec.py` — Self-Attentive Sequential Recommendation. Two variants:
- **SASRec** — item ID embeddings only
- **SASRecWithGenre** — item ID + 12-dim genre one-hot fused via a learned linear projection

```bash
# Base SASRec
python src/training/sasrec.py \
    --start 2018-01-01 --end 2018-06-30 --loss bpr

# With genre fusion
python src/training/sasrec.py \
    --start 2018-01-01 --end 2018-06-30 --loss bpr --use_genre

# Full config
python src/training/sasrec.py \
    --start 2018-01-01 --end 2018-06-30 \
    --loss bpr --dim 64 --n_layers 2 --n_heads 2 \
    --max_len 200 --dropout 0.2 --epochs 30 --lr 0.001 \
    --use_genre --eval_k 5 10 20
```

| Argument | Default | Description |
|---|---|---|
| `--start` | required | Training window start `YYYY-MM-DD` |
| `--end` | required | Training window end `YYYY-MM-DD` |
| `--loss` | `bpr` | Loss: `bpr` or `bce` |
| `--dim` | `64` | Embedding dimension |
| `--n_layers` | `2` | Transformer encoder layers |
| `--n_heads` | `2` | Attention heads |
| `--max_len` | `200` | Max sequence length (left-padded) |
| `--dropout` | `0.2` | Dropout rate |
| `--lr` | `0.001` | Learning rate |
| `--epochs` | `30` | Training epochs |
| `--use_genre` | `False` | Enable 12-dim genre one-hot fusion |
| `--min_user_ratings` | `5` | Min ratings per user (co-filter) |
| `--min_item_ratings` | `10` | Min ratings per item (co-filter) |
| `--min_rating` | `3.0` | Minimum rating to treat as positive |
| `--eval_k` | `5 10 20` | K values for Recall/NDCG |
| `--eval_months` | `5` | Months to evaluate after training end |

**Output — `train_logs/SASREC/{start}_{end}_{timestamp}_{loss}_dim{d}/`:**
```
model.pt               ← trained model weights
encoder.pkl            ← user/item index mappings
recs_window_01.csv     ← top-N recommendations for eval month 1
...
recs_window_05.csv
eval_summary.csv       ← Recall@K, NDCG@K per window
loss_curve.png
```

---

## Two-Stage Pipeline

```
Stage 1 — MF Retrieval
  Scores all items via P_u · Q_i
  Saves top-N candidates per user → recs_window_XX.csv

         ↓  top-500 candidate pool per user

Stage 2 — SASRec Re-ranking
  Reads user training sequence
  Runs SASRec forward pass → h_u (last-position hidden state)
  Scores candidates: h_u · item_emb[i] for i in pool
  Re-ranks pool by sequential relevance
```

**Known limitation:** MF candidate pool coverage is ~19% for a 6-month training window — 81% of ground truth items are never retrieved by MF. SASRec re-ranking achieves 6–7x lift over random within the pool but is bounded by retrieval coverage. Replacing MF with a broader retriever is the recommended next step.

---

## Evaluation — SASRec on MF Candidates

`src/eval/sasrec_eval_candidates.py` runs two evaluations against the MF candidate pool:

- **Eval 1 (Batch)** — SASRec re-ranks MF top-N candidates; measures Recall@K and NDCG@K vs ground truth per eval window
- **Eval 2 (Sequential)** — Sliding next-item prediction; SASRec sees training sequence + partial eval month, predicts next item within MF pool; measures HitRate@K per month and aggregate with random baseline and lift

```bash
python src/eval/sasrec_eval_candidates.py \
    --start 2018-01-01 --end 2018-06-30 \
    --dim 64 --max_len 200 --n_heads 2 --n_layers 2 \
    --top_n 500 --eval_k 5 10 --eval2_months 2 \
    --min_eval_ratings 10
```

| Argument | Default | Description |
|---|---|---|
| `--start` / `--end` | required | Must match an existing MF and SASRec run |
| `--dim`, `--max_len`, `--n_heads`, `--n_layers` | — | Must match the trained SASRec model exactly |
| `--top_n` | `300` | Top-N MF candidates to re-rank |
| `--eval_k` | `5 10` | K values for all metrics |
| `--eval2_months` | `2` | Months to include in sequential eval |
| `--min_eval_ratings` | `10` | Min ground truth ratings to qualify a user |

**Output — `eval_results/SASREC_ON_CANDIDATES/{start}_{end}_{timestamp}_top{N}/`:**
```
eval1_batch_summary.csv         ← Recall@K, NDCG@K per window
eval2_sequential_summary.csv    ← HitRate@K per month + aggregate + random baseline + lift
eval1_recall_over_windows.png
eval1_ndcg_over_windows.png
eval2_hitrate_per_month.png
eval.log
```

---

## Analysis Scripts

All scripts in `src/analysis/` save outputs to `analysis/{script_name}/{run_folder}/`.

### MF Recommendation Analysis

```bash
python src/analysis/rec_analysis.py \
    --logroot train_logs/MF --datadir data/ml-32m
```

Produces per run:
- **Q1** — Genre-heavy user case studies: watch history distribution vs top-20 recommendations
- **Q2** — Popularity bias histogram: fraction of recs in top-500 popular movies
- **Q4** — Genre consistency heatmap: recommendation genre distribution for selected users

### MF Bias Analysis

```bash
python src/analysis/bias_analysis.py \
    --logroot train_logs/MF --datadir data/ml-32m --topn 20
```

Produces per run:
- Top/bottom N users: model bias vs raw average rating
- Top/bottom N movies: model bias vs raw average rating
- Genre-level bias vs raw average rating (with error bars)
- Item bias vs popularity scatter (Pearson correlation annotated)

### Temporal Drift Analysis

```bash
python src/analysis/drift_analysis.py \
    --start 2018-01-01 --datadir data/ml-32m \
    --logroot train_logs/MF --min_ratings 10 --topn_movies 200
```

Measures cumulative drift from training baseline across training + eval months:
- **Item popularity drift** — Jaccard distance between top-N movies of month *i* vs month 1
- **User preference drift** — mean per-user JSD between genre distribution of month *i* vs month 1

Produces individual and combined drift curves with training/eval region shading.

### Embedding Dimensionality Reduction

```bash
python src/analysis/dim_reduction.py \
    --start 2018-01-01 --end 2018-06-30 --datadir data/ml-32m
```

PCA and UMAP projections of MF and SASRec item/user embeddings, coloured by genre and popularity.

### SASRec Attention Analysis

```bash
python src/analysis/sasrec_analysis.py \
    --start 2018-01-01 --end 2018-06-30 \
    --loss bpr --dim 64 --n_heads 2 --n_layers 2 --max_len 50
```

Produces:
- **Attention heatmaps** — last-layer attention bar charts for sampled users (which past items the model attends to most)
- **Genre order experiment** — same 20 Action + 10 Comedy movies in 3 different orderings; shows SASRec is order-sensitive while MF is not

### MF vs SASRec Comparison

```bash
python src/analysis/mf_sasrec_compare.py \
    --start 2018-01-01 --end 2018-06-30 \
    --mf_root train_logs/MF --sasr_root train_logs/SASREC \
    --ndcg_k 10 --n_shift 3
```

Produces:
- **Length-wise NDCG@K** — both models compared across user sequence length buckets
- **Genre-shift case studies** — users whose taste shifted mid-window; shows SASRec adapts to recent preferences while MF averages over the full history

---

## Run Folder Naming Convention

All training runs are saved as:
```
{start}_{end}_{YYYYMMDD_HHMMSS}_{loss}_dim{d}/
e.g. 2018-01-01_2018-06-30_20260401_211517_bpr_dim64/
```

When multiple runs exist for the same date range, scripts always pick the **most recent** matching run automatically.