"""
sasrec.py
---------
SASRec training with BCE or BPR loss.
Supports base SASRec and SASRecWithGenre.
Integrated with MLflow tracking via ExperimentTracker.

Usage examples:
    # Base SASRec BPR
    python sasrec.py --start 2016-01-01 --end 2018-06-30 \
        --eval_start 2018-07-01 --eval_end 2018-12-31 --loss bpr

    # SASRec with genre BCE
    python sasrec.py --start 2016-01-01 --end 2018-06-30 \
        --eval_start 2018-07-01 --eval_end 2018-12-31 --loss bce --use_genre
"""

import argparse
import logging
import pickle
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data.db_simulator import MovieLensDB
from src.training.matrix_factorisation import Encoder
from src.training.sasrec_architecture import SASRec, SASRecWithGenre
from src.tracking.tracker import ExperimentTracker
from src.tracking.config import EXPERIMENT_NAMES
from src.artifacts.local_store import LocalArtifactStore


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =========================================================
# Data loading
# =========================================================

def load_training_data(
    db, start, end,
    min_rating=3.0, min_user_ratings=10, min_item_ratings=10,
):
    logger.info(f"Loading data from {start} to {end} ...")
    df = db.get_ratings_by_daterange(start, end).copy()
    logger.info(f"Raw interactions: {len(df):,}")

    df = df[df["rating"] >= min_rating].copy()
    logger.info(f"After rating >= {min_rating}: {len(df):,}")

    prev_len = -1
    iteration = 0
    while prev_len != len(df):
        prev_len    = len(df)
        iteration  += 1
        user_counts = df["userId"].value_counts()
        item_counts = df["movieId"].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        valid_items = item_counts[item_counts >= min_item_ratings].index
        df = df[df["userId"].isin(valid_users) & df["movieId"].isin(valid_items)]

    logger.info(
        f"After co-filtering (iters={iteration}): {len(df):,} interactions | "
        f"{df['userId'].nunique():,} users | {df['movieId'].nunique():,} items"
    )
    return df.reset_index(drop=True)


# =========================================================
# Dataset
# =========================================================

class SASRecDataset(Dataset):
    def __init__(self, df, enc, db, max_len=200, n_neg=4, use_genre=False):
        self.max_len   = max_len
        self.n_neg     = n_neg
        self.enc       = enc
        self.use_genre = use_genre
        self.db        = db

        df = df.sort_values(["userId", "timestamp"])

        self.sequences = []
        self.all_items = np.array(list(enc.id_item.keys()))

        self.user_pos = {}
        for uid, grp in df.groupby("userId"):
            u = enc.user_id[uid]
            self.user_pos[u] = set(enc.item_id[m] for m in grp["movieId"].values)

        for uid, grp in df.groupby("userId"):
            u        = enc.user_id[uid]
            items    = [enc.item_id[i] + 1 for i in grp["movieId"].values]
            orig_ids = list(grp["movieId"].values)

            if len(items) < 2:
                continue

            full      = items[-(max_len + 1):]
            full_orig = orig_ids[-(max_len + 1):]

            seq      = full[:-1]
            targets  = full[1:]
            seq_orig = full_orig[:-1]

            pad_len  = max_len - len(seq)
            seq      = [0] * pad_len + seq
            targets  = [-1] * pad_len + targets
            seq_orig = [0] * pad_len + seq_orig

            self.sequences.append((u, seq, targets, seq_orig))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        u, seq, targets, seq_orig = self.sequences[idx]

        pos_set = self.user_pos.get(u, set())
        negs = []
        for t in targets:
            if t == -1:
                negs.append([-1] * self.n_neg)
                continue
            pos_negs = []
            while len(pos_negs) < self.n_neg:
                j = int(np.random.choice(self.all_items))
                if j not in pos_set:
                    pos_negs.append(j + 1)
            negs.append(pos_negs)

        if self.use_genre:
            genre_vecs = self.db.get_genre_vectors_batch(seq_orig)
            return (
                torch.LongTensor(seq),
                torch.LongTensor(targets),
                torch.LongTensor(negs),
                torch.FloatTensor(genre_vecs),
            )

        return (
            torch.LongTensor(seq),
            torch.LongTensor(targets),
            torch.LongTensor(negs),
        )


# =========================================================
# Training
# =========================================================

def lr_lambda(step):
    warmup_steps = 100
    return step / warmup_steps if step < warmup_steps else 1.0


def train_one_epoch(model, loader, optimizer, scheduler, device, loss_name, use_genre=False):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, leave=False):
        if use_genre:
            seq, targets, negs, genre_seq = batch
            genre_seq = genre_seq.to(device)
        else:
            seq, targets, negs = batch

        seq     = seq.to(device)
        targets = targets.to(device)
        negs    = negs.to(device)

        if use_genre:
            h, _ = model(seq, genre_seq)
        else:
            h, _ = model(seq)

        mask         = (targets != -1)
        safe_targets = targets.clamp(min=0)
        pos_emb      = model.item_emb(safe_targets)
        pos_logits   = (h * pos_emb).sum(dim=-1)

        safe_negs  = negs.clamp(min=0)
        neg_emb    = model.item_emb(safe_negs)
        neg_logits = (h.unsqueeze(2) * neg_emb).sum(dim=-1)

        if loss_name == "bce":
            pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pos_logits, torch.ones_like(pos_logits), reduction="none"
            )
            neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits), reduction="none"
            ).mean(dim=-1)
            loss = ((pos_loss + neg_loss) * mask).sum() / mask.sum()
        else:
            diff = pos_logits.unsqueeze(2) - neg_logits
            loss = -torch.log(torch.sigmoid(diff) + 1e-10)
            mask_expanded = mask.unsqueeze(2).expand_as(loss)
            loss = (loss * mask_expanded).sum() / mask_expanded.sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================
# Inference helper
# =========================================================

def get_user_representation(model, enc, db, history_orig_ids, max_len, use_genre, device):
    """
    Build padded sequence from history, run through model,
    return last hidden state (dim,) as numpy array.
    Skips items not present in enc.item_id.
    """
    known_ids = [m for m in history_orig_ids if m in enc.item_id]
    known_ids = known_ids[-max_len:]

    items   = [enc.item_id[m] + 1 for m in known_ids]
    pad_len = max_len - len(items)
    seq     = [0] * pad_len + items
    seq_t   = torch.LongTensor(seq).unsqueeze(0).to(device)

    with torch.no_grad():
        if use_genre:
            padded_orig = [0] * pad_len + known_ids
            genre_vecs  = db.get_genre_vectors_batch(padded_orig)
            genre_t     = torch.FloatTensor(genre_vecs).unsqueeze(0).to(device)
            h = model(seq_t, genre_t)[0].cpu().numpy()[0, -1]
        else:
            h = model(seq_t)[0].cpu().numpy()[0, -1]

    return h


# =========================================================
# Evaluation
# =========================================================

def evaluate(
    model, enc, db, train_df,
    eval_start, eval_end,
    min_rating, min_eval_ratings, eval_k,
    top_n_recs, max_len, use_genre, device,
) -> dict:
    """
    Single-window evaluation:

    Recall@K  — GT items retrieved in top-K (averaged per user)
    NDCG@K    — position-aware ranking quality (averaged per user)
    HitRate@K — sequential next-item prediction over eval window:
                slide seq forward for each eval interaction,
                predict top-K, check if next item is hit.
                hit_rate@K = total_hits / total_steps
    """
    model.eval()
    logger.info(f"Evaluating on window: {eval_start} → {eval_end}")

    eval_raw = db.get_ratings_by_daterange(eval_start, eval_end)
    eval_raw = eval_raw[eval_raw["rating"] >= min_rating]
    eval_raw = eval_raw[eval_raw["userId"].isin(enc.user_id)]
    eval_raw = eval_raw[eval_raw["movieId"].isin(enc.item_id)]

    if eval_raw.empty:
        logger.warning("No eval data found in the given window.")
        return {}

    eval_raw = eval_raw.sort_values(["userId", "timestamp"])

    train_history = {}
    for uid_orig, grp in train_df.groupby("userId"):
        train_history[uid_orig] = list(grp.sort_values("timestamp")["movieId"].values)

    all_item_indices = np.array(list(enc.id_item.keys()))
    emb_w            = model.item_emb.weight.detach().cpu().numpy()
    user_eval_counts = eval_raw.groupby("userId")["movieId"].count()

    gt_per_user = {
        uid: set(grp["movieId"].values)
        for uid, grp in eval_raw.groupby("userId")
    }

    # ── Recall@K and NDCG@K — one-shot recs from train history ──────────────
    recs = {}
    for uid_orig in user_eval_counts.index:
        hist   = train_history.get(uid_orig, [])
        seen   = set(enc.item_id[m] for m in hist if m in enc.item_id)
        unseen = all_item_indices[~np.isin(all_item_indices, list(seen))]
        if len(unseen) == 0:
            continue
        h              = get_user_representation(model, enc, db, hist, max_len, use_genre, device)
        scores         = emb_w[unseen + 1] @ h
        top_idx        = np.argsort(scores)[::-1][:top_n_recs]
        recs[uid_orig] = [enc.id_item[unseen[i]] for i in top_idx]

    # ── HitRate@K — sequential next-item sliding window ─────────────────────
    hit_steps = {k: {"hits": 0, "steps": 0} for k in eval_k}

    for uid_orig, grp in eval_raw.groupby("userId"):
        eval_items_known = [
            m for m in grp.sort_values("timestamp")["movieId"].values
            if m in enc.item_id
        ]
        if not eval_items_known:
            continue

        sliding_hist = list(train_history.get(uid_orig, []))

        for next_item in eval_items_known:
            seen_set = set(enc.item_id[m] for m in sliding_hist if m in enc.item_id)
            unseen   = all_item_indices[~np.isin(all_item_indices, list(seen_set))]

            if len(unseen) == 0:
                sliding_hist.append(next_item)
                continue

            h      = get_user_representation(model, enc, db, sliding_hist, max_len, use_genre, device)
            scores = emb_w[unseen + 1] @ h

            for k in eval_k:
                top_idx   = np.argsort(scores)[::-1][:k]
                top_items = set(enc.id_item[unseen[i]] for i in top_idx)
                hit_steps[k]["hits"]  += int(next_item in top_items)
                hit_steps[k]["steps"] += 1

            sliding_hist.append(next_item)

    # ── collect metrics ──────────────────────────────────────────────────────
    metrics = {}

    for min_er in min_eval_ratings:
        qualifying = user_eval_counts[user_eval_counts >= min_er].index
        if len(qualifying) == 0:
            logger.info(f"  min_eval_ratings={min_er}: no qualifying users")
            continue

        for k in eval_k:
            recalls, ndcgs = [], []
            for uid_orig in qualifying:
                gt        = gt_per_user.get(uid_orig, set())
                user_recs = recs.get(uid_orig, [])
                if not gt or not user_recs:
                    continue

                top_k = set(user_recs[:k])
                hits  = len(top_k & gt)
                recalls.append(hits / min(len(gt), k))

                idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(gt), k)))
                dcg  = sum(
                    1.0 / np.log2(r + 2)
                    for r, item in enumerate(user_recs[:k])
                    if item in gt
                )
                ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

            mean_recall = float(np.mean(recalls)) if recalls else 0.0
            mean_ndcg   = float(np.mean(ndcgs))   if ndcgs   else 0.0

            metrics[f"recall_at_{k}_min{min_er}"] = mean_recall
            metrics[f"ndcg_at_{k}_min{min_er}"]   = mean_ndcg
            metrics[f"n_users_min{min_er}"]        = len(recalls)

            logger.info(
                f"  min_ratings={min_er:2d} | K={k:2d} | "
                f"Recall@{k}={mean_recall:.4f} | NDCG@{k}={mean_ndcg:.4f} | "
                f"users={len(recalls)}"
            )

    for k in eval_k:
        steps = hit_steps[k]["steps"]
        hits  = hit_steps[k]["hits"]
        hr    = hits / steps if steps > 0 else 0.0
        metrics[f"hit_rate_at_{k}"] = hr
        logger.info(f"  HitRate@{k}={hr:.4f} | hits={hits} / steps={steps}")

    return metrics


# =========================================================
# Artifact helpers
# =========================================================

def save_artifacts(tmpdir, model, enc, args, losses):
    torch.save(model.state_dict(), tmpdir / "model.pt")
    with open(tmpdir / "encoder.pkl", "wb") as f:
        pickle.dump(enc, f)
    with open(tmpdir / "args.pkl", "wb") as f:
        pickle.dump(vars(args), f)
    _plot_losses(losses, tmpdir, args)
    logger.info(f"Artifacts prepared in {tmpdir}")


def _plot_losses(losses, out_dir, args):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    genre_label = " + genre" if args.use_genre else ""
    ax.set_title(
        f"SASRec{genre_label} | loss={args.loss} | dim={args.dim} | "
        f"lr={args.lr} | max_len={args.max_len}"
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)


# =========================================================
# Argument parsing
# =========================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--start",            type=str,   required=True)
    p.add_argument("--end",              type=str,   required=True)
    p.add_argument("--loss",             type=str,   default="bce",  choices=["bce", "bpr"])
    p.add_argument("--dim",              type=int,   default=64)
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--lr",               type=float, default=0.001)
    p.add_argument("--max_len",          type=int,   default=200)
    p.add_argument("--batch_size",       type=int,   default=256)
    p.add_argument("--n_neg",            type=int,   default=4)
    p.add_argument("--n_heads",          type=int,   default=2)
    p.add_argument("--n_layers",         type=int,   default=2)
    p.add_argument("--min_rating",       type=float, default=3.0)
    p.add_argument("--min_u_rat",        type=int,   default=10)
    p.add_argument("--min_i_rat",        type=int,   default=10)
    p.add_argument("--eval_start",       type=str,   required=True)
    p.add_argument("--eval_end",         type=str,   required=True)
    p.add_argument("--eval_k",           type=int,   nargs="+", default=[5, 10, 15, 20])
    p.add_argument("--top_n_recs",       type=int,   default=50)
    p.add_argument("--min_eval_ratings", type=int,   nargs="+", default=[10])
    p.add_argument("--use_genre",        action="store_true")
    p.add_argument("--n_genres",         type=int,   default=12)

    return p.parse_args()


# =========================================================
# Main
# =========================================================

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    assert args.dim % args.n_heads == 0, \
        f"dim ({args.dim}) must be divisible by n_heads ({args.n_heads})"

    genre_suffix = "_genre" if args.use_genre else ""
    run_name     = f"sasrec_{args.loss}_dim{args.dim}{genre_suffix}"

    store  = LocalArtifactStore()
    params = {
        "start":            args.start,
        "end":              args.end,
        "eval_start":       args.eval_start,
        "eval_end":         args.eval_end,
        "loss":             args.loss,
        "dim":              args.dim,
        "epochs":           args.epochs,
        "lr":               args.lr,
        "max_len":          args.max_len,
        "batch_size":       args.batch_size,
        "n_neg":            args.n_neg,
        "n_heads":          args.n_heads,
        "n_layers":         args.n_layers,
        "min_rating":       args.min_rating,
        "min_u_rat":        args.min_u_rat,
        "min_i_rat":        args.min_i_rat,
        "eval_k":           str(args.eval_k),
        "min_eval_ratings": str(args.min_eval_ratings),
        "top_n_recs":       args.top_n_recs,
        "use_genre":        args.use_genre,
        "n_genres":         args.n_genres,
    }

    with ExperimentTracker(
        experiment_name=EXPERIMENT_NAMES["sasrec"],
        run_name=run_name,
        store=store,
        tags={"model_type": "sasrec", "genre": str(args.use_genre)}
    ) as tracker:

        tracker.log_params(params)
        logger.info(f"MLflow run started | run_id={tracker.run_id} | run_name={run_name}")

        db = MovieLensDB()
        db.load_data()

        train_df = load_training_data(
            db, args.start, args.end,
            min_rating=args.min_rating,
            min_user_ratings=args.min_u_rat,
            min_item_ratings=args.min_i_rat,
        )

        enc = Encoder()
        enc.encode(train_df)
        logger.info(f"Encoded {enc.n_users:,} users and {enc.n_items:,} items")
        tracker.set_tag("n_users", str(enc.n_users))
        tracker.set_tag("n_items", str(enc.n_items))

        dataset = SASRecDataset(
            train_df, enc, db, args.max_len, args.n_neg, use_genre=args.use_genre
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True
        )

        if args.use_genre:
            logger.info("Using SASRecWithGenre")
            model = SASRecWithGenre(
                enc.n_items, args.dim, args.max_len,
                n_genres=args.n_genres,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
            ).to(device)
        else:
            logger.info("Using base SASRec")
            model = SASRec(
                enc.n_items, args.dim, args.max_len,
                n_heads=args.n_heads,
                n_layers=args.n_layers,
            ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        losses = []
        t0     = time.time()

        for epoch in range(1, args.epochs + 1):
            t_ep = time.time()
            loss = train_one_epoch(
                model, loader, optimizer, scheduler,
                device, args.loss, use_genre=args.use_genre
            )
            losses.append(float(loss))
            tracker.log_metric("train_loss", float(loss), step=epoch)
            logger.info(f"Epoch {epoch:3d}/{args.epochs} | loss={loss:.4f} | time={time.time()-t_ep:.1f}s")

        tracker.set_tag("training_time_sec", str(round(time.time() - t0, 1)))

        eval_metrics = evaluate(
            model=model, enc=enc, db=db, train_df=train_df,
            eval_start=args.eval_start, eval_end=args.eval_end,
            min_rating=args.min_rating,
            min_eval_ratings=args.min_eval_ratings,
            eval_k=args.eval_k, top_n_recs=args.top_n_recs,
            max_len=args.max_len, use_genre=args.use_genre, device=device,
        )

        if eval_metrics:
            tracker.log_metrics(eval_metrics, step=args.epochs)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            save_artifacts(tmpdir, model, enc, args, losses)
            tracker.log_artifacts(str(tmpdir), artifact_path=None)

        logger.info(f"Run complete | run_id={tracker.run_id}")


if __name__ == "__main__":
    main()