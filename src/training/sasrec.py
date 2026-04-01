# sasrec.py

import argparse
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys 
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.data.db_simulator import MovieLensDB
from src.training.matrix_factorisation import Encoder
from src.training.sasrec_architecture import SASRec, SASRecWithGenre


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =========================================================
# Dataset
# =========================================================
def load_training_data(
    db: MovieLensDB,
    start: str,
    end: str,
    min_rating: float = 3.0,
    min_user_ratings: int = 10,
    min_item_ratings: int = 10,
) -> pd.DataFrame:
    """
    Pull ratings from db within [start, end].
    Applies rating filter and iterative co-filtering.
    Returns dataframe with: userId, movieId, rating, timestamp
    """
    logger.info(f"Loading data from {start} to {end} ...")
    df = db.get_ratings_by_daterange(start, end).copy()
    logger.info(f"Raw interactions: {len(df):,}")

    df = df[df["rating"] >= min_rating].copy()
    logger.info(f"After rating >= {min_rating}: {len(df):,}")

    prev_len  = -1
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


class SASRecDataset(Dataset):
    def __init__(self, df, enc, db, max_len=200, n_neg=4, use_genre=False):
        self.max_len   = max_len
        self.n_neg     = n_neg
        self.enc       = enc
        self.use_genre = use_genre
        self.db        = db            # needed for genre lookup

        df = df.sort_values(["userId", "timestamp"])

        self.sequences = []            # (u_idx, seq, targets, orig_movie_ids_padded)
        self.all_items = np.array(list(enc.id_item.keys()))

        self.user_pos = {}
        for uid, grp in df.groupby("userId"):
            u = enc.user_id[uid]
            self.user_pos[u] = set(enc.item_id[m] for m in grp["movieId"].values)

        for uid, grp in df.groupby("userId"):
            u        = enc.user_id[uid]
            items    = [enc.item_id[i] + 1 for i in grp["movieId"].values]
            # keep original movieIds (for genre lookup) aligned with items
            orig_ids = list(grp["movieId"].values)

            if len(items) < 2:
                continue

            full         = items[-(max_len + 1):]
            full_orig    = orig_ids[-(max_len + 1):]

            seq          = full[:-1]
            targets      = full[1:]
            seq_orig     = full_orig[:-1]   # original movieIds for the input seq

            pad_len      = max_len - len(seq)
            seq          = [0] * pad_len + seq
            targets      = [-1] * pad_len + targets
            seq_orig     = [0] * pad_len + seq_orig   # 0 → zero genre vector in db

            self.sequences.append((u, seq, targets, seq_orig))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        u, seq, targets, seq_orig = self.sequences[idx]

        # negative sampling (unchanged)
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
            # genre vectors for each position in the input sequence
            # seq_orig has 0 for padded positions → db returns zero vector
            genre_vecs = self.db.get_genre_vectors_batch(seq_orig)  # (L, 12)
            return (
                torch.LongTensor(seq),
                torch.LongTensor(targets),
                torch.LongTensor(negs),
                torch.FloatTensor(genre_vecs),   # (L, 12)
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
    if step < warmup_steps:
        return step / warmup_steps
    return 1.0

def train_one_epoch(model, loader, optimizer, scheduler, device, loss_name, use_genre=False):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader):
        if use_genre:
            seq, targets, negs, genre_seq = batch
            genre_seq = genre_seq.to(device)   # (B, L, 12)
        else:
            seq, targets, negs = batch

        seq     = seq.to(device)
        targets = targets.to(device)
        negs    = negs.to(device)

        # ── forward ───────────────────────────────────────────────────
        if use_genre:
            h, _ = model(seq, genre_seq)   # SASRecWithGenre
        else:
            h, _ = model(seq)              # SASRec

        # ── mask + scoring (unchanged from original) ──────────────────
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

        elif loss_name == "bpr":
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
# Evaluation (same logic as MF)
# =========================================================

def evaluate(model, enc, train_df, db, args, log_dir, device):
    model.eval()
    all_item_indices = np.array(list(enc.id_item.keys()))
    summary_rows     = []

    # build seen_per_user from training window
    seen_per_user = {}
    for uid, grp in train_df.groupby("userId"):
        u = enc.user_id[uid]
        seen_per_user[u] = set(enc.item_id[m] for m in grp["movieId"])

    end_dt = pd.Timestamp(args.end)

    for m in range(1, args.eval_months + 1):
        ws = (end_dt + pd.DateOffset(months=m-1)).strftime("%Y-%m-%d")
        we = (end_dt + pd.DateOffset(months=m)).strftime("%Y-%m-%d")
        logger.info(f"\nEval window {m}: {ws} → {we}")

        eval_df = db.get_ratings_by_daterange(ws, we)
        eval_df = eval_df[eval_df["rating"] >= args.min_rating]
        eval_df = eval_df[eval_df["userId"].isin(enc.user_id)]
        eval_df = eval_df[eval_df["movieId"].isin(enc.item_id)]

        if eval_df.empty:
            logger.info(f"  No eval data for window {m}, skipping.")
            continue

        user_eval_counts = eval_df.groupby("userId")["movieId"].count()
        gt_per_user = {
            uid: set(grp["movieId"].values)
            for uid, grp in eval_df.groupby("userId")
        }

        # ── generate top-N recs ──────────────────────────────────────────
        recs_rows = []
        for uid_orig, count in user_eval_counts.items():
            u      = enc.user_id[uid_orig]
            seen   = seen_per_user.get(u, set())
            unseen = all_item_indices[~np.isin(all_item_indices, list(seen))]

            if len(unseen) == 0:
                continue

            # build sequence
            # ── build sequence ─────────────────────────────────────────
            hist     = train_df[train_df["userId"] == uid_orig].sort_values("timestamp")
            orig_ids = list(hist["movieId"].values)[-args.max_len:]
            items    = [enc.item_id[i] + 1 for i in orig_ids]
            pad_len  = args.max_len - len(items)
            seq      = [0] * pad_len + items
            seq_t    = torch.LongTensor(seq).unsqueeze(0).to(device)

            with torch.no_grad():
                if args.use_genre:
                    seq_orig_padded = [0] * pad_len + orig_ids
                    genre_vecs = db.get_genre_vectors_batch(seq_orig_padded)   # (L, 12)
                    genre_t    = torch.FloatTensor(genre_vecs).unsqueeze(0).to(device)
                    h = model(seq_t, genre_t)[0].cpu().numpy()[0][-1]          # (dim,)
                else:
                    h = model(seq_t)[0].cpu().numpy()[0][-1]                   # (dim,)

            # score unseen items — shift indices by 1 for embedding lookup
            emb_w       = model.item_emb.weight.detach().cpu().numpy()
            item_vecs   = emb_w[unseen + 1]         # (n_unseen, dim)
            scores      = item_vecs @ h              # (n_unseen,)
            top_idx     = np.argsort(scores)[::-1][:args.top_n_recs]
            top_items   = unseen[top_idx]
            top_scores  = scores[top_idx]

            for rank, (item_idx, score) in enumerate(
                zip(top_items, top_scores), start=1
            ):
                recs_rows.append({
                    "userId":         uid_orig,
                    "movieId":        enc.id_item[item_idx],
                    "score":          round(float(score), 6),
                    "rank":           rank,
                    "eval_window":    m,
                    "window_start":   ws,
                    "window_end":     we,
                    "n_eval_ratings": int(count),
                })

        if not recs_rows:
            logger.info(f"  No recs generated for window {m}.")
            continue

        recs_df   = pd.DataFrame(recs_rows)
        recs_path = log_dir / f"recs_window_{m:02d}.csv"
        recs_df.to_csv(recs_path, index=False)
        logger.info(f"  Saved {len(recs_df):,} rows → {recs_path.name}")

        # ── metrics ───────────────────────────────────────────────────────
        for min_er in args.min_eval_ratings:
            qualifying = user_eval_counts[user_eval_counts >= min_er].index
            if len(qualifying) == 0:
                continue

            for k in args.eval_k:
                recalls, ndcgs = [], []

                for uid_orig in qualifying:
                    gt = gt_per_user.get(uid_orig, set())
                    if not gt:
                        continue

                    user_recs   = recs_df[recs_df["userId"] == uid_orig].sort_values("rank")
                    top_k_items = list(user_recs.head(k)["movieId"].values)

                    hits   = len(set(top_k_items) & gt)
                    recall = hits / min(len(gt), k)
                    recalls.append(recall)

                    idcg = sum(1.0/np.log2(r+2) for r in range(min(len(gt), k)))
                    dcg  = sum(
                        1.0/np.log2(r+2)
                        for r, item in enumerate(top_k_items)
                        if item in gt
                    )
                    ndcgs.append(dcg/idcg if idcg > 0 else 0.0)

                mean_recall = float(np.mean(recalls))
                mean_ndcg   = float(np.mean(ndcgs))

                logger.info(
                    f"  Window {m} | min_ratings={min_er:2d} | K={k:2d} | "
                    f"Recall@{k}={mean_recall:.4f} | NDCG@{k}={mean_ndcg:.4f} | "
                    f"users={len(recalls)}"
                )
                summary_rows.append({
                    "eval_window":      m,
                    "window_start":     ws,
                    "window_end":       we,
                    "min_eval_ratings": min_er,
                    "K":                k,
                    "n_users":          len(recalls),
                    "recall":           round(mean_recall, 6),
                    "ndcg":             round(mean_ndcg, 6),
                })

        # update seen_per_user with this window's interactions
        for uid_orig, grp in eval_df.groupby("userId"):
            if uid_orig in enc.user_id:
                u = enc.user_id[uid_orig]
                if u not in seen_per_user:
                    seen_per_user[u] = set()
                seen_per_user[u].update(
                    enc.item_id[mid]
                    for mid in grp["movieId"]
                    if mid in enc.item_id
                )

    return summary_rows

def _plot_eval_summary(summary_df: pd.DataFrame, log_dir: Path, args):
    for metric in ["recall", "ndcg"]:
        fig, axes = plt.subplots(
            1, len(args.min_eval_ratings),
            figsize=(6 * len(args.min_eval_ratings), 5),
            sharey=False,
        )
        if len(args.min_eval_ratings) == 1:
            axes = [axes]

        for ax, min_er in zip(axes, args.min_eval_ratings):
            sub = summary_df[summary_df["min_eval_ratings"] == min_er]
            for k in args.eval_k:
                ksub = sub[sub["K"] == k].sort_values("eval_window")
                ax.plot(
                    ksub["eval_window"], ksub[metric],
                    marker="o", markersize=4, label=f"K={k}",
                )
            ax.set_title(f"min_ratings={min_er}")
            ax.set_xlabel("Eval window (months after training end)")
            ax.set_ylabel(f"{metric.upper()}@K")
            ax.legend()
            ax.grid(alpha=0.3)

        fig.suptitle(
            f"{metric.upper()}@K over time | SASRec | dim={args.dim}",
            fontsize=13,
        )
        plt.tight_layout()
        path = log_dir / f"{metric}_over_time.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"Plot saved → {path}")

def _plot_losses(losses, log_dir, args):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(1, len(losses)+1), losses, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"SASRec | dim={args.dim} | lr={args.lr} | "
        f"max_len={args.max_len} | n_neg={args.n_neg}"
    )
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(log_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    logger.info(f"Loss curve saved → {log_dir / 'loss_curve.png'}")

# =========================================================
# Main
# =========================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--max_len", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_neg", type=int, default=4)
    p.add_argument("--eval_months", type=int, default=5)
    p.add_argument("--eval_k", type=int, nargs="+", default=[5,10,20])
    p.add_argument("--top_n_recs", type=int, default=50)
    p.add_argument("--min_u_rat", type=int, default=10)
    p.add_argument("--min_i_rat", type=int, default=10)
    p.add_argument("--min_rating", type=float, default=3.0)
    p.add_argument("--log_root", type=str, default="train_logs/SASREC")
    p.add_argument("--min_eval_ratings", type=int, nargs="+", default=[10])
    p.add_argument("--data_dir", type=str, default="data/ml-32m")
    p.add_argument("--n_heads", type=int, default=2)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--loss", type=str, choices=["bce", "bpr"], default="bce")
    # ── genre args ─────────────────────────────────────────────
    p.add_argument("--use_genre",  action="store_true",
                help="Use SASRecWithGenre with genre embedding fusion")
    p.add_argument("--n_genres",   type=int, default=12,
                help="Genre vocabulary size (must match GENRE_VOCAB in db_simulator)")


    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    genre_tag = "_genre" if args.use_genre else ""
    log_dir   = Path(args.log_root) / \
            f"{args.start}_{args.end}_{run_tag}_{args.loss}_dim{args.dim}{genre_tag}"
    log_dir.mkdir(parents=True, exist_ok=True)

    db = MovieLensDB(data_dir=args.data_dir)
    db.load_data()

    train_df = load_training_data(
        db, args.start, args.end,
        min_rating=args.min_rating,
        min_user_ratings=args.min_u_rat,
        min_item_ratings=args.min_i_rat,
    )

    enc = Encoder()
    enc.encode(train_df)
    
    # dataset — pass db and use_genre flag
    dataset = SASRecDataset(train_df, enc, db, args.max_len, args.n_neg,
                            use_genre=args.use_genre)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True)

    assert args.dim % args.n_heads == 0, \
        f"dim ({args.dim}) must be divisible by n_heads ({args.n_heads})"

    # model — pick class based on flag
    if args.use_genre:
        logger.info("Using SASRecWithGenre (genre embedding fusion enabled)")
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
    for epoch in range(1, args.epochs+1):
        loss = train_one_epoch(model, loader, optimizer, scheduler, device, args.loss, use_genre=args.use_genre)
        losses.append(loss)
        logger.info(f"Epoch {epoch}: loss={loss:.4f}")
    
    _plot_losses(losses, log_dir, args)

    # eval_results = evaluate(model, enc, train_df, db, args, device)
    summary_rows = evaluate(model, enc, train_df, db, args, log_dir, device)

    # save
    torch.save(model.state_dict(), log_dir / "model.pt")
    with open(log_dir / "encoder.pkl", "wb") as f:
        pickle.dump(enc, f)

    # pd.DataFrame(eval_results).to_csv(log_dir / "eval.csv", index=False)
    # logger.info(f"Saved results to {log_dir}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(log_dir / "eval_summary.csv", index=False)
        logger.info(f"Eval summary saved → {log_dir / 'eval_summary.csv'}")
        _plot_eval_summary(summary_df, log_dir, args)


if __name__ == "__main__":
    main()