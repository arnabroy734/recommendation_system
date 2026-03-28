import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head Self Attention
# ─────────────────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.dim     = dim
        self.n_heads = n_heads
        self.d_head  = dim // n_heads

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        """
        x        : (B, L, dim)
        pad_mask : (B, L) boolean — True where position is padding
        returns  : output (B, L, dim) and attention scores (B, n_heads, L, L)
        """
        B, L, _ = x.shape

        # ── project ───────────────────────────────────────────────────
        Q = self.W_q(x)   # (B, L, dim)
        K = self.W_k(x)
        V = self.W_v(x)

        # ── split heads ───────────────────────────────────────────────
        # (B, L, dim) → (B, n_heads, L, d_head)
        Q = Q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        # ── scaled dot product attention ──────────────────────────────
        scale  = self.d_head ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale   # (B, n_heads, L, L)

        # ── causal mask — upper triangle = -inf ───────────────────────
        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device),
            diagonal=1
        )   # (L, L)

        # pad mask — mask BOTH pad keys AND pad queries
        if pad_mask is not None:
            # mask pad as keys — real tokens cannot attend to pad
            key_mask = pad_mask.unsqueeze(1).unsqueeze(2)   # (B, 1, 1, L)
            key_mask = key_mask.expand(B, self.n_heads, L, L)

            # mask pad as queries — pad tokens cannot attend to anything
            query_mask = pad_mask.unsqueeze(1).unsqueeze(-1) # (B, 1, L, 1)
            query_mask = query_mask.expand(B, self.n_heads, L, L)

            # combine — either pad key or pad query → -inf
            pad_combined = (key_mask | query_mask).float() * float("-inf")
            pad_combined = torch.nan_to_num(pad_combined, nan=0.0)

            scores = scores + causal.unsqueeze(0).unsqueeze(0) + pad_combined
        else:
            scores = scores + causal.unsqueeze(0).unsqueeze(0)


        # ── softmax + dropout ─────────────────────────────────────────
        attn   = torch.softmax(scores, dim=-1)   # (B, n_heads, L, L)
        attn   = self.dropout(attn)

        # ── weighted sum ──────────────────────────────────────────────
        out = torch.matmul(attn, V)              # (B, n_heads, L, d_head)
        out = out.transpose(1, 2).contiguous().view(B, L, self.dim)   # (B, L, dim)
        out = self.W_o(out)



        return out, attn


# ─────────────────────────────────────────────────────────────────────────────
# Feed Forward Block
# ─────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim=None, dropout=0.1):
        super().__init__()
        ff_dim = ff_dim or dim * 2   # standard transformer uses 4x expansion
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Encoder Layer (Pre-LN — more stable than Post-LN)
# ─────────────────────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    Pre-LayerNorm variant:
        x = x + Attn(LN(x))
        x = x + FFN(LN(x))
    More stable gradients than Post-LN during early training.
    """
    def __init__(self, dim, n_heads, ff_dim=None, dropout=0.1):
        super().__init__()
        self.attn    = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.ff      = FeedForward(dim, ff_dim, dropout)
        self.norm1   = nn.LayerNorm(dim)
        self.norm2   = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None):
        # ── self attention with residual ──────────────────────────────
        attn_out, attn_scores = self.attn(self.norm1(x), pad_mask)
        x = x + self.dropout(attn_out)
        # ── feed forward with residual ────────────────────────────────
        x = x + self.ff(self.norm2(x))
        return x, attn_scores


# ─────────────────────────────────────────────────────────────────────────────
# Encoder Block (stack of EncoderLayers)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    def __init__(self, dim, n_heads, n_layers, ff_dim=None, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(dim, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)   # final norm after all layers

    def forward(self, x, pad_mask=None):
        attn_scores = []
        for layer in self.layers:
            x, attn_score = layer(x, pad_mask)
            attn_scores.append(attn_score)
        return self.norm(x), attn_scores


# ─────────────────────────────────────────────────────────────────────────────
# SASRec Model
# ─────────────────────────────────────────────────────────────────────────────

class SASRec(nn.Module):
    def __init__(self, n_items, dim, max_len, n_heads=2, n_layers=2,
                 ff_dim=None, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_len, dim)
        self.dropout  = nn.Dropout(dropout)

        self.encoder  = TransformerEncoder(dim, n_heads, n_layers, ff_dim, dropout)

        # initialise embeddings
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.pos_emb.weight,  mean=0, std=0.01)

    def forward(self, seq):
        """
        seq : (B, L) — item indices, 0 = padding
        returns h : (B, L, dim) — contextual representations
        """
        B, L = seq.shape

        # ── pad mask — True where seq == 0 ───────────────────────────
        pad_mask = (seq == 0)   # (B, L)

        # ── embeddings ────────────────────────────────────────────────
        pos = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x   = self.item_emb(seq) + self.pos_emb(pos)

        # ── zero out pad token embeddings explicitly ───────────────────
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        x = self.dropout(x)
        

        # ── transformer encoder ───────────────────────────────────────
        h, attn_scores = self.encoder(x, pad_mask)   # (B, L, dim)

        return h, attn_scores