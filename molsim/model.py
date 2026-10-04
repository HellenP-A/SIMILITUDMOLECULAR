"""Red Siamese con encoder CNN-1D multi-kernel compartido y 4 cabezas de regresión.

Mejoras respecto al PIA-02:
  * Combinación de embeddings simétrica por construcción: [a+b, |a-b|, a*b]
    (sim(A,B) == sim(B,A) exactamente, no solo aproximadamente).
  * Pooling enmascarado (max + promedio) que ignora el padding.
  * Tronco compartido + 4 cabezas: las métricas comparten información (multi-tarea).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim=64, channels=128, kernels=(3, 5, 7), out_dim=256, dropout=0.15):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(emb_dim, channels, k, padding=k // 2),
                nn.BatchNorm1d(channels),
                nn.GELU(),
                nn.Conv1d(channels, channels, k, padding=k // 2),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            )
            for k in kernels
        )
        self.proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * channels * len(kernels), out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != 0).unsqueeze(1).float()  # (B,1,L)
        x = self.emb(tokens).transpose(1, 2)  # (B,E,L)
        h = torch.cat([br(x) for br in self.branches], dim=1)  # (B,C*k,L)
        h_max = (h - (1.0 - mask) * 1e4).amax(dim=2)
        h_mean = (h * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1.0)
        return self.proj(torch.cat([h_max, h_mean], dim=1))


class SiameseSimilarity(nn.Module):
    def __init__(self, vocab_size: int, n_metrics: int = 4, emb_dim=64, channels=128, kernels=(3, 5, 7),
                 out_dim=256, hidden=256, dropout=0.15):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, n_metrics=n_metrics, emb_dim=emb_dim, channels=channels,
                           kernels=list(kernels), out_dim=out_dim, hidden=hidden, dropout=dropout)
        self.encoder = Encoder(vocab_size, emb_dim, channels, kernels, out_dim, dropout)
        self.trunk = nn.Sequential(nn.Linear(3 * out_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1)) for _ in range(n_metrics)
        )

    def combine(self, ea: torch.Tensor, eb: torch.Tensor) -> torch.Tensor:
        z = torch.cat([ea + eb, (ea - eb).abs(), ea * eb], dim=1)
        h = self.trunk(z)
        return torch.sigmoid(torch.cat([head(h) for head in self.heads], dim=1))

    def forward(self, tok_a: torch.Tensor, tok_b: torch.Tensor) -> torch.Tensor:
        return self.combine(self.encoder(tok_a), self.encoder(tok_b))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
