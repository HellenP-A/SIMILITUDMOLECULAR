"""Métricas de evaluación: regresión (R², MAE, RMSE, Pearson, Spearman, Kendall) y ranking (NDCG@k, Precision@k)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "pearson": float(stats.pearsonr(y_true, y_pred)[0]),
        "spearman": float(stats.spearmanr(y_true, y_pred)[0]),
        "kendall": float(stats.kendalltau(y_true, y_pred)[0]),
        "n": int(len(y_true)),
    }


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """NDCG@k usando la similitud real como relevancia graduada."""
    order = np.argsort(-y_pred)[:k]
    ideal = np.sort(y_true)[::-1][:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((y_true[order] * disc[: len(order)]).sum())
    idcg = float((ideal * disc[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else float("nan")


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Fracción del top-k real que el modelo recupera en su top-k."""
    top_true = set(np.argsort(-y_true)[:k])
    top_pred = set(np.argsort(-y_pred)[:k])
    return len(top_true & top_pred) / k


def ranking(y_true: np.ndarray, y_pred: np.ndarray, ks=(10, 50, 100)) -> dict:
    out = {"spearman": float(stats.spearmanr(y_true, y_pred)[0])}
    for k in ks:
        if len(y_true) >= k:
            out[f"ndcg@{k}"] = ndcg_at_k(y_true, y_pred, k)
            out[f"precision@{k}"] = precision_at_k(y_true, y_pred, k)
    return out
