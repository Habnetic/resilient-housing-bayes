from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_posterior_params(idata_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idata = az.from_netcdf(idata_path)
    posterior = idata.posterior

    alpha = posterior["alpha"].values.reshape(-1)
    beta_e = posterior["beta_E"].values.reshape(-1)
    beta_h = posterior["beta_H"].values.reshape(-1)

    return alpha, beta_e, beta_h


def compute_posterior_probabilities(
    alpha: np.ndarray,
    beta_e: np.ndarray,
    beta_h: np.ndarray,
    e: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """
    Returns posterior probabilities with shape:
    (n_draws, n_assets)
    """
    logit = (
        alpha[:, None]
        + beta_e[:, None] * e[None, :]
        + beta_h[:, None] * h[None, :]
    )
    return sigmoid(logit)


def compute_topk_membership(
    p_draws: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute top-k membership probability and rank standard deviation.

    Parameters
    ----------
    p_draws:
        Array of shape (n_draws, n_assets).
    k:
        Number of top-ranked assets.

    Returns
    -------
    topk_prob:
        Probability each asset appears in top-k.
    rank_std:
        Standard deviation of asset rank across posterior draws.
    """
    n_draws, n_assets = p_draws.shape

    k = min(k, n_assets)

    topk_counts = np.zeros(n_assets, dtype=np.int32)
    rank_matrix = np.empty((n_draws, n_assets), dtype=np.int32)

    for s in range(n_draws):
        order = np.argsort(-p_draws[s])
        ranks = np.empty(n_assets, dtype=np.int32)
        ranks[order] = np.arange(1, n_assets + 1)

        rank_matrix[s] = ranks
        topk_counts[order[:k]] += 1

    topk_prob = topk_counts / n_draws
    rank_std = rank_matrix.std(axis=0)

    return topk_prob, rank_std


def summarize_decision_metrics(
    topk_prob: np.ndarray,
    rank_std: np.ndarray,
    k: int,
    borderline_low: float = 0.2,
    borderline_high: float = 0.8,
) -> dict:
    n = len(topk_prob)

    borderline = (topk_prob > borderline_low) & (topk_prob < borderline_high)
    stable_inclusion = topk_prob >= borderline_high
    stable_exclusion = topk_prob <= borderline_low

    return {
        "k": int(k),
        "n_assets": int(n),
        "borderline_low": float(borderline_low),
        "borderline_high": float(borderline_high),
        "borderline_count": int(borderline.sum()),
        "borderline_share": float(borderline.mean()),
        "stable_inclusion_count": int(stable_inclusion.sum()),
        "stable_inclusion_share": float(stable_inclusion.mean()),
        "stable_exclusion_count": int(stable_exclusion.sum()),
        "stable_exclusion_share": float(stable_exclusion.mean()),
        "rank_std_mean": float(rank_std.mean()),
        "rank_std_p50": float(np.quantile(rank_std, 0.50)),
        "rank_std_p90": float(np.quantile(rank_std, 0.90)),
        "rank_std_p95": float(np.quantile(rank_std, 0.95)),
    }


def save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")