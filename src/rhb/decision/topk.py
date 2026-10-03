from __future__ import annotations

import numpy as np


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