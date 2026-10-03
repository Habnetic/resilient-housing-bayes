from __future__ import annotations

import numpy as np

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