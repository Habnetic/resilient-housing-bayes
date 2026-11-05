"""Pure evaluation and decision metrics for the M1-C simulation study.

Use the Phase 3 draw-wise top-k membership definition without modifying it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from rhb.decision.phase3_decision_metrics import compute_topk_membership


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Select the k greatest scores using the original Phase 3 pilot rule."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not 0 < k <= scores.size:
        raise ValueError("Expected one-dimensional scores and 1 <= k <= n")
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite scores")
    return np.argpartition(scores, -k)[-k:]


def predictive_scores(
    posterior_probabilities: np.ndarray,
    outcomes: np.ndarray,
    true_probabilities: np.ndarray,
) -> dict[str, float]:
    """Evaluate held-out predictive accuracy, with access to synthetic truth."""
    draws = np.asarray(posterior_probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    truth = np.asarray(true_probabilities, dtype=float)

    if draws.ndim != 2 or y.shape != (draws.shape[1],) or truth.shape != y.shape:
        raise ValueError("Incompatible held-out predictive arrays")
    if draws.shape[0] < 1 or not np.isfinite(draws).all():
        raise ValueError("Expected finite posterior probability draws")
    if not np.all((draws >= 0) & (draws <= 1)):
        raise ValueError("Posterior probabilities outside [0, 1]")
    if not np.all(np.isin(y, [0, 1])) or not np.all((truth >= 0) & (truth <= 1)):
        raise ValueError("Invalid outcomes or synthetic ground truth")

    mean = draws.mean(axis=0)
    # Bernoulli likelihood integrated over posterior draws; no fitted test labels.
    clipped = np.clip(draws, 1e-12, 1 - 1e-12)
    loglik = np.where(y[None, :] == 1, np.log(clipped), np.log1p(-clipped))
    elpd = float((logsumexp(loglik, axis=0) - np.log(draws.shape[0])).sum())

    return {
        "rmse_vs_true_p": float(np.sqrt(np.mean((mean - truth) ** 2))),
        "brier": float(np.mean((mean - y) ** 2)),
        "heldout_elpd": elpd,
        "heldout_mean_log_score": elpd / len(y),
    }


def decision_metrics(
    posterior_probabilities: np.ndarray,
    bldg_ids: np.ndarray,
    true_probabilities: np.ndarray,
    k: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute posterior top-k stability and oracle-relative selection loss."""
    draws = np.asarray(posterior_probabilities, dtype=float)
    ids = np.asarray(bldg_ids)
    truth = np.asarray(true_probabilities, dtype=float)

    if draws.ndim != 2 or draws.shape[1] != len(ids) or len(truth) != len(ids):
        raise ValueError("Incompatible decision arrays")
    if len(ids) != len(np.unique(ids)):
        raise ValueError("Duplicate bldg_id in test population")
    if not np.isfinite(draws).all() or not np.isfinite(truth).all():
        raise ValueError("Nonfinite probabilities")
    if not np.all((draws >= 0) & (draws <= 1)) or not np.all((truth >= 0) & (truth <= 1)):
        raise ValueError("Probabilities outside [0, 1]")

    inclusion, rank_std = compute_topk_membership(draws, k=k)
    if inclusion.shape != (len(ids),) or not np.isclose(inclusion.sum(), k, atol=1e-6):
        raise AssertionError("Draw-wise top-k membership must sum to k")

    means = draws.mean(axis=0)
    selected = np.zeros(len(ids), dtype=bool)
    selected[topk_indices(means, k)] = True
    oracle = np.zeros(len(ids), dtype=bool)
    oracle[topk_indices(truth, k)] = True

    oracle_value = float(truth[oracle].sum())
    selected_value = float(truth[selected].sum())
    regret = max(0.0, oracle_value - selected_value)

    frame = pd.DataFrame({
        "bldg_id": ids,
        "p_true": truth,
        "p_mean": means,
        "p_topk": inclusion,
        "rank_std": rank_std,
        "selected_by_p_mean": selected,
        "selected_by_oracle": oracle,
    })
    metrics: dict[str, Any] = {
        "k": int(k),
        "borderline_count": int(((inclusion > 0.2) & (inclusion < 0.8)).sum()),
        "oracle_overlap": int(np.logical_and(selected, oracle).sum()),
        "oracle_expected_events": oracle_value,
        "selected_expected_events": selected_value,
        "expected_regret": regret,
    }
    return frame, metrics


def compare_decisions(base: pd.DataFrame, extended: pd.DataFrame, k: int) -> dict[str, Any]:
    """Compare B0 vs B1 using a common, identically ordered test population."""
    if not base["bldg_id"].equals(extended["bldg_id"]):
        raise ValueError("B0 and B1 test IDs or order differ")
    if not np.allclose(base["p_true"], extended["p_true"], rtol=0, atol=1e-12):
        raise ValueError("B0 and B1 synthetic truths differ")

    a = base["selected_by_p_mean"].to_numpy(dtype=bool)
    b = extended["selected_by_p_mean"].to_numpy(dtype=bool)
    if a.sum() != k or b.sum() != k:
        raise AssertionError("Both models must select exactly k assets")

    diff = np.abs(base["p_topk"].to_numpy() - extended["p_topk"].to_numpy())
    truth = base["p_true"].to_numpy(dtype=float)
    oracle_idx = topk_indices(truth, k)
    oracle_mask = np.zeros(len(truth), dtype=bool)
    oracle_mask[oracle_idx] = True
    oracle_value = float(truth[oracle_mask].sum())
    regret_a = max(0.0, oracle_value - float(truth[a].sum()))
    regret_b = max(0.0, oracle_value - float(truth[b].sum()))

    return {
        "k": int(k),
        "overlap": int(np.logical_and(a, b).sum()),
        "overlap_fraction": float(np.logical_and(a, b).sum() / k),
        "replaced_assets": int(k - np.logical_and(a, b).sum()),
        "borderline_B0": int(((base["p_topk"] > 0.2) & (base["p_topk"] < 0.8)).sum()),
        "borderline_B1": int(((extended["p_topk"] > 0.2) & (extended["p_topk"] < 0.8)).sum()),
        "mean_abs_inclusion_change": float(diff.mean()),
        "max_inclusion_change": float(diff.max()),
        "oracle_overlap_B0": int(np.logical_and(a, oracle_mask).sum()),
        "oracle_overlap_B1": int(np.logical_and(b, oracle_mask).sum()),
        "expected_regret_B0": regret_a,
        "expected_regret_B1": regret_b,
        "regret_improvement_B0_minus_B1": float(regret_a - regret_b),
    }
