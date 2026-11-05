"""Lightweight tests for M1-C decision and predictive scoring."""
import numpy as np
import pandas as pd
import pytest

from rhb.decision.m1c_metrics import (
    compare_decisions,
    decision_metrics,
    predictive_scores,
    topk_indices,
)


def test_topk_indices_select_highest():
    assert set(topk_indices(np.array([0.2, 0.9, 0.1]), 1)) == {1}
    with pytest.raises(ValueError):
        topk_indices(np.array([0.1, 0.2]), 0)


def test_predictive_scores_exact_values():
    draws = np.array([[0.1, 0.9], [0.3, 0.7]])
    result = predictive_scores(draws, np.array([0, 1]), np.array([0.2, 0.8]))
    assert result["rmse_vs_true_p"] == pytest.approx(0.0, abs=1e-12)
    assert result["brier"] == pytest.approx(0.04)
    assert np.isfinite(result["heldout_elpd"])


def test_regret_and_oracle_overlap():
    # Posterior consistently picks building 10; synthetic truth picks 20.
    draws = np.array([[0.9, 0.2, 0.1], [0.8, 0.3, 0.2]])
    truth = np.array([0.7, 0.9, 0.1])
    frame, metrics = decision_metrics(draws, np.array([10, 20, 30]), truth, k=1)
    assert metrics["oracle_overlap"] == 0
    assert metrics["expected_regret"] == pytest.approx(0.2)
    assert frame["p_topk"].sum() == pytest.approx(1.0)


def test_compare_decisions():
    base = pd.DataFrame({
        "bldg_id": [10, 20, 30], "p_true": [0.7, 0.9, 0.1],
        "p_topk": [1.0, 0.0, 0.0], "selected_by_p_mean": [True, False, False],
    })
    extended = pd.DataFrame({
        "bldg_id": [10, 20, 30], "p_true": [0.7, 0.9, 0.1],
        "p_topk": [0.0, 1.0, 0.0], "selected_by_p_mean": [False, True, False],
    })
    result = compare_decisions(base, extended, k=1)
    assert result["overlap"] == 0
    assert result["expected_regret_B0"] == pytest.approx(0.2)
    assert result["expected_regret_B1"] == pytest.approx(0.0)
    assert result["max_inclusion_change"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        compare_decisions(base, extended.iloc[::-1], k=1)
