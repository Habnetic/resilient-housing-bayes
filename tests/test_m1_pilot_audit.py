"""Small pure-function checks for the M1 pilot audit. Run: pytest -q tests/test_m1_pilot_audit.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rhb.pipelines.audit_m1_pilots import compare_decisions, topk_indices


def test_topk_indices_select_highest_probabilities() -> None:
    values = np.array([0.1, 0.7, 0.3, 0.9])
    assert set(topk_indices(values, 2)) == {1, 3}


def test_topk_indices_reject_invalid_k() -> None:
    with pytest.raises(ValueError):
        topk_indices(np.array([0.1, 0.2]), 3)


def test_compare_decisions_overlap_and_regret() -> None:
    base = pd.DataFrame({
        "bldg_id": [1, 2, 3, 4],
        "selected_by_p_mean": [True, True, False, False],
        "p_topk": [1.0, 1.0, 0.0, 0.0],
    })
    extended = pd.DataFrame({
        "bldg_id": [1, 2, 3, 4],
        "selected_by_p_mean": [True, False, True, False],
        "p_topk": [1.0, 0.0, 1.0, 0.0],
    })
    truth = np.array([0.9, 0.4, 0.8, 0.1])
    out = compare_decisions(base, extended, k=2, oracle_probabilities=truth)
    assert out["overlap"] == 1
    assert out["base_oracle_overlap"] == 1
    assert out["extended_oracle_overlap"] == 2
    assert out["base_expected_regret"] == pytest.approx(0.4)
    assert out["extended_expected_regret"] == pytest.approx(0.0)
