"""Source-free helper tests. Run in the original repository Python 3.11 .venv."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.audit_m1g2_full import (exact_expected_scores, grouped_bins,
                                      normalize_weights, prior_reweight,
                                      weighted_quantile)


def test_normalized_weights_and_ess():
    w, ess = normalize_weights(np.zeros(100))
    assert w.sum() == pytest.approx(1)
    assert ess == pytest.approx(100)
    assert weighted_quantile(np.array([1., 2., 3.]), np.array([.2, .3, .5]), (.5,))[0] <= 3


def test_prior_reweight_identity():
    x = np.linspace(-1, 1, 1000)
    result = prior_reweight(x, base_sigma=.5, new_sigma=.5)
    assert result['importance_ess'] == pytest.approx(1000)
    assert result['gamma_weighted_mean'] == pytest.approx(0)


def test_exact_scores_oracle_is_best():
    p = np.array([.1, .2, .5, .9])
    optimal = exact_expected_scores(p, p)
    wrong = exact_expected_scores(p, np.array([.8, .8, .8, .2]))
    assert optimal['expected_logloss'] < wrong['expected_logloss']
    assert optimal['expected_brier'] < wrong['expected_brier']


def test_group_bins_constant():
    codes, names = grouped_bins(np.ones(20))
    assert names == ['all_equal']
    assert codes.sum() == 0


def test_nonfinite_prior_weights_rejected():
    with pytest.raises(AssertionError):
        normalize_weights(np.array([0., np.inf]))
