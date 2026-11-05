"""Unit tests for M1-D1 configurations and paired sensitivity summaries."""
import pandas as pd
import pytest

from rhb.pipelines.run_m1d import sigma_label, validate_config
from rhb.pipelines.summarize_m1d import summarize


def test_prior_config_includes_baseline():
    config = {
        "experiment": "M1-D1", "city": "RTM",
        "baseline_gamma_prior_sigma": 0.5,
        "gamma_prior_sigmas": [0.25, 0.5, 1.0],
    }
    assert validate_config(config) == [0.25, 0.5, 1.0]
    assert sigma_label(0.25) == "sigma_0p25"
    assert sigma_label(1.0) == "sigma_1"


def test_reject_duplicate_or_missing_baseline():
    config = {
        "experiment": "M1-D1", "city": "RTM",
        "baseline_gamma_prior_sigma": 0.5,
        "gamma_prior_sigmas": [0.25, 0.25, 1.0],
    }
    with pytest.raises(ValueError):
        validate_config(config)
    config["gamma_prior_sigmas"] = [0.25, 1.0]
    with pytest.raises(ValueError):
        validate_config(config)


def test_paired_regret_summary():
    df = pd.DataFrame({
        "gamma_true": [-0.8, -0.8],
        "simulation_seed": [101, 202],
        "gamma_prior_sigma": [0.25, 0.25],
        "gamma_posterior_mean": [-0.6, -0.7],
        "gamma_covered": [True, True],
        "expected_regret_B0": [0.2, 0.3],
        "expected_regret_B1": [0.1, 0.4],
        "overlap_B1_baseline": [9, 8],
        "max_inclusion_change_B1_baseline": [0.1, 0.3],
        "heldout_elpd_B0": [-150, -160],
        "heldout_elpd_B1": [-140, -155],
        "rhat_max": [1.0, 1.02],
        "ess_bulk_min": [500, 500],
        "ess_tail_min": [500, 500],
        "divergences": [0, 0],
        "prior_predictive_rate_p50_B1": [0.45, 0.47],
    })
    result = summarize(df).iloc[0]
    assert result["mean_expected_regret_B0"] == pytest.approx(0.25)
    assert result["mean_expected_regret_B1"] == pytest.approx(0.25)
    assert result["mean_paired_regret_improvement"] == pytest.approx(0.0)
    assert result["B1_better"] == 1
    assert result["B0_better"] == 1
    assert result["diagnostic_flag_count"] == 1


def test_duplicate_scenarios_rejected():
    row = {
        "gamma_true": 0.0, "simulation_seed": 101, "gamma_prior_sigma": 0.5,
        "gamma_posterior_mean": 0.0, "gamma_covered": True,
        "expected_regret_B0": 0.0, "expected_regret_B1": 0.0,
        "overlap_B1_baseline": 10, "max_inclusion_change_B1_baseline": 0.0,
        "heldout_elpd_B0": -100.0, "heldout_elpd_B1": -100.0,
        "rhat_max": 1.0, "ess_bulk_min": 900, "ess_tail_min": 700,
        "divergences": 0, "prior_predictive_rate_p50_B1": 0.5,
    }
    with pytest.raises(ValueError):
        summarize(pd.DataFrame([row, row]))
