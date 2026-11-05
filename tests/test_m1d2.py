"""M1-D2 isolation, validation and paired-summary tests (no NUTS required)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rhb.experiments.m1d2_prior_predictive import (
    draw_prior_predictive,
    predictor_summary,
    prior_summary,
    reference_training_set,
    validate_config,
)
from rhb.pipelines.summarize_m1d2 import compare_to_reference, summarize


def fixture_config() -> dict:
    return {
        "experiment": "M1-D2",
        "city": "RTM",
        "reference_prior_id": "P0",
        "fixed_beta_sigma": 2.5,
        "fixed_gamma_sigma": 0.5,
        "prior_draws": 500,
        "prior_batch_size": 50,
        "prior_seed": 101,
        "intercept_priors": [
            {"id": "P0", "mean": 0.0, "sigma": 2.5},
            {"id": "P1", "mean": -3.0, "sigma": 1.0},
            {"id": "P2", "mean": -3.0, "sigma": 0.5},
        ],
    }


def test_config_reference_and_priors():
    assert len(validate_config(fixture_config())) == 3


def test_reject_changed_baseline_or_slopes():
    cfg = fixture_config()
    cfg["intercept_priors"][0]["mean"] = -1
    with pytest.raises(ValueError, match="P0 must exactly"):
        validate_config(cfg)
    cfg = fixture_config()
    cfg["fixed_beta_sigma"] = 1.0
    with pytest.raises(ValueError, match="slope priors"):
        validate_config(cfg)


def test_reject_duplicate_prior():
    cfg = fixture_config()
    cfg["intercept_priors"][2]["id"] = "P1"
    with pytest.raises(ValueError, match="distinct"):
        validate_config(cfg)


def test_no_test_outcome_leakage():
    rows = []
    for gamma in (0.0, -0.4):
        for seed in (101, 202):
            for i in range(6):
                rows.append({
                    "gamma_true": gamma, "simulation_seed": seed,
                    "bldg_id": i, "E_hat_v0": (i - 3) / 4,
                    "H_pluvial_v1_logrel": i / 200,
                    "T": i / 5, "split": "train" if i < 4 else "test",
                    "Y_m1c": int(seed % 2 == 0 and i % 2 == 0),
                })
    df = pd.DataFrame(rows)
    train = reference_training_set(df)
    altered = df.copy()
    altered.loc[altered["split"].eq("test"), "Y_m1c"] = 1
    assert len(train) == 4
    pd.testing.assert_frame_equal(train, reference_training_set(altered))
    prior = {"id": "P1", "mean": -3.0, "sigma": 1.0}
    a = draw_prior_predictive(train, prior, draws=40, seed=10, batch_size=10)
    b = draw_prior_predictive(reference_training_set(altered), prior, draws=40, seed=10, batch_size=10)
    pd.testing.assert_frame_equal(a, b)


def test_reference_population_rejects_changed_covariate():
    rows = []
    for seed in (101, 202):
        for i in range(3):
            rows.append({
                "gamma_true": 0, "simulation_seed": seed, "bldg_id": i,
                "E_hat_v0": float(i + (seed == 202 and i == 1)),
                "H_pluvial_v1_logrel": 0.01, "T": 0.1,
                "split": "train" if i != 2 else "test", "Y_m1c": 0,
            })
    with pytest.raises(ValueError, match="Covariate E_hat_v0 changed"):
        reference_training_set(pd.DataFrame(rows))


def test_prior_predictive_symmetric_zero_covariates():
    x = pd.DataFrame({
        "E_hat_v0": np.zeros(20), "H_pluvial_v1_logrel": np.zeros(20),
        "T": np.zeros(20),
    })
    result = draw_prior_predictive(
        x, {"id": "P0", "mean": 0.0, "sigma": 0.0001},
        draws=300, seed=123, batch_size=32,
    )
    assert abs(result["mean_event_probability"].median() - 0.5) < 0.001
    assert len(result) == 300
    assert (result["sampled_event_rate"].between(0, 1)).all()
    assert prior_summary(result).iloc[0]["draws"] == 300
    assert len(predictor_summary(x)) == 3


def sample_comparison_rows() -> pd.DataFrame:
    rows = []
    for seed in (101, 202):
        for pid, regret, elpd in (("P0", 0.10, -120.0), ("P1", 0.07 if seed == 101 else 0.12, -118.0)):
            rows.append({
                "gamma_true": -0.4, "simulation_seed": seed, "prior_id": pid,
                "alpha_prior_mean": 0.0 if pid == "P0" else -3.0,
                "alpha_prior_sigma": 2.5 if pid == "P0" else 1.0,
                "alpha_posterior_mean": -2.5 if pid == "P0" else -3.1,
                "alpha_covered": True,
                "gamma_posterior_mean": -0.35, "gamma_covered": True,
                "expected_regret_B0": 0.2, "expected_regret_B1": regret,
                "overlap_B1_P0": 10 if pid == "P0" else 9,
                "max_inclusion_change_B1_P0": 0.0 if pid == "P0" else 0.2,
                "heldout_elpd_B1": elpd, "heldout_elpd_B0": -130.0,
                "brier_B1": 0.1, "rmse_vs_true_p_B1": 0.03,
                "rhat_max": 1.0, "ess_bulk_min": 800.0,
                "ess_tail_min": 550.0, "divergences": 0,
                "train_damage_rate": 0.06,
                "posterior_predictive_rate_p50_B1": 0.06,
            })
    return pd.DataFrame(rows)


def test_paired_reference_comparison():
    paired = compare_to_reference(sample_comparison_rows())
    result = summarize(paired)
    p1 = result.loc[result["prior_id"].eq("P1")].iloc[0]
    assert p1["replications"] == 2
    assert p1["mean_expected_regret_B1"] == pytest.approx(0.095)
    assert p1["mean_regret_change_vs_P0"] == pytest.approx(-0.005)
    assert p1["prior_better_than_P0"] == 1
    assert p1["prior_worse_than_P0"] == 1
    assert p1["mean_topk_overlap_with_P0"] == 9


def test_reject_duplicate_and_missing_prior_variant():
    df = sample_comparison_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        compare_to_reference(pd.concat([df, df.iloc[[0]]], ignore_index=True))
    incomplete = df.loc[~((df["simulation_seed"] == 202) & (df["prior_id"] == "P1"))]
    with pytest.raises(ValueError, match="Missing prior variant"):
        compare_to_reference(incomplete)
