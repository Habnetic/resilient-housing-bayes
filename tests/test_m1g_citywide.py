"""Synthetic, source-free unit checks for full-universe M1-G2 scoring logic."""
import numpy as np
import pandas as pd
import pytest

from rhb.pipelines.m1g_citywide import (
    comparison_frame, gamma_label, posterior_indices, predictive_rate_check,
    scenario_for, score_draws, topk_indices,
)


def population(n: int = 12) -> pd.DataFrame:
    t50 = np.linspace(-2, 2, n)
    return pd.DataFrame({
        "bldg_id": [f"A{i:03d}" for i in range(n)],
        "E_hat_v0": np.linspace(-0.75, 0.95, n),
        "H_pluvial_v1_logrel": np.linspace(0.7, -0.35, n),
        "T50": t50,
        "T": np.arcsinh(t50 / .5),
    })


def cfg():
    return {"terrain_scale_m": .5, "outcome_seed": 22,
            "alpha_generating": -3., "beta_E_generating": 1., "beta_H_generating": 1.,
            "baseline_prior_sigma": 2.5, "terrain_prior_sigma": .5}


def draws(b1: bool = False):
    x = {"alpha": np.array([-3, -2.8, -3.2, -3.1]),
         "beta_E": np.array([1., 1.2, .8, 1.]),
         "beta_H": np.array([1., 1.1, .9, 1.])}
    if b1:
        x["gamma_T"] = np.array([-.8, -.75, -.85, -.82])
    return x


def test_full_count_and_inclusion_sum():
    frame = scenario_for(population(), "DON", -.8, cfg())
    result = score_draws(frame, draws(b1=True), 3)
    assert len(result) == 12
    assert result["selected_by_p_mean"].sum() == 3
    assert result["selected_by_oracle"].sum() == 3
    assert result["p_topk"].sum() == pytest.approx(3)
    assert result["p_topk"].between(0, 1).all()
    assert (result["rank_sd"] >= 0).all()


def test_tie_breaking_is_deterministic():
    assert topk_indices(np.array([.1, .9, .9, .8]), 2).tolist() == [1, 2]
    assert topk_indices(np.array([.1, .9, .9, .8]), 4).tolist() == [1, 2, 3, 0]


def test_negative_control_reuses_outcome_uniforms():
    p = population()
    a = scenario_for(p, "HAM", 0., cfg())
    b = scenario_for(p, "HAM", -.8, cfg())
    assert (a["p_true"].to_numpy() != b["p_true"].to_numpy()).any()
    assert np.allclose(a["p_true"], 1. / (1. + np.exp(3 - p["E_hat_v0"] - p["H_pluvial_v1_logrel"])))
    assert np.array_equal(a["Y_synthetic"], scenario_for(p, "HAM", 0., cfg())["Y_synthetic"])


def test_matched_comparison_partitions_universe():
    frame = scenario_for(population(), "DON", -.8, cfg())
    a = score_draws(frame, draws(b1=False), 3)
    b = score_draws(frame, draws(b1=True), 3)
    out, metrics = comparison_frame(a, b, 3)
    assert len(out) == 12
    assert metrics["overlap"] + metrics["entries_B1"] == 3
    assert metrics["entries_B1"] == metrics["exits_B1"]
    assert out["decision_change"].isin({"stayed_selected", "entered_B1", "left_B1", "not_selected"}).all()
    assert np.allclose(out["delta_p_topk_B1_minus_B0"], out["p_topk_B1"] - out["p_topk_B0"])


def test_both_models_same_coefficients_mean_zero_delta():
    frame = scenario_for(population(), "DON", 0., cfg())
    a = score_draws(frame, draws(), 3)
    b = score_draws(frame, draws(), 3)
    out, metrics = comparison_frame(a, b, 3)
    assert metrics["overlap"] == 3
    assert metrics["entries_B1"] == 0
    assert np.allclose(out["delta_p_topk_B1_minus_B0"], 0)


def test_reject_mismatched_building_order():
    frame = scenario_for(population(), "DON", -.8, cfg())
    a = score_draws(frame, draws(), 3)
    b = a.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="order"):
        comparison_frame(a, b, 3)


def test_reject_duplicate_buildings_and_nonfinite_probabilities():
    frame = scenario_for(population(), "DON", 0., cfg())
    frame.loc[1, "bldg_id"] = frame.loc[0, "bldg_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        score_draws(frame, draws(), 3)
    with pytest.raises(ValueError, match="Invalid"):
        topk_indices(np.array([np.nan, 0.5]), 1)


def test_prior_and_posterior_rate_checks_are_explicitly_synthetic():
    frame = scenario_for(population(), "DON", -.8, cfg())
    record = predictive_rate_check(frame, draws(True), cfg(), seed=123, n_check=4)
    assert record["observed_synthetic_rate"] == pytest.approx(float(frame["Y_synthetic"].mean()))
    assert record["status"] == "recorded_not_scientifically_reviewed"
    assert 0 <= record["posterior_predictive_rate"]["p05"] <= 1


def test_posterior_indices_preserves_chain_strata():
    import xarray as xr
    posterior = xr.Dataset({"alpha": (("chain", "draw"), np.arange(20).reshape(2, 10)),
                            "beta_E": (("chain", "draw"), np.ones((2, 10))),
                            "beta_H": (("chain", "draw"), np.ones((2, 10)))})
    result = posterior_indices(posterior, 3)
    assert len(result["alpha"]) == 6
    assert np.array_equal(result["alpha"], [0, 4, 9, 10, 14, 19])


def test_gamma_labels_are_preregistered():
    assert gamma_label(0.) == "gamma_0"
    assert gamma_label(-.8) == "gamma_neg0p8"
    with pytest.raises(ValueError):
        gamma_label(-.4)
