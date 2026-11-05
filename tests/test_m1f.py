"""Pure tests for M1-F sample design, paired truth, and paired summaries.

These tests do not execute PyMC and do not require actual citywide inputs.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from rhb.experiments.m1f_population_sampling import (
    derived_seed, generate_scenarios, select_populations,
    validate_config, validate_scenarios,
)
from rhb.pipelines.run_m1f import cases_for
from rhb.pipelines.summarize_m1f import (
    check_result_tables, diagnostics, summarize_pairs, with_predictive_contrasts,
)


def config() -> dict:
    return {
        "experiment": "M1-F", "city": "RTM", "population_ids": ["pop01", "pop02", "pop03"],
        "population_seed": 51, "split_seed": 52,
        "sample_size": 5000, "train_size": 4000, "test_size": 1000,
        "gamma_true_values": [0.0, -0.8], "simulation_seeds": [101, 202],
        "generator_alpha": -3.0, "generator_beta_E": 1.0, "generator_beta_H": 1.0,
        "terrain_transform_scale_m": 0.5, "gamma_prior_sigma": 0.5,
        "selection_share": 0.01, "expected_reference_sample_size": 5000,
        "expected_measured_assets": 220456, "outcome_seed_prefix": "test",
        "mcmc": {"chains": 2, "tune": 500, "draws": 500, "target_accept": 0.9},
    }


@pytest.fixture(scope="module")
def fake_samples() -> tuple[dict, pd.DataFrame, pd.DataFrame, set[int]]:
    cfg = config()
    n = 22000
    v = np.arange(n)
    universe = pd.DataFrame({
        "bldg_id": v + 1,
        "E_hat_v0": np.sin(v / 50),
        "H_pluvial_v1_logrel": np.cos(v / 100) * 0.005,
        "T50": np.sin(v / 70) * 0.2,
    })
    universe["T"] = np.arcsinh(universe["T50"] / 0.5)
    excluded = set(range(1, 5001))
    samples = select_populations(universe, excluded, cfg)
    cases = generate_scenarios(samples, cfg)
    return cfg, samples, cases, excluded


def test_valid_and_frozen_design() -> None:
    validate_config(config())
    bad = config()
    bad["gamma_true_values"] = [0.0, -0.4]
    with pytest.raises(ValueError):
        validate_config(bad)


def test_reject_duplicate_populations() -> None:
    cfg = config()
    cfg["population_ids"] = ["pop01", "pop01", "pop02"]
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_reject_other_python_inference_configuration() -> None:
    cfg = config()
    cfg["mcmc"]["draws"] = 100
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_population_exclusion_disjointness_and_split(fake_samples) -> None:
    cfg, sample, _, excluded = fake_samples
    assert len(sample) == 15000
    assert sample["bldg_id"].is_unique
    assert not set(sample["bldg_id"]) & excluded
    assert sample.groupby("population_id").size().to_dict() == dict.fromkeys(cfg["population_ids"], 5000)
    assert (sample["split"] == "train").sum() == 12000


def test_reproducible_populations(fake_samples) -> None:
    cfg, sampled, _, excluded = fake_samples
    universe = pd.concat([sampled, sampled], ignore_index=True).drop_duplicates("bldg_id")
    # Additional unused eligible assets to make the selection possible.
    extra = universe.copy()
    extra["bldg_id"] += 100000
    bigger = pd.concat([universe, extra], ignore_index=True)
    a = select_populations(bigger, excluded, cfg)
    b = select_populations(bigger, excluded, cfg)
    pd.testing.assert_frame_equal(a, b)


def test_paired_outcomes_and_generator(fake_samples) -> None:
    cfg, _, cases, excluded = fake_samples
    assert len(cases) == 60000
    validate_scenarios(cases, cfg, excluded)
    g0 = cases.query('population_id == "pop01" and gamma_true == 0 and simulation_seed == 101')
    g8 = cases.query('population_id == "pop01" and gamma_true == -0.8 and simulation_seed == 101')
    assert g0["bldg_id"].reset_index(drop=True).equals(g8["bldg_id"].reset_index(drop=True))
    np.testing.assert_allclose(g8["p_true"], expit(-3 + g8["E_hat_v0"] + g8["H_pluvial_v1_logrel"] - 0.8 * g8["T"]))
    assert derived_seed(cfg["outcome_seed_prefix"], "pop01", 101) == derived_seed(cfg["outcome_seed_prefix"], "pop01", 101)


def test_corrupt_synthetic_label_detected(fake_samples) -> None:
    cfg, _, cases, excluded = fake_samples
    changed = cases.copy()
    changed.at[0, "Y_m1f"] = 1 - changed.at[0, "Y_m1f"]
    with pytest.raises(ValueError, match="Paired outcome"):
        validate_scenarios(changed, cfg, excluded)


def test_training_split_leak_detected(fake_samples) -> None:
    cfg, _, cases, excluded = fake_samples
    changed = cases.copy()
    changed.at[0, "split"] = "test" if changed.at[0, "split"] == "train" else "train"
    with pytest.raises(ValueError, match="Invalid train/test split"):
        validate_scenarios(changed, cfg, excluded)


def test_four_smoke_and_twenty_four_full_fits() -> None:
    assert len(cases_for(config(), smoke=True)) == 2
    assert len(cases_for(config(), smoke=False)) == 12


def tiny_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    cases = pd.DataFrame({
        "population_id": ["pop01", "pop01"], "gamma_true": [0., -0.8],
        "simulation_seed": [101, 101], "k": [10, 10], "overlap": [10, 8],
        "expected_regret_B0": [0.0, 0.25], "expected_regret_B1": [0.0, 0.05],
        "regret_improvement_B0_minus_B1": [0.0, 0.2],
        "oracle_overlap_B0": [10, 8], "oracle_overlap_B1": [10, 10],
        "borderline_B0": [2, 2], "borderline_B1": [2, 4],
        "mean_abs_inclusion_change": [0.0, 0.01], "max_inclusion_change": [0.0, 0.5],
    })
    rows = []
    for _, row in cases.iterrows():
        for model in ("B0", "B1"):
            rows.append({"population_id": row["population_id"], "gamma_true": row["gamma_true"],
                         "simulation_seed": row["simulation_seed"], "model": model,
                         "expected_regret": row[f"expected_regret_{model}"],
                         "rhat_max": 1.0, "ess_bulk_min": 500, "ess_tail_min": 600, "divergences": 0,
                         "heldout_elpd": -100 if model == "B0" else -99,
                         "brier": 0.08 if model == "B0" else 0.07,
                         "rmse_vs_true_p": 0.02 if model == "B0" else 0.01,
                         "gamma_posterior_mean": (-0.7 if row["gamma_true"] < 0 else 0.01) if model == "B1" else None,
                         "gamma_covered": model == "B1",
                         "posterior_predictive_rate_p50": 0.06,
                         "prior_predictive_rate_p50": 0.50,
                         "train_damage_rate": 0.06})
    return pd.DataFrame(rows), cases


def test_paired_regret_aggregation_and_coverage() -> None:
    fit, cases = tiny_tables()
    check_result_tables(fit, cases, populations=1, seeds=1)
    by = summarize_pairs(cases, ["gamma_true"])
    a = by.loc[by["gamma_true"] == -0.8].iloc[0]
    assert a["B1_wins"] == 1 and a["mean_regret_B1"] == pytest.approx(0.05)
    assert a["mean_paired_regret_improvement"] == pytest.approx(0.2)
    zero = by.loc[by["gamma_true"] == 0].iloc[0]
    assert zero["ties"] == 1


def test_detect_missing_fit() -> None:
    fit, cases = tiny_tables()
    with pytest.raises(ValueError):
        check_result_tables(fit.iloc[:-1], cases, populations=1, seeds=1)


def test_detect_wrong_paired_regret() -> None:
    fit, cases = tiny_tables()
    changed = cases.copy()
    changed.loc[1, "regret_improvement_B0_minus_B1"] = -0.2
    with pytest.raises(ValueError, match="Regret improvements"):
        check_result_tables(fit, changed, populations=1, seeds=1)


def test_diagnostic_flags() -> None:
    fit, _ = tiny_tables()
    fit.loc[0, "ess_tail_min"] = 399
    flags = diagnostics(fit, {"max_rhat": 1.01, "min_bulk_ess": 400, "min_tail_ess": 400})
    assert len(flags) == 1 and bool(flags.iloc[0]["flag_tail_ess"])


def test_predictive_metrics_pairing() -> None:
    fits, cases = tiny_tables()
    matched = with_predictive_contrasts(cases, fits)
    assert len(matched) == 2
    assert np.allclose(matched["delta_heldout_elpd_B1_minus_B0"], 1.0)
    assert np.allclose(matched["delta_brier_B1_minus_B0"], -0.01)
    summary = summarize_pairs(matched, ["gamma_true"])
    assert "mean_gamma_estimate_B1" in summary.columns
    assert summary.loc[summary["gamma_true"] == -0.8, "mean_gamma_estimate_B1"].item() == pytest.approx(-0.7)


def test_measured_terrain_join_rejects_missing_feature(tmp_path, monkeypatch) -> None:
    from rhb.experiments.m1f_population_sampling import load_universe
    fpath = tmp_path / "features.parquet"
    tpath = tmp_path / "terrain.parquet"
    fpath.touch()
    tpath.touch()
    features = pd.DataFrame({
        "bldg_id": [1, 2], "E_hat_v0": [0., 1.], "H_pluvial_v1_logrel": [0.1, 0.2],
    })
    terrain = pd.DataFrame({
        "bldg_id": [1, 3, 4], "T50": [0.1, 0.2, None],
        "terrain_measured": [True, True, False],
    })
    monkeypatch.setattr(pd, "read_parquet", lambda path, columns: {
        fpath: features, tpath: terrain,
    }[path][columns].copy())
    with pytest.raises(ValueError, match="without matching Phase 3"):
        load_universe(fpath, tpath, expected_count=2)


def test_measured_terrain_join_transform(tmp_path, monkeypatch) -> None:
    from rhb.experiments.m1f_population_sampling import load_universe
    fpath = tmp_path / "features.parquet"
    tpath = tmp_path / "terrain.parquet"
    fpath.touch()
    tpath.touch()
    features = pd.DataFrame({
        "bldg_id": [2, 1], "E_hat_v0": [0., 1.], "H_pluvial_v1_logrel": [0.1, 0.2],
    })
    terrain = pd.DataFrame({
        "bldg_id": [1, 2, 3], "T50": [0.5, -0.5, None],
        "terrain_measured": [True, True, False],
    })
    monkeypatch.setattr(pd, "read_parquet", lambda path, columns: {
        fpath: features, tpath: terrain,
    }[path][columns].copy())
    out = load_universe(fpath, tpath, expected_count=2)
    assert out["bldg_id"].tolist() == [1, 2]
    np.testing.assert_allclose(out["T"], [np.arcsinh(1.0), np.arcsinh(-1.0)])
