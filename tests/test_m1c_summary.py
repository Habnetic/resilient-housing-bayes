"""Regression tests for the M1-C aggregated-results pipeline."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rhb.pipelines.summarize_m1c import summarize


def _example_outputs(tmp_path):
    source = tmp_path / "full"
    source.mkdir()
    fit_rows = []
    compare_rows = []
    for gamma, r0, r1 in ((0.0, 0.01, 0.01), (-0.8, 0.25, 0.02)):
        compare_rows.append({
            "gamma_true": gamma, "simulation_seed": 101, "k": 10,
            "overlap": 10 if gamma == 0 else 8,
            "oracle_overlap_B0": 10 if gamma == 0 else 8,
            "oracle_overlap_B1": 10,
            "borderline_B0": 2, "borderline_B1": 2,
            "expected_regret_B0": r0, "expected_regret_B1": r1,
            "mean_abs_inclusion_change": 0.01, "max_inclusion_change": 0.55,
        })
        for model, regret in (("B0", r0), ("B1", r1)):
            fit_rows.append({
                "gamma_true": gamma, "simulation_seed": 101, "model": model,
                "expected_regret": regret,
                "gamma_posterior_mean": gamma + 0.1 if model == "B1" else None,
                "gamma_hdi_low": gamma - 0.2 if model == "B1" else None,
                "gamma_hdi_high": gamma + 0.2 if model == "B1" else None,
                "divergences": 0,
                "rhat_max": 1.02 if (gamma == -0.8 and model == "B0") else 1.0,
                "ess_bulk_min": 450,
                "ess_tail_min": 390 if (gamma == -0.8 and model == "B1") else 500,
                "rmse_vs_true_p": 0.03 if model == "B0" else 0.02,
                "brier": 0.06 if model == "B0" else 0.059,
                "heldout_elpd": -220.0 if model == "B0" else -218.0,
            })
    pd.DataFrame(fit_rows).to_csv(source / "fit_results.csv", index=False)
    pd.DataFrame(compare_rows).to_csv(source / "scenario_comparisons.csv", index=False)
    (source / "manifest.json").write_text(
        json.dumps({"scenario_count": 2, "model_fits": 4, "stage": "full"}),
        encoding="utf-8",
    )
    return source


def test_summary_preserves_paired_regret_and_flags(tmp_path):
    source = _example_outputs(tmp_path)
    report = summarize(source, source / "summary")
    rows = pd.read_csv(source / "summary/summary_by_gamma.csv")
    signal = rows.loc[rows["gamma_true"] == -0.8].iloc[0]
    assert signal["mean_paired_regret_improvement"] == pytest.approx(0.23)
    assert signal["regret_reduction_percent_of_mean"] == pytest.approx(92.0)
    assert signal["mean_heldout_elpd_gain_B1_minus_B0"] == pytest.approx(2.0)
    assert signal["gamma_95_hdi_covered_count"] == 1
    assert report["n_flagged_fits"] == 2
    assert len(pd.read_csv(source / "summary/diagnostic_flags.csv")) == 2


def test_summary_fails_on_inconsistent_regret(tmp_path):
    source = _example_outputs(tmp_path)
    cases = pd.read_csv(source / "scenario_comparisons.csv")
    cases.loc[cases["gamma_true"] == -0.8, "expected_regret_B1"] = 0.14
    cases.to_csv(source / "scenario_comparisons.csv", index=False)
    with pytest.raises(ValueError, match="Inconsistent B1 regret"):
        summarize(source, source / "summary")


def test_summary_fails_on_missing_fit(tmp_path):
    source = _example_outputs(tmp_path)
    fits = pd.read_csv(source / "fit_results.csv")
    fits.iloc[:-1].to_csv(source / "fit_results.csv", index=False)
    with pytest.raises(ValueError, match="two fits"):
        summarize(source, source / "summary")
