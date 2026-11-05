r"""M1-F: aggregate paired decision quality across independent asset samples and outcome seeds.

From repository root:
    .\.venv\Scripts\python.exe -m rhb.pipelines.summarize_m1f --smoke
    .\.venv\Scripts\python.exe -m rhb.pipelines.summarize_m1f

Only aggregates existing generated CSVs. No inference or population changes.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rhb.experiments.m1f_population_sampling import OUTPUT_DIR, check_python, sha256

LOG = logging.getLogger("rhb.m1f.summary")


def check_result_tables(fits: pd.DataFrame, cases: pd.DataFrame, populations: int, seeds: int) -> None:
    """Reject incomplete or inconsistent model comparisons before aggregating."""
    expected_cases = populations * seeds * 2
    if len(cases) != expected_cases or len(fits) != expected_cases * 2:
        raise ValueError(f"Expected {expected_cases} cases / {2 * expected_cases} fits")
    keys = ["population_id", "gamma_true", "simulation_seed"]
    if cases.duplicated(keys).any() or fits.duplicated(keys + ["model"]).any():
        raise ValueError("Duplicate model or paired case")
    if set(fits["model"]) != {"B0", "B1"} or not (fits.groupby(keys)["model"].nunique() == 2).all():
        raise ValueError("Every case requires both B0 and B1")
    fit_keys = set(map(tuple, fits[keys].to_numpy()))
    comparison_keys = set(map(tuple, cases[keys].to_numpy()))
    if fit_keys != comparison_keys:
        raise ValueError("Fit and comparison case sets do not match")
    if not (cases["k"] == 10).all() or not (cases["overlap"].between(0, 10)).all():
        raise ValueError("Unexpected budget or impossible overlap")
    for name in ("expected_regret_B0", "expected_regret_B1"):
        if not np.isfinite(cases[name].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid {name}")
    if not np.allclose(
        cases["expected_regret_B0"].to_numpy(dtype=float)
        - cases["expected_regret_B1"].to_numpy(dtype=float),
        cases["regret_improvement_B0_minus_B1"].to_numpy(dtype=float), rtol=0, atol=1e-8,
    ):
        raise ValueError("Regret improvements inconsistent with component regrets")
    for model, name in (("B0", "expected_regret_B0"), ("B1", "expected_regret_B1")):
        joined = fits.loc[fits["model"] == model, keys + ["expected_regret"]].merge(
            cases[keys + [name]], on=keys, validate="one_to_one"
        )
        if not np.allclose(joined["expected_regret"], joined[name], rtol=0, atol=1e-8):
            raise ValueError("Fit decision regret does not match paired comparison")




def with_predictive_contrasts(cases: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """Attach matched held-out scores and B1 coefficient recovery per case."""
    keys = ["population_id", "gamma_true", "simulation_seed"]
    scores = ["heldout_elpd", "brier", "rmse_vs_true_p"]
    for name in scores + ["gamma_posterior_mean", "gamma_covered",
                          "posterior_predictive_rate_p50", "prior_predictive_rate_p50",
                          "train_damage_rate"]:
        if name not in fits.columns:
            raise ValueError(f"Required M1-C fit metric missing: {name}")
    result = cases.copy()
    for model in ("B0", "B1"):
        cols = scores + (["gamma_posterior_mean", "gamma_covered",
                          "posterior_predictive_rate_p50", "prior_predictive_rate_p50",
                          "train_damage_rate"] if model == "B1" else [])
        sub = fits.loc[fits["model"] == model, keys + cols].copy()
        sub = sub.rename(columns={col: f"{col}_{model}" for col in cols})
        result = result.merge(sub, on=keys, how="left", validate="one_to_one")
    for name in scores:
        result[f"delta_{name}_B1_minus_B0"] = result[f"{name}_B1"] - result[f"{name}_B0"]
    if result[[f"delta_{name}_B1_minus_B0" for name in scores]].isna().any().any():
        raise ValueError("Missing held-out metric in paired fits")
    result["gamma_bias_B1"] = result["gamma_posterior_mean_B1"] - result["gamma_true"]
    return result

def diagnostics(fits: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    result = fits.copy()
    result["flag_rhat"] = result["rhat_max"].isna() | (result["rhat_max"] > float(thresholds["max_rhat"]))
    result["flag_bulk_ess"] = result["ess_bulk_min"].isna() | (result["ess_bulk_min"] < float(thresholds["min_bulk_ess"]))
    result["flag_tail_ess"] = result["ess_tail_min"].isna() | (result["ess_tail_min"] < float(thresholds["min_tail_ess"]))
    result["flag_divergences"] = result["divergences"].isna() | (result["divergences"] > 0)
    return result.loc[result[["flag_rhat", "flag_bulk_ess", "flag_tail_ess", "flag_divergences"]].any(axis=1)].copy()


def summarize_pairs(cases: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Keep population and outcome-seed uncertainty distinct in reporting."""
    rows = []
    for key, group in cases.groupby(by, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        delta = group["regret_improvement_B0_minus_B1"].to_numpy(dtype=float)
        row = {name: value for name, value in zip(by, keys)}
        row.update({
            "cases": int(len(group)),
            "distinct_populations": int(group["population_id"].nunique()),
            "distinct_outcome_seeds": int(group["simulation_seed"].nunique()),
            "mean_regret_B0": float(group["expected_regret_B0"].mean()),
            "mean_regret_B1": float(group["expected_regret_B1"].mean()),
            "mean_paired_regret_improvement": float(delta.mean()),
            "B1_wins": int((delta > 1e-10).sum()),
            "ties": int((np.abs(delta) <= 1e-10).sum()),
            "B0_wins": int((delta < -1e-10).sum()),
            "mean_topk_overlap_B0_B1": float(group["overlap"].mean()),
            "mean_oracle_overlap_B0": float(group["oracle_overlap_B0"].mean()),
            "mean_oracle_overlap_B1": float(group["oracle_overlap_B1"].mean()),
            "mean_borderline_B0": float(group["borderline_B0"].mean()),
            "mean_borderline_B1": float(group["borderline_B1"].mean()),
            "mean_abs_inclusion_change": float(group["mean_abs_inclusion_change"].mean()),
            "max_inclusion_change": float(group["max_inclusion_change"].max()),
        })
        if "delta_heldout_elpd_B1_minus_B0" in group.columns:
            row.update({
                "mean_heldout_elpd_delta_B1_minus_B0": float(group["delta_heldout_elpd_B1_minus_B0"].mean()),
                "mean_brier_delta_B1_minus_B0": float(group["delta_brier_B1_minus_B0"].mean()),
                "mean_rmse_delta_B1_minus_B0": float(group["delta_rmse_vs_true_p_B1_minus_B0"].mean()),
                "mean_gamma_estimate_B1": float(group["gamma_posterior_mean_B1"].mean()),
                "mean_gamma_bias_B1": float(group["gamma_bias_B1"].mean()),
                "gamma_coverage_count_B1": int(group["gamma_covered_B1"].fillna(False).astype(bool).sum()),
                "mean_prior_rate_p50_B1": float(group["prior_predictive_rate_p50_B1"].mean()),
                "mean_ppc_rate_p50_B1": float(group["posterior_predictive_rate_p50_B1"].mean()),
                "mean_observed_train_rate_B1": float(group["train_damage_rate_B1"].mean()),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    check_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Aggregate four-fit smoke run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    stage = "smoke" if args.smoke else "full"
    manifest_path = OUTPUT_DIR / f"{stage}_manifest.json"
    fit_path = OUTPUT_DIR / f"{stage}_fit_results.csv"
    comp_path = OUTPUT_DIR / f"{stage}_comparisons.csv"
    for path in (manifest_path, fit_path, comp_path):
        if not path.is_file():
            raise FileNotFoundError(f"M1-F {stage} result missing: {path}. Run run_m1f first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "M1-F" or manifest.get("stage") != stage:
        raise ValueError("Invalid M1-F fit manifest")
    if manifest["output_hashes"] != {"fit_results": sha256(fit_path), "comparisons": sha256(comp_path)}:
        raise ValueError("M1-F fit tables were modified after inference")
    cfg = manifest["config"]
    fits, cases = pd.read_csv(fit_path), pd.read_csv(comp_path)
    check_result_tables(fits, cases,
                        populations=(1 if args.smoke else len(cfg["population_ids"])),
                        seeds=(1 if args.smoke else len(cfg["simulation_seeds"])))
    matched = with_predictive_contrasts(cases, fits)
    by_gamma = summarize_pairs(matched, ["gamma_true"])
    by_population = summarize_pairs(matched, ["population_id", "gamma_true"])
    flags = diagnostics(fits, cfg["diagnostic_limits"])
    out = OUTPUT_DIR / f"{stage}_summary"
    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary_by_gamma": out / "summary_by_gamma.csv",
        "summary_by_population_gamma": out / "summary_by_population_gamma.csv",
        "diagnostic_flags": out / "diagnostic_flags.csv",
        "matched_cases": out / "matched_cases.csv",
    }
    by_gamma.to_csv(outputs["summary_by_gamma"], index=False)
    by_population.to_csv(outputs["summary_by_population_gamma"], index=False)
    flags.to_csv(outputs["diagnostic_flags"], index=False)
    matched.to_csv(outputs["matched_cases"], index=False)
    lines = [
        "# Habnetic M1-F | Population robustness computational summary", "",
        f"Stage: **{stage}**. Scenarios: **{len(cases)}**; B0/B1 fits: **{len(fits)}**.",
        f"Numerical diagnostic flags: **{len(flags)}**.", "",
        "All outcomes and oracle probabilities are synthetic. Populations are disjoint samples",
        "from the *same* Rotterdam inventory. Two outcomes per population are correlated",
        "through their shared buildings, features, and train/test split; they are not",
        "six independent geographical validations.", "",
        "## Decision summaries", "",
        "```text", by_gamma.round(6).to_string(index=False), "```", "",
        "## Per-population summaries", "",
        "```text", by_population.round(6).to_string(index=False), "```", "",
        "## Interpretation rule", "",
        "Review each population and its two outcome seeds before asserting robustness.",
        "No population or spatial transfer claim follows from these tables.",
        "Review prior/PPC diagnostics and posterior samples before confirmation.", "",
        "**What to test next:** Spatial-blocked holdout and HAM/DON feature portability",
        "are separate stages, not evidence supplied by M1-F.", "",
    ]
    report_path = out / "summary_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    summary_manifest = {
        "experiment": "M1-F", "stage": stage,
        "scenarios": int(len(cases)), "fits": int(len(fits)),
        "numerical_flags": int(len(flags)),
        "input_hashes": {"manifest": sha256(manifest_path),
                         "fits": sha256(fit_path), "comparisons": sha256(comp_path)},
        "output_hashes": {p: sha256(f) for p, f in outputs.items()}
                         | {"summary_report": sha256(report_path)},
        "caveat": "Not geographical transfer, empirical damage validation or spatially blocked holdout",
    }
    (out / "summary_manifest.json").write_text(
        json.dumps(summary_manifest, indent=2, allow_nan=False), encoding="utf-8"
    )
    LOG.info("M1-F %s SUMMARY COMPLETE: %d paired cases, %d fits, %d flags, %s",
             stage.upper(), len(cases), len(fits), len(flags), out)
    print(by_gamma.to_string(index=False))
    print("\nPer-population:")
    print(by_population.to_string(index=False))


if __name__ == "__main__":
    main()
