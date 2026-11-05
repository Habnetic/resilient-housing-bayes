"""M1-D2 intercept prior sensitivity, reusing M1-C synthetic cases and reference fits.

Mandatory first step: python -m rhb.experiments.m1d2_prior_predictive
Then: python -m rhb.pipelines.run_m1d2 --smoke --dry-run
      python -m rhb.pipelines.run_m1d2 --smoke
      python -m rhb.pipelines.run_m1d2 --dry-run
      python -m rhb.pipelines.run_m1d2

P0 B1 and B0 are authenticated M1-C references. P1/P2/P3 are new B1 fits.
No changes to M1-C, Phase 3, or M1-D1. All outcomes are synthetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from rhb.decision.m1c_metrics import compare_decisions, decision_metrics, predictive_scores
from rhb.experiments.m1d2_prior_predictive import (
    AUDIT_DIR,
    CONFIG_PATH,
    OUTPUT_ROOT,
    validate_config,
)
from rhb.pipelines.run_m1c import (
    DATA_DIR as M1C_DIR,
    git_head,
    label,
    posterior_probabilities,
    read_inputs,
    sha256,
)
from rhb.pipelines.run_m1d import reference_fit

LOG = logging.getLogger("rhb.m1d2.run")


def sampler_seed(gamma: float, seed: int, prior_id: str) -> int:
    """Stable M1-D2 seed, distinct from M1-C and M1-D1."""
    payload = f"M1-D2-v0.1|{gamma:g}|{seed}|{prior_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def audit_gate(source_hashes: dict[str, str]) -> dict[str, Any]:
    """Stop if a prior audit is absent or based on changed inputs/code."""
    manifest_path = AUDIT_DIR / "audit_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Run `python -m rhb.experiments.m1d2_prior_predictive` first."
        )
    audit = json.loads(manifest_path.read_text(encoding="utf-8"))
    from rhb.experiments import m1d2_prior_predictive as prior_module

    if audit.get("experiment") != "M1-D2" or audit.get("stage") != "prior_predictive":
        raise ValueError("Unexpected prior audit manifest")
    if audit.get("source_m1c_fingerprints") != source_hashes:
        raise ValueError("M1-C data/source changed after prior predictive audit")
    if audit.get("config_sha256") != sha256(CONFIG_PATH):
        raise ValueError("M1-D2 prior config changed after audit; rerun audit explicitly")
    if audit.get("prior_audit_code_sha256") != sha256(Path(prior_module.__file__)):
        raise ValueError("Prior predictive implementation changed after audit")
    outputs = {
        "draws": AUDIT_DIR / "prior_predictive_draws.parquet",
        "summary": AUDIT_DIR / "prior_predictive_summary.csv",
        "predictors": AUDIT_DIR / "predictor_summary.csv",
        "observed": AUDIT_DIR / "observed_synthetic_train_rates.csv",
    }
    for name, path in outputs.items():
        if not path.is_file() or audit.get("outputs_sha256", {}).get(name) != sha256(path):
            raise ValueError(f"Prior predictive artifact missing/changed: {path}")
    return audit


def model_with_intercept_prior(
    e: np.ndarray,
    h: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
    prior: dict[str, Any],
    fixed_beta_sigma: float,
    fixed_gamma_sigma: float,
) -> pm.Model:
    """B1 logistic likelihood; only alpha prior differs from frozen M1-C."""
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=float(prior["mean"]), sigma=float(prior["sigma"]))
        beta_e = pm.Normal("beta_E", mu=0.0, sigma=fixed_beta_sigma)
        beta_h = pm.Normal("beta_H", mu=0.0, sigma=fixed_beta_sigma)
        gamma_t = pm.Normal("gamma_T", mu=0.0, sigma=fixed_gamma_sigma)
        logits = alpha + beta_e * e + beta_h * h + gamma_t * t
        pm.Bernoulli("Y_obs", logit_p=logits, observed=y)
    return model


def fit_fingerprints(
    source_hashes: dict[str, str], prior: dict[str, Any], audit_sha256: str,
) -> dict[str, str]:
    """Prevent stale cached fits from being used with changed assumptions."""
    return {
        **source_hashes,
        "m1d2_prior_id": str(prior["id"]),
        "m1d2_alpha_mean": f"{float(prior['mean']):.17g}",
        "m1d2_alpha_sigma": f"{float(prior['sigma']):.17g}",
        "m1d2_config_sha256": sha256(CONFIG_PATH),
        "m1d2_runner_sha256": sha256(Path(__file__)),
        "m1d2_prior_audit_manifest_sha256": audit_sha256,
    }


def fit_one(
    group: pd.DataFrame,
    c_config: dict[str, Any],
    d_config: dict[str, Any],
    hashes: dict[str, str],
    prior: dict[str, Any],
    gamma_true: float,
    simulation_seed: int,
    fit_dir: Path,
    k: int,
    force: bool,
) -> dict[str, Any]:
    """Fit one new B1 variant; write posterior, test decisions, diagnostics."""
    fit_dir.mkdir(parents=True, exist_ok=True)
    record_path = fit_dir / "fit_record.json"
    posterior_path = fit_dir / "idata.nc"
    decisions_path = fit_dir / "test_decisions.parquet"
    if record_path.is_file() and not force:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("fingerprints") != hashes:
            raise ValueError(f"Stale M1-D2 cache at {fit_dir}; review then pass --force")
        if not posterior_path.is_file() or not decisions_path.is_file():
            raise ValueError(f"Incomplete M1-D2 cache at {fit_dir}; review then pass --force")
        LOG.info("Reusing %s", fit_dir)
        return record

    train = group.loc[group["split"].eq("train")].reset_index(drop=True)
    test = group.loc[group["split"].eq("test")].reset_index(drop=True)
    if len(train) != int(c_config["train_size"]) or len(test) != int(c_config["test_size"]):
        raise ValueError("Train/test split differs from M1-C")
    e = train["E_hat_v0"].to_numpy(dtype=float)
    h = train["H_pluvial_v1_logrel"].to_numpy(dtype=float)
    t = train["T"].to_numpy(dtype=float)
    y = train["Y_m1c"].to_numpy(dtype=int)
    model = model_with_intercept_prior(
        e, h, t, y, prior, float(d_config["fixed_beta_sigma"]),
        float(d_config["fixed_gamma_sigma"]),
    )
    mcmc = c_config["mcmc"]
    seed = sampler_seed(gamma_true, simulation_seed, prior["id"])
    LOG.info("Fitting gamma=%g seed=%d intercept=%s", gamma_true, simulation_seed, prior["id"])
    with model:
        prior_replications = pm.sample_prior_predictive(samples=100, random_seed=seed)
        idata = pm.sample(
            draws=int(mcmc["draws"]),
            tune=int(mcmc["tune"]),
            chains=int(mcmc["chains"]),
            cores=min(2, int(mcmc["chains"])),
            target_accept=float(mcmc["target_accept"]),
            random_seed=seed,
            progressbar=False,
        )
        posterior_replications = pm.sample_posterior_predictive(
            idata, var_names=["Y_obs"], random_seed=seed, progressbar=False,
        )

    summary = az.summary(
        idata, var_names=["alpha", "beta_E", "beta_H", "gamma_T"], hdi_prob=0.95,
    )
    p_test = posterior_probabilities(idata, test, extended=True)
    scores = predictive_scores(
        p_test, test["Y_m1c"].to_numpy(dtype=int), test["p_true"].to_numpy(dtype=float),
    )
    decisions, decision_info = decision_metrics(
        p_test, test["bldg_id"].to_numpy(), test["p_true"].to_numpy(dtype=float), k=k,
    )
    prior_rates = (
        prior_replications.prior_predictive["Y_obs"].values
        .reshape(-1, len(train)).mean(axis=1)
    )
    posterior_rates = (
        posterior_replications.posterior_predictive["Y_obs"].values
        .reshape(-1, len(train)).mean(axis=1)
    )
    percentiles = (5, 50, 95)
    hdi_low, hdi_high = "hdi_2.5%", "hdi_97.5%"
    record: dict[str, Any] = {
        "gamma_true": float(gamma_true),
        "simulation_seed": int(simulation_seed),
        "model": "B1",
        "prior_id": prior["id"],
        "alpha_prior_mean": float(prior["mean"]),
        "alpha_prior_sigma": float(prior["sigma"]),
        "sampler_seed": int(seed),
        "train_n": len(train),
        "test_n": len(test),
        "train_damage_rate": float(y.mean()),
        "test_damage_rate": float(test["Y_m1c"].mean()),
        "divergences": int(idata.sample_stats["diverging"].sum().item()),
        "rhat_max": float(summary["r_hat"].max()),
        "ess_bulk_min": float(summary["ess_bulk"].min()),
        "ess_tail_min": float(summary["ess_tail"].min()),
        "alpha_posterior_mean": float(summary.loc["alpha", "mean"]),
        "alpha_hdi_low": float(summary.loc["alpha", hdi_low]),
        "alpha_hdi_high": float(summary.loc["alpha", hdi_high]),
        "gamma_posterior_mean": float(summary.loc["gamma_T", "mean"]),
        "gamma_hdi_low": float(summary.loc["gamma_T", hdi_low]),
        "gamma_hdi_high": float(summary.loc["gamma_T", hdi_high]),
        "gamma_covered": bool(summary.loc["gamma_T", hdi_low] <= gamma_true <= summary.loc["gamma_T", hdi_high]),
        "alpha_covered": bool(summary.loc["alpha", hdi_low] <= -3.0 <= summary.loc["alpha", hdi_high]),
        "prior_predictive_rate_p05": float(np.percentile(prior_rates, percentiles[0])),
        "prior_predictive_rate_p50": float(np.percentile(prior_rates, percentiles[1])),
        "prior_predictive_rate_p95": float(np.percentile(prior_rates, percentiles[2])),
        "posterior_predictive_rate_p05": float(np.percentile(posterior_rates, percentiles[0])),
        "posterior_predictive_rate_p50": float(np.percentile(posterior_rates, percentiles[1])),
        "posterior_predictive_rate_p95": float(np.percentile(posterior_rates, percentiles[2])),
        "fingerprints": hashes,
        **scores,
        **decision_info,
    }
    az.to_netcdf(idata, posterior_path)
    summary.to_csv(fit_dir / "posterior_summary.csv")
    decisions.to_parquet(decisions_path, index=False)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    LOG.info("Completed %s: regret=%.5g alpha_mean=%.3f",
             fit_dir, record["expected_regret"], record["alpha_posterior_mean"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Only first synthetic outcome seed")
    parser.add_argument("--dry-run", action="store_true", help="Validate preregistration and references, no fitting")
    parser.add_argument("--force", action="store_true", help="Explicitly recompute only new M1-D2 fits")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    d_cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    priors = validate_config(d_cfg)
    c_cfg, scenarios, source_hashes = read_inputs()
    if float(c_cfg["gamma_prior_sigma"]) != float(d_cfg["fixed_gamma_sigma"]):
        raise ValueError("M1-C gamma prior differs from M1-D2 frozen reference")
    audit_gate(source_hashes)
    audit_hash = sha256(AUDIT_DIR / "audit_manifest.json")
    stage = "smoke" if args.smoke else "full"
    gamma_values = [float(x) for x in c_cfg["gamma_true_values"]]
    seeds = [int(x) for x in c_cfg["simulation_seeds"]]
    if args.smoke:
        seeds = seeds[:1]
    k = max(1, int(round(int(c_cfg["test_size"]) * float(c_cfg["selection_share"]))))
    n_new = len(gamma_values) * len(seeds) * (len(priors) - 1)
    LOG.info("M1-D2 %s: %d scenarios x %d intercept priors; %d new B1 fits, k=%d",
             stage, len(gamma_values) * len(seeds), len(priors), n_new, k)

    source_stage = M1C_DIR / stage / "fits"
    # Full provenance verification, including for dry-run.
    for gamma in gamma_values:
        for seed in seeds:
            case_dir = source_stage / label(gamma, seed)
            reference_fit(case_dir, "B0", source_hashes)
            reference_fit(case_dir, "B1", source_hashes)
            if args.dry_run:
                LOG.info("DRY RUN gamma=%g seed=%d: reuse P0; fit %s",
                         gamma, seed, ",".join(x["id"] for x in priors[1:]))
    if args.dry_run:
        return

    out = OUTPUT_ROOT / stage
    rows: list[dict[str, Any]] = []
    for gamma in gamma_values:
        for seed in seeds:
            case_dir = source_stage / label(gamma, seed)
            record_b0, decisions_b0 = reference_fit(case_dir, "B0", source_hashes)
            record_baseline, decisions_baseline = reference_fit(case_dir, "B1", source_hashes)
            group = scenarios.loc[
                np.isclose(scenarios["gamma_true"], gamma, rtol=0, atol=1e-12)
                & scenarios["simulation_seed"].eq(seed)
            ].reset_index(drop=True)
            if len(group) != int(c_cfg["sample_size"]):
                raise ValueError(f"Incorrect synthetic case size: gamma={gamma}, seed={seed}")

            for prior in priors:
                if prior["id"] == "P0":
                    record, decisions = record_baseline, decisions_baseline
                    source = "M1-C baseline B1"
                    # M1-C stores alpha summaries separately, not in fit_record.json.
                    reference_summary_path = case_dir / "B1" / "posterior_summary.csv"
                    reference_summary = pd.read_csv(reference_summary_path, index_col=0)
                    alpha_info = reference_summary.loc["alpha"]
                    record = {
                        **record,
                        "alpha_posterior_mean": float(alpha_info["mean"]),
                        "alpha_hdi_low": float(alpha_info["hdi_2.5%"]),
                        "alpha_hdi_high": float(alpha_info["hdi_97.5%"]),
                        "alpha_covered": bool(
                            alpha_info["hdi_2.5%"] <= -3.0 <= alpha_info["hdi_97.5%"]
                        ),
                    }
                else:
                    destination = out / "fits" / label(gamma, seed) / prior["id"] / "B1"
                    record = fit_one(
                        group, c_cfg, d_cfg, fit_fingerprints(source_hashes, prior, audit_hash),
                        prior, gamma, seed, destination, k, args.force,
                    )
                    decisions = pd.read_parquet(destination / "test_decisions.parquet")
                    source = "M1-D2 new B1"
                cmp_b0 = compare_decisions(decisions_b0, decisions, k=k)
                cmp_baseline = compare_decisions(decisions_baseline, decisions, k=k)
                row: dict[str, Any] = {
                    "gamma_true": gamma,
                    "simulation_seed": seed,
                    "prior_id": prior["id"],
                    "alpha_prior_mean": float(prior["mean"]),
                    "alpha_prior_sigma": float(prior["sigma"]),
                    "fit_source": source,
                    "alpha_posterior_mean": record.get("alpha_posterior_mean", np.nan),
                    "alpha_hdi_low": record.get("alpha_hdi_low", np.nan),
                    "alpha_hdi_high": record.get("alpha_hdi_high", np.nan),
                    "alpha_covered": record.get("alpha_covered", False),
                    "gamma_posterior_mean": record["gamma_posterior_mean"],
                    "gamma_covered": record["gamma_covered"],
                    "rhat_max": record["rhat_max"],
                    "ess_bulk_min": record["ess_bulk_min"],
                    "ess_tail_min": record["ess_tail_min"],
                    "divergences": record["divergences"],
                    "prior_predictive_rate_p50_B1": record["prior_predictive_rate_p50"],
                    "posterior_predictive_rate_p50_B1": record["posterior_predictive_rate_p50"],
                    "train_damage_rate": record["train_damage_rate"],
                    "rmse_vs_true_p_B0": record_b0["rmse_vs_true_p"],
                    "rmse_vs_true_p_B1": record["rmse_vs_true_p"],
                    "brier_B0": record_b0["brier"],
                    "brier_B1": record["brier"],
                    "heldout_elpd_B0": record_b0["heldout_elpd"],
                    "heldout_elpd_B1": record["heldout_elpd"],
                    **cmp_b0,
                    "overlap_B1_P0": cmp_baseline["overlap"],
                    "max_inclusion_change_B1_P0": cmp_baseline["max_inclusion_change"],
                    "mean_abs_inclusion_change_B1_P0": cmp_baseline["mean_abs_inclusion_change"],
                    "regret_shift_B1_minus_P0": (
                        float(cmp_b0["expected_regret_B1"])
                        - float(record_baseline["expected_regret"])
                    ),
                }
                if not np.isclose(row["expected_regret_B0"], record_b0["expected_regret"], atol=1e-9):
                    raise AssertionError("B0 regret differs from authenticated M1-C reference")
                if prior["id"] == "P0" and not np.isclose(row["expected_regret_B1"], record_baseline["expected_regret"], atol=1e-9):
                    raise AssertionError("P0 regret differs from authenticated M1-C reference")
                rows.append(row)
                LOG.info("gamma=%g seed=%d %s regret=%.5g overlap P0=%d/%d",
                         gamma, seed, prior["id"], row["expected_regret_B1"], row["overlap_B1_P0"], k)

    out.mkdir(parents=True, exist_ok=True)
    comparison_path = out / "comparison_results.csv"
    pd.DataFrame(rows).sort_values(
        ["gamma_true", "simulation_seed", "prior_id"],
        ascending=[False, True, True],
    ).to_csv(comparison_path, index=False)
    manifest = {
        "experiment": "M1-D2",
        "stage": stage,
        "scenario_count": len(gamma_values) * len(seeds),
        "comparison_rows": len(rows),
        "new_B1_fits": n_new,
        "reused_M1C_B1_fits": len(gamma_values) * len(seeds),
        "git_head": git_head(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "dependencies": {p: metadata.version(p) for p in ("pymc", "arviz", "numpy", "pandas", "scipy")},
        "source_m1c_fingerprints": source_hashes,
        "audit_manifest_sha256": audit_hash,
        "config_sha256": sha256(CONFIG_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "comparison_results_sha256": sha256(comparison_path),
        "m1c_config": c_cfg,
        "m1d2_config": d_cfg,
        "note": "Synthetic prior sensitivity only; oracle-centred priors rely on a known generator.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("M1-D2 %s COMPLETE: %d comparisons, %d new fits, %s",
             stage.upper(), len(rows), n_new, out)


if __name__ == "__main__":  # essential for Windows multiprocess sampling
    main()
