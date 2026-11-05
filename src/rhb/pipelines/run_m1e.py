"""M1-E: targeted computational confirmation of frozen M1-C and M1-D2 synthetic fits.

From the repository root, with src on PYTHONPATH:
  python -m rhb.pipelines.run_m1e --audit-only
  python -m rhb.pipelines.run_m1e --dry-run
  python -m rhb.pipelines.run_m1e --smoke
  python -m rhb.pipelines.run_m1e

One cache across smoke/full; original experiment data and models remain untouched.
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

from rhb.decision.m1c_metrics import decision_metrics, predictive_scores
from rhb.models.logistic_m1 import build_m1_model
from rhb.models.logistic_phase3 import build_phase3_model
from rhb.pipelines.m1e_checks import decision_delta, diagnostic_flags, sampler_seed, validate_plan
from rhb.pipelines.run_m1c import (
    DATA_DIR as M1C_DIR,
    ROOT, git_head, label, posterior_probabilities, read_inputs, sha256,
)
from rhb.pipelines.run_m1d import reference_fit
from rhb.pipelines.run_m1d2 import (
    audit_gate, fit_fingerprints, model_with_intercept_prior,
)
from rhb.experiments.m1d2_prior_predictive import (
    AUDIT_DIR as M1D2_AUDIT_DIR,
    CONFIG_PATH as M1D2_CONFIG_PATH,
    OUTPUT_ROOT as M1D2_ROOT,
)

LOG = logging.getLogger("rhb.m1e")
CONFIG_PATH = ROOT / "configs/m1e_computational_confirmation_rtm.json"
OUT = ROOT / "outputs/phase4/RTM/m1e"


def file_provenance(path: Path) -> dict[str, str]:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def reference_case(case: dict[str, Any], source_hashes: dict[str, str],
                   d2_config: dict[str, Any], audit_hash: str) -> tuple[dict[str, Any], pd.DataFrame, Path]:
    """Authenticate the untouched original fit and its source assumptions."""
    case_path = label(float(case["gamma_true"]), int(case["simulation_seed"]))
    if case["source"] == "M1-C":
        src = M1C_DIR / "full" / "fits" / case_path / case["model"]
        record, decisions = reference_fit(src.parent, case["model"], source_hashes)
    elif case["source"] == "M1-D2":
        src = M1D2_ROOT / "full" / "fits" / case_path / case["prior_id"] / "B1"
        for name in ("fit_record.json", "idata.nc", "test_decisions.parquet"):
            if not (src / name).is_file():
                raise FileNotFoundError(f"Missing original M1-D2 artifact: {src / name}")
        record = json.loads((src / "fit_record.json").read_text(encoding="utf-8"))
        prior = next((p for p in d2_config["intercept_priors"] if p["id"] == case["prior_id"]), None)
        if prior is None or record.get("fingerprints") != fit_fingerprints(source_hashes, prior, audit_hash):
            raise ValueError(f"M1-D2 reference provenance does not match: {src}")
        decisions = pd.read_parquet(src / "test_decisions.parquet")
    else:
        raise ValueError(f"Unknown reference source: {case['source']}")
    if not (src / "idata.nc").is_file():
        raise FileNotFoundError(f"Missing original posterior draws at {src}")
    if case["model"] != record["model"] or float(record["gamma_true"]) != float(case["gamma_true"]) or int(record["simulation_seed"]) != int(case["simulation_seed"]):
        raise ValueError(f"Reference fit identity differs from registered case: {src}")
    return record, decisions, src


def original_diagnostics(case: dict[str, Any], record: dict[str, Any], src: Path,
                         limits: dict[str, Any], trace_plots: bool) -> dict[str, Any]:
    """Recalculate diagnostics from existing posterior draws, no new inference."""
    idata = az.from_netcdf(src / "idata.nc")
    names = ["alpha", "beta_E", "beta_H"] + (["gamma_T"] if case["model"] == "B1" else [])
    summary = az.summary(idata, var_names=names, hdi_prob=0.95)
    result = {
        "case_id": case["case_id"], "source": case["source"],
        "gamma_true": case["gamma_true"], "simulation_seed": case["simulation_seed"],
        "model": case["model"], "prior_id": case["prior_id"],
        "rhat_max": float(summary["r_hat"].max()),
        "ess_bulk_min": float(summary["ess_bulk"].min()),
        "ess_tail_min": float(summary["ess_tail"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum().item()),
        "record_rhat_max": record["rhat_max"],
        "record_ess_bulk_min": record["ess_bulk_min"],
        "record_ess_tail_min": record["ess_tail_min"],
        "posterior_chains": int(idata.posterior.sizes["chain"]),
        "posterior_draws_per_chain": int(idata.posterior.sizes["draw"]),
    }
    result["warnings"] = ";".join(diagnostic_flags(result, limits))
    if trace_plots:
        import matplotlib.pyplot as plt
        destination = OUT / "reference_traces" / f"{case['case_id']}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        az.plot_trace(idata, var_names=names, compact=True)
        plt.suptitle(f"Original: {case['case_id']}")
        plt.savefig(destination, dpi=125, bbox_inches="tight")
        plt.close("all")
    return result


def fingerprint(case: dict[str, Any], repeat: int, original: Path,
                source_hashes: dict[str, str], config_hash: str) -> dict[str, str]:
    """Cache identity includes original evidence and code/model definitions."""
    return {
        **source_hashes,
        "m1e_config_sha256": config_hash,
        "m1e_runner_sha256": sha256(Path(__file__)),
        "m1e_checks_sha256": sha256(Path(__file__).with_name("m1e_checks.py")),
        "m1d2_config_sha256": sha256(M1D2_CONFIG_PATH),
        "m1d2_runner_sha256": sha256(Path(model_with_intercept_prior.__code__.co_filename)),
        "m1e_case_id": case["case_id"],
        "m1e_repeat": str(repeat),
        "original_record_sha256": sha256(original / "fit_record.json"),
        "original_idata_sha256": sha256(original / "idata.nc"),
        "original_decisions_sha256": sha256(original / "test_decisions.parquet"),
    }


def fit_one(case: dict[str, Any], repeat: int, group: pd.DataFrame,
            c_cfg: dict[str, Any], d2_cfg: dict[str, Any], e_cfg: dict[str, Any],
            hashes: dict[str, str], fit_dir: Path, force: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    """Refit the *same* likelihood/priors with four longer chains."""
    record_path = fit_dir / "fit_record.json"
    decisions_path = fit_dir / "test_decisions.parquet"
    idata_path = fit_dir / "idata.nc"
    if record_path.is_file() and not force:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("fingerprints") != hashes:
            raise ValueError(f"Stale confirmation cache: {fit_dir}. Investigate before --force")
        if not decisions_path.is_file() or not idata_path.is_file() or not (fit_dir / "trace.png").is_file():
            raise ValueError(f"Incomplete confirmation cache: {fit_dir}")
        LOG.info("Reusing completed confirmation: %s", fit_dir)
        return record, pd.read_parquet(decisions_path)

    train = group.loc[group["split"].eq("train")].reset_index(drop=True)
    test = group.loc[group["split"].eq("test")].reset_index(drop=True)
    if len(train) != int(c_cfg["train_size"]) or len(test) != int(c_cfg["test_size"]):
        raise ValueError("Original train/test split changed")
    e = train["E_hat_v0"].to_numpy(dtype=float)
    h = train["H_pluvial_v1_logrel"].to_numpy(dtype=float)
    t = train["T"].to_numpy(dtype=float)
    y = train["Y_m1c"].to_numpy(dtype=int)
    if case["model"] == "B0":
        model = build_phase3_model(e, h, y)
        extended = False
    elif case["prior_id"] == "P0":
        model = build_m1_model(e, h, t, y, gamma_sigma=float(c_cfg["gamma_prior_sigma"]))
        extended = True
    else:
        prior = next(p for p in d2_cfg["intercept_priors"] if p["id"] == case["prior_id"])
        model = model_with_intercept_prior(e, h, t, y, prior,
            fixed_beta_sigma=float(d2_cfg["fixed_beta_sigma"]),
            fixed_gamma_sigma=float(d2_cfg["fixed_gamma_sigma"]))
        extended = True
    seed = sampler_seed(case["case_id"], repeat)
    mcmc = e_cfg["mcmc"]
    LOG.info("M1-E fitting %s, repeat %d, seed %d", case["case_id"], repeat, seed)
    with model:
        prior_rep = pm.sample_prior_predictive(samples=100, random_seed=seed)
        idata = pm.sample(draws=int(mcmc["draws"]), tune=int(mcmc["tune"]),
            chains=int(mcmc["chains"]), cores=int(mcmc["cores"]),
            target_accept=float(mcmc["target_accept"]),
            random_seed=seed, progressbar=False)
        ppc = pm.sample_posterior_predictive(idata, var_names=["Y_obs"],
            random_seed=seed, progressbar=False)

    names = ["alpha", "beta_E", "beta_H"] + (["gamma_T"] if extended else [])
    summ = az.summary(idata, var_names=names, hdi_prob=0.95)
    pred = posterior_probabilities(idata, test, extended)
    scores = predictive_scores(pred, test["Y_m1c"].to_numpy(dtype=int), test["p_true"].to_numpy(dtype=float))
    decisions, dmetrics = decision_metrics(pred, test["bldg_id"].to_numpy(),
        test["p_true"].to_numpy(dtype=float), k=int(e_cfg["selection_k"]))
    prior_rate = prior_rep.prior_predictive["Y_obs"].values.reshape(-1, len(train)).mean(axis=1)
    ppc_rate = ppc.posterior_predictive["Y_obs"].values.reshape(-1, len(train)).mean(axis=1)
    record: dict[str, Any] = {
        "case_id": case["case_id"], "repeat": repeat, "gamma_true": case["gamma_true"],
        "simulation_seed": case["simulation_seed"], "model": case["model"],
        "prior_id": case["prior_id"], "original_source": case["source"],
        "sampler_seed": seed, "chains": int(mcmc["chains"]),
        "draws_per_chain": int(mcmc["draws"]), "tune_per_chain": int(mcmc["tune"]),
        "rhat_max": float(summ["r_hat"].max()),
        "ess_bulk_min": float(summ["ess_bulk"].min()),
        "ess_tail_min": float(summ["ess_tail"].min()),
        "divergences": int(idata.sample_stats["diverging"].sum().item()),
        "prior_predictive_rate_p05": float(np.percentile(prior_rate, 5)),
        "prior_predictive_rate_p50": float(np.percentile(prior_rate, 50)),
        "prior_predictive_rate_p95": float(np.percentile(prior_rate, 95)),
        "posterior_predictive_rate_p05": float(np.percentile(ppc_rate, 5)),
        "posterior_predictive_rate_p50": float(np.percentile(ppc_rate, 50)),
        "posterior_predictive_rate_p95": float(np.percentile(ppc_rate, 95)),
        "observed_train_rate": float(y.mean()),
        "alpha_posterior_mean": float(summ.loc["alpha", "mean"]),
        "gamma_posterior_mean": float(summ.loc["gamma_T", "mean"]) if extended else None,
        "fingerprints": hashes,
        **scores, **dmetrics,
    }
    fit_dir.mkdir(parents=True, exist_ok=True)
    # Persist only posterior inference, not the large 4,000x4,000 PPC array.
    az.to_netcdf(idata, idata_path)
    summ.to_csv(fit_dir / "posterior_summary.csv")
    decisions.to_parquet(decisions_path, index=False)
    import matplotlib.pyplot as plt
    az.plot_trace(idata, var_names=names, compact=True)
    plt.suptitle(f"M1-E: {case['case_id']}, repeat={repeat}")
    plt.savefig(fit_dir / "trace.png", dpi=125, bbox_inches="tight")
    plt.close("all")
    record_path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    LOG.info("Completed %s repeat=%d: regret=%.6f, Rhat=%.3f, tail ESS=%.0f, divergences=%d",
        case["case_id"], repeat, record["expected_regret"], record["rhat_max"],
        record["ess_tail_min"], record["divergences"])
    return record, decisions


def make_plan(args: argparse.Namespace, cases: list[dict[str, Any]]) -> list[tuple[dict[str, Any], int]]:
    plan = [(c, rep) for c in cases for rep in range(1, int(c["repeats"]) + 1)]
    if args.smoke:
        selected = {"m1c_b1_gneg0p8_s101", "boundary_p0_gneg0p4_s101",
                    "boundary_p1_gneg0p4_s101", "boundary_p2_gneg0p4_s101"}
        plan = [(c, rep) for c, rep in plan if c["case_id"] in selected and rep == 1]
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true", help="Read existing traces and validate original references; no fitting")
    parser.add_argument("--trace-plots", action="store_true", help="Also plot original traces during --audit-only")
    parser.add_argument("--dry-run", action="store_true", help="Check source provenance and list fits; no inference")
    parser.add_argument("--smoke", action="store_true", help="Four targeted fits; shared cache with full run")
    parser.add_argument("--force", action="store_true", help="Rerun selected confirmation fits after inspecting caches")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.audit_only and (args.smoke or args.dry_run or args.force):
        parser.error("--audit-only must run alone (optionally with --trace-plots)")
    if args.trace_plots and not args.audit_only:
        parser.error("--trace-plots requires --audit-only")

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    d2_cfg = json.loads(M1D2_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    c_cfg, scenarios, source_hashes = read_inputs()
    cases = validate_plan(cfg, c_cfg, d2_cfg)
    audit_gate(source_hashes)
    audit_hash = sha256(M1D2_AUDIT_DIR / "audit_manifest.json")
    originals: dict[str, tuple[dict[str, Any], pd.DataFrame, Path]] = {}
    for case in cases:
        originals[case["case_id"]] = reference_case(case, source_hashes, d2_cfg, audit_hash)
    plan = make_plan(args, cases)
    LOG.info("M1-E %s: %d registered original cases, %d confirmation fits (full 9, smoke 4)",
             "SMOKE" if args.smoke else "FULL", len(cases), len(plan))
    if args.dry_run:
        for case, repeat in plan:
            LOG.info("DRY RUN case=%s repeat=%s source=%s prior=%s", case["case_id"], repeat, case["source"], case["prior_id"])
        return
    if args.audit_only:
        rows = []
        for case in cases:
            record, _, src = originals[case["case_id"]]
            result = original_diagnostics(case, record, src, cfg["diagnostics"], args.trace_plots)
            rows.append(result)
            LOG.info("AUDIT %s: Rhat %.2f, bulk ESS %.0f, tail ESS %.0f, flags=%s",
                case["case_id"], result["rhat_max"], result["ess_bulk_min"],
                result["ess_tail_min"], result["warnings"] or "none")
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT / "reference_audit.csv", index=False)
        (OUT / "reference_audit_manifest.json").write_text(json.dumps({
            "experiment": "M1-E", "stage": "original_trace_audit",
            "outcome_evidence": "synthetic",
            "code_commit": git_head(), "source_m1c_fingerprints": source_hashes,
            "config_sha256": sha256(CONFIG_PATH),
            "audit_sha256": sha256(OUT / "reference_audit.csv"),
            "trace_plots_generated": bool(args.trace_plots),
        }, indent=2), encoding="utf-8")
        LOG.info("M1-E REFERENCE AUDIT COMPLETE: %s", OUT / "reference_audit.csv")
        return

    config_hash = sha256(CONFIG_PATH)
    results = []
    for case, repeat in plan:
        gamma, seed = float(case["gamma_true"]), int(case["simulation_seed"])
        group = scenarios.loc[
            np.isclose(scenarios["gamma_true"], gamma, rtol=0, atol=1e-12)
            & scenarios["simulation_seed"].eq(seed)
        ].reset_index(drop=True)
        if len(group) != int(c_cfg["sample_size"]):
            raise ValueError(f"Missing synthetic scenario for {case['case_id']}")
        ref_record, ref_decisions, src = originals[case["case_id"]]
        dest = OUT / "fits" / case["case_id"] / f"repeat_{repeat}"
        hashes = fingerprint(case, repeat, src, source_hashes, config_hash)
        record, decisions = fit_one(case, repeat, group, c_cfg, d2_cfg, cfg, hashes, dest, args.force)
        delta = decision_delta(ref_decisions, decisions, k=int(cfg["selection_k"]))
        results.append({k: v for k, v in record.items() if k != "fingerprints"} | {
            "original_rhat_max": ref_record["rhat_max"],
            "original_ess_bulk_min": ref_record["ess_bulk_min"],
            "original_ess_tail_min": ref_record["ess_tail_min"],
            "original_divergences": ref_record["divergences"],
            "original_expected_regret": ref_record["expected_regret"],
            "original_overlap": delta["overlap"],
            "original_max_abs_inclusion_shift": delta["max_abs_inclusion_shift"],
            "original_selected_only_ids": delta["selected_only_a_ids"],
            "confirmation_selected_only_ids": delta["selected_only_b_ids"],
            "diagnostic_flags": ";".join(diagnostic_flags(record, cfg["diagnostics"])),
        })
    stage = "smoke" if args.smoke else "full"
    data_path = OUT / f"{stage}_results.csv"
    pd.DataFrame(results).to_csv(data_path, index=False)
    version_names = ("pymc", "arviz", "numpy", "pandas", "scipy")
    manifest = {
        "experiment": "M1-E", "stage": stage, "outcome_evidence": "synthetic",
        "status": "draft", "code_commit": git_head(),
        "reproduction_command": f"python -m rhb.pipelines.run_m1e{' --smoke' if args.smoke else ''}",
        "result_rows": len(results), "config_sha256": config_hash,
        "source_m1c_fingerprints": source_hashes,
        "m1d2_config_sha256": sha256(M1D2_CONFIG_PATH),
        "m1d2_audit_manifest_sha256": audit_hash,
        "python": sys.version.split()[0],
        "dependencies": {name: metadata.version(name) for name in version_names},
        "results": {"path": data_path.relative_to(OUT).as_posix(), "sha256": sha256(data_path)},
        "note": "Four-chain computation-only refits. Synthetic outcomes. No empirical risk validation.",
    }
    (OUT / f"{stage}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOG.info("M1-E %s COMPLETE: %d confirmation fits, %s", stage.upper(), len(results), data_path)


if __name__ == "__main__":
    main()  # Windows multiprocessing requires the __main__ guard
