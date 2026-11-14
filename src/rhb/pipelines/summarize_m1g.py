"""M1-G scientific QA, matched comparisons, and draft publication manifests.

    python -m rhb.pipelines.summarize_m1g --city HAM --smoke
    python -m rhb.pipelines.summarize_m1g --city HAM
    python -m rhb.pipelines.summarize_m1g --city DON

The draft manifests are *not published*. Review MCMC flags and spatial sampling
before issuing a tag or exposing the results in the Operations Center.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rhb.experiments.m1f_population_sampling import sha256
from rhb.experiments.m1g_cross_city_sampling import ROOT, load_config, output_dir
from rhb.pipelines.summarize_m1f import (
    check_result_tables, diagnostics, summarize_pairs, with_predictive_contrasts,
)

LOGGER = logging.getLogger("rhb.m1g.summary")


def read_coverage(value: object) -> bool | float:
    """Interpret CSV boolean fields without converting the string 'False' to True."""
    if pd.isna(value):
        return np.nan
    if str(value).strip().lower() == "true":
        return True
    if str(value).strip().lower() == "false":
        return False
    raise ValueError(f"Unknown gamma coverage flag: {value!r}")


def verify_decision_files(city: str, fits: pd.DataFrame, cfg: dict[str, Any]) -> None:
    """Verify actual held-out candidate sets and sum of posterior inclusion weights."""
    from rhb.pipelines.run_m1c import label
    out = output_dir(city)
    for gamma, seed in ((g, s) for g in cfg["gamma_true_values"] for s in cfg["simulation_seeds"]):
        files = [out / "fits" / label(gamma, seed) / model / "test_decisions.parquet"
                 for model in ("B0", "B1")]
        if any(not p.is_file() for p in files):
            raise FileNotFoundError(f"Missing M1-G decision file: {files}")
        a, b = [pd.read_parquet(p) for p in files]
        if not a["bldg_id"].equals(b["bldg_id"]):
            raise ValueError(f"{city}: B0/B1 candidate IDs do not match")
        if len(a) != cfg["test_size"]:
            raise ValueError(f"{city}: wrong held-out population")
        for df in (a, b):
            if not df["bldg_id"].is_unique or not np.isfinite(df["p_topk"]).all():
                raise ValueError("Missing/nonfinite/duplicated decision results")
            if abs(float(df["p_topk"].sum()) - 10.0) > 1e-6:
                raise ValueError(f"{city}: posterior inclusion sum must equal k=10")
            if int(df["selected_by_p_mean"].sum()) != 10:
                raise ValueError(f"{city}: selected set must contain exactly k=10")


def summarize_city(city: str, smoke: bool) -> dict[str, Any]:
    cfg = load_config()
    stage = "smoke" if smoke else "full"
    out = output_dir(city)
    fit_path = out / f"{stage}_fit_results.csv"
    comp_path = out / f"{stage}_comparisons.csv"
    run_manifest = out / f"{stage}_manifest.json"
    for p in (fit_path, comp_path, run_manifest):
        if not p.is_file():
            raise FileNotFoundError(f"{city}: missing {stage} inference artifact: {p}")
    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    if (manifest.get("experiment"), manifest.get("city"), manifest.get("stage")) != ("M1-G", city, stage):
        raise ValueError("Incorrect M1-G run manifest")
    if manifest.get("output_hashes") != {"fit_results": sha256(fit_path), "comparisons": sha256(comp_path)}:
        raise ValueError("M1-G fit or comparison results changed after inference")
    fits = pd.read_csv(fit_path)
    comparisons = pd.read_csv(comp_path)
    fits["gamma_covered"] = fits["gamma_covered"].map(read_coverage)
    check_result_tables(fits, comparisons, populations=1, seeds=1 if smoke else 2)
    if not fits["city"].eq(city).all() or not comparisons["city"].eq(city).all():
        raise ValueError("Mixed cities in M1-G result table")
    if not smoke:
        verify_decision_files(city, fits, cfg)
    matched = with_predictive_contrasts(comparisons, fits)
    by_gamma = summarize_pairs(matched, ["gamma_true"])
    flags = diagnostics(fits, cfg["diagnostic_limits"])
    # Does not turn a bad fit into a scientific success.
    qa = {
        "complete": len(comparisons) == (2 if smoke else 4),
        "all_compared": True,
        "flagged_fits": int(len(flags)),
        "rhat_max": float(fits["rhat_max"].max()),
        "ess_bulk_min": float(fits["ess_bulk_min"].min()),
        "ess_tail_min": float(fits["ess_tail_min"].min()),
        "divergences_total": int(fits["divergences"].sum()),
        "scientific_review_required": True,
    }
    dest = out / f"{stage}_summary"
    dest.mkdir(parents=True, exist_ok=True)
    paths = {
        "matched_cases": dest / "matched_cases.csv",
        "summary_by_gamma": dest / "summary_by_gamma.csv",
        "diagnostic_flags": dest / "diagnostic_flags.csv",
    }
    matched.to_csv(paths["matched_cases"], index=False)
    by_gamma.to_csv(paths["summary_by_gamma"], index=False)
    flags.to_csv(paths["diagnostic_flags"], index=False)
    report = dest / "summary_report.md"
    report.write_text("\n".join([
        f"# Habnetic | M1-G {city} {stage} scientific report (draft)", "",
        f"Matched synthetic cases: **{len(comparisons)}**; MCMC fits: **{len(fits)}**; numeric flags: **{len(flags)}**.", "",
        "**Not an empirical prediction or a trained-model geographic transfer.**",
        "B0 and B1 are independently refitted within the city on shared synthetic outcomes.", "The Phase 3 posterior artifacts are contextual baselines, not paired M1-G fits.", "",
        "## Matched synthetic contrasts", "", "```text", by_gamma.to_string(index=False), "```", "",
        "## Convergence and limitations", "",
        f"R-hat max: {qa['rhat_max']:.4f}; bulk ESS min: {qa['ess_bulk_min']:.1f}; ",
        f"tail ESS min: {qa['ess_tail_min']:.1f}; total divergences: {qa['divergences_total']}.", "",
        "Sampler: 2 chains, 500 tune and 500 draws; diagnostics must be reviewed before scientific approval.",
        "Prior predictive and posterior predictive event-rate checks are recorded in each fit_record.json.",
        "Only one random asset sample per city, 2 outcome seeds. No spatially blocked holdout.",
        "No observed outcomes or independent terrain-elevation survey accuracy checks.",
        "Gamma prior sigma 0.5 follows M1-C; a wider alternative prior was studied in M1-D,",
        "but not refitted in these cities. Sensitivity is therefore an open limitation.", "",
        "## Review gate", "",
        "Review flags, PPC, posterior traces, and city-specific data distributions.",
        "Do not tag/publish a 'validated' claim solely because this file exists.", "",
    ]) + "\n", encoding="utf-8")
    paths["summary_report"] = report
    summary_manifest = {
        "experiment": "M1-G", "city": city, "stage": stage,
        "cases": int(len(comparisons)), "fits": int(len(fits)),
        "qa": qa,
        "inputs_sha256": {"run_manifest": sha256(run_manifest),
                          "fit_results": sha256(fit_path), "comparisons": sha256(comp_path)},
        "output_sha256": {name: sha256(path) for name, path in paths.items()},
    }
    manifest_path = dest / "summary_manifest.json"
    manifest_path.write_text(json.dumps(summary_manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # Draft adapter for Experiment Artifact Contract v0.1, not a public bundle.
    draft = {
        "schema_version": "0.1.0",
        "experiment_id": f"phase4-m1g-{city.lower()}-{stage}-v0.1",
        "phase": "phase4", "milestone": "M1-G", "domain": "urban-flood-risk", "study_area": city,
        "status": "draft", "analysis_kind": "posterior_inference", "outcome_evidence": "synthetic",
        "code": {"repository": "Habnetic/resilient-housing-bayes", "commit": manifest.get("git_head")},
        "baseline": {"experiment_id": "phase4-m1c-rerun-B0", "note": "Matched B0 refit in same M1-G synthetic case, not precomputed Phase 3 baseline"},
        "inputs": [
            {"name": key, "sha256": val} for key, val in manifest["input_code_fingerprints"].items()
            if "sha256" in key
        ],
        "configuration": {"path": None, "seeds": {"outcome": cfg["simulation_seeds"]}, "environment_path": None},
        "reproduction": {"command": f"python -m rhb.pipelines.run_m1g --city {city} && python -m rhb.pipelines.summarize_m1g --city {city}"},
        "artifacts": {
            name: {"path": path.name, "media_type": "text/csv" if path.suffix == ".csv" else "text/markdown", "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "checks": {"artifact_integrity": "pass", "inference_diagnostics": "flagged" if len(flags) else "numerical_thresholds_pass",
                   "prior_predictive": "recorded_not_reviewed", "posterior_predictive": "recorded_not_reviewed",
                   "decision_comparison": "complete_not_scientifically_reviewed"},
        "limitations": ["Synthetic Bernoulli outcomes, not observed flood impacts", "Two chains and one non-spatial random asset sample per city",
                        "Independent within-city refits, not model coefficient transfer", "No cross-city gamma prior sensitivity rerun"],
    }
    (dest / "experiment_manifest_DRAFT.json").write_text(json.dumps(draft, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    LOGGER.info("%s %s summary complete: %d cases, %d flags (scientific review still required)",
                city, stage, len(comparisons), len(flags))
    return {"city": city, **qa, "summary_path": str(dest)}


def cross_city_summary() -> None:
    """Concatenate checked *computational* summaries; no statistical pooling claim."""
    rows = []
    for city in ("HAM", "DON"):
        out = output_dir(city) / "full_summary"
        manifest = out / "summary_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing completed {city} summary: {manifest}")
        record = json.loads(manifest.read_text(encoding="utf-8"))
        table = out / "summary_by_gamma.csv"
        if sha256(table) != record["output_sha256"]["summary_by_gamma"]:
            raise ValueError("City summary was modified")
        frame = pd.read_csv(table)
        if len(frame) != 2 or set(frame["gamma_true"]) != {0., -0.8}:
            raise ValueError("Not exactly two fixed gamma scenarios")
        frame.insert(0, "city", city)
        rows.append(frame)
    outdir = ROOT / "outputs/phase4/m1g_cross_city_summary"
    outdir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(rows, ignore_index=True)
    output = outdir / "summary_by_city_gamma.csv"
    combined.to_csv(output, index=False)
    (outdir / "summary_manifest.json").write_text(json.dumps({
        "experiment": "M1-G", "status": "draft", "outcome_evidence": "synthetic",
        "cities": ["HAM", "DON"], "cases_per_city": 4, "models_per_city": 8,
        "summary_sha256": sha256(output),
        "limitation": "Synthetic within-city refits, not estimated cross-city transferability of learned coefficients",
    }, indent=2) + "\n", encoding="utf-8")
    print("\nM1-G CROSS-CITY NUMERICAL SUMMARY (DRAFT)")
    print(combined[["city", "gamma_true", "mean_paired_regret_improvement", "mean_heldout_elpd_delta_B1_minus_B0", "B1_wins", "B0_wins"]].to_string(index=False))
    print("\nOutput:", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("HAM", "DON", "all"), required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.city == "all":
        if args.smoke:
            raise ValueError("No pooled smoke report; summarize individual cities")
        cross_city_summary()
    else:
        record = summarize_city(args.city, smoke=args.smoke)
        print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
