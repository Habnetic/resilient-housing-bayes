"""Summarize M1-E diagnostics and paired selection stability without rerunning PyMC.

  python -m rhb.pipelines.summarize_m1e --smoke
  python -m rhb.pipelines.summarize_m1e
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rhb.pipelines.m1e_checks import decision_delta, diagnostic_flags, validate_plan
from rhb.pipelines.run_m1e import CONFIG_PATH, OUT, M1D2_CONFIG_PATH, read_inputs, sha256


def make_boundary_comparisons(out: Path, config: dict[str, Any], repeats: int) -> pd.DataFrame:
    """Compare P0/P1/P2 within each MCMC replicate and each prior across replicates."""
    prefix = "boundary_"
    ids = {"P0": "boundary_p0_gneg0p4_s101",
           "P1": "boundary_p1_gneg0p4_s101",
           "P2": "boundary_p2_gneg0p4_s101"}
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for prior_id, case_id in ids.items():
        for repeat in range(1, repeats + 1):
            path = out / "fits" / case_id / f"repeat_{repeat}" / "test_decisions.parquet"
            if not path.is_file():
                raise FileNotFoundError(f"Missing M1-E boundary refit decisions: {path}")
            frames[(prior_id, repeat)] = pd.read_parquet(path)
    rows: list[dict[str, Any]] = []
    k = int(config["selection_k"])
    for repeat in range(1, repeats + 1):
        for prior in ("P1", "P2"):
            rows.append({"comparison": "between_priors", "a": "P0", "b": prior,
                         "repeat_a": repeat, "repeat_b": repeat,
                         **decision_delta(frames[("P0", repeat)], frames[(prior, repeat)], k)})
    if repeats == 2:
        for prior in ("P0", "P1", "P2"):
            rows.append({"comparison": "same_prior_independent_mcmc", "a": prior, "b": prior,
                         "repeat_a": 1, "repeat_b": 2,
                         **decision_delta(frames[(prior, 1)], frames[(prior, 2)], k)})
    return pd.DataFrame(rows)


def summarize_run(stage: str) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    m1c_config, _, _ = read_inputs()
    m1d2_config = json.loads(M1D2_CONFIG_PATH.read_text(encoding="utf-8-sig"))
    cases = validate_plan(config, m1c_config, m1d2_config)
    path = OUT / f"{stage}_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run `python -m rhb.pipelines.run_m1e{' --smoke' if stage == 'smoke' else ''}` first")
    df = pd.read_csv(path)
    keys = ["case_id", "repeat"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate M1-E confirmation fits")
    expected = {(c["case_id"], repeat) for c in cases for repeat in range(1, int(c["repeats"]) + 1)}
    if stage == "smoke":
        allowed = {"m1c_b1_gneg0p8_s101", "boundary_p0_gneg0p4_s101",
                   "boundary_p1_gneg0p4_s101", "boundary_p2_gneg0p4_s101"}
        expected = {(case_id, 1) for case_id, _ in expected if case_id in allowed}
    found = set(zip(df["case_id"].astype(str), df["repeat"].astype(int)))
    if found != expected:
        raise ValueError(f"Expected {len(expected)} specified fits, found {len(found)}; missing={expected - found}")
    flags = []
    for _, row in df.iterrows():
        flags.append(";".join(diagnostic_flags(row.to_dict(), config["diagnostics"])))
    df["computed_diagnostic_flags"] = flags
    boundary = make_boundary_comparisons(OUT, config, repeats=1 if stage == "smoke" else 2)
    output = OUT / ("smoke_summary" if stage == "smoke" else "full_summary")
    output.mkdir(parents=True, exist_ok=True)
    reviewed_path = output / "fit_diagnostics.csv"
    boundary_path = output / "boundary_comparisons.csv"
    flags_path = output / "diagnostic_flags.csv"
    df.to_csv(reviewed_path, index=False)
    boundary.to_csv(boundary_path, index=False)
    df.loc[df["computed_diagnostic_flags"].ne("")].to_csv(flags_path, index=False)
    n_fail = int(df["computed_diagnostic_flags"].ne("").sum())
    n_swaps = int((boundary.loc[boundary["comparison"].eq("same_prior_independent_mcmc"), "replaced_assets"] > 0).sum())
    report = (
        "# M1-E | Computational confirmation (automated summary)\n\n"
        f"**Stage:** {stage}; **synthetic outcome evidence only**.\n\n"
        f"**Fit count:** {len(df)}; diagnostic warnings: {n_fail}.\n\n"
        "## Refitted parameters and diagnostics\n\n"
        "```text\n" + df[["case_id", "repeat", "rhat_max", "ess_bulk_min", "ess_tail_min", "divergences",
                             "expected_regret", "original_expected_regret", "original_overlap",
                             "computed_diagnostic_flags"]].to_string(index=False) + "\n```\n\n"
        "## Decision boundary comparisons\n\n"
        "```text\n" + boundary[["comparison", "a", "b", "repeat_a", "repeat_b", "overlap",
                                      "replaced_assets", "max_abs_inclusion_shift",
                                      "expected_events_change_b_minus_a"]].to_string(index=False) + "\n```\n\n"
        f"Independent-repeat within-prior selection disagreements: {n_swaps}.\n\n"
        "## Required manual inspection\n\n"
        "Review `trace.png` for each refit, original flags and boundary member IDs. "
        "Do not declare M1-E complete solely from automated thresholds; decide whether P0/P1/P2 "
        "swaps reproduce in both independent MCMC repeats.\n\n"
        "M1-E provides no observed flood data, geographical validation or physical effect estimate.\n"
    )
    (output / "summary_report.md").write_text(report, encoding="utf-8")
    hashes = {p.name: sha256(p) for p in (reviewed_path, boundary_path, flags_path, output / "summary_report.md")}
    (output / "summary_manifest.json").write_text(json.dumps({
        "experiment": "M1-E", "stage": stage, "outcome_evidence": "synthetic", "status": "draft",
        "inputs": {"results_sha256": sha256(path), "config_sha256": sha256(CONFIG_PATH)},
        "artifacts_sha256": hashes, "diagnostic_warnings": n_fail,
        "manual_trace_review": "pending", "boundary_review": "pending",
    }, indent=2), encoding="utf-8")
    return {"stage": stage, "fits": len(df), "diagnostic_warnings": n_fail,
            "boundary_comparison_rows": len(boundary),
            "within_prior_selection_disagreements": n_swaps,
            "output": output}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    r = summarize_run("smoke" if args.smoke else "full")
    print("M1-E SUMMARY:", r)
    print("Inspect fit_diagnostics.csv, boundary_comparisons.csv and trace.png files; manual review remains required.")


if __name__ == "__main__":
    main()
