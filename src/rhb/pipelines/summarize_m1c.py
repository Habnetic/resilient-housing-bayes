"""Summarize M1-C paired synthetic experiments without rerunning PyMC.

Run from the resilient-housing-bayes repository root::

    python -m rhb.pipelines.summarize_m1c
    python -m rhb.pipelines.summarize_m1c --stage smoke

Reads fit_results.csv, scenario_comparisons.csv and manifest.json from
outputs/phase4/RTM/m1c/<stage>/. Writes only to <stage>/summary/.

The experiment uses a fixed population/split, so its five seeds measure
outcome-realization variability, not geographic or population generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
LOG = logging.getLogger("rhb.m1c.summary")
KEYS = ["gamma_true", "simulation_seed"]
FIT_REQUIRED = {
    "gamma_true", "simulation_seed", "model", "expected_regret",
    "gamma_posterior_mean", "gamma_hdi_low", "gamma_hdi_high",
    "divergences", "rhat_max", "ess_bulk_min", "ess_tail_min",
    "rmse_vs_true_p", "brier", "heldout_elpd",
}
COMPARE_REQUIRED = {
    "gamma_true", "simulation_seed", "k", "overlap",
    "oracle_overlap_B0", "oracle_overlap_B1", "borderline_B0",
    "borderline_B1", "expected_regret_B0", "expected_regret_B1",
    "mean_abs_inclusion_change", "max_inclusion_change",
}
R_HAT_WARNING = 1.01
ESS_WARNING = 400.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_unique(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    if frame.duplicated(columns).any():
        raise ValueError(f"Duplicate {label} rows for keys {columns}")


def _numeric(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    for col in columns:
        if col not in frame.columns:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if frame[col].isna().any():
            raise ValueError(f"Missing numeric values in {label}.{col}")


def _read_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fit_path = input_dir / "fit_results.csv"
    comparison_path = input_dir / "scenario_comparisons.csv"
    manifest_path = input_dir / "manifest.json"
    for path in (fit_path, comparison_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing M1-C artifact: {path}")
    fits = pd.read_csv(fit_path)
    comparisons = pd.read_csv(comparison_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    missing_fit = FIT_REQUIRED - set(fits.columns)
    missing_cmp = COMPARE_REQUIRED - set(comparisons.columns)
    if missing_fit or missing_cmp:
        raise ValueError(
            f"Missing fit columns: {sorted(missing_fit)}; "
            f"comparison columns: {sorted(missing_cmp)}"
        )

    _numeric(
        fits,
        {"gamma_true", "simulation_seed", "expected_regret", "divergences",
         "rhat_max", "ess_bulk_min", "ess_tail_min", "rmse_vs_true_p",
         "brier", "heldout_elpd"},
        "fits",
    )
    _numeric(
        comparisons,
        {"gamma_true", "simulation_seed", "k", "overlap", "oracle_overlap_B0",
         "oracle_overlap_B1", "borderline_B0", "borderline_B1",
         "expected_regret_B0", "expected_regret_B1",
         "mean_abs_inclusion_change", "max_inclusion_change"},
        "comparisons",
    )
    _assert_unique(fits, KEYS + ["model"], "fit")
    _assert_unique(comparisons, KEYS, "comparison")

    if len(fits) != 2 * len(comparisons):
        raise ValueError("Expected two fits (B0/B1) per scenario")
    if manifest.get("model_fits") != len(fits):
        raise ValueError("Fit count does not match the original run manifest")
    if manifest.get("scenario_count") != len(comparisons):
        raise ValueError("Scenario count does not match the original run manifest")
    if set(fits["model"].dropna().unique()) != {"B0", "B1"}:
        raise ValueError("Only paired B0/B1 fits are supported")
    if fits[KEYS].isna().any().any() or comparisons[KEYS].isna().any().any():
        raise ValueError("Missing scenario identity keys")
    if fits["divergences"].lt(0).any():
        raise ValueError("Invalid negative divergence count")
    if fits["expected_regret"].lt(-1e-10).any():
        raise ValueError("Expected regret must be nonnegative")
    return fits, comparisons, manifest


def _diagnostic_flags(fits: pd.DataFrame) -> pd.DataFrame:
    flags = fits.copy()
    flags["flag_rhat"] = flags["rhat_max"] > R_HAT_WARNING
    flags["flag_bulk_ess"] = flags["ess_bulk_min"] < ESS_WARNING
    flags["flag_tail_ess"] = flags["ess_tail_min"] < ESS_WARNING
    flags["flag_divergences"] = flags["divergences"] > 0
    flags["has_diagnostic_flag"] = flags[
        ["flag_rhat", "flag_bulk_ess", "flag_tail_ess", "flag_divergences"]
    ].any(axis=1)
    columns = KEYS + [
        "model", "rhat_max", "ess_bulk_min", "ess_tail_min", "divergences",
        "flag_rhat", "flag_bulk_ess", "flag_tail_ess", "flag_divergences",
        "has_diagnostic_flag",
    ]
    return flags.loc[flags["has_diagnostic_flag"], columns].copy()


def build_paired_cases(fits: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    """Match models on gamma/seed and verify consistent decision metrics."""
    b0 = fits.loc[fits["model"] == "B0"].drop(columns=["model"])
    b1 = fits.loc[fits["model"] == "B1"].drop(columns=["model"])
    if len(b0) != len(b1) or len(b0) != len(comparisons):
        raise ValueError("Unpaired M1-C scenario records")
    paired = comparisons.merge(b0, on=KEYS, how="inner", validate="one_to_one", suffixes=("", "_fit_B0"))
    # Explicitly suffix all non-key B1 columns to keep model ownership clear.
    b1 = b1.rename(columns={col: f"{col}_fit_B1" for col in b1.columns if col not in KEYS})
    paired = paired.merge(b1, on=KEYS, how="inner", validate="one_to_one")
    if len(paired) != len(comparisons):
        raise ValueError("Fit/comparison scenario keys do not match")

    # Since original comparison and fit records both include expected regret,
    # the B0 fit field is automatically suffix-labelled by pandas due to collision.
    regret0 = paired["expected_regret"]
    regret1 = paired["expected_regret_fit_B1"]
    if not np.allclose(regret0, paired["expected_regret_B0"], atol=1e-8, rtol=0):
        raise ValueError("Inconsistent B0 regret between fit and decision comparison")
    if not np.allclose(regret1, paired["expected_regret_B1"], atol=1e-8, rtol=0):
        raise ValueError("Inconsistent B1 regret between fit and decision comparison")

    paired["regret_improvement"] = paired["expected_regret_B0"] - paired["expected_regret_B1"]
    paired["B1_better"] = paired["regret_improvement"] > 1e-9
    paired["B0_better"] = paired["regret_improvement"] < -1e-9
    paired["regret_tie"] = ~(paired["B1_better"] | paired["B0_better"])

    for column in ("gamma_posterior_mean_fit_B1", "gamma_hdi_low_fit_B1", "gamma_hdi_high_fit_B1"):
        paired[column] = pd.to_numeric(paired[column], errors="coerce")
        if paired[column].isna().any():
            raise ValueError(f"Missing posterior terrain parameter: {column}")
    paired["gamma_bias"] = paired["gamma_posterior_mean_fit_B1"] - paired["gamma_true"]
    paired["gamma_abs_error"] = paired["gamma_bias"].abs()
    paired["gamma_covered_95"] = (
        (paired["gamma_hdi_low_fit_B1"] <= paired["gamma_true"])
        & (paired["gamma_true"] <= paired["gamma_hdi_high_fit_B1"])
    )
    paired["elpd_gain_B1_minus_B0"] = paired["heldout_elpd_fit_B1"] - paired["heldout_elpd"]
    paired["brier_change_B1_minus_B0"] = paired["brier_fit_B1"] - paired["brier"]
    paired["rmse_change_B1_minus_B0"] = paired["rmse_vs_true_p_fit_B1"] - paired["rmse_vs_true_p"]
    return paired.sort_values(KEYS, ascending=[False, True]).reset_index(drop=True)


def summarize_by_gamma(paired: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """Aggregate *paired* outcome replicates, without treating models as independent."""
    rows: list[dict[str, Any]] = []
    for gamma, g in paired.groupby("gamma_true", sort=False):
        gf = fits.loc[np.isclose(fits["gamma_true"], gamma, atol=1e-12, rtol=0)]
        mean0 = float(g["expected_regret_B0"].mean())
        mean1 = float(g["expected_regret_B1"].mean())
        reduction = float(100 * (mean0 - mean1) / mean0) if mean0 > 1e-12 else float("nan")
        rows.append({
            "gamma_true": float(gamma),
            "replications": len(g),
            "mean_gamma_estimate_B1": float(g["gamma_posterior_mean_fit_B1"].mean()),
            "mean_gamma_bias_B1": float(g["gamma_bias"].mean()),
            "mean_gamma_absolute_error_B1": float(g["gamma_abs_error"].mean()),
            "gamma_95_hdi_covered_count": int(g["gamma_covered_95"].sum()),
            "mean_expected_regret_B0": mean0,
            "mean_expected_regret_B1": mean1,
            "mean_paired_regret_improvement": float(g["regret_improvement"].mean()),
            "regret_reduction_percent_of_mean": reduction,
            "B1_wins": int(g["B1_better"].sum()),
            "ties": int(g["regret_tie"].sum()),
            "B0_wins": int(g["B0_better"].sum()),
            "mean_model_topk_overlap": float(g["overlap"].mean()),
            "mean_oracle_overlap_B0": float(g["oracle_overlap_B0"].mean()),
            "mean_oracle_overlap_B1": float(g["oracle_overlap_B1"].mean()),
            "mean_borderline_B0": float(g["borderline_B0"].mean()),
            "mean_borderline_B1": float(g["borderline_B1"].mean()),
            "mean_max_inclusion_change": float(g["max_inclusion_change"].mean()),
            "mean_heldout_elpd_gain_B1_minus_B0": float(g["elpd_gain_B1_minus_B0"].mean()),
            "mean_brier_change_B1_minus_B0": float(g["brier_change_B1_minus_B0"].mean()),
            "mean_rmse_change_B1_minus_B0": float(g["rmse_change_B1_minus_B0"].mean()),
            "max_rhat": float(gf["rhat_max"].max()),
            "min_bulk_ess": float(gf["ess_bulk_min"].min()),
            "min_tail_ess": float(gf["ess_tail_min"].min()),
            "total_divergences": int(gf["divergences"].sum()),
            "flagged_fits": int((
                (gf["rhat_max"] > R_HAT_WARNING)
                | (gf["ess_bulk_min"] < ESS_WARNING)
                | (gf["ess_tail_min"] < ESS_WARNING)
                | (gf["divergences"] > 0)
            ).sum()),
        })
    return pd.DataFrame(rows).sort_values("gamma_true", ascending=False).reset_index(drop=True)


def _markdown_report(summary: pd.DataFrame, n_fits: int, stage: str, flagged: pd.DataFrame) -> str:
    lines = [
        "# M1-C | Automatically generated summary",
        "",
        f"**Stage:** {stage}. **Model fits:** {n_fits}. **Source:** M1-C persisted CSV outputs.",
        "",
        "| True γ | Replicates | Mean γ estimate | Coverage count | Mean regret B0 | Mean regret B1 | B1 wins/ties/B0 wins | Mean top-k overlap |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.gamma_true:g} | {row.replications} | {row.mean_gamma_estimate_B1:+.4f} "
            f"| {row.gamma_95_hdi_covered_count}/{row.replications} "
            f"| {row.mean_expected_regret_B0:.6f} "
            f"| {row.mean_expected_regret_B1:.6f} "
            f"| {row.B1_wins}/{row.ties}/{row.B0_wins} "
            f"| {row.mean_model_topk_overlap:.2f}/10 |"
        )
    lines.extend([
        "",
        f"**Diagnostic flags:** {len(flagged)} fits flagged using R-hat > {R_HAT_WARNING}, "
        f"bulk/tail ESS < {ESS_WARNING:g} or divergences > 0.",
        "",
        "**Scope:** Synthetic outcomes, one fixed population and split. Replicates differ by "
        "outcome seed only; scenarios share paired random uniforms. The percent regret reduction "
        "is the ratio of group means, not a mean of individual percentages. "
        "Coverage counts are not reliable estimates of nominal 95% coverage at n=5.",
        "",
        "**Next:** Inspect flagged traces, prior sensitivity and geographically blocked data splits.",
        "",
    ])
    return "\n".join(lines)


def summarize(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Validate input artifacts, aggregate paired cases, and persist reports."""
    fits, comparisons, manifest = _read_inputs(input_dir)
    paired = build_paired_cases(fits, comparisons)
    summary = summarize_by_gamma(paired, fits)
    flagged = _diagnostic_flags(fits)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / "paired_cases.csv", index=False)
    summary.to_csv(output_dir / "summary_by_gamma.csv", index=False)
    flagged.to_csv(output_dir / "diagnostic_flags.csv", index=False)
    (output_dir / "summary_report.md").write_text(
        _markdown_report(summary, len(fits), str(manifest.get("stage", "unknown")), flagged),
        encoding="utf-8",
    )
    source = {
        "experiment": "M1-C",
        "source_stage": manifest.get("stage"),
        "source_git_head": manifest.get("git_head"),
        "source_python": manifest.get("python"),
        "source_dependencies": manifest.get("dependencies"),
        "source_manifest_sha256": _sha256(input_dir / "manifest.json"),
        "source_fits_sha256": _sha256(input_dir / "fit_results.csv"),
        "source_comparisons_sha256": _sha256(input_dir / "scenario_comparisons.csv"),
        "summary_script_sha256": _sha256(Path(__file__)),
        "n_scenarios": len(paired),
        "n_fits": len(fits),
        "n_flagged_fits": len(flagged),
        "scope": "Synthetic fixed-population paired outcome replication, no flood validation",
    }
    (output_dir / "summary_manifest.json").write_text(
        json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), default="full")
    parser.add_argument("--input-dir", type=Path, default=None, help="Override source directory")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override summary output directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    input_dir = args.input_dir or ROOT / "outputs/phase4/RTM/m1c" / args.stage
    output_dir = args.output_dir or input_dir / "summary"
    result = summarize(input_dir, output_dir)
    LOG.info("M1-C summary COMPLETE: %s scenarios, %s fits, %s diagnostic flags", result["n_scenarios"], result["n_fits"], result["n_flagged_fits"])
    LOG.info("Written to %s", output_dir)


if __name__ == "__main__":
    main()
