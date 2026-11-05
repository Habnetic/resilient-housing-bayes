"""Reproduce M1-A and M1-B pilot decision audits from persisted InferenceData.

Run from the repository root:
    python -m rhb.pipelines.audit_m1_pilots

This module does not refit models and does not change Phase 3 artifacts.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

from rhb.decision.phase3_decision_metrics import compute_topk_membership

ROOT = Path(__file__).resolve().parents[3]
PHASE3 = ROOT / "outputs/phase3/RTM/phase3_features_scaled.parquet"
A_DIR = ROOT / "outputs/phase4/RTM/m1a_pilot"
B_DIR = ROOT / "outputs/phase4/RTM/m1b_pilot"
TOPO_SCALE_METRES = 0.5
BORDERLINE_LOWER = 0.2
BORDERLINE_UPPER = 0.8


def topk_indices(p_mean: np.ndarray, k: int) -> np.ndarray:
    """Apply the same mean-probability/argpartition selection used in the pilots."""
    if p_mean.ndim != 1 or not 0 < k <= len(p_mean):
        raise ValueError("p_mean must be 1-D, and 1 <= k <= n_assets")
    if not np.isfinite(p_mean).all():
        raise ValueError("Nonfinite posterior mean probability")
    return np.argpartition(p_mean, -k)[-k:]


def posterior_decisions(
    df: pd.DataFrame, idata_path: Path, *, include_topography: bool, k: int
) -> pd.DataFrame:
    """Compute asset-level posterior-mean selection and draw-wise top-k inclusion."""
    if df.empty or not df["bldg_id"].is_unique:
        raise ValueError("Expected nonempty data with unique bldg_id")
    idata = az.from_netcdf(idata_path)
    posterior = idata.posterior
    e = df["E_hat_v0"].to_numpy(dtype=float)
    h = df["H_pluvial_v1_logrel"].to_numpy(dtype=float)

    def param(name: str) -> np.ndarray:
        return posterior[name].values.reshape(-1, 1)

    logits = param("alpha") + param("beta_E") * e[None, :] + param("beta_H") * h[None, :]
    if include_topography:
        t = np.arcsinh(df["T50"].to_numpy(dtype=float) / TOPO_SCALE_METRES)
        logits = logits + param("gamma_T") * t[None, :]

    probabilities = expit(logits)
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite posterior predicted probabilities")

    inclusion, rank_std = compute_topk_membership(probabilities, k=k)
    if not np.all((inclusion >= 0) & (inclusion <= 1)):
        raise AssertionError("Posterior inclusion probabilities outside [0, 1]")
    if not np.isclose(inclusion.sum(), k, atol=1e-6):
        raise AssertionError("Posterior inclusion probabilities do not sum to k")

    p_mean = probabilities.mean(axis=0)
    selected = np.zeros(len(df), dtype=bool)
    selected[topk_indices(p_mean, k)] = True

    out = pd.DataFrame({
        "bldg_id": df["bldg_id"].to_numpy(),
        "p_mean": p_mean,
        "p_topk": inclusion,
        "rank_std": rank_std,
        "selected_by_p_mean": selected,
    })
    if "p_true_m1b" in df.columns:
        out["p_true_m1b"] = df["p_true_m1b"].to_numpy(dtype=float)
    return out


def _borderline_count(metrics: pd.DataFrame) -> int:
    prob = metrics["p_topk"]
    return int(((prob > BORDERLINE_LOWER) & (prob < BORDERLINE_UPPER)).sum())


def compare_decisions(
    base: pd.DataFrame,
    extended: pd.DataFrame,
    *,
    k: int,
    oracle_probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize agreement and, when truth is known, expected selection regret."""
    if not base["bldg_id"].equals(extended["bldg_id"]):
        raise ValueError("Model outputs have mismatched asset order")
    a0 = set(base.loc[base["selected_by_p_mean"], "bldg_id"].tolist())
    a1 = set(extended.loc[extended["selected_by_p_mean"], "bldg_id"].tolist())
    if len(a0) != k or len(a1) != k:
        raise AssertionError("Each selection must contain exactly k unique buildings")

    difference = np.abs(extended["p_topk"].to_numpy() - base["p_topk"].to_numpy())
    summary: dict[str, Any] = {
        "assets": len(base),
        "k": k,
        "overlap": len(a0 & a1),
        "overlap_fraction": len(a0 & a1) / k,
        "exits": sorted(a0 - a1),
        "entries": sorted(a1 - a0),
        "borderline_base": _borderline_count(base),
        "borderline_extended": _borderline_count(extended),
        "mean_absolute_inclusion_probability_change": float(difference.mean()),
        "maximum_inclusion_probability_change": float(difference.max()),
        "decision_rule": "select k largest posterior mean event probabilities",
        "inclusion_rule": "fraction of posterior draws in per-draw top-k",
        "borderline_rule": "strict 0.2 < p_topk < 0.8",
    }
    if oracle_probabilities is not None:
        truth = np.asarray(oracle_probabilities, dtype=float)
        if len(truth) != len(base):
            raise ValueError("Oracle probabilities must match decision population")
        oracle_idx = topk_indices(truth, k)
        oracle = set(base.iloc[oracle_idx]["bldg_id"].tolist())
        oracle_total = float(truth[oracle_idx].sum())
        for name, metrics, ids in [("base", base, a0), ("extended", extended, a1)]:
            selected = metrics["selected_by_p_mean"].to_numpy(dtype=bool)
            expected = float(truth[selected].sum())
            summary[f"{name}_oracle_overlap"] = len(ids & oracle)
            summary[f"{name}_expected_selected_events"] = expected
            summary[f"{name}_expected_regret"] = max(0.0, oracle_total - expected)
        summary["oracle_expected_selected_events"] = oracle_total
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(args: list[str]) -> str | None:
    command = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return command.stdout.strip() if command.returncode == 0 else None


def provenance(inputs: list[Path]) -> dict[str, Any]:
    versions = {}
    for package in ["arviz", "numpy", "pandas", "pymc", "scipy"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "python": sys.version.split()[0],
        "versions": versions,
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_status_short": _git(["status", "--short"]),
        "inputs_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in inputs},
        "topography_transformation": "asinh(T50_m / 0.5 m)",
        "note": "Inputs may be ignored by Git: hashes verify identity, not availability.",
    }


def run_a() -> None:
    sample_path = A_DIR / "pilot_sample.csv"
    columns = ["bldg_id", "E_hat_v0", "H_pluvial_v1_logrel"]
    source = pd.read_parquet(PHASE3, columns=columns)
    sample = pd.read_csv(sample_path)
    df = sample.merge(source, on="bldg_id", how="left", validate="one_to_one", sort=False)
    if len(df) != 5000 or df[["E_hat_v0", "H_pluvial_v1_logrel", "T50"]].isna().any().any():
        raise ValueError("M1-A pilot population mismatch or missing predictors")

    k = 50
    base_path, extended_path = A_DIR / "A0_idata.nc", A_DIR / "A1_idata.nc"
    base = posterior_decisions(df, base_path, include_topography=False, k=k)
    extended = posterior_decisions(df, extended_path, include_topography=True, k=k)
    base.to_csv(A_DIR / "A0_decisions.csv", index=False)
    extended.to_csv(A_DIR / "A1_decisions.csv", index=False)
    summary = compare_decisions(base, extended, k=k)
    summary["provenance"] = provenance([PHASE3, sample_path, base_path, extended_path])
    (A_DIR / "decision_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"M1-A: overlap={summary['overlap']}/{k}; borderline={summary['borderline_base']} -> {summary['borderline_extended']}")


def run_b() -> None:
    dataset_path = B_DIR / "m1b_dataset.parquet"
    df = pd.read_parquet(dataset_path)
    df = df.loc[df["split"].eq("test")].reset_index(drop=True)
    if len(df) != 1000:
        raise ValueError("M1-B test population mismatch")

    k = 10
    base_path, extended_path = B_DIR / "B0_idata.nc", B_DIR / "B1_idata.nc"
    base = posterior_decisions(df, base_path, include_topography=False, k=k)
    extended = posterior_decisions(df, extended_path, include_topography=True, k=k)
    base.to_csv(B_DIR / "B0_decisions.csv", index=False)
    extended.to_csv(B_DIR / "B1_decisions.csv", index=False)
    summary = compare_decisions(base, extended, k=k, oracle_probabilities=df["p_true_m1b"].to_numpy())
    summary["hazard_std_test"] = float(df["H_pluvial_v1_logrel"].std(ddof=0))
    summary["provenance"] = provenance([dataset_path, base_path, extended_path])
    (B_DIR / "decision_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"M1-B: overlap={summary['overlap']}/{k}; oracle overlap={summary['base_oracle_overlap']} -> {summary['extended_oracle_overlap']}")
    print(f"M1-B: expected regret={summary['base_expected_regret']:.6f} -> {summary['extended_expected_regret']:.6f}")


def main() -> None:
    run_a()
    run_b()
    print("M1 PILOT DECISION AUDITS: COMPLETE")


if __name__ == "__main__":
    main()
