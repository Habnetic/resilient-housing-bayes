"""Pure M1-E plan validation and selection comparison helpers."""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd


def validate_plan(config: dict[str, Any], m1c: dict[str, Any], m1d2: dict[str, Any]) -> list[dict[str, Any]]:
    """Ensure this is a small, explicit confirmation of *registered* original cases."""
    if (config.get("experiment"), config.get("city"), config.get("reference_stage")) != ("M1-E", "RTM", "full"):
        raise ValueError("M1-E accepts only full Rotterdam references")
    mcmc = config["mcmc"]
    for key in ("chains", "tune", "draws", "cores"):
        if not isinstance(mcmc[key], int) or isinstance(mcmc[key], bool) or mcmc[key] < 1:
            raise ValueError(f"MCMC {key} must be a positive integer")
    if mcmc["chains"] < 4 or mcmc["draws"] < 1000 or mcmc["tune"] < 1000:
        raise ValueError("M1-E confirmation needs >=4 chains, >=1000 draws and tune")
    if not 0.5 < float(mcmc["target_accept"]) < 1.0:
        raise ValueError("target_accept must be in (0.5, 1)")
    if mcmc["cores"] > mcmc["chains"]:
        raise ValueError("cores must not exceed chains")
    k = max(1, int(round(int(m1c["test_size"]) * float(m1c["selection_share"]))))
    if int(config["selection_k"]) != k:
        raise ValueError("Selection k differs from frozen M1-C")
    if m1c["mcmc"]["chains"] >= mcmc["chains"]:
        raise ValueError("Confirmation must use more chains than exploratory M1-C")
    if float(m1c["gamma_prior_sigma"]) != float(m1d2["fixed_gamma_sigma"]):
        raise ValueError("Gamma prior mismatch between M1-C and M1-D2")
    priors = {p["id"]: p for p in m1d2["intercept_priors"]}
    if set(priors) != {"P0", "P1", "P2", "P3"}:
        raise ValueError("M1-D2 registered prior settings changed")
    if (float(priors["P0"]["mean"]), float(priors["P0"]["sigma"])) != (0.0, 2.5):
        raise ValueError("P0 is no longer the original M1-C prior")
    all_cases = list(config["flagged_cases"]) + list(config["boundary_cases"])
    if len(config["flagged_cases"]) != 3 or len(config["boundary_cases"]) != 3:
        raise ValueError("Expected exactly three diagnostic and three boundary cases")
    expected_flags = {("M1-C", "B1", "P0", -0.8, 101),
                      ("M1-C", "B0", "B0", -0.8, 404),
                      ("M1-D2", "B1", "P2", -0.8, 505)}
    expected_boundary = {("M1-C", "B1", "P0", -0.4, 101),
                         ("M1-D2", "B1", "P1", -0.4, 101),
                         ("M1-D2", "B1", "P2", -0.4, 101)}
    key = lambda c: (c["source"], c["model"], c["prior_id"], float(c["gamma_true"]), int(c["simulation_seed"]))
    if {key(c) for c in config["flagged_cases"]} != expected_flags:
        raise ValueError("Diagnostic case registrations differ from original warnings")
    if {key(c) for c in config["boundary_cases"]} != expected_boundary:
        raise ValueError("Boundary case registrations differ from P0/P1/P2 comparison")
    if len({x["case_id"] for x in all_cases}) != len(all_cases):
        raise ValueError("Duplicate case_id")
    for c in all_cases:
        if not set(("case_id", "gamma_true", "simulation_seed", "model", "prior_id", "source", "repeats")).issubset(c):
            raise ValueError("Incomplete case definition")
        if int(c["simulation_seed"]) not in m1c["simulation_seeds"] or float(c["gamma_true"]) not in m1c["gamma_true_values"]:
            raise ValueError("Case absent from frozen M1-C generator")
        if int(c["repeats"]) != (2 if c in config["boundary_cases"] else 1):
            raise ValueError("Unexpected confirmation repeat count")
    return all_cases


def sampler_seed(case_id: str, repeat: int) -> int:
    """Stable independent sampler seeds per variant and confirmation repeat."""
    payload = f"M1-E-v0.1|{case_id}|{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def decision_delta(a: pd.DataFrame, b: pd.DataFrame, k: int) -> dict[str, Any]:
    """Compare two decision frames on identical assets, preserving original order."""
    required = {"bldg_id", "p_true", "p_topk", "selected_by_p_mean"}
    if not required.issubset(a.columns) or not required.issubset(b.columns):
        raise ValueError("Decision tables do not contain required columns")
    if not a["bldg_id"].equals(b["bldg_id"]) or not a["bldg_id"].is_unique:
        raise ValueError("Decision tables differ in asset ordering or duplicates")
    if not np.allclose(a["p_true"], b["p_true"], atol=1e-12, rtol=0):
        raise ValueError("Comparison must use the same synthetic truth")
    am = a["selected_by_p_mean"].to_numpy(dtype=bool)
    bm = b["selected_by_p_mean"].to_numpy(dtype=bool)
    if am.sum() != k or bm.sum() != k:
        raise ValueError("Inconsistent top-k selection budget")
    p_a = a["p_topk"].to_numpy(dtype=float)
    p_b = b["p_topk"].to_numpy(dtype=float)
    if not np.isfinite(p_a).all() or not np.isfinite(p_b).all():
        raise ValueError("Nonfinite inclusion probabilities")
    truth = a["p_true"].to_numpy(dtype=float)
    ids = a["bldg_id"].to_numpy()
    return {
        "overlap": int(np.logical_and(am, bm).sum()),
        "replaced_assets": int(k - np.logical_and(am, bm).sum()),
        "max_abs_inclusion_shift": float(np.max(np.abs(p_a - p_b))),
        "mean_abs_inclusion_shift": float(np.mean(np.abs(p_a - p_b))),
        "selected_only_a_ids": ";".join(map(str, ids[np.logical_and(am, ~bm)])),
        "selected_only_b_ids": ";".join(map(str, ids[np.logical_and(~am, bm)])),
        "expected_events_a": float(truth[am].sum()),
        "expected_events_b": float(truth[bm].sum()),
        "expected_events_change_b_minus_a": float(truth[bm].sum() - truth[am].sum()),
    }


def diagnostic_flags(row: dict[str, Any], limits: dict[str, Any]) -> list[str]:
    """Return diagnostic warnings; an unreported diagnostic is not a pass."""
    req = ("rhat_max", "ess_bulk_min", "ess_tail_min", "divergences")
    if any(k not in row or not math.isfinite(float(row[k])) for k in req):
        return ["missing_or_nonfinite_diagnostics"]
    result = []
    if float(row["rhat_max"]) > float(limits["max_rhat"]):
        result.append("rhat")
    if float(row["ess_bulk_min"]) < float(limits["min_bulk_ess"]):
        result.append("bulk_ess")
    if float(row["ess_tail_min"]) < float(limits["min_tail_ess"]):
        result.append("tail_ess")
    if float(row["divergences"]) > float(limits["max_divergences"]):
        result.append("divergences")
    return result
