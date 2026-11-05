"""Tests for M1-E registration and selection stability helpers (no PyMC fits)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rhb.pipelines.m1e_checks import (
    decision_delta, diagnostic_flags, sampler_seed, validate_plan,
)

ROOT = Path(__file__).resolve().parents[1]
E_CONFIG = json.loads((ROOT / "configs/m1e_computational_confirmation_rtm.json").read_text(encoding="utf-8"))


def m1c_fixture():
    return {"test_size": 1000, "selection_share": 0.01,
            "gamma_true_values": [0.0, -0.4, -0.8],
            "simulation_seeds": [101, 202, 303, 404, 505],
            "mcmc": {"chains": 2}, "gamma_prior_sigma": 0.5}


def d2_fixture():
    return {"fixed_gamma_sigma": 0.5, "intercept_priors": [
        {"id": "P0", "mean": 0, "sigma": 2.5},
        {"id": "P1", "mean": -3, "sigma": 1},
        {"id": "P2", "mean": -3, "sigma": 0.5},
        {"id": "P3", "mean": -1.5, "sigma": 1},
    ]}


def test_registered_nine_fits_only():
    cases = validate_plan(E_CONFIG, m1c_fixture(), d2_fixture())
    assert len(cases) == 6
    assert sum(c["repeats"] for c in cases) == 9


def test_invalid_budget_rejected():
    cfg = copy.deepcopy(E_CONFIG)
    cfg["selection_k"] = 50
    with pytest.raises(ValueError, match="Selection k"):
        validate_plan(cfg, m1c_fixture(), d2_fixture())


def test_not_allow_extra_unregistered_case():
    cfg = copy.deepcopy(E_CONFIG)
    cfg["boundary_cases"][0]["simulation_seed"] = 202
    with pytest.raises(ValueError, match="Boundary case"):
        validate_plan(cfg, m1c_fixture(), d2_fixture())


def test_four_chains_required():
    cfg = copy.deepcopy(E_CONFIG)
    cfg["mcmc"]["chains"] = 2
    with pytest.raises(ValueError, match="at least|>=4"):
        validate_plan(cfg, m1c_fixture(), d2_fixture())


def test_sampler_seeds_stable_and_distinct():
    assert sampler_seed("boundary_p0_gneg0p4_s101", 1) == sampler_seed("boundary_p0_gneg0p4_s101", 1)
    assert sampler_seed("boundary_p0_gneg0p4_s101", 1) != sampler_seed("boundary_p0_gneg0p4_s101", 2)
    assert sampler_seed("boundary_p0_gneg0p4_s101", 1) != sampler_seed("boundary_p1_gneg0p4_s101", 1)


def frame(ids, selected, inclusion):
    return pd.DataFrame({
        "bldg_id": ids, "p_true": [.2, .7, .8, .1],
        "selected_by_p_mean": selected, "p_topk": inclusion,
    })


def test_decision_swap_and_inclusion_shift():
    a = frame([1, 2, 3, 4], [False, True, True, False], [.0, .8, .9, .3])
    b = frame([1, 2, 3, 4], [True, False, True, False], [.4, .3, .9, .4])
    d = decision_delta(a, b, k=2)
    assert d["overlap"] == 1
    assert d["replaced_assets"] == 1
    assert d["selected_only_a_ids"] == "2"
    assert d["selected_only_b_ids"] == "1"
    assert d["max_abs_inclusion_shift"] == pytest.approx(.5)
    assert d["expected_events_change_b_minus_a"] == pytest.approx(-.5)


def test_reject_mismatched_asset_order_or_truth():
    a = frame([1, 2, 3, 4], [False, True, True, False], [.0, .8, .9, .3])
    b = a.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="ordering"):
        decision_delta(a, b, 2)
    b = a.copy()
    b.loc[0, "p_true"] = 0.21
    with pytest.raises(ValueError, match="synthetic truth"):
        decision_delta(a, b, 2)


def test_diagnostic_flags_strict_threshold():
    limits = E_CONFIG["diagnostics"]
    assert diagnostic_flags({"rhat_max": 1.01, "ess_bulk_min": 400,
                             "ess_tail_min": 400, "divergences": 0}, limits) == []
    assert diagnostic_flags({"rhat_max": 1.02, "ess_bulk_min": 399,
                             "ess_tail_min": 373, "divergences": 1}, limits) == [
                                 "rhat", "bulk_ess", "tail_ess", "divergences"]


def test_missing_diagnostic_is_never_pass():
    assert diagnostic_flags({"rhat_max": np.nan}, E_CONFIG["diagnostics"]) == ["missing_or_nonfinite_diagnostics"]
