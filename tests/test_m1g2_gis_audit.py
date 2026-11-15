"""Source-free regression tests for GIS audit tabular helper functions."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_m1g2_gis.py"
spec = importlib.util.spec_from_file_location("m1g2_gis_audit", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_ids_detect_missing_and_duplicates():
    assert mod.normal_ids(pd.Series([" A ", "B"])).tolist() == ["A", "B"]
    with pytest.raises(ValueError, match="Duplicate"):
        mod.normal_ids(pd.Series(["A", " A "]))
    with pytest.raises(ValueError, match="Missing"):
        mod.normal_ids(pd.Series(["A", None]))


def test_exclusion_set_is_exact():
    original = pd.Series(["A", "B", "C", "D"])
    eligible = pd.Series(["A", "C"])
    assert mod.missing_ids(original, eligible).tolist() == ["B", "D"]


def test_relative_extreme_export_and_profile():
    n = 110
    frame = pd.DataFrame({"bldg_id": [f"i{i}" for i in range(n)],
        "T50": np.linspace(-25, 30, n), "T": np.linspace(-4, 4, n),
        "E_hat_v0": np.ones(n), "H_pluvial_v1_logrel": np.zeros(n)})
    p = mod.terrain_profile(frame, "DON")
    assert p["min_m"] == -25 and p["max_m"] == 30
    r = mod.don_extreme_rows(frame, n_each=1)
    assert (r["T50"].abs() > 20).all()
    assert len(r) == int((frame["T50"].abs() > 20).sum())


def test_correlations_and_scales():
    df = pd.DataFrame({"E_hat_v0": [0, 1, 2],
       "H_pluvial_v1_logrel": [2, 1, 0], "T": [0, 1, 2]})
    scales, corr = mod.predictor_diagnostics(df, "DON")
    assert set(scales["predictor"]) == {"E_hat_v0", "H_pluvial_v1_logrel", "T"}
    assert corr.loc[corr["predictor"] == "E_hat_v0", "H_pluvial_v1_logrel"].iat[0] == pytest.approx(-1)
