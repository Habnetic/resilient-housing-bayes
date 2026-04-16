from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any, Dict, List

import pandas as pd


# ---------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceScaler:
    reference_city: str
    feature_order: List[str]
    means: Dict[str, float]
    stds: Dict[str, float]
    ddof: int = 0
    version: str = "phase3_v1"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.feature_order:
            raise ValueError("feature_order cannot be empty")

        for col in self.feature_order:
            if col not in self.means:
                raise ValueError(f"Missing mean for {col}")
            if col not in self.stds:
                raise ValueError(f"Missing std for {col}")
            if self.stds[col] <= 0:
                raise ValueError(f"Non-positive std for {col}: {self.stds[col]}")


# ---------------------------------------------------------------------
# Fit scaler (RTM only)
# ---------------------------------------------------------------------

def fit_reference_scaler(
    df_ref: pd.DataFrame,
    cols: List[str],
    reference_city: str = "RTM",
    ddof: int = 0,
) -> ReferenceScaler:

    if not cols:
        raise ValueError("cols cannot be empty")

    missing = [c for c in cols if c not in df_ref.columns]
    if missing:
        raise KeyError(f"Missing columns in reference df: {missing}")

    means = {}
    stds = {}

    for col in cols:
        s = df_ref[col]

        if s.isna().any():
            raise ValueError(f"Column {col} has NaNs in reference data")

        mean_val = float(s.mean())
        std_val = float(s.std(ddof=ddof))

        if std_val <= 0:
            raise ValueError(f"Column {col} has std <= 0")

        means[col] = mean_val
        stds[col] = std_val

    scaler = ReferenceScaler(
        reference_city=reference_city,
        feature_order=list(cols),
        means=means,
        stds=stds,
        ddof=ddof,
    )

    scaler.validate()
    return scaler


# ---------------------------------------------------------------------
# Apply scaler (ANY city)
# ---------------------------------------------------------------------

def apply_reference_scaler(
    df: pd.DataFrame,
    scaler: ReferenceScaler,
    suffix: str = "_z",
) -> pd.DataFrame:

    scaler.validate()

    missing = [c for c in scaler.feature_order if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in input df: {missing}")

    out = df.copy()

    for col in scaler.feature_order:
        if out[col].isna().any():
            raise ValueError(f"Column {col} has NaNs in input data")

        out[f"{col}{suffix}"] = (
            (out[col] - scaler.means[col]) / scaler.stds[col]
        )

    return out


# ---------------------------------------------------------------------
# Exposure proxy (IMPORTANT: sign convention)
# ---------------------------------------------------------------------

def derive_e_hat_v0(df: pd.DataFrame) -> pd.DataFrame:
    """
    E_hat_v0 = mean of scaled exposure components
    with negative sign for distance to water.
    """

    required = [
        "dist_to_water_m_z",
        "water_len_density_250m_z",
        "water_len_density_500m_z",
        "water_len_density_1000m_z",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for E_hat_v0: {missing}")

    out = df.copy()

    out["E_hat_v0"] = (
        -out["dist_to_water_m_z"]
        + out["water_len_density_250m_z"]
        + out["water_len_density_500m_z"]
        + out["water_len_density_1000m_z"]
    ) / 4.0

    return out


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def save_reference_scaler(scaler: ReferenceScaler, path: str | Path) -> None:
    scaler.validate()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(scaler.to_dict(), f, indent=2)


def load_reference_scaler(path: str | Path) -> ReferenceScaler:
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scaler = ReferenceScaler(**data)
    scaler.validate()
    return scaler


# ---------------------------------------------------------------------
# Debug helper (optional but useful)
# ---------------------------------------------------------------------

def scaler_summary_df(scaler: ReferenceScaler) -> pd.DataFrame:
    rows = []

    for col in scaler.feature_order:
        rows.append(
            {
                "feature": col,
                "mean_ref": scaler.means[col],
                "std_ref": scaler.stds[col],
                "reference_city": scaler.reference_city,
            }
        )

    return pd.DataFrame(rows)