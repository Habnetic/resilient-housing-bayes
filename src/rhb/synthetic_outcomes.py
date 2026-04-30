from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def add_synthetic_damage_outcome(
    df: pd.DataFrame,
    exposure_col: str = "E_hat_v0",
    hazard_col: str = "H_pluvial_v1_logrel",
    output_col: str = "Y_damage",
    probability_col: str = "p_damage_true",
    alpha: float = -3.0,
    beta_e: float = 1.0,
    beta_h: float = 1.0,
    seed: int = 20260430,
) -> pd.DataFrame:
    required = [exposure_col, hazard_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for synthetic outcome: {missing}")

    out = df.copy()

    x = (
        alpha
        + beta_e * out[exposure_col].to_numpy()
        + beta_h * out[hazard_col].to_numpy()
    )

    p = sigmoid(x)

    rng = np.random.default_rng(seed)
    y = rng.binomial(n=1, p=p)

    out[probability_col] = p
    out[output_col] = y.astype(int)

    return out