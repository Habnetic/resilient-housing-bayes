from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from rhb.decision.topk import compute_topk_membership
from rhb.decision.stability import summarize_decision_metrics

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_posterior_params(idata_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idata = az.from_netcdf(idata_path)
    posterior = idata.posterior

    alpha = posterior["alpha"].values.reshape(-1)
    beta_e = posterior["beta_E"].values.reshape(-1)
    beta_h = posterior["beta_H"].values.reshape(-1)

    return alpha, beta_e, beta_h


def compute_posterior_probabilities(
    alpha: np.ndarray,
    beta_e: np.ndarray,
    beta_h: np.ndarray,
    e: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """
    Returns posterior probabilities with shape:
    (n_draws, n_assets)
    """
    logit = (
        alpha[:, None]
        + beta_e[:, None] * e[None, :]
        + beta_h[:, None] * h[None, :]
    )
    return sigmoid(logit)


def save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")