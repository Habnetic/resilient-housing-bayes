from __future__ import annotations

import numpy as np
import pymc as pm


def build_m1_model(
    E: np.ndarray,
    H: np.ndarray,
    T: np.ndarray,
    y: np.ndarray,
    gamma_sigma: float = 0.5,
) -> pm.Model:
    """Bayesian logistic model with transformed relative elevation."""

    with pm.Model() as model:

        # Phase 3 priors: unchanged
        alpha = pm.Normal("alpha", mu=0.0, sigma=2.5)
        beta_E = pm.Normal("beta_E", mu=0.0, sigma=2.5)
        beta_H = pm.Normal("beta_H", mu=0.0, sigma=2.5)

        # M1 topographic effect
        gamma_T = pm.Normal(
            "gamma_T",
            mu=0.0,
            sigma=gamma_sigma,
        )

        # Linear predictor
        logit_p = (
            alpha
            + beta_E * E
            + beta_H * H
            + gamma_T * T
        )

        # Likelihood
        pm.Bernoulli(
            "Y_obs",
            logit_p=logit_p,
            observed=y,
        )

    return model
