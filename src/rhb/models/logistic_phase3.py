from __future__ import annotations

import pymc as pm


def build_phase3_model(
    E,
    H,
    y,
    alpha_mu: float = 0.0,
    alpha_sigma: float = 2.5,
    beta_e_mu: float = 0.0,
    beta_e_sigma: float = 2.5,
    beta_h_mu: float = 0.0,
    beta_h_sigma: float = 2.5,
):
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=alpha_mu, sigma=alpha_sigma)
        beta_E = pm.Normal("beta_E", mu=beta_e_mu, sigma=beta_e_sigma)
        beta_H = pm.Normal("beta_H", mu=beta_h_mu, sigma=beta_h_sigma)

        logit_p = alpha + beta_E * E + beta_H * H

        pm.Bernoulli(
            "Y_obs",
            logit_p=logit_p,
            observed=y,
        )

    return model