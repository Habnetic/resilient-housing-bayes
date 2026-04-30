from __future__ import annotations

import pymc as pm


def build_phase3_model(E, H, y):
    with pm.Model() as model:
        # Priors
        alpha = pm.Normal("alpha", mu=0, sigma=2.5)
        beta_E = pm.Normal("beta_E", mu=0, sigma=2.5)
        beta_H = pm.Normal("beta_H", mu=0, sigma=2.5)

        # Linear predictor
        logit_p = alpha + beta_E * E + beta_H * H

        # Likelihood (use logit directly, no sigmoid needed here)
        pm.Bernoulli(
            "Y_obs",
            logit_p=logit_p,
            observed=y,
        )

    return model