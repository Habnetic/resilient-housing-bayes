# RTM Phase 0 — Inference Notes (Structural Validation)

last_updated: 2026-02-05  
scope: Rotterdam (RTM)  
phase: 0 (structural / synthetic)  

This document records the **first complete end-to-end inference pass** of the
RTM pipeline.  
Its purpose is **pipeline validation**, not real-world risk estimation.

---

## 1) What this phase is

RTM Phase 0 exists to answer one question only:

> *Can the Habnetic pipeline support meaningful Bayesian inference once an
exposure, a hazard signal, and an outcome exist?*

The answer, based on this phase, is **yes**.

---

## 2) Inputs used

### Exposure (deterministic, frozen)

- `E_hat` — deterministic latent exposure proxy  
- Derived from water proximity metrics:
  - distance to water
  - water length density at 250 / 500 / 1000 m
- Standardized and averaged
- **Not a hazard, not a probability, not a risk score**

Source:
- `outputs/rtm/water_exposure_Ehat_v0.parquet`

---

### Hazard (proxy, v0)

- `H_pluvial_v0`
- Constant forcing (`H = 1.0` for all buildings)

This represents **hazard presence**, not hazard intensity.

Purpose:
- activate the exposure → outcome relationship
- keep inference identifiable
- avoid coupling to real rainfall data prematurely

---

### Outcome (synthetic, v0)

- `Y_damage_v0` ∈ {0, 1}
- Binary damage proxy
- Generated probabilistically as a function of `E_hat`

This outcome:
- is **synthetic**
- is **not calibrated**
- does **not** represent real flood damage

Its only role is to:
- provide a likelihood
- allow Bayesian inference to operate

---

## 3) Model definition

A simple logistic regression:

logit(p_i) = α + β_E · E_hat_i + β_H · H_pluvial_i
Y_i ~ Bernoulli(p_i)


Priors:
- `α ~ Normal(0, 2)`
- `β_E ~ Normal(0, 1)`
- `β_H ~ Normal(0, 1)`

Inference method:
- ADVI (for speed and structural testing)
- Subsampled population (3,000 buildings)

---

## 4) Results (high-level)

- Posterior estimates are stable
- No divergences observed
- Exposure coefficient (`β_E`) is positive and identifiable
- Hazard coefficient (`β_H`) behaves as expected under constant forcing
- Posterior predictive checks show observed outcomes lying well within
  posterior predictive mass

This confirms:
- the likelihood is well-defined
- the model is numerically stable
- the pipeline supports inference end-to-end

---

## 5) Posterior predictive check (PPC)

The PPC shows:
- binary outcome mass at 0 and 1
- observed damage rate aligned with posterior predictive mean
- uncertainty consistent with synthetic outcome construction

This is **expected** behavior for a Bernoulli outcome and confirms that
the model is internally coherent.

---

## 6) What this phase does NOT claim

RTM Phase 0 does **not**:
- estimate flood probability
- quantify real damage risk
- represent calibrated hazard intensity
- support decision-making or policy claims

Any such interpretation would be incorrect.

---

## 7) Why proxies were used

Real hazard and damage data were intentionally **not** used in Phase 0 in order
to:

- decouple pipeline correctness from data availability
- validate modeling logic before adding realism
- avoid premature claims or overfitting
- ensure reproducibility and interpretability

Real data enters **only after** the inference machinery is proven to work.

---

## 8) Exit condition (Phase 0)

Phase 0 is considered **complete** when:

- Exposure is deterministic and frozen
- A hazard signal exists (even as a proxy)
- An outcome exists (even if synthetic)
- Bayesian inference runs end-to-end
- PPC confirms internal consistency

All conditions are now met.

---

## 9) Next phase (Phase 1)

Phase 1 will replace **one** proxy at a time:

1. Introduce a real hazard intensity proxy (e.g. rainfall extremes)
2. Keep outcome simple or semi-synthetic
3. Re-run inference
4. Evaluate sensitivity and stability

Only after that will real damage data be considered.

---

## 10) Status

**RTM Phase 0: CLOSED**

Further work proceeds under Phase 1.
