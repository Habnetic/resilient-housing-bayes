# Habnetic — Next Steps (Control Tower)

last_updated: 2026-02-05  
owner: C. Price  
scope: RTM (Rotterdam) pipeline + immediate modeling steps

This file lives in the parent `Habnetic/` folder on purpose.  
It is the **one place to read before touching any repo**.

---

## Where we are right now (RTM)

✅ Spatial backbone complete  
- RTM boundary, buildings, hydrography normalized, clipped, validated  
- CRS harmonized (EPSG:28992), geometry hygiene enforced

✅ Water exposure priors computed  
- Distance to nearest water  
- Water length density at 250 m / 500 m / 1000 m  
- Computed for **all 221,324 buildings**

✅ Stable building identifier fixed  
- Deterministic `bldg_id` propagated end-to-end  
- No implicit index joins, no hidden geometry state

✅ Notebook 01 — data exploration  
- Distributions validated  
- Correlations understood and documented

✅ Notebook 03 — deterministic exposure completed  
- Latent exposure proxy `E_hat` computed deterministically  
- Exported with explicit assumptions and metadata

⚠️ Notebook 03 — Bayesian section intentionally disabled  
- Confirmed: **no outcome = no learning**  
- Inference deferred by design

✅ Notebook 04 — hazard scaffolding started  
- First pluvial hazard proxy introduced  
- Deterministic, placeholder forcing only  
- No physical flood modeling implied

✅ Environment stabilized  
- venv + ipykernel wired correctly  
- Notebooks runnable and reproducible

---

## Immediate state (just completed)

### Deterministic exposure index (RTM v0)

A **deterministic latent exposure proxy** has been locked in:

- `E_hat` is **not a hazard**
- `E_hat` is **not a probability**
- `E_hat` is **not a risk score**

It is a **relative exposure ranking** suitable for:
- clustering
- conditioning
- future hazard coupling

### Deterministic hazard proxy (RTM v0)

A **first pluvial hazard forcing proxy** has been introduced:

- `H_pluvial_v0` is **not flooding**
- `H_pluvial_v0` is **not intensity-calibrated**
- `H_pluvial_v0` is a **structural placeholder**

Purpose:
- enable Exposure → Hazard → Outcome wiring
- prevent premature Bayesian inference
- freeze interfaces before adding realism

### Artifacts produced

**Data**
- `data/processed/RTM/priors/building_water_proximity.parquet` ✅

**Model outputs**
- `resilient-housing-bayes/outputs/rtm/water_exposure_Ehat_v0.parquet` ✅
- `resilient-housing-bayes/outputs/rtm/water_exposure_Ehat_v0_stats.json` ✅
- `resilient-housing-bayes/outputs/rtm/hazard_pluvial_v0.parquet` ✅

---

## v0 deterministic definition (frozen)

Let:

- `d = dist_to_water_m`
- `rho_r = water_len_density_{r}m`

Transforms:

- `x_d     = -log(d + eps)`
- `x_250   = log(rho_250 + eps)`
- `x_500   = log(rho_500 + eps)`
- `x_1000  = log(rho_1000 + eps)`

Standardization:

- `z_k = (x_k - mean(x_k)) / std(x_k)`

Deterministic exposure proxy:

E_hat = (z_d + z_250 + z_500 + z_1000) / 4


Interpretation constraints:
- relative exposure only
- no probability implied
- valid **only within RTM context**

---

## Next step (after a pause, not immediately)

### Phase 1b — Outcome proxy (gate to inference)

Introduce **one minimal outcome variable**, e.g.:

- binary damage flag (synthetic)
- ordinal downtime class
- toy continuous loss index

Requirements:
- joined by `bldg_id`
- explicitly labeled synthetic (if synthetic)
- no calibration claims

**Bayesian inference becomes legal only after this step.**

---

## Phase 2 — Exposure + Hazard Observatory (v0)

### Goal

Create a **read-only observatory** that:

- visualizes deterministic artifacts (`E_hat`, `H_pluvial_v0`)
- records provenance
- produces human-readable run artifacts

This is **not** a hazard dashboard.  
This is **not** a decision support tool.

---

### Allowed scope (strict)

The v0 observatory may show:
- spatial view of buildings + water network
- deterministic exposure and hazard proxies
- run metadata (timestamp, code version, seed)
- assumptions and interpretation constraints
- structured run log

The v0 observatory must **not**:
- claim flood probability or risk
- show posteriors or uncertainty bands
- include sliders, toggles, or controls
- re-run inference or mutate inputs

---

### Implementation principle

The observatory consumes **immutable pipeline artifacts**.

Minimum required artifacts per run:
- `state_snapshot.json`
- map snapshot(s)
- references to parquet outputs
- structured run log

---

### Deliverable (v0)

- One **static HTML report per run**
- Generated automatically at pipeline completion
- Stored under:

runs/<run_id>/


This report is the **canonical human interface** to RTM v0.

---

## What NOT to do (still true)

- Do **not** claim flood probability.
- Do **not** enable Bayesian inference without an outcome.
- Do **not** introduce a second hazard yet.
- Do **not** commit large generated artifacts without policy (LFS or equivalent).

---

## Canonical docs to update

- `Habnetic/docs/references/exposure/rtm_water_exposure_v0.md`
  - frozen exposure definition
  - hazard placeholder note
  - interpretation constraints
  - links to generated artifacts
