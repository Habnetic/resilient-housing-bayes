# RTM Pipeline — Status + Finish Scheme

last_updated: 2026-02-05  
scope: Rotterdam (RTM)

This is the **map of the territory**.  
If you’re confused, read this before opening QGIS or writing more code.

---

## 1) RTM pipeline overview (end-to-end)

### A. Normalize sources (reproducible)
- boundary (CBS → EPSG:28992)
- buildings (OSM → EPSG:28992, schema filtered)
- hydrography (TOP50NL `waterdeel_lijn` → EPSG:28992)

### B. Derive study-area layers (reproducible)
- clip buildings to RTM boundary
- clip hydrography to RTM boundary

### C. Compute priors (deterministic)
- water exposure metrics per building:
  - distance to nearest water
  - water length density within 250 / 500 / 1000 m buffers

### D. Validate priors
- statistical sanity (quantiles, ranges, distributions)
- spatial sanity (QGIS join + graduated styling, manual plausibility check)


### E. Define latent exposure proxy (v0)
- deterministic index `E_hat` from transformed + standardized priors
- export as key-value table by stable building ID (`bldg_id`)

### F. Exposure observatory (v0)
- export run artifacts (state snapshot, logs, figures)
- generate static HTML report for human inspection
- visualize deterministic exposure (`E_hat`) with full provenance

### F2. Define hazard interface (v0)
- introduce first hazard variable (`H_pluvial_v0`)
- deterministic placeholder forcing
- no physical calibration or flood modeling
- establishes Exposure → Hazard interface

### G. Couple to hazard + impact (v1+)
- introduce hazard intensity (rainfall, flood depth, etc.)
- introduce outcome proxy (damage, downtime, cost, displacement)
- Bayesian inference becomes meaningful only here (likelihood exists)

---

## 2) Where we stand (today)

### Completed ✅

**Data repo**
- `processed/RTM/normalized/*` created
- `processed/RTM/derived/buildings_rtm.gpkg`  
  → **221,324 buildings**
- `processed/RTM/derived/hydrography_rtm.gpkg`
- `processed/RTM/priors/building_water_proximity.parquet`  
  → deterministic priors for all buildings  
  → stable `bldg_id` enforced end-to-end

**Resilient-Housing-Bayes**
- notebook 01: EDA completed  
  - distributions validated  
  - correlations understood and documented
- notebook 03 (Part A): **deterministic `E_hat` completed**
  - transforms applied
  - z-score standardization verified
  - `E_hat` exported
  - stats + metadata written
- notebook 03 (Part B): **explicitly disabled**
  - confirmed: no outcome ⇒ no learning
- notebook 03: output paths anchored to repo root (outputs/rtm/)
- notebook 04: hazard interface established
  - deterministic pluvial hazard placeholder (`H_pluvial_v0`)
  - constant forcing, no calibration
  - joined by `bldg_id`



**Model outputs**
- `outputs/rtm/water_exposure_Ehat_v0.parquet`
- `outputs/rtm/water_exposure_Ehat_v0_stats.json`
- `outputs/rtm/hazard_pluvial_v0.parquet`


**QGIS spatial sanity check completed**
  - E_hat joined to buildings_rtm.gpkg via bldg_id
  - graduated quantile styling
  - expected harbor / canal / inland gradients confirmed

**Environment**
- venv + ipykernel fixed
- notebooks fully runnable and reproducible

---

### In progress ⚠️
- Exposure observatory v0
  - artifact-driven
  - read-only
  - static HTML report per run

---

### Not started ❌
- Hazard intensity calibration (real data, ERA5, etc.)
- Outcome / impact definition
- Full Bayesian model with likelihood
- Multi-hazard composition


(All intentionally deferred until v0 is frozen.)

---

## 3) Definition of Done — RTM v0

RTM v0 is done when **all** are true:

1) `building_water_proximity.parquet` exists for all buildings ✅  
2) Deterministic `E_hat` exists and is exported ✅  
3) QGIS join + styling confirms expected spatial patterns (manual check)  
4) Docs specify:
   - features
   - transforms
   - `E_hat` formula
   - interpretation constraints
5) Resilient-Housing-Bayes contains:
   - updated notebook 03
   - outputs written under `outputs/rtm/`
6) One static exposure observatory report exists for an RTM run

At present: items 1–5 complete, item 6 pending.
RTM Phase 0 exposure proxy is frozen.


---

## 4) Final v0 file layout (locked)

### data/
- `processed/RTM/priors/building_water_proximity.parquet` ✅

### resilient-housing-bayes/
- `outputs/rtm/water_exposure_Ehat_v0.parquet` ✅
- `outputs/rtm/water_exposure_Ehat_v0_stats.json` ✅

### docs/
- `docs/references/exposure/rtm_water_exposure_v0.md` (update with E_hat export)

### runs/ (v0 observatory)
- `runs/<run_id>/state_snapshot.json`
- `runs/<run_id>/report.html`
- `runs/<run_id>/figures/`

---

## 5) Next milestone after v0 (v1)

Pick **one** hazard track only:

### Option 1 — Pluvial proxy
- rainfall extremes (ERA5 hourly precipitation)
- imperviousness / drainage proxy (OSM or Copernicus)

### Option 2 — Fluvial / coastal proxy
- flood maps or water levels (if available)
- elevation / distance-to-defense proxies

Then add **one** simple outcome:
- synthetic damage class
- or downtime proxy

Only then does Bayesian inference stop being decorative and start being useful.

---
