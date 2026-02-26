# 📘 Resilient Housing Bayes — Documentation

This repository contains **modeling logic and Bayesian inference workflows** only.

All canonical definitions, assumptions, and pipeline status documents live outside this repository.

---

## 🔗 Canonical Sources (Habnetic Ecosystem)

### Exposure Definitions
RTM Water Exposure v0  
→ `Habnetic/docs/references/exposure/rtm_water_exposure_v0.md`

Exposure variables are defined and versioned in the **Habnetic/docs** repository.  
This repository consumes exposure artifacts but does not define them.

---

### Hazard Definitions
RTM Pluvial Hazard v1 (ERA5-Land)  
→ `Habnetic/docs/references/hazard/rtm_pluvial_v1.md`

Hazard semantics, units, aggregation rules, and interpretation constraints are defined upstream.  
Any change to hazard definition must occur in the docs repository, not here.

---

### Data Artifacts
Produced in: `Habnetic/data`

Key RTM artifacts:

- `processed/RTM/priors/building_water_proximity.parquet`
- `processed/RTM/hazards/pluvial/H_pluvial_v1_grid.nc`
- `processed/RTM/hazards/pluvial/H_pluvial_v1_buildings.parquet`

This repository reads those artifacts but does not generate them.

---

## 🧠 Scope of This Repository

This repository is responsible for:

- Constructing generative Bayesian models
- Defining likelihoods
- Performing inference (PyMC / ArviZ)
- Running posterior predictive checks
- Computing decision-level quantities (ranking stability, top-k probability, etc.)

It is **not responsible for**:

- Data normalization
- Hazard computation
- Exposure definition
- CRS policy
- Pipeline orchestration

---

## 📂 Notebook Structure

1. Data exploration (EDA of priors + hazard)
2. Synthetic generation (prior predictive reasoning)
3. Model definition (generative structure)
4. Inference and validation
5. Visualization and communication

Skipping steps is discouraged.

---

## 🧩 Architectural Principle

Separation of concerns:

- **Habnetic/docs** → Definitions, semantics, interpretation constraints  
- **Habnetic/data** → Raw + processed datasets and hazard generation  
- **resilient-housing-bayes** → Bayesian modeling and inference  

This separation ensures:

- Reproducibility  
- Version clarity  
- Explicit phase boundaries  
- No hidden semantic drift  

---

If something appears undefined here, it is probably defined upstream.