# 🪐 Resilient Housing Bayes

**Research code for Bayesian decision stability and probabilistic prioritisation under uncertainty.**

Part of the **[Habnetic](https://habnetic.org)** open research project.

---

## Overview

**Resilient Housing Bayes** is the primary research and experimentation repository behind Habnetic.

The project studies how uncertainty in evidence and probabilistic models propagates into prioritisation decisions. Rather than treating a ranking or selected set as deterministic, the workflow estimates posterior quantities such as the probability that an asset belongs to a constrained top-k selection.

The current reference application is **urban flood-risk prioritisation**.

Rotterdam is used as the baseline case, followed by cross-city stress testing in Hamburg and Donostia-San Sebastián.

The broader Habnetic research direction is domain-general:

```text
uncertain evidence
        ↓
posterior quantities
        ↓
constrained prioritisation
        ↓
decision stability
        ↓
sensitivity / value of information
```

This repository contains the research implementation used to develop, test, and validate those ideas.

It is **not intended to define the final public Habnetic Python API**. Reusable components developed here may later be extracted into a dedicated library once their interfaces and assumptions are sufficiently stable.

---

## Position within the Habnetic ecosystem

Responsibilities are intentionally separated across repositories:

- **[Habnetic/docs](https://github.com/Habnetic/docs)**  
  Concepts, methodology, definitions, research decisions, and project documentation.
- **[Habnetic/data](https://github.com/Habnetic/data)**  
  Canonical raw, processed, and derived datasets.
- **resilient-housing-bayes**  
  Research models, experiments, inference, diagnostics, decision analysis, sensitivity analysis, and reporting code.
- **[Habnetic/habnetic.github.io](https://github.com/Habnetic/habnetic.github.io)**  
  Public project website.

Some historical notebooks in this repository predate the current separation between modelling, data, and documentation. They are retained where useful for reproducibility and development history.

---

## Research status

### Phases 0–2

Early work established:

- exposure and hazard proxies;
- synthetic outcome generation;
- Bayesian risk estimation;
- posterior ranking;
- top-k membership probabilities;
- decision-boundary stability;
- hazard perturbation experiments.

These experiments remain in the repository as part of the research history.

### Phase 3

Phase 3 extended the framework to cross-city transfer and stress testing using:

- **Rotterdam (RTM)** as the baseline;
- **Hamburg (HAM)**;
- **Donostia-San Sebastián (DON)**.

The analysis evaluates whether posterior prioritisation behaviour remains stable when a fixed model specification is transferred across substantially different urban contexts.

The resulting preprint is:

> Martinez Mugica, M. (2026).  
> *Posterior-Based Decision Stability under Cross-City Stress Testing in Urban Flood Risk Prioritisation.*  
> Zenodo. https://doi.org/10.5281/zenodo.23110714

### Phase 4

Current development extends the reference model beyond the simplified Phase 3 specification.

Planned work includes:

- topographic information;
- more realistic hazard representation;
- improved vulnerability and exposure modelling;
- observed outcome validation;
- sensitivity analysis across model assumptions;
- continued development of reusable decision-stability components.

Complexity is added incrementally so that each extension can be compared against the previous model rather than folded into a single opaque specification.

---

## Core decision quantities

A central quantity is the posterior probability that an asset belongs to a constrained selected set:

```text
P(asset i ∈ Top-k | data)
```

From these posterior membership probabilities, the project studies quantities such as:

- stable inclusion;
- stable exclusion;
- uncertain decision boundaries;
- borderline-set size;
- top-k overlap;
- sensitivity to k;
- sensitivity to model assumptions;
- cross-city decision behaviour.

The purpose is not merely to estimate risk, but to quantify **uncertainty in the decision produced from that risk estimate**.

---

## Bayesian workflow

Research models follow a common structure:

1. Generative model
2. Likelihood
3. Priors
4. Prior predictive checks
5. Posterior inference
6. Convergence diagnostics
7. Posterior predictive checks
8. Posterior decision quantities
9. Sensitivity analysis
10. Interpretation

Diagnostics include, where applicable:

- R-hat;
- effective sample size;
- trace behaviour;
- prior predictive sanity checks;
- posterior predictive checks;
- alternate prior or likelihood specifications.

---

## Repository structure

```text
resilient-housing-bayes/
│
├── docs/
│   ├── README.md
│   └── archive/
│       ├── phase2_implementation_plan.md
│       └── phase3_implementation_plan.md
│
├── figures/
│   └── selected publication / research figures
│
├── notebooks/
│   ├── historical research notebooks
│   ├── deprecated/
│   └── phase3/
│
├── outputs/
│   ├── rtm/
│   │   └── selected historical reproducibility artifacts
│   └── phase3/
│       ├── RTM/
│       ├── HAM/
│       └── DON/
│
├── scripts/
│   └── phase3/
│       └── prepare_qgis_layers.py
│
├── src/
│   ├── rhb/
│   │   ├── decision/
│   │   ├── models/
│   │   ├── pipelines/
│   │   └── reports/
│   └── rtm/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Generated outputs, caches, posterior NetCDF files, raster previews, and other reproducible artifacts are ignored by default.

Selected processed inputs and historical artifacts are intentionally versioned where they are required to reproduce earlier experiments.

---

## Source-code organisation

The current `src/rhb/` package separates several research concerns:

```text
decision/
    posterior decision metrics

models/
    Bayesian model definitions

pipelines/
    model execution and analysis workflows

reports/
    diagnostics, sensitivity analyses, figures, maps, and tables
```

Some modules remain explicitly Phase 3-specific.

Reusable decision-stability logic will only be promoted into more general Habnetic components once its domain assumptions and interfaces have been tested beyond the original flood-risk implementation.

---

## Notebooks

The notebooks preserve the chronological development of the project from early exploratory models through Phase 3.

They include:

- exploratory data analysis;
- synthetic outcome experiments;
- model development;
- posterior inference;
- sensitivity analysis;
- deterministic versus posterior ranking comparisons;
- decision-stability experiments;
- hazard perturbation experiments;
- Phase 3 validation and closure work.

Notebooks are retained primarily as **research records and reproducibility material**.

New reusable logic should preferentially be implemented as testable modules under `src/`, with notebooks used for exploration, validation, and documented experiments.

---

## Reproducibility

The project aims to keep research results reproducible while avoiding unnecessary versioning of generated artifacts.

The repository therefore distinguishes between:

**Versioned**

- source code;
- notebooks;
- configuration and metadata;
- selected processed inputs required for reproduction;
- selected research outputs;
- archived implementation plans.

**Generated locally**

- caches;
- posterior NetCDF files;
- temporary figures;
- raster previews;
- intermediate outputs;
- QGIS export layers;
- model-run artifacts that can be regenerated.

Historical experiments may have slightly different structures because the repository architecture evolved during development.

---

## Dependencies

Primary scientific stack:

- Python 3.11+
- PyMC
- ArviZ
- NumPy
- pandas
- GeoPandas
- Matplotlib

Install the current environment with:

```bash
pip install -r requirements.txt
```

For reproducible research runs, use the repository's recorded dependency versions and dataset versions.

---

## Development principles

The project follows several practical rules:

- correctness before performance;
- explicit uncertainty rather than hidden safety factors;
- simple falsifiable models before additional complexity;
- prior predictive checking before inference;
- posterior predictive checking after inference;
- sensitivity analysis for important modelling assumptions;
- small, testable Python functions rather than notebook-only logic;
- reproducible data and environment versions;
- clear separation between exploratory research and reusable library code.

---

## Stewardship

This repository was founded and is currently stewarded by **Mikel Martínez Mugica**.

**ORCID:** https://orcid.org/0009-0006-5170-4405

Development is conducted openly under permissive open-source licensing.

Contributions, discussion, replication, and collaboration are welcome while maintaining conceptual and methodological coherence across the Habnetic project.

---

## Related repositories

- https://github.com/Habnetic/data
- https://github.com/Habnetic/docs
- https://github.com/Habnetic/habnetic.github.io
- https://github.com/Habnetic/habnetic-papers

---

## Links

🌐 Website: https://habnetic.org

🆔 ORCID: https://orcid.org/0009-0006-5170-4405

📫 Email: info@habnetic.org

📄 Phase 3 preprint: https://doi.org/10.5281/zenodo.23110714

---

## Citation

For the Phase 3 research, please cite:

> Martinez Mugica, M. (2026).  
> *Posterior-Based Decision Stability under Cross-City Stress Testing in Urban Flood Risk Prioritisation*.  
> Zenodo. https://doi.org/10.5281/zenodo.23110714

For reuse of the software itself, refer to this repository and the corresponding release or commit used in the analysis.

---

## License

Unless stated otherwise, source code in this repository is released under the **MIT License**.

The **Habnetic** name, logo, visual identity, and branding assets are not covered by the MIT License and may not be reused without permission.

---

© 2026 Habnetic
