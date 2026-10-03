# Resilient Housing Bayes — Repository Documentation

This repository contains the research code, Bayesian modelling workflows,
decision-stability analysis, and reproducibility material behind Habnetic.

Canonical project-wide definitions and methodology are maintained in:

- `Habnetic/docs` — concepts, methodology, definitions, decisions
- `Habnetic/data` — canonical datasets and data-processing artifacts
- `resilient-housing-bayes` — research models, experiments, inference,
  diagnostics, decision analysis, and reporting code

## Historical material

The notebooks and archived implementation plans preserve the development
history of Phases 0–3.

Some historical notebooks generated intermediate exposure, hazard, and
synthetic-outcome artifacts inside this repository. These are retained where
needed for reproducibility and should not be interpreted as the current
cross-repository architecture.

Archived plans are stored under:

```text
docs/archive/

Current development

New reusable decision-stability logic should be implemented as testable Python
modules under src/, while notebooks remain primarily for exploration,
validation, and documented experiments.

Phase-specific scripts belong under:

scripts/<phase>/

Generated outputs should not normally be committed unless they are explicitly
required for reproducibility.


For the root `README.md`, I would **not yet rewrite the whole thing manually in this step**. First finish the structural move. Then we rewrite it based on the actual final tree, because otherwise we get to update the README twice, which is apparently how civilizations collapse.

So now do:

```powershell
New-Item -ItemType Directory -Force scripts\phase3
git mv prepare_phase3_qgis_layers.py scripts\phase3\prepare_qgis_layers.py

