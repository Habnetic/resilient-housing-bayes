# 🪐 Resilient Housing Bayes

**Bayesian framework for posterior decision stability and robust prioritisation under uncertainty.**

Part of the **[Habnetic](https://habnetic.org)** open research project.

---

# Overview

**Resilient Housing Bayes** is the primary research repository behind Habnetic.

It develops Bayesian methods for estimating **posterior decision stability**, allowing prioritisation decisions to be expressed as probability distributions rather than deterministic rankings.

The current implementation focuses on **urban flood prioritisation**, using Rotterdam as the baseline case study followed by cross-city transfer experiments in Hamburg and Donostia-San Sebastián.

The repository emphasizes transparent, reproducible Bayesian workflows built on open data and open-source software.

---

# Position within the Habnetic ecosystem

This repository is responsible for the **probabilistic modelling layer**.

It consumes canonical exposure definitions and processed datasets generated elsewhere within Habnetic while implementing the statistical models, inference procedures, diagnostics, and visualisations.

Responsibilities are intentionally separated:

* **Habnetic/docs** → conceptual framework, methodology, definitions
* **Habnetic/data** → raw, processed, and derived datasets
* **resilient-housing-bayes** → Bayesian models, inference, posterior analysis, visualisation

---

# Stewardship

This repository was founded and is currently stewarded by **Mikel Martínez Mugica**.

Development is conducted openly under permissive open-source licenses. Strategic direction, conceptual coherence, and project governance currently remain under the stewardship of **Mikel Martínez Mugica**.

Contributions, discussion, and collaboration are welcome while maintaining a coherent long-term research direction.

---

# Current capabilities

* Bayesian modelling with PyMC
* Posterior decision probability estimation
* Cross-city transfer experiments
* Posterior predictive checks
* Reproducible data pipelines
* Publication-quality figures and visualisations

---

# Repository structure

```text
resilient-housing-bayes/
│
├── data/
├── docs/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_synthetic_generation.ipynb
│   ├── 03_model_definition.ipynb
│   ├── 04_inference_and_validation.ipynb
│   └── 05_visualization.ipynb
│
└── src/
```

The notebooks follow a structured workflow:

1. Data exploration
2. Prior predictive reasoning
3. Bayesian model definition
4. Posterior inference and validation
5. Visualisation and communication

Skipping intermediate steps is discouraged to preserve reproducibility.

---

# Current research status

Current work includes:

* Exposure proxy construction
* Hazard proxy integration
* Posterior decision stability estimation
* Cross-city transfer evaluation
* Robust prioritisation analysis

Future releases will introduce fully Bayesian hazard modelling and hierarchical uncertainty propagation.

---

# Dependencies

* Python 3.11+
* PyMC
* ArviZ
* NumPy
* pandas
* GeoPandas
* Matplotlib
* Plotly (optional)

Installation:

```bash
pip install -r requirements.txt
```

---

# Related repositories

* https://github.com/Habnetic/data
* https://github.com/Habnetic/docs
* https://github.com/Habnetic/habnetic.github.io

---

# Links

🌐 Website: https://habnetic.org

🆔 ORCID: https://orcid.org/0009-0006-5170-4405

📫 Email: [info@habnetic.org](mailto:info@habnetic.org)

---

# Citation

If you use this repository, please cite:

> Martínez Mugica, M. (2026). *Resilient Housing Bayes*. Habnetic. https://habnetic.org

---

# License

Unless stated otherwise, the contents of this repository are released under the **MIT License**.

The **Habnetic** name, logo, visual identity, and branding assets are **not** covered by the MIT License and may not be reused without permission.

---

© 2026 Habnetic — Open research for posterior decision stability.
