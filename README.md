# 🪐 Resilient Housing Bayes

**Bayesian simulation framework for modeling housing resilience and urban-scale risk.**  
Part of the **[Habnetic](https://habnetic.org)** open research initiative.

---

## 🧭 Overview
**Resilient Housing Bayes** is Habnetic’s foundational research module exploring how buildings and housing systems behave under uncertainty.  
It integrates **Bayesian inference**, **synthetic data generation**, and **stochastic hazard simulation** to estimate fragility, downtime, and recovery costs at building and urban scales.

The goal is to provide transparent, reproducible probabilistic models that can inform both **resilience policy** and **adaptive design strategies** — for habitats on Earth and, eventually, beyond it.

---

## 🔗 Position in the Habnetic ecosystem

This repository **does not define exposure variables or datasets**.

Instead, it:
- **consumes canonical exposure priors** defined in `Habnetic/docs`
- **reads derived datasets** generated in `Habnetic/data`
- **implements Bayesian models** using those definitions

Separation of concerns:
- **Habnetic/docs** → conceptual definitions, assumptions, interpretation constraints  
- **Habnetic/data** → raw, normalized, and derived datasets  
- **resilient-housing-bayes** → probabilistic models, inference, validation

---

## ⚙️ Features
- 🧩 **Probabilistic modeling** using PyMC and ArviZ  
- 🧠 **Synthetic dataset generation** for resilience testing  
- 📈 **Fragility curve estimation** for multiple hazard types  
- 🌀 **Monte Carlo simulation** for reconstruction cost and downtime  
- 🧱 **Modular, extensible structure** for urban-scale adaptation  

---

## 🧩 Repository Structure

```
resilient-housing-bayes/
│   .gitignore
│   LICENSE
│   README.md
│
├───data/
├───docs/
├───notebooks/
│       01_data_exploration.ipynb
│       02_synthetic_generation.ipynb
│       03_model_definition.ipynb
│       04_inference_and_validation.ipynb
│       05_visualization.ipynb
│
└───src/
```
Notebooks follow a strict progression:
1. **Data exploration** (understanding priors and distributions)
2. **Synthetic generation** (prior predictive reasoning)
3. **Model definition** (generative structure)
4. **Inference and validation**
5. **Visualization and communication**

Skipping steps is discouraged.

At present, the repository focuses on validating exposure inputs and model structure; inference results are exploratory and subject to change.

---

## 📦 Dependencies
- Python 3.11+
- [PyMC](https://www.pymc.io/)
- [ArviZ](https://python.arviz.org/)
- NumPy, pandas, geopandas
- Matplotlib (default), Plotly optional

Install the environment:

```bash
pip install -r requirements.txt

```

---

## 🧠 Roadmap (module scope)
**Phase I — Synthetic Modeling**  
Develop Bayesian fragility models for simplified housing typologies.  

**Phase II — Open Data Integration**  
Incorporate open hazard and climate data from NASA, ESA, and Copernicus.  

**Phase III — Urban-Scale Simulation**  
Extend probabilistic inference to city-scale networks and interdependencies.

---

## 📚 Conceptual references

All exposure variables used in this repository follow **canonical definitions**
maintained in the Habnetic documentation repository.

Current references:
- **RTM — Water Exposure Prior (v0)**  
  https://github.com/Habnetic/docs/blob/main/references/exposure/rtm_water_exposure_v0.md

No exposure variable is defined locally in this repository.
Any change in exposure semantics must occur upstream in `Habnetic/docs`.

---

## 🧪 Citation & Acknowledgement
All code and research materials are released under the **MIT License**.  
If you use this work, please cite:

> **Habnetic (2026)**. *Resilient Housing Bayes: Bayesian Modeling Framework for Housing Resilience.*  
> Habnetic Open Research Lab.  
> https://github.com/Habnetic/resilient-housing-bayes

---

## Linked repositories:
- [Habnetic Data](https://github.com/Habnetic/data)
- [Habnetic Docs](https://github.com/Habnetic/docs)
- [Public Site](https://habnetic.org)

---

## 🌍 Links
- 🌐 [Habnetic Website](https://habnetic.org)
- 🧭 [Habnetic Organization](https://github.com/Habnetic)
- 📫 Contact: [info@habnetic.org](mailto:info@habnetic.org)

---

## License

Unless otherwise stated, the contents of this repository are licensed under the MIT License.

The Habnetic name and logo are not licensed for reuse or endorsement.

---

© 2026 Habnetic — Open Research for Resilient Futures
