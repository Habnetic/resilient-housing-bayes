# Habnetic | Phase 4 M1-F: Population robustness protocol

**Version:** 0.1.0, 5 November 2026<br>
**Status:** registered *exploratory* experiment; results not yet available.<br>
**Scope:** three disjoint Rotterdam building populations with synthetic Bernoulli outcomes. No observed loss data and no spatial transfer claims.

## Goal and reason for the experiment

M1-C and M1-D use the same 5,000 Rotterdam buildings while varying Bernoulli outcomes or prior assumptions. M1-F asks whether the original **conditional synthetic finding** about including informative terrain persists when the **building inventory sample** changes. It does *not* independently validate spatial generalization or the physical meaning of T50.

## Pre-registered design

| Property | Frozen specification |
|---|---|
| Asset universe | 220,456 Rotterdam assets with measured T50 (RTM 5 m AHN5 terrain features), matched by unique `bldg_id` to Phase 3 E/H features |
| Reference exclusion | Remove **all 5,000 M1-C sampled buildings** before drawing M1-F populations |
| New populations | `pop01`, `pop02`, `pop03`, disjoint random samples without replacement |
| Population size | 5,000 buildings per population |
| Split | 4,000 training, 1,000 test assets per population; fixed for both effects and both outcome seeds |
| Synthetic effects | `gamma_true = 0` (negative control), `gamma_true = -0.8` (known signal) |
| Outcome realizations | Seeds 101, 202 per population and effect; paired random uniforms across effects for each population/seed |
| Models | B0 (E+H), B1 (E+H+T) |
| Total | 3 populations × 2 effects × 2 outcome seeds × 2 models = **24 fits** |
| Smoke | `pop01`, both effects, seed 101: **4 fits**. Full run reuses completed smoke fits. |
| Decision budget | Top k=10 among 1,000 held-out buildings (1%); ranking by posterior mean event probability |
| Sampling | Two NUTS chains, 500 tune + 500 draws per chain, target_accept=0.90; exploratory M1-C equivalence |

The three samples are disjoint **from each other and from M1-C**, but drawn without replacement from one urban inventory. They should not be described as independent cities, or as full spatial-blocked validations. Two seeds within a population reuse all covariates and the same split and thus must not be presented as independent population replicates. No cross-city comparison is performed in M1-F.

## Model, generative story, priors and quantities

**Synthetic outcome generator** (unchanged M1-C):

```
T_i = asinh(T50_i / 0.5 m)
logit(p_true_i) = -3 + E_i + H_i + gamma_true * T_i
Y_i ~ Bernoulli(p_true_i)
```

`E=E_hat_v0`, `H=H_pluvial_v1_logrel`; all features retain the existing Phase 3 scales. For each population/seed, identical uniform random numbers generate paired Bernoulli outcomes at gamma 0 and -0.8. The true probabilities are only used to evaluate decisions and predictive error, never to fit the models.

**Likelihoods:** B0 `Y ~ Bernoulli(sigmoid(alpha + beta_E * E + beta_H * H))`; B1 adds `gamma_T * T`.

**Unchanged priors:** `alpha, beta_E, beta_H ~ Normal(0, 2.5)` independently; B1 `gamma_T ~ Normal(0, 0.5)` independently. These are the original M1-C priors. Their aggregate prior predictive rates are overly diffuse; this weakness was studied separately in M1-D2 and is not altered here. The hazard predictor H has very low variation and its coefficient can remain weakly identified despite MCMC convergence.

**Primary posterior quantity:** `p_i` on each held-out building. Draw-wise top-10 inclusion `P(i in Top10 | Y, model)`, selected top-10 based on posterior mean p, oracle top-10 based on known synthetic `p_true`.

**Primary comparison:** `expected_regret = sum(p_true for oracle top-k) - sum(p_true for model top-k)` and paired `regret_B0 - regret_B1`. This equals expected prevented events **only under a hypothetical common, completely effective intervention**; it is not real avoided flood loss.

**Secondary checks:** B0/B1 top-10 overlap, oracle overlap, borderline inclusion counts, maximum/mean change in posterior inclusion probabilities, held-out Brier, log predictive density integrated across posterior draws, RMSE to known synthetic p, coefficient recovery for gamma, prior and posterior predictive aggregate event rates.

## Data source, exclusion and leakage safeguards

The population generator reads, without editing:

- `outputs/phase3/RTM/phase3_features_scaled.parquet` in `resilient-housing-bayes` (E, H, bldg_id).
- `../data/runs/rtm_m1_citywide/terrain_features_m1_5m.parquet` in the sibling `Habnetic/data` repository (`T50`, `terrain_measured`, `bldg_id`).
- `outputs/phase4/RTM/m1c/m1c_scenarios.parquet` to exclude its 5,000 previously selected assets (no M1-C outcomes are reused).

Enforce unique IDs in raw inputs; 220,456 genuinely measured terrain assets; finite E/H/T; one-to-one join; exact population and split counts; no within-population train/test overlap; disjoint populations; no overlap with M1-C; same IDs, order and split across gamma/seed cases. Store SHA-256 checksums for all inputs, generation code, config, population ID file and scenario parquet. The fit runner rejects changed inputs or stale checkpoints.

## Prior predictive, posterior predictive, diagnostics and sensitivity

- **Prior predictive:** retain training-set event-rate 5th/50th/95th percentiles from the original M1-C `run_fit` implementation. Compare with observed synthetic rate; do not claim these priors are calibrated.
- **Posterior predictive:** analogous training-rate summaries; an aggregate rate alone is insufficient for a complete PPC.
- **Convergence:** flag R-hat > 1.01, bulk or tail ESS < 400, any divergences. Two chains are exploratory; any serious warnings require investigation before M1 closure. Check trace behaviour on problematic cases; assess numerical stability of borderline selections when relevant.
- **Model comparison:** held-out log predictive density, Brier and oracle-based regret. Fixed 4,000/1,000 splits mean no LOO or WAIC is needed in this screen.
- **Sensitivity:** intentionally vary *population membership* only; keep the synthetic generator, likelihoods, priors, transforms and budget constant. Separate spatial blocking, alternative population sizes and terrain quality filtering remain untested.

## Acceptance and failure reporting

M1-F is acceptable as an *exploratory robustness screen* when all 12 paired cases and 24 model fits are accounted for, no data leakage occurs, reproducibility hashes match, and diagnostic anomalies are investigated. Report all three per-population results and the within-population variation across two outcome seeds. Do **not** use a majority of wins as evidence of universal transfer.

If B1 does not improve synthetic regret for a particular population when gamma = -0.8, preserve that failure and characterize differences in predictor/terrain distributions and selection margins. If gamma = 0 nevertheless changes rankings or regret, report that negative-control sensitivity. Do not redefine effects or select samples after seeing results.

## Reproduce (repository root, correct Python 3.11 environment)

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
& .\.venv\Scripts\python.exe -m pytest tests/test_m1f.py -q
& .\.venv\Scripts\python.exe -m rhb.experiments.m1f_population_sampling
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1f --smoke --dry-run
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1f --smoke
& .\.venv\Scripts\python.exe -m rhb.pipelines.summarize_m1f --smoke
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1f --dry-run
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1f
& .\.venv\Scripts\python.exe -m rhb.pipelines.summarize_m1f
```

All generated files are written under `outputs/phase4/RTM/m1f/` and remain excluded from Git. Expect `m1f_scenarios.parquet`, `population_ids.csv`, `sampling_manifest.json`, `fits/pop*/gamma*/B{0,1}/{idata.nc,fit_record.json,test_decisions.parquet,posterior_summary.csv}`, `smoke_{fit_results,comparisons,manifest}` and `full_{fit_results,comparisons,manifest}`, plus `full_summary/{summary_by_gamma.csv,summary_by_population_gamma.csv,diagnostic_flags.csv,summary_report.md,summary_manifest.json}`.

**What to test next:** review per-population and per-seed results, investigate any failures/diagnostics, and write `M1_F_Results.md` before issuing an M1 closure statement. Spatial-blocked transfer and HAM/DON terrain portability are later research steps, not claims established here.
