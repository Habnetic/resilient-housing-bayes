# Habnetic | Phase 4 M1-E: Computational confirmation results

**Version:** 0.2.0, 5 November 2026<br>
**Status:** Nine refits and the automated summary completed; three originally flagged traces visually reviewed; **six boundary trace plots still require individual visual inspection** before final sign-off. Exploratory computational confirmation, not empirical flood-risk validation.<br>
**Protocol:** `docs/phase4/M1_E_Experimental_Protocol.md`<br>
**Configuration:** `configs/m1e_computational_confirmation_rtm.json`<br>
**Implementation:** `src/rhb/pipelines/run_m1e.py`, `src/rhb/pipelines/summarize_m1e.py`, `src/rhb/pipelines/m1e_checks.py`<br>
**Evidence:** `outputs/phase4/RTM/m1e/` (Git-ignored; preserve the source artifacts and manifests). This report records the actual terminal output and the three user-supplied trace plots reviewed on 5 November 2026; it does not invent unseen fit-level scores or input hashes.

## Research question

Do longer, four-chain MCMC runs resolve three previously observed numerical warnings? Was the single M1-D2 P0-versus-P1/P2 top-10 membership change (`gamma_true=-0.4`, outcome seed 101) reproducible, or did finite Monte Carlo sampling plausibly contribute?

## Experimental specification

- **Generative story (unchanged):** synthetic `Y_i ~ Bernoulli(p_true_i)`, where `logit(p_true_i) = -3 + E_i + H_i + gamma_true*T_i`, with `T_i=asinh(T50_i/0.5 m)`. In the refits, the outcomes and generator are *not* changed.
- **Likelihood (unchanged):** Bernoulli logistic regression. B0 includes E and H; B1 adds T. The M1-D2 P0, P1 and P2 cases differ only in the prior on the intercept.
- **Priors (unchanged):** B0 and B1 P0 use `alpha, beta_E, beta_H ~ Normal(0,2.5)`; B1 uses `gamma_T ~ Normal(0,0.5)`. P1 uses `alpha ~ Normal(-3,1.0)`; P2 uses `alpha ~ Normal(-3,0.5)`. P1/P2 centres exploit the known synthetic generator and are not field-elicited priors.
- **Population and decision (unchanged):** 5,000 fixed Rotterdam buildings, 4,000 training and 1,000 test assets; selection of top `k=10` by posterior mean probability. Oracle-relative expected regret is computed against known synthetic `p_true` under an equal-benefit idealization. Held-out labels are not fitted.
- **Inference:** four NUTS chains, 1,000 tuning and 1,000 retained draws **per chain**, `target_accept=0.92`, separate stable seeds for independent repeats.
- **Pre-specified numerical checks:** flag maximum R-hat `>1.01`, minimum bulk or tail ESS `<400`, or any divergences. These are screening rules rather than proof of validity. Trace and decision inspections are separate requirements.

## 1. Source reference audit

The audit read existing reference posterior artifacts and reproduced the three original warnings plus the three clean boundary reference fits:

| Original case | Maximum R-hat | Minimum bulk ESS | Minimum tail ESS | Original flag |
|---|---:|---:|---:|---|
| M1-C B1, gamma -0.8, seed 101 | 1.00 | 868 | 394 | Tail ESS < 400 |
| M1-C B0, gamma -0.8, seed 404 | 1.02 | 522 | 579 | R-hat > 1.01 |
| M1-D2 B1/P2, gamma -0.8, seed 505 | 1.01 | 531 | 373 | Tail ESS < 400 |
| Boundary B1/P0, gamma -0.4, seed 101 | 1.01 | 664 | 632 | None |
| Boundary B1/P1, gamma -0.4, seed 101 | 1.00 | 789 | 511 | None |
| Boundary B1/P2, gamma -0.4, seed 101 | 1.01 | 835 | 573 | None |

**Audit source:** `outputs/phase4/RTM/m1e/reference_audit.csv`, `reference_audit_manifest.json`; obtain exact SHA-256 digests from the generated manifests, not from this narrative. Repeated references to the same M1-C fit across M1-D1 and M1-D2 are not independent warnings.

## 2. Four-chain computational confirmation

The complete run reported **nine confirmation fits** (three flagged-case refits and two independent repetitions of P0/P1/P2 at the decision boundary). The automated summary reported **zero diagnostic warnings** and **zero divergences**.

| Case | Repeat | R-hat max (reported) | Bulk ESS min | Tail ESS min | Divergences | Synthetic expected regret | Overlap vs original top-10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1-C B1, gamma -0.8, seed 101 | 1 | 1.00 | 3,441 | 2,618 | 0 | 0.000000 | 10/10 |
| M1-C B0, gamma -0.8, seed 404 | 1 | 1.00 | 2,394 | 2,180 | 0 | 0.253009 | 10/10 |
| M1-D2 B1/P2, gamma -0.8, seed 505 | 1 | 1.00 | 3,651 | 2,535 | 0 | 0.000000 | 10/10 |
| Boundary P0, gamma -0.4, seed 101 | 1 | 1.00 | 3,839 | 2,861 | 0 | 0.020406 | 9/10 |
| Boundary P0, gamma -0.4, seed 101 | 2 | 1.00 | 3,550 | 2,756 | 0 | 0.020406 | 9/10 |
| Boundary P1, gamma -0.4, seed 101 | 1 | 1.00 | 3,350 | 2,797 | 0 | 0.020406 | 10/10 |
| Boundary P1, gamma -0.4, seed 101 | 2 | 1.00 | 3,649 | 2,800 | 0 | 0.020406 | 10/10 |
| Boundary P2, gamma -0.4, seed 101 | 1 | 1.00 | 3,583 | 2,961 | 0 | 0.020406 | 10/10 |
| Boundary P2, gamma -0.4, seed 101 | 2 | 1.00 | 3,506 | 2,765 | 0 | 0.020406 | 10/10 |

**Aggregate numerical checks:** 9/9 fits satisfy the registered thresholds, with minimum bulk ESS **2,394**, minimum tail ESS **2,180**, zero divergences. R-hat values displayed above are rounded from exported summaries; the no-flag result is calculated using the pipeline's actual diagnostic check, not the displayed rounding.

**Scope of confirmation:** These refits resolve the prior computational warnings on the **specified cases**. They do not show that every model in all earlier experiments is perfectly sampled, nor that any coefficient is physically meaningful.

## 3. Reproducibility of the prior-sensitive boundary

### 3.1 Original exploratory M1-D2 observation

At `gamma_true=-0.4`, seed 101, original two-chain inference reported:

| Prior | Original regret | Original relationship |
|---|---:|---|
| P0 | 0.000000 | Different by one selected asset from original P1/P2 |
| P1 | 0.020406 | Original P1/P2 selection |
| P2 | 0.020406 | Original P1/P2 selection |

In the original five-seed M1-D2 table, this single scenario accounted for the mean P1/P2 versus P0 regret difference (+0.004081). The original result remains correctly recorded in `M1_D2_Results.md` as an **observed numerical result**, but it requires qualification.

### 3.2 Independent four-chain repeats

All **six** boundary refits returned synthetic expected regret **0.0204055404959859**. The two repeats under each prior selected **identical top-10 assets**.

| Comparison | Repeat(s) | Top-10 overlap | Assets replaced | Maximum absolute difference in posterior inclusion probability |
|---|---|---:|---:|---:|
| P0 versus P1 | 1 | 10/10 | 0 | 0.01100 |
| P0 versus P2 | 1 | 10/10 | 0 | 0.01125 |
| P0 versus P1 | 2 | 10/10 | 0 | 0.01125 |
| P0 versus P2 | 2 | 10/10 | 0 | 0.01825 |
| P0 repeat 1 versus repeat 2 | 1 vs 2 | 10/10 | 0 | 0.00975 |
| P1 repeat 1 versus repeat 2 | 1 vs 2 | 10/10 | 0 | 0.00700 |
| P2 repeat 1 versus repeat 2 | 1 vs 2 | 10/10 | 0 | 0.01550 |

**Key observation:** The new P0 selections overlap the original P0 by only **9/10**, whereas the new P1 and P2 selections overlap their respective originals **10/10**. Under the more intensive computation, P0, P1 and P2 agree. Consequently the isolated original M1-D2 top-10 difference is **not reproduced**; it cannot support a reliable claim of intercept-prior-induced selection harm.

**Interpretation:** Finite MCMC approximation near a discrete top-*k* boundary is a plausible cause of the original swap, although this is not a formal decomposition of sampling variance and does not exclude an implementation or rounding contribution. Moreover, a numerically better-supported selection need not have lower oracle-relative regret: the common confirmed selection has regret **0.020406**, while the original P0 selection reported zero.

**Not yet audited:** The exact swapped asset IDs and the difference in their posterior mean probabilities have **not** been independently extracted from the stored decision tables in this report. Avoid claims about the physical location or characteristics of these two assets until that check is performed.

## 4. Visual trace review

Three trace plots supplied to the research log have been visually inspected:

| Fit | Review | Observations |
|---|---|---|
| M1-C B0, gamma -0.8, seed 404 | **Pass (visual)** | Four overlapping chains, stationary-looking traces, no obvious stuck chains; beta_H posterior diffuse |
| M1-C B1, gamma -0.8, seed 101 | **Pass (visual)** | No visible drift or persistent chain separation; beta_H broad, gamma_T negative |
| M1-D2 P2, gamma -0.8, seed 505 | **Pass (visual)** | Density and trace behaviour broadly consistent across chains; beta_H remains diffuse |

The **six boundary trace images exist**, but their individual visual review has **not** been recorded. The automated summaries alone cannot replace that inspection. Complete and record it before changing the document status to *fully reviewed*.

**Scientific caution:** `H_pluvial_v1_logrel` had a training SD of only about **0.007618**; a broad `beta_H` posterior is compatible with poor identification despite excellent R-hat/ESS. This is a covariate/parameter-scale problem, **not** itself an MCMC convergence warning.

## 5. Predictive checks and limits of the present evidence

- **Prior predictive:** M1-D2 established that P1/P2 greatly reduce the gross prior predictive rate mismatch relative to P0, but these priors are centred on the known synthetic intercept. M1-E does not change priors or provide new prior elicitation.
- **Posterior predictive:** `run_m1e.py` records posterior predictive information. The supplied terminal summary does not include the fit-level PPC quantiles; this document makes **no independent claim** that every detailed PPC passed. Inspect the stored fit records if a broader posterior predictive statement is needed.
- **Held-out prediction:** The M1-E pipeline records held-out metrics; no numerical held-out score differences were supplied for this documentation update. Do not invent or infer them from regret alone.
- **Model comparison:** This is a fixed, held-out train/test study. No LOO/WAIC comparison is claimed as a M1-E result.
- **Physical interpretation:** These are synthetic outcomes with a known generator; no observed flood losses, drainage mechanisms, vulnerability, spatial generalization, missingness sensitivity or recovery outcomes were evaluated.
- **Population robustness:** The buildings, split and outcome seeds remain unchanged from M1-C/M1-D2. A successful MCMC confirmation is *not* a new population replication.

## 6. Reproducibility and follow-up checks

From the repository root, run with the **project-local Python 3.11 environment** (avoid accidentally using `../data/.venv`, which is Python 3.8):

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
& .\.venv\Scripts\python.exe -m pytest tests/test_m1e.py -q
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1e --audit-only --trace-plots
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1e --dry-run
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1e --smoke
& .\.venv\Scripts\python.exe -m rhb.pipelines.summarize_m1e --smoke
& .\.venv\Scripts\python.exe -m rhb.pipelines.run_m1e
& .\.venv\Scripts\python.exe -m rhb.pipelines.summarize_m1e
```

**Input sources:** original M1-C scenario parquet/fit artifacts, M1-D2 audit manifest and referenced fit artifacts, M1-E config. **Generated evidence:** `outputs/phase4/RTM/m1e/reference_audit.csv`, `reference_audit_manifest.json`, `full_results.csv`, `full_manifest.json`, `full_summary/{fit_diagnostics.csv,boundary_comparisons.csv,diagnostic_flags.csv,summary_manifest.json}`, and nine NetCDF/trace/decision fit directories under `fits/`. Preserve files and manifest hashes in separately controlled research storage because `outputs/**` is Git-ignored.

**Required remaining QA:** (1) visually inspect six boundary trace plots; (2) rerun the nine M1-E tests under the project-local Python 3.11 executable (earlier passing output used the sibling Python 3.8 environment); (3) check source/result manifest hashes and extract the original-versus-confirmed swapped asset IDs if a detailed boundary narrative is to be published.

## Decision

**M1-E computational results accepted for targeted exploratory confirmation**, with 9/9 numerically passing refits and a reproduced common selection across both independent boundary repeats. The original M1-D2 claim of a prior-induced boundary swap is **downgraded to an unconfirmed exploratory observation**. Full documentation sign-off remains contingent on the limited manual QA items above. M1-F population robustness remains a separate experiment.

**What to test next:** Close the remaining trace review and provenance QA; then run a small, preregistered M1-F population/split sensitivity study. Do not expand the intercept-prior grid without a specific new hypothesis.
