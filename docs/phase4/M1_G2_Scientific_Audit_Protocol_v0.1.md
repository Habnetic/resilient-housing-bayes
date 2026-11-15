# Habnetic | M1-G2 scientific acceptance audit

**Protocol version:** 0.1, 15 November 2026  
**Status:** prospective verification plan; no new audit results asserted.  
**Reference experiment:** 3 cities × 2 injected terrain effects × 2 fits = 12 fitted posteriors; 569,741 unique terrain-eligible buildings.  
**Evidence:** synthetic Bernoulli outcomes, not observed flood damage.

## Scientific question and estimands

Does matched full-population inclusion of a transformed terrain predictor alter top-1% decisions and synthetic oracle regret under a controlled synthetic data-generating mechanism? Negative control γ=0; positive control γ=-0.8. B0 removes terrain; B1 includes it. All models are separately fitted within each city, never pooled.

The initial synthetic story is `logit(p_true_i) = -3 + E_i + H_i + gamma_true*T_i`, `Y_i ~ Bernoulli(p_true_i)`; `T_i=asinh(T50_i / 0.5 m)`. B0 logit = α+βE E+βH H, B1 adds γT T. Parameters α, βE, βH each have Normal(0,2.5) prior; B1 γT has Normal(0,0.5). The observed likelihood is Bernoulli logistic, not hydrodynamic flooding. Check all conditions with original frozen config and code.

The decision estimand is each building's posterior probability of being included in the entire-city top-k at fixed k=round(0.01N), approximated from 1,000 posterior draws. The `selected_by_p_mean` set ranks posterior mean event probabilities, which differs from inclusion-probability ranking. Synthetic oracle regret is the expected difference in selected synthetic event-probability sums, not financial loss.

## Verification gates

**Gate 1: frozen inputs and full universe.** Revalidate 569,741 eligible, ID order, reproducible synthetic outcomes, hashes, T50 transformation; note 868 RTM exclusions, not imputed.

**Gate 2: fitting.** All 12 original four-chain NUTS traces must be recoverable from NetCDF and meet R-hat ≤1.01, bulk/tail ESS ≥400, 0 divergences. Inspect trace PNGs and BFMI (flag below .3). No convergence conclusion from short smoke fits.

**Gate 3: parameter recovery.** Report all B1 coefficient means and 95% HDIs against known generative parameters (-3,1,1,γ). A single 95% interval can miss truth by sampling variation; do not treat a miss as proof of failure. B0 under γ=-0.8 is intentionally misspecified and should not be expected to recover generating βs.

**Gate 4: predictive behaviour.** Review saved aggregate rate prior/posterior predictive summaries. Additionally replicate synthetic event rates in quintiles separately by E, H, T, compare observed synthetic rates to 95% replicate intervals. These are in-sample checks; a few binwise misses are expected by chance. Use analytical expected Brier/log loss on independent replicated Bernoulli outcomes under known synthetic p_true for better reference comparisons, not as external empirical validation.

**Gate 5: selection integrity.** Recompute and verify per-building `p_topk`, ∑p_topk=k, exact k selected, ID match, entries=exits, matched synthetic oracle and regrets. Quantify draw-split sensitivity and avoid overstating differences smaller than Monte Carlo noise.

**Gate 6: prior sensitivity.** For B1, use prior-ratio posterior importance reweighting for Gaussian γ priors of σ=0.25 and σ=1.0 versus original σ=0.5. Report importance ESS, γ shift and selected-set changes; if ESS is low, perform actual NUTS refit in a new provenance track. This approximation checks prior-width sensitivity, not the sensitivity to a new physical likelihood.

**Gate 7: limitations and empirical gap.** Validate local elevation CRS/vertical datums and outliers using actual source metadata; inspect RTM exclusions. No historical damage calibration exists in this experiment. Do not confuse old flood-damage map labels with predictions under modified 2026 terrain, buildings and defences. Document this as an M2/M3 task.

## Acceptance

**Automated pass ≠ scientific acceptance.** Accept M1-G2 for the scoped methodological claim only after gates 1–6 are reviewed, and gate 7's unresolved physical data caveats are explicitly bounded in the paper. The posterior audit must be independently run on the actual workstation. Keep the new audit outside the original six frozen scenario directories. Tag only after manually reviewing the report and archived scientific outputs.

## Reproduction

From the original root with its Python 3.11 virtual environment:

```powershell
.\RUN_M1G2_SCIENTIFIC_AUDIT.ps1
```

Expected new outputs under `outputs/phase4/_m1g2_scientific_audit_v0.1/`:

`AUDIT_REPORT.md`, `audit_results.json`, and 12 trace PNGs. The audit produces no new M1-G2 posterior fits.

## What to test next

Compare stratified PPC and expected synthetic replicate log scores to M1-G1's held-out conclusions. If robust, finish the geographic source audit, archive, commit and tag the accepted synthetic M1-G2 milestone.
