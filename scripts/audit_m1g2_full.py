"""Read-only scientific QA for the frozen Habnetic M1-G2 full-city experiment.

Run from repository root: python scripts/audit_m1g2_full.py
Requires existing M1-G2 environment, src on PYTHONPATH, and frozen outputs.
Writes only to outputs/phase4/_m1g2_scientific_audit_v0.1/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))
from rhb.pipelines import m1g_citywide as eng  # noqa: E402

LOG = logging.getLogger('m1g2.audit')
CITIES = ('DON', 'RTM', 'HAM')
SCENARIOS = ((0.0, 'gamma_0'), (-0.8, 'gamma_neg0p8'))
BAD_IDS = {'', 'nan', 'none', '<na>', 'nat'}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def near(a: float, b: float, *, tol: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= tol


def p_diagnostic(summary: pd.DataFrame, name: str) -> dict[str, float]:
    require(name in summary.index, f'Posterior summary lacks parameter {name}')
    row = summary.loc[name]
    return {k: float(row[k]) for k in ('mean', 'sd', 'hdi_2.5%', 'hdi_97.5%',
                                       'ess_bulk', 'ess_tail', 'r_hat')}


def normalize_weights(logweights: np.ndarray) -> tuple[np.ndarray, float]:
    logweights = np.asarray(logweights, dtype=float)
    require(logweights.ndim == 1 and len(logweights) >= 2 and np.isfinite(logweights).all(), 'Bad logweights')
    w = np.exp(logweights - logsumexp(logweights))
    ess = float(1.0 / np.dot(w, w))
    return w, ess


def weighted_quantile(v: np.ndarray, w: np.ndarray, quantiles=(0.025, 0.975)) -> list[float]:
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)
    idx = np.argsort(v)
    v, w = v[idx], w[idx]
    cum = np.cumsum(w)
    return [float(np.interp(q, cum, v)) for q in quantiles]


def prior_reweight(gamma: np.ndarray, base_sigma: float = 0.5,
                   new_sigma: float = 0.25) -> dict[str, Any]:
    """Approximate alternate-prior posterior via importance reweighting; no refit."""
    x = np.asarray(gamma, dtype=np.float64).reshape(-1)
    # Uniform normalizing constants cancel; use Gaussian log prior ratio.
    logw = -0.5 * x * x * (1 / new_sigma**2 - 1 / base_sigma**2)
    weights, ess = normalize_weights(logw)
    qlo, qhi = weighted_quantile(x, weights)
    return {'prior_sigma': new_sigma, 'gamma_weighted_mean': float(np.dot(x, weights)),
            'gamma_interval_equal_tail_95': [qlo, qhi],
            'importance_ess': ess, 'importance_ess_fraction': ess / len(x),
            'n_draws': int(len(x)), 'method': 'self-normalised posterior importance reweighting; not exact refit'}


def grouped_bins(values: np.ndarray, bins: int = 5) -> tuple[np.ndarray, list[str]]:
    vals = np.asarray(values, dtype=float)
    require(np.isfinite(vals).all(), 'Nonfinite predictor in PPC bins')
    cuts = np.unique(np.quantile(vals, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 2:
        return np.zeros(len(vals), dtype=int), ['all_equal']
    groups = np.searchsorted(cuts[1:-1], vals, side='right').astype(np.int16)
    labels = [f'bin{i+1}_of_{len(cuts)-1}' for i in range(len(cuts)-1)]
    return groups, labels


def compare_with_manifest(comparison: pd.DataFrame, man: dict, k: int,
                          b0: pd.DataFrame, b1: pd.DataFrame) -> dict[str, Any]:
    n = len(comparison)
    require(len(b0) == len(b1) == n, 'Scoring/comparison lengths differ')
    for x in (b0, b1):
        require(x['bldg_id'].astype(str).equals(comparison['bldg_id'].astype(str)), 'Score/comparison ID order mismatch')
        require(x['bldg_id'].is_unique, 'Scoring contains duplicate IDs')
        require(x['p_mean'].between(0, 1).all(), 'Invalid p_mean range')
        require(x['p_topk'].between(0, 1).all(), 'Invalid p_topk range')
        require(near(float(x['p_topk'].sum()), k, tol=1e-5), 'Probability of inclusion does not sum to k')
        require(x['selected_by_p_mean'].sum() == k, 'Selected-by-mean != k')
        require(x['selected_by_oracle'].sum() == k, 'Selected-by-oracle != k')
    require((b0['selected_by_oracle'] == b1['selected_by_oracle']).all(), 'Different synthetic oracle')
    require(np.allclose(b0['p_true_synthetic'], b1['p_true_synthetic'], atol=0, rtol=0), 'Different synthetic truth')
    require(np.allclose(comparison['T50_m'], b0['T50_m'], atol=0, rtol=0), 'T50 mismatch')
    require(np.allclose(comparison['p_topk_B0'], b0['p_topk'], atol=0, rtol=0), 'B0 top-k probability mismatch')
    require(np.allclose(comparison['p_topk_B1'], b1['p_topk'], atol=0, rtol=0), 'B1 top-k probability mismatch')
    require(np.allclose(comparison['delta_p_topk_B1_minus_B0'],
                        comparison['p_topk_B1'] - comparison['p_topk_B0'], atol=1e-12, rtol=0), 'Delta top-k mismatch')
    a = comparison['selected_B0'].astype(bool).to_numpy()
    b = comparison['selected_B1'].astype(bool).to_numpy()
    require(int(a.sum()) == int(b.sum()) == k, 'Comparison selected flags != k')
    overlap = int((a & b).sum())
    entries = int((~a & b).sum())
    exits = int((a & ~b).sum())
    expected_groups = np.select([a & b, ~a & b, a & ~b],
                                ['stayed_selected', 'entered_B1', 'left_B1'], default='not_selected')
    require(np.array_equal(comparison['decision_change'].to_numpy(), expected_groups), 'Decision-group mapping wrong')
    metrics = man['metrics']
    for key, actual in {'n': n, 'k': k, 'overlap': overlap, 'entries_B1': entries,
                        'exits_B1': exits,
                        'borderline_B0': int(((b0.p_topk > .2) & (b0.p_topk < .8)).sum()),
                        'borderline_B1': int(((b1.p_topk > .2) & (b1.p_topk < .8)).sum())}.items():
        require(int(metrics[key]) == actual, f'Comparison metric mismatch: {key}')
    require(entries == exits, 'Entry/exit imbalance')
    require(near(float(np.abs(comparison['delta_p_topk_B1_minus_B0']).mean()),
                 float(metrics['mean_abs_delta_inclusion']), tol=1e-10),
            'Mean absolute inclusion-probability difference mismatch')
    for model, sel, key in [('B0', a, 'regret_B0'), ('B1', b, 'regret_B1')]:
        truth = b0['p_true_synthetic'].to_numpy(float)
        oracle = b0['selected_by_oracle'].astype(bool).to_numpy()
        regret = max(0., float(truth[oracle].sum()) - float(truth[sel].sum()))
        require(near(regret, metrics[key], tol=1e-6), f'Incorrect {model} regret')
    return {'n': n, 'k': k, 'overlap': overlap, 'entries': entries,
            'exits': exits, 'replacement_fraction': float(entries / k),
            'borderline_B0': int(metrics['borderline_B0']),
            'borderline_B1': int(metrics['borderline_B1']),
            'regret_B0': float(metrics['regret_B0']),
            'regret_B1': float(metrics['regret_B1'])}


def exact_expected_scores(ptrue: np.ndarray, p_estimate: np.ndarray) -> dict[str, float]:
    """Expected loss on an independent replication from the *known synthetic* DGP."""
    truth = np.asarray(ptrue, dtype=float)
    pred = np.clip(np.asarray(p_estimate, dtype=float), 1e-12, 1 - 1e-12)
    require(np.isfinite(truth).all() and np.isfinite(pred).all(), 'Invalid expected score inputs')
    return {'expected_logloss': float(np.mean(-truth * np.log(pred) - (1 - truth) * np.log1p(-pred))),
            'expected_brier': float(np.mean(truth * (1 - pred) ** 2 + (1 - truth) * pred**2)),
            'mean_abs_bias_vs_truth': float(np.mean(np.abs(pred - truth))),
            'interpretation': 'known-DGP expected replicate loss at same covariates; not field validation'}


def ppc_by_strata(frame: pd.DataFrame, draw: dict[str, np.ndarray], seed: int,
                  n_draws: int = 100) -> dict[str, Any]:
    """Within-city feature-quintile Bernoulli posterior predictive replicated counts.

    Uses a bounded number of posterior parameter draws and one Bernoulli replication
    for each. Each replicate uses the *entire* city. B0 has no terrain term.
    """
    rng = np.random.default_rng(seed)
    e, h, t = [frame[c].to_numpy(dtype=float) for c in ('E_hat_v0', 'H_pluvial_v1_logrel', 'T')]
    y = frame['Y_synthetic'].to_numpy(dtype=np.int8)
    axes = {'exposure': e, 'hazard': h, 'terrain': t}
    codes: dict[str, np.ndarray] = {}
    labels: dict[str, list[str]] = {}
    hist: dict[str, np.ndarray] = {}
    sizes: dict[str, np.ndarray] = {}
    for axis, values in axes.items():
        codes[axis], labels[axis] = grouped_bins(values, 5)
        hist[axis] = np.bincount(codes[axis], weights=y, minlength=len(labels[axis]))
        sizes[axis] = np.bincount(codes[axis], minlength=len(labels[axis]))
    total = len(draw['alpha'])
    sampleidx = np.linspace(0, total - 1, min(n_draws, total), dtype=int)
    reps = {axis: np.zeros((len(sampleidx), len(labels[axis])), dtype=float) for axis in axes}
    for j, i in enumerate(sampleidx):
        lp = draw['alpha'][i] + draw['beta_E'][i] * e + draw['beta_H'][i] * h
        if 'gamma_T' in draw:
            lp = lp + draw['gamma_T'][i] * t
        simulated = (rng.random(len(y)) < expit(lp)).astype(np.int8)
        for axis in axes:
            reps[axis][j] = np.bincount(codes[axis], weights=simulated,
                                         minlength=len(labels[axis])) / np.maximum(sizes[axis], 1)
    outcome: dict[str, Any] = {}
    for axis in axes:
        series = []
        for j, label in enumerate(labels[axis]):
            low, mid, high = np.quantile(reps[axis][:, j], [0.025, 0.5, 0.975]).tolist()
            observed = float(hist[axis][j] / sizes[axis][j])
            series.append({'bin': label, 'n': int(sizes[axis][j]), 'observed_rate': observed,
                           'predicted_p025': low, 'predicted_median': mid,
                           'predicted_p975': high,
                           'outside_95_predictive_interval': bool(observed < low or observed > high)})
        outcome[axis] = series
    return {'n_posterior_replications': len(sampleidx),
            'axes': outcome, 'check': 'in-sample synthetic feature-stratified PPC',
            'warning': 'binwise 95% coverage is not joint; a few misses can occur by chance'}


def topk_sensitivity(frame: pd.DataFrame, draws: dict[str, np.ndarray],
                     reference: pd.DataFrame, k: int,
                     prior_widths: tuple[float, ...] = ()) -> dict[str, Any]:
    """Re-score full city with interleaved posterior half-chains.

    B1 also applies prior importance weights on the same score draws. No refit.
    The two halves are distinct selected-draw subsets, not independent MCMC chains.
    """
    e, h, t = [frame[x].to_numpy(dtype=np.float64) for x in ('E_hat_v0', 'H_pluvial_v1_logrel', 'T')]
    n = len(frame)
    ndraw = len(draws['alpha'])
    require(ndraw % 2 == 0, 'Expected even number of score draws')
    pair = [np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)]
    wcounts = {sigma: np.zeros(n, dtype=np.float64) for sigma in prior_widths}
    weighted_risk = {sigma: np.zeros(n, dtype=np.float64) for sigma in prior_widths}
    weights: dict[float, np.ndarray] = {}
    prior_summaries: dict[str, Any] = {}
    for sigma in prior_widths:
        pr = prior_reweight(draws['gamma_T'], new_sigma=sigma)
        prior_summaries[str(sigma)] = pr
        lg = -.5 * draws['gamma_T'] ** 2 * (1 / sigma**2 - 1 / .5**2)
        weights[sigma], _ = normalize_weights(lg)
    for i in range(ndraw):
        lp = draws['alpha'][i] + draws['beta_E'][i] * e + draws['beta_H'][i] * h
        if 'gamma_T' in draws:
            lp = lp + draws['gamma_T'][i] * t
        # Use the exact frozen scoring transform so floating-point ties match.
        p = expit(lp)
        selected = eng.topk_indices(p, k)
        pair[i % 2][selected] += 1
        for sigma in prior_widths:
            wcounts[sigma][selected] += weights[sigma][i]
            weighted_risk[sigma] += weights[sigma][i] * p
    avg = (pair[0] + pair[1]) / ndraw
    require(np.allclose(avg, reference['p_topk'].to_numpy(float), atol=1e-12, rtol=0),
            'Recomputed 1,000-draw inclusion != frozen score output')
    p1 = pair[0] / (ndraw // 2)
    p2 = pair[1] / (ndraw // 2)
    d = np.abs(p1 - p2)
    result: dict[str, Any] = {
        'n_scoring_draws': ndraw, 'split_draws_each': ndraw // 2,
        'half_split_abs_delta_mean': float(d.mean()),
        'half_split_abs_delta_p99': float(np.quantile(d, .99)),
        'half_split_abs_delta_max': float(d.max()),
        'n_buildings_half_split_delta_gt_0p05': int((d > .05).sum()),
        'n_buildings_half_split_delta_gt_0p10': int((d > .10).sum()),
        'reference_recomputed_exactly': True,
        'warning': 'split posterior draws are serially correlated; these are sensitivity summaries, not SE estimates'
    }
    for sigma in prior_widths:
        ws = wcounts[sigma]
        require(near(float(ws.sum()), k, tol=1e-5), 'Prior-weighted inclusion did not sum to k')
        sel = np.zeros(n, dtype=bool)
        sel[eng.topk_indices(weighted_risk[sigma], k)] = True
        baseline = reference['selected_by_p_mean'].astype(bool).to_numpy()
        prior_summaries[str(sigma)].update({
            'weighted_topk_mean_abs_delta': float(np.abs(ws - reference['p_topk'].to_numpy(float)).mean()),
            'selected_replacements': int((~baseline & sel).sum()),
            'n_selected': int(sel.sum()),
        })
    result['alternate_priors_approximate'] = prior_summaries
    return result


def capture_trace(idata: Any, names: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    posterior = idata.posterior
    fig, axes = plt.subplots(len(names), 2, figsize=(11, 2.6 * len(names)),
                             squeeze=False, constrained_layout=True)
    for row, name in enumerate(names):
        data = np.asarray(posterior[name].values, dtype=float)
        for chain in range(data.shape[0]):
            axes[row, 0].plot(np.arange(data.shape[1]), data[chain], linewidth=.5, alpha=.65,
                              label=f'chain {chain+1}')
            axes[row, 1].hist(data[chain], bins=45, histtype='step', density=True,
                              linewidth=1.1, alpha=.9)
        axes[row, 0].set_title(f'{name}: trace')
        axes[row, 1].set_title(f'{name}: marginal posterior')
        axes[row, 0].set_xlabel('post-warmup draw')
    axes[0, 0].legend(fontsize=8, loc='best')
    fig.savefig(path, dpi=120)
    plt.close(fig)


def module_versions() -> dict[str, str]:
    packages = ('pymc', 'arviz', 'numpy', 'pandas', 'scipy', 'pyarrow', 'xarray', 'matplotlib')
    out = {}
    for pkg in packages:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = 'NOT_INSTALLED'
    return out


def get_git(args: list[str]) -> str:
    try:
        p = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True,
                           encoding='utf-8', timeout=12, check=True)
        return p.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def audit_all(*, quick: bool, output_dir: Path) -> dict[str, Any]:
    cfg = eng.read_config()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / 'traces'
    trace_dir.mkdir(exist_ok=True)
    audit: dict[str, Any] = {
        'audit_version': '0.1.0',
        'audit_time_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'status': 'unreviewed',
        'evidence': 'synthetic labels generated by fixed logistic model',
        'scientific_limitations': [
            'No observed flood-damage labels or validation',
            'No spatially independent external validation',
            'No elevation vertical-datum/city source CRS validation from local artefacts alone',
            'No refitting under alternate prior or alternate likelihood; reweighting is approximate',
            'The joint posterior predictive spatial correlation pattern is not assessed',
            'In-sample PPC and oracle expectation cannot establish empirical forecast accuracy',
        ],
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'packages': module_versions(), 'git_commit': get_git(['rev-parse', 'HEAD']),
                        'git_status_short': get_git(['status', '--short'])},
        'code_hashes': eng.code_fingerprint(),
        'cities': {}, 'critical_failures': [], 'warnings': [],
    }
    require(sys.version_info[:2] == (3, 11), 'Scientific run requires Python 3.11')
    if eng.digest(eng.CONFIG_PATH) != '068b3a5d12b5e2bd410ce6a092cfba520901e2892bedbfa2c71e6b9eeff405c4':
        audit['warnings'].append('Configuration SHA differs from original M1-G2 run; reconcile before acceptance')
    if eng.digest(Path(eng.__file__)) != 'e3481e2cf32c0aeb0e72b5cff017f5e3c52062f1b5aac55b87df9d4cf045fc15':
        audit['warnings'].append('Engine SHA differs from original M1-G2 run; reconcile before acceptance')
    for city in CITIES:
        LOG.info('AUDIT %s: source/terrain/scenario integrity', city)
        universe, sources = eng.load_city(city, cfg)
        n = len(universe)
        require(n == cfg['cities'][city]['expected_eligible'], f'{city}: wrong eligibility count')
        require(universe['bldg_id'].is_unique and not universe['bldg_id'].astype(str).str.lower().isin(BAD_IDS).any(),
                f'{city}: missing/duplicated ID')
        b = universe['T50'].to_numpy(float)
        profile = {'min': float(b.min()), 'p001': float(np.quantile(b, .001)),
                   'p01': float(np.quantile(b, .01)), 'median': float(np.median(b)),
                   'p99': float(np.quantile(b, .99)), 'p999': float(np.quantile(b, .999)),
                   'max': float(b.max()), 'below_minus20m': int((b < -20).sum()),
                   'above_plus20m': int((b > 20).sum()),
                   'T_transformed_min': float(universe['T'].min()),
                   'T_transformed_max': float(universe['T'].max()),
                   'review': 'Physical interpretation/vertical datum/source DEM QA requires independent GIS review'}
        LOG.info('  T50 range %.3f..%.3f m', profile['min'], profile['max'])
        root = eng.city_dir(city)
        prep = read_json(root / 'preparation_manifest.json')
        require(prep['city'] == city and prep['eligible_count'] == n, f'{city}: preparation metadata incorrect')
        require(prep['source_hashes'] == sources, f'{city}: source SHA differs from frozen manifest')
        require(prep['code_hashes'] == eng.code_fingerprint(), f'{city}: code hash differs from frozen manifest')
        city_record = {'n_eligible': n, 'phase3_n': cfg['cities'][city]['expected_phase3'],
                       'excluded': cfg['cities'][city]['expected_phase3'] - n,
                       'source_sha256': sources, 'terrain_profile': profile, 'scenarios': {}}
        for gamma, label in SCENARIOS:
            LOG.info('  SCENARIO %s', label)
            frame, _ = eng.read_frozen(city, gamma, cfg)
            require(frame['bldg_id'].astype(str).equals(universe['bldg_id']), 'Scenario/universe ID order mismatch')
            require(np.isfinite(frame['p_true'].to_numpy(float)).all() and
                    frame['p_true'].between(0, 1).all(), 'Invalid synthetic true p')
            require(frame['Y_synthetic'].isin([0, 1]).all(), 'Synthetic outcome not Bernoulli')
            scenario_record: dict[str, Any] = {
                'gamma_true': gamma,
                'n_synthetic_events': int(frame['Y_synthetic'].sum()),
                'event_rate': float(frame['Y_synthetic'].mean()),
                'mean_true_probability': float(frame['p_true'].mean()),
                'models': {}}
            fit_dir = root / label
            compare_manifest = read_json(fit_dir / 'comparison_manifest.json')
            cp_file = fit_dir / 'citywide_comparison.parquet'
            require(eng.digest(cp_file) == compare_manifest['comparison_sha256'], 'Comparison SHA mismatch')
            require(compare_manifest['gamma_true'] == gamma, 'Wrong comparison gamma')
            comparison = pd.read_parquet(cp_file)
            model_outputs = {}
            for model in ('B0', 'B1'):
                LOG.info('    %s: hashes, MCMC, trace, diagnostics, scoring PPC', model)
                md = fit_dir / model
                fit = read_json(md / 'fit_diagnostics.json')
                scoring = read_json(md / 'scoring_manifest.json')
                require(fit['n_fit'] == n and not fit['smoke'] and fit['model'] == model,
                        f'{city}/{label}/{model}: not a full-population fit')
                require(eng.digest(md / 'idata.nc') == fit['idata_sha256'], 'NetCDF SHA mismatch')
                require(fit['code_hashes'] == eng.code_fingerprint(), 'Fit/code hash mismatch')
                require(eng.digest(md / 'citywide_decisions.parquet') == scoring['result_sha256'], 'Scoring SHA mismatch')
                require(scoring['n'] == n and scoring['k'] == max(1, round(.01 * n)), 'Score n/k mismatch')
                require(scoring['scoring_draws'] == 1000, 'Expected 1,000 score draws')
                decisions = pd.read_parquet(md / 'citywide_decisions.parquet')
                require(decisions['bldg_id'].astype(str).equals(frame['bldg_id']), 'Score/scenario IDs differ')
                idata = az.from_netcdf(md / 'idata.nc')
                names = ['alpha', 'beta_E', 'beta_H'] + (['gamma_T'] if model == 'B1' else [])
                summ = az.summary(idata, var_names=names, hdi_prob=.95)
                require(summ.index.tolist() == names, 'Parameter order mismatch')
                reported = pd.read_csv(md / 'posterior_summary.csv', index_col=0)
                for key in names:
                    for col in ('mean', 'sd', 'r_hat', 'ess_bulk', 'ess_tail', 'hdi_2.5%', 'hdi_97.5%'):
                        require(abs(float(reported.loc[key, col]) - float(summ.loc[key, col])) < .0051,
                                f'{city}/{label}/{model}: saved summary discrepancy {key}/{col}')
                divergences = int(idata.sample_stats['diverging'].sum().item())
                require(divergences == fit['divergences'], 'Saved/recomputed divergences differ')
                limit = cfg['diagnostics']
                rhat_max = float(summ['r_hat'].max())
                eb = float(summ['ess_bulk'].min())
                et = float(summ['ess_tail'].min())
                require(rhat_max <= limit['rhat_max'] and eb >= limit['ess_bulk_min'] and
                        et >= limit['ess_tail_min'] and divergences <= limit['divergences_max'],
                        f'{city}/{label}/{model}: failed convergence criterion')
                require(abs(rhat_max - float(fit['rhat_max'])) < .01, 'Saved/recomputed R-hat mismatch')
                bfmi = az.bfmi(idata)
                min_bfmi = float(np.min(bfmi))
                if min_bfmi < .3:
                    audit['warnings'].append(f'{city}/{label}/{model}: BFMI {min_bfmi:.3f} < .3')
                capture_trace(idata, names, trace_dir / f'{city}_{label}_{model}.png')
                draws = eng.posterior_indices(idata.posterior, cfg['scoring']['draws_per_chain'])
                param = {k: p_diagnostic(summ, k) for k in names}
                if model == 'B1':
                    for key, truth in [('alpha', cfg['alpha_generating']),
                                       ('beta_E', cfg['beta_E_generating']),
                                       ('beta_H', cfg['beta_H_generating']),
                                       ('gamma_T', gamma)]:
                        interval = param[key]
                        if not interval['hdi_2.5%'] <= truth <= interval['hdi_97.5%']:
                            audit['warnings'].append(
                                f'{city}/{label}/B1: 95% HDI for {key} does not contain injected {truth}; review model/MC variation')
                check = scoring['predictive_checks']
                require(check['check_kind'] == 'aggregate_event_rate_only', 'Unexpected original PPC method')
                require(near(float(check['observed_synthetic_rate']), float(frame['Y_synthetic'].mean())),
                        'Original PPC observed synthetic rate differs from scenario')
                for kind in ('prior_predictive_rate', 'posterior_predictive_rate'):
                    q = check[kind]
                    require(0 <= q['p05'] <= q['p50'] <= q['p95'] <= 1,
                            f'Invalid original {kind} quantiles')
                post = check['posterior_predictive_rate']
                original_ppc_outside = not (post['p05'] <= check['observed_synthetic_rate'] <= post['p95'])
                if original_ppc_outside:
                    audit['warnings'].append(f'{city}/{label}/{model}: synthetic global event rate outside original 90% PPC band')
                record: dict[str, Any] = {
                    'original_aggregate_predictive_check': check,
                    'global_rate_outside_original_90pct_ppc': original_ppc_outside,
                    'n_fit': n, 'rhat_max': rhat_max, 'ess_bulk_min': eb,
                    'ess_tail_min': et, 'divergences': divergences, 'bfmi_min': min_bfmi,
                    'posterior_parameters': param, 'file_sha256': {
                        'idata': fit['idata_sha256'], 'scoring': scoring['result_sha256']},
                    'scoring_draws': len(draws['alpha']),
                    'synthetic_expected_replicate_scores': exact_expected_scores(
                        frame['p_true'].to_numpy(float), decisions['p_mean'].to_numpy(float)),
                }
                if not quick:
                    record['stratified_ppc'] = ppc_by_strata(
                        frame, draws, seed=int(hashlib.sha256(f'{city}|{gamma}|{model}|PPC'.encode()).hexdigest()[:8], 16))
                    # Full score split, + approximate sensitivity to alternate gamma priors.
                    record['scoring_sensitivity'] = topk_sensitivity(
                        frame, draws, decisions, scoring['k'],
                        prior_widths=(.25, 1.0) if model == 'B1' else ())
                    for sigma, info in record['scoring_sensitivity'].get('alternate_priors_approximate', {}).items():
                        if info['importance_ess_fraction'] < .1 or info['importance_ess'] < 200:
                            audit['warnings'].append(f'{city}/{label}/{model}: prior sigma={sigma} importance ESS low; refit required')
                scenario_record['models'][model] = record
                model_outputs[model] = decisions
                idata.close() if hasattr(idata, 'close') else None
                del idata, decisions
            scenario_record['comparison'] = compare_with_manifest(
                comparison, compare_manifest, max(1, round(cfg['selection_share'] * n)),
                model_outputs['B0'], model_outputs['B1'])
            scenario_record['comparison']['expected_replicate_delta_B1_minus_B0_logloss'] = (
                scenario_record['models']['B1']['synthetic_expected_replicate_scores']['expected_logloss'] -
                scenario_record['models']['B0']['synthetic_expected_replicate_scores']['expected_logloss'])
            city_record['scenarios'][label] = scenario_record
        audit['cities'][city] = city_record
    audit['status'] = 'AUTOMATED_CHECKS_PASS_WITH_SCOPED_LIMITATIONS'
    return audit


def fmt(value: Any, digits: int = 3) -> str:
    return f'{value:.{digits}f}' if isinstance(value, (float, np.floating)) else str(value)


def write_outputs(audit: dict[str, Any], dest: Path) -> None:
    (dest / 'audit_results.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# Habnetic | M1-G2 scientific audit (automated evidence)', '',
             f"**Run:** {audit['audit_time_utc']}  ",
             f"**Status:** {audit['status']}  ",
             '**Scope:** Frozen 3-city synthetic experiment; no empirical flood-damage validation.', '',
             '## Provenance', '',
             f"- Scientific Git commit: `{audit['environment']['git_commit']}`",
             f"- Python: `{audit['environment']['python'].splitlines()[0]}`",
             '- M1-G2 engine/config/source checksums checked against existing manifests.', '',
             '## Citywide results', '',
             '| City | Scenario γ | N | k | Replacements | Borderline B0/B1 | Regret B0/B1 | Δ expected logloss B1−B0 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for city, record in audit['cities'].items():
        for label, item in record['scenarios'].items():
            m = item['comparison']
            lines.append(f"| {city} | {item['gamma_true']} | {record['n_eligible']:,} | {m['k']:,} | "
                         f"{m['entries']:,} ({100*m['replacement_fraction']:.2f}%) | "
                         f"{m['borderline_B0']}/{m['borderline_B1']} | "
                         f"{m['regret_B0']:.6f}/{m['regret_B1']:.6f} | "
                         f"{m['expected_replicate_delta_B1_minus_B0_logloss']:+.8f} |")
    lines += ['', '## B1 coefficient recovery and fitting', '',
              '| City | γ true | γ mean | γ 95% HDI | Max R-hat | Min ESS bulk/tail | BFMI min |',
              '|---|---:|---:|---|---:|---:|---:|']
    for city, record in audit['cities'].items():
        for label, item in record['scenarios'].items():
            m = item['models']['B1']
            g = m['posterior_parameters']['gamma_T']
            lines.append(f"| {city} | {item['gamma_true']} | {g['mean']:.3f} | "
                         f"[{g['hdi_2.5%']:.3f}, {g['hdi_97.5%']:.3f}] | "
                         f"{m['rhat_max']:.3f} | {m['ess_bulk_min']:.0f}/{m['ess_tail_min']:.0f} | {m['bfmi_min']:.3f} |")
    lines += ['', '## Full audit checks', '',
              'Automated: original raw-input fingerprints, deterministic synthetic generation, '
              'building-ID alignment, parquet SHA256 checks, complete fit cardinalities, '
              'netCDF/CSV posteriors, MCMC diagnostics, per-draw top-k sums, rankings and regret.',
              'Trace plots are in `traces/`.', '']
    if any('stratified_ppc' in model for c in audit['cities'].values()
           for s in c['scenarios'].values() for model in s['models'].values()):
        lines += ['### Feature-stratified synthetic posterior predictive checks', '',
                  '| City | γ | Model | Missed 95% intervals (E/H/T bins) | Maximum split-draw Δp_topk |',
                  '|---|---:|---|---|---:|']
        for city, c in audit['cities'].items():
            for scenario in c['scenarios'].values():
                for name, model in scenario['models'].items():
                    pp = model['stratified_ppc']['axes']
                    miss = '/'.join(str(sum(v['outside_95_predictive_interval'] for v in pp[k]))
                                    for k in ('exposure', 'hazard', 'terrain'))
                    sens = model['scoring_sensitivity']['half_split_abs_delta_max']
                    lines.append(f"| {city} | {scenario['gamma_true']} | {name} | {miss} | {sens:.4f} |")
        lines += ['', '### Approximate terrain-prior sensitivity (B1, importance reweighting)', '',
                  '| City | γ | Prior σ | Importance ESS | Weighted γ | Top-k replacements |',
                  '|---|---:|---:|---:|---:|---:|']
        for city, c in audit['cities'].items():
            for scenario in c['scenarios'].values():
                priors = scenario['models']['B1']['scoring_sensitivity']['alternate_priors_approximate']
                for sigma, vals in priors.items():
                    lines.append(f"| {city} | {scenario['gamma_true']} | {sigma} | "
                                 f"{vals['importance_ess']:.1f} | {vals['gamma_weighted_mean']:.4f} | "
                                 f"{vals['selected_replacements']} |")
    lines += ['', '## Remaining *manual / external* scientific acceptance gates', '',
              '1. Confirm vertical datums, CRS, original DEM footprint, RTM 868 excluded locations '
              'and DON extreme T50 in the source GIS.',
              '2. Review trace PNGs for mixing and trends; automated R-hat/ESS alone is not complete review.',
              '3. Interpret stratified PPC failures relative to intentionally misspecified B0 under γ = -0.8.',
              '4. If importance ESS is inadequate, refit B1 under an alternative gamma prior; '
              'importance weighting is an approximation, not a true refit.',
              '5. Compare with M1-G1 held-out evaluations and independently validate Phase 3 baselines.',
              '6. Archive NetCDF, Parquet and manifests off-Git with checksums; tag only after the review.', '',
              '## Interpretation boundary', '',
              'These synthetic labels are generated with a logistic terrain effect and B1 contains '
              'that exact covariate. Recovery of γ and reduced oracle regret under γ = -0.8 are '
              'positive controls, not validation on observed flood impacts. γ = 0 is a negative control. '
              'The expected replicate log loss is calculated analytically from the known synthetic '
              'generating probabilities on the same geographic feature rows.', '',
              '## Warnings', '']
    lines += [f'- {w}' for w in audit['warnings']] or ['- No automated warnings.']
    lines += ['', '## What to test next', '',
              'Resolve the manual gates and compare feature-stratified PPC to M1-G1 held-out results '
              'before assigning a scientific-acceptance Git tag.', '']
    (dest / 'AUDIT_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true',
                        help='Checks artifacts, MCMC, traces and oracle metrics; skip expensive PPC/scoring/prior sensitivity')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/phase4/_m1g2_scientific_audit_v0.1')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    dest = args.output.resolve()
    if any(dest.is_relative_to(eng.city_dir(city).resolve()) for city in CITIES):
        raise SystemExit('Refusing to write audit inside a frozen city folder')
    try:
        result = audit_all(quick=args.quick, output_dir=dest)
    except Exception as exc:
        LOG.exception('AUDIT FAILED: %s', exc)
        (dest / 'audit_failure.txt').write_text(f'{type(exc).__name__}: {exc}\n', encoding='utf-8')
        return 1
    write_outputs(result, dest)
    LOG.info('AUTOMATED AUDIT PASS. Report: %s', dest / 'AUDIT_REPORT.md')
    LOG.info('Scientific acceptance is NOT automatic: GIS, trace and methodological review remain.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
