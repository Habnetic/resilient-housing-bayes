"""Resume-safe M1-G2 run orchestrator (no changes to scientific implementation).

Run from repository root: python scripts/run_m1g2_all.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'm1g_citywide_v0.1.json'
ENGINE = ROOT / 'src' / 'rhb' / 'pipelines' / 'm1g_citywide.py'
CITIES = ('DON', 'RTM', 'HAM')
GAMMAS = ((-0.8, 'gamma_neg0p8'), (0.0, 'gamma_0'))
LOG = logging.getLogger('m1g2.runner')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def verify_prerequisites() -> dict:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError('Use the repository Python 3.11 virtual environment')
    if not CONFIG.exists() or not ENGINE.exists():
        raise RuntimeError('M1-G2 v0.1 must already be installed in repository root')
    cfg = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    if cfg.get('experiment') != 'M1-G2' or cfg.get('gamma_scenarios') != [0.0, -0.8]:
        raise RuntimeError('Unexpected M1-G2 experiment configuration')
    if set(cfg['cities']) != set(CITIES):
        raise RuntimeError('Expected RTM/HAM/DON in configuration')
    import pymc  # noqa: F401
    import arviz  # noqa: F401
    import pyarrow  # noqa: F401
    return cfg


def run(*flags: str, log_path: Path) -> None:
    cmd = [sys.executable, '-m', 'rhb.pipelines.m1g_citywide', *flags]
    LOG.info('RUN: %s', ' '.join(cmd))
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    with log_path.open('a', encoding='utf-8') as stream:
        stream.write('\n' + '=' * 78 + '\n')
        stream.write(f'{dt.datetime.now(dt.timezone.utc).isoformat()} | {cmd}\n')
        stream.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                             encoding='utf-8', errors='replace')
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            stream.write(line)
        code = p.wait()
        if code:
            raise RuntimeError(f'Command exited {code}: {" ".join(cmd)}; see {log_path}')


def check_fit(fit_dir: Path, expected_n: int, limits: dict) -> None:
    diag_file = fit_dir / 'fit_diagnostics.json'
    idata_file = fit_dir / 'idata.nc'
    if not diag_file.exists() or not idata_file.exists():
        raise RuntimeError(f'Incomplete fit outputs in {fit_dir}; do not overwrite. Inspect manually.')
    d = json.loads(diag_file.read_text(encoding='utf-8'))
    if d['smoke'] or d['n_fit'] != expected_n:
        raise RuntimeError(f'Not a full-city fit: {fit_dir}')
    if d['idata_sha256'] != sha256(idata_file):
        raise RuntimeError(f'Fit file checksum mismatch: {fit_dir}')
    failed = []
    if d['rhat_max'] > limits['rhat_max']:
        failed.append(f'R-hat {d["rhat_max"]:.4f} > {limits["rhat_max"]}')
    if d['ess_bulk_min'] < limits['ess_bulk_min']:
        failed.append(f'Bulk ESS {d["ess_bulk_min"]:.0f} < {limits["ess_bulk_min"]}')
    if d['ess_tail_min'] < limits['ess_tail_min']:
        failed.append(f'Tail ESS {d["ess_tail_min"]:.0f} < {limits["ess_tail_min"]}')
    if d['divergences'] > limits['divergences_max']:
        failed.append(f'Divergences {d["divergences"]} > {limits["divergences_max"]}')
    if failed:
        raise RuntimeError(f'Diagnostics failed for {fit_dir}: {"; ".join(failed)}. Halt for review.')
    LOG.info('PASS full fit: %s | n=%s Rhat=%.4f bulkESS=%.0f tailESS=%.0f divergences=%s',
             fit_dir, expected_n, d['rhat_max'], d['ess_bulk_min'],
             d['ess_tail_min'], d['divergences'])


def check_output(path: Path, expected_n: int, *, checksum_manifest: Path, checksum_key: str) -> None:
    if not path.exists() or not checksum_manifest.exists():
        raise RuntimeError(f'Incomplete output/manifest {path}; inspect manually, do not overwrite.')
    m = json.loads(checksum_manifest.read_text(encoding='utf-8'))
    if m.get('n', m.get('n_eligible')) != expected_n or m[checksum_key] != sha256(path):
        raise RuntimeError(f'Count/hash mismatch: {path}')


def main() -> int:
    parser = argparse.ArgumentParser(description='Run M1-G2 full-city B0/B1 fits, scores and comparisons for 3 cities, 2 gamma scenarios')
    parser.add_argument('--city', choices=[*CITIES, 'all'], default='all', help='Default: all three cities')
    parser.add_argument('--skip-smoke', action='store_true', help='Skip recommended small Donostia smoke tests')
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = verify_prerequisites()
    run_dir = ROOT / 'outputs' / 'phase4' / '_m1g2_runner'
    run_dir.mkdir(parents=True, exist_ok=True)
    logfile = run_dir / 'm1g2_runner.log'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(logfile, encoding='utf-8')])
    LOG.info('M1-G2 | original engine SHA256 %s | config SHA256 %s', sha256(ENGINE), sha256(CONFIG))
    cities = CITIES if args.city == 'all' else (args.city,)
    for city in cities:
        run('--city', city, '--check', log_path=logfile)
        dest = ROOT / 'outputs' / 'phase4' / city / 'm1g_citywide_v0.1'
        manifest = dest / 'preparation_manifest.json'
        if not manifest.exists():
            if dest.exists() and any(dest.glob('*_scenario.parquet')):
                raise RuntimeError(f'Partial preparation in {dest}; inspect rather than overwrite')
            run('--city', city, '--prepare', log_path=logfile)
        else:
            LOG.info('Already prepared; engine verifies SHA256/provenance on every fit or score: %s', city)

        if city == 'DON' and not args.skip_smoke:
            smoke_root = dest / 'smoke_gamma_neg0p8'
            for model in ('B0', 'B1'):
                if (smoke_root / model / 'fit_diagnostics.json').exists():
                    LOG.info('Smoke output already exists: %s', model)
                elif (smoke_root / model).exists():
                    raise RuntimeError(f'Partial smoke directory: {smoke_root / model}')
                else:
                    run('--city', city, '--smoke', '--gamma', '-0.8', '--model', model, log_path=logfile)
            LOG.info('Smoke succeeded (not counted as full city fit).')
        for gamma, label in GAMMAS:
            scenario = dest / label
            for model in ('B0', 'B1'):
                fitdir = scenario / model
                if not fitdir.exists():
                    run('--city', city, '--fit', '--gamma', str(gamma), '--model', model, log_path=logfile)
                check_fit(fitdir, cfg['cities'][city]['expected_eligible'], cfg['diagnostics'])
                score = fitdir / 'citywide_decisions.parquet'
                if not score.exists():
                    run('--city', city, '--score', '--gamma', str(gamma), '--model', model, log_path=logfile)
                check_output(score, cfg['cities'][city]['expected_eligible'],
                             checksum_manifest=fitdir / 'scoring_manifest.json', checksum_key='result_sha256')
            comparison = scenario / 'citywide_comparison.parquet'
            if not comparison.exists():
                run('--city', city, '--compare', '--gamma', str(gamma), log_path=logfile)
            check_output(comparison, cfg['cities'][city]['expected_eligible'],
                         checksum_manifest=scenario / 'comparison_manifest.json', checksum_key='comparison_sha256')
            LOG.info('COMPLETE: %s %s | %s', city, label, comparison.relative_to(ROOT))
    LOG.info('ALL REQUESTED CITY/SCENARIO ANALYSES COMPLETE. These are synthetic comparisons, not Phase 3 observed validations or map tiles.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, KeyError, ValueError) as exc:
        logging.getLogger('m1g2.runner').error('STOP: %s', exc)
        raise SystemExit(1) from exc
