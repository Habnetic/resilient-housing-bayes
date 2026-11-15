# Execute from the root of resilient-housing-bayes with activated Python 3.11 .venv.
$ErrorActionPreference = 'Stop'
if (-not (Test-Path '.\configs\m1g_citywide_v0.1.json')) {
    throw 'Open PowerShell in the resilient-housing-bayes repository root with M1-G2 v0.1 installed.'
}
$env:PYTHONPATH = (Resolve-Path '.\src').Path
python -m pytest tests/test_m1g.py tests/test_m1g_citywide.py -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed. Stopping.' }
python .\scripts\run_m1g2_all.py
if ($LASTEXITCODE -ne 0) { throw 'Full-city runner stopped. Inspect outputs/phase4/_m1g2_runner/m1g2_runner.log; restart to resume safe completed steps.' }
Write-Host 'M1-G2 finished for DON, RTM and HAM; gamma -0.8 and 0; B0/B1 fit/score/compare.'
