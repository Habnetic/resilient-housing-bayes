# Uses repository-specific Python explicitly, ignoring any mistakenly activated data/.venv.
$ErrorActionPreference = 'Stop'
$repo = $PSScriptRoot
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "Missing project interpreter: $python" }
Set-Location $repo
$env:PYTHONPATH = (Join-Path $repo 'src')
Write-Host "M1-G2 GIS audit using: $python"
& $python --version
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python -m pytest tests/test_m1g2_gis_audit.py -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python scripts/audit_m1g2_gis.py @args
exit $LASTEXITCODE

