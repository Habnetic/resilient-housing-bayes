$ErrorActionPreference = "Stop"
$env:PYTHONPATH = (Resolve-Path .\src).Path

Write-Host "M1-G2 scientific audit: unit tests"
python -m pytest tests/test_m1g2_scientific_audit.py tests/test_m1g_citywide.py -q
if ($LASTEXITCODE -ne 0) { throw "Unit tests failed; audit not started." }

Write-Host "M1-G2 scientific audit: all cities / all scenarios (read-only frozen outputs)"
python .\scripts\audit_m1g2_full.py
if ($LASTEXITCODE -ne 0) { throw "Scientific audit failed; inspect the terminal and audit_failure.txt." }

Write-Host "Scientific audit report: .\outputs\phase4\_m1g2_scientific_audit_v0.1\AUDIT_REPORT.md"
