# Habnetic | Phase 4 M1 artifact archive and provenance manifest
# Run from any PowerShell directory:
#   & .\scripts\finalize_m1_archive.ps1
# Requires Windows PowerShell 5.1+ or PowerShell 7+, Git and Compress-Archive.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$outputDir = Join-Path $repoRoot 'outputs\phase4\RTM'
$archiveDir = [System.IO.Path]::GetFullPath((Join-Path $repoRoot '..\m1-archives'))
$archivePath = Join-Path $archiveDir 'Habnetic_Phase4_M1_Outputs_v1.0.zip'
$manifestPath = Join-Path $repoRoot 'docs\phase4\M1_Archive_Manifest.md'

if (-not (Test-Path -LiteralPath $outputDir -PathType Container)) {
    throw "M1 output directory not found: $outputDir"
}

# Inputs are external to the M1 ZIP. A checksum is not a backup.
$sourceItems = @(
    @{
        RelativePath = 'outputs/phase3/RTM/phase3_features_scaled.parquet'
        AbsolutePath = (Join-Path $repoRoot 'outputs\phase3\RTM\phase3_features_scaled.parquet')
    },
    @{
        RelativePath = '../data/runs/rtm_m1_citywide/terrain_features_m1_5m.parquet'
        AbsolutePath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot '..\data\runs\rtm_m1_citywide\terrain_features_m1_5m.parquet'))
    }
)

foreach ($source in $sourceItems) {
    if (-not (Test-Path -LiteralPath $source.AbsolutePath -PathType Leaf)) {
        throw "Required input missing: $($source.RelativePath). Update the script with the correct path before proceeding."
    }
}

# Preserve a previously created archive; create it only if absent.
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
    Write-Host "Creating archive: $archivePath"
    Compress-Archive -Path $outputDir -DestinationPath $archivePath -CompressionLevel Optimal
}
else {
    Write-Host "Using existing archive: $archivePath"
}

# Independently verify every archived file against the current directory.
# This detects outdated archives as well as missing, extra or corrupted entries.
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$files = @(Get-ChildItem -LiteralPath $outputDir -Recurse -File)
$expected = @{}
$expectedBytes = [long]0
foreach ($file in $files) {
    $relative = $file.FullName.Substring($outputDir.Length).TrimStart('\', '/') -replace '\\', '/'
    $zipName = "RTM/$relative"
    if ($expected.ContainsKey($zipName)) {
        throw "Duplicate archive member: $zipName"
    }
    $expected[$zipName] = $file.FullName
    $expectedBytes += $file.Length
}

$sha = [System.Security.Cryptography.SHA256]::Create()
$zip = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
$verifiedCount = 0
$verifiedBytes = [long]0
try {
    foreach ($entry in $zip.Entries) {
        $name = $entry.FullName.Replace('\', '/')
        if ($name.EndsWith('/')) { continue }
        if (-not $expected.ContainsKey($name)) {
            throw "Archive contains a file absent from source directory: $name"
        }
        $localFile = $expected[$name]
        $localHash = (Get-FileHash -LiteralPath $localFile -Algorithm SHA256).Hash
        $stream = $entry.Open()
        try {
            $archiveHash = [System.BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '')
        }
        finally {
            $stream.Dispose()
        }
        if ($localHash -ne $archiveHash) {
            throw "Archive byte-content mismatch: $name. Do not commit the manifest until corrected."
        }
        $verifiedCount++
        $verifiedBytes += $entry.Length
    }
}
finally {
    $zip.Dispose()
    $sha.Dispose()
}

if ($verifiedCount -ne $files.Count -or $verifiedBytes -ne $expectedBytes) {
    throw "Archive verification failed. Source=$($files.Count) files/$expectedBytes bytes; ZIP=$verifiedCount files/$verifiedBytes bytes."
}

$archiveFile = Get-Item -LiteralPath $archivePath
$archiveHash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
$gitHead = (& git -C $repoRoot rev-parse --short HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Unable to identify the current Git commit.' }
$utc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
$bt = [char]96

$lines = New-Object 'System.Collections.Generic.List[string]'
$lines.Add('# Habnetic | Phase 4 M1 Archive Manifest')
$lines.Add('')
$lines.Add('**Status:** Local archive verified; durable off-machine storage pending confirmation.')
$lines.Add("**Manifest generated (UTC):** $utc")
$lines.Add("**Closure commit:** ${bt}f2b1a8d${bt}")
$lines.Add("**Repository HEAD when verified:** $bt$gitHead$bt")
$lines.Add("**Branch at closure:** ${bt}phase4/m1-topography${bt}")
$lines.Add("**Scope:** Rotterdam M1-A through M1-F, controlled synthetic topographic validation only.")
$lines.Add('')
$lines.Add('## Computational output archive')
$lines.Add('')
$lines.Add("- Source directory: ${bt}outputs/phase4/RTM/${bt} (Git-ignored).")
$lines.Add("- Archive location: ${bt}../m1-archives/Habnetic_Phase4_M1_Outputs_v1.0.zip${bt} (relative to repository root).")
$lines.Add("- Archive name: ${bt}$($archiveFile.Name)${bt}.")
$lines.Add("- Archive SHA-256: ${bt}$archiveHash${bt}.")
$lines.Add("- ZIP size: $($archiveFile.Length) bytes.")
$lines.Add("- Archived files: $verifiedCount.")
$lines.Add("- Total uncompressed file size: $verifiedBytes bytes.")
$lines.Add('- Integrity validation: all ZIP file entries were individually SHA-256 checked against the current source files; paths, count and uncompressed byte total match.')
$lines.Add('')
$lines.Add('## Principal external inputs')
$lines.Add('')
$lines.Add('| Input path (relative to repository root) | Size (bytes) | SHA-256 |')
$lines.Add('| --- | ---: | --- |')
foreach ($source in $sourceItems) {
    $item = Get-Item -LiteralPath $source.AbsolutePath
    $hash = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $lines.Add("| $bt$($source.RelativePath)$bt | $($item.Length) | $bt$hash$bt |")
}
$lines.Add('')
$lines.Add('These input Parquet files are **not** included in the M1 output ZIP. Preserve separate, verified copies of the source datasets and any required upstream generation inputs.')
$lines.Add('')
$lines.Add('## Scientific evidence and reproduction')
$lines.Add('')
$lines.Add("- Experimental summary: ${bt}docs/phase4/M1_Methodological_Review.md${bt}.")
$lines.Add("- Computational confirmation: ${bt}docs/phase4/M1_E_Results.md${bt}.")
$lines.Add("- Population robustness: ${bt}docs/phase4/M1_F_Results.md${bt}.")
$lines.Add("- Closure statement and limitations: ${bt}docs/phase4/M1_Closure_Report.md${bt}.")
$lines.Add("- M1-C input scenarios: ${bt}outputs/phase4/RTM/m1c/m1c_scenarios.parquet${bt} (inside the output archive).")
$lines.Add("- M1-F generated scenarios and manifests: ${bt}outputs/phase4/RTM/m1f/${bt} (inside the output archive).")
$lines.Add('- The outputs are synthetic research evidence. They do not validate actual flood damage or geographical transfer.')
$lines.Add('')
$lines.Add('## Restore and verify')
$lines.Add('')
$lines.Add("1. Check out the tagged M1 code and verify the manifest's source hashes against the matching input datasets.")
$lines.Add("2. Extract the ZIP at repository root so it recreates ${bt}outputs/phase4/RTM/${bt}.")
$lines.Add("3. Validate the ZIP SHA-256 against this manifest before extracting.")
$lines.Add("4. Run the targeted tests, e.g. ${bt}.\.venv\Scripts\python.exe -m pytest tests/test_m1c.py tests/test_m1d.py tests/test_m1d2.py tests/test_m1e.py tests/test_m1f.py -q${bt}, in the pinned project environment.")
$lines.Add('')
$lines.Add('## Storage and release conditions')
$lines.Add('')
$lines.Add('- Local working archive: verified at the path above.')
$lines.Add('- Durable storage (Drive / research archive): **NOT YET RECORDED**. Add an immutable location and recheck the uploaded SHA-256 before claiming durable preservation.')
$lines.Add('- Source datasets: **NOT BACKED UP BY THIS MANIFEST**. Their hashes identify versions; preserve the bytes separately.')
$lines.Add("- Permanent tag ${bt}phase4-m1-complete-v1.0${bt}: create only after the durable archive and required inputs are preserved and verified.")
$lines.Add('')
$lines.Add('**What to test next:** HAM/DON feature portability and cross-city transfer are separate research checkpoints, not demonstrated by this archive.')

[System.IO.Directory]::CreateDirectory((Split-Path -Parent $manifestPath)) | Out-Null
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($manifestPath, ($lines -join "`n") + "`n", $utf8NoBom)

Write-Host "Verified: $verifiedCount files; $verifiedBytes uncompressed bytes; SHA-256 $archiveHash"
Write-Host "Wrote manifest: $manifestPath"
Write-Host 'Note: copy the ZIP and external input files to durable storage before creating the release tag.'
