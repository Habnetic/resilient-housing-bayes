# Habnetic | Phase 4 M1 Archive Manifest

**Status:** Local archive verified; durable off-machine storage pending confirmation.
**Manifest generated (UTC):** 2026-11-05T22:43:23Z
**Closure commit:** `f2b1a8d`
**Repository HEAD when verified:** `f2b1a8d`
**Branch at closure:** `phase4/m1-topography`
**Scope:** Rotterdam M1-A through M1-F, controlled synthetic topographic validation only.

## Computational output archive

- Source directory: `outputs/phase4/RTM/` (Git-ignored).
- Archive location: `../m1-archives/Habnetic_Phase4_M1_Outputs_v1.0.zip` (relative to repository root).
- Archive name: `Habnetic_Phase4_M1_Outputs_v1.0.zip`.
- Archive SHA-256: `f18e802d4283069bc4aa8cd96d884e20df0f0e54606f2e390512cd3f7f541462`.
- ZIP size: 33699125 bytes.
- Archived files: 761.
- Total uncompressed file size: 55509296 bytes.
- Integrity validation: all ZIP file entries were individually SHA-256 checked against the current source files; paths, count and uncompressed byte total match.

## Principal external inputs

| Input path (relative to repository root) | Size (bytes) | SHA-256 |
| --- | ---: | --- |
| `outputs/phase3/RTM/phase3_features_scaled.parquet` | 22630561 | `3f948d7aa1a0a1d7c8e7cf1b2f9d82dbba76cb75d2a79a26c8c57a395157fd04` |
| `../data/runs/rtm_m1_citywide/terrain_features_m1_5m.parquet` | 11514691 | `242045b495f547ddaa56f9e130415841831ca85f89de584ff44b3038d0c34711` |

These input Parquet files are **not** included in the M1 output ZIP. Preserve separate, verified copies of the source datasets and any required upstream generation inputs.

## Scientific evidence and reproduction

- Experimental summary: `docs/phase4/M1_Methodological_Review.md`.
- Computational confirmation: `docs/phase4/M1_E_Results.md`.
- Population robustness: `docs/phase4/M1_F_Results.md`.
- Closure statement and limitations: `docs/phase4/M1_Closure_Report.md`.
- M1-C input scenarios: `outputs/phase4/RTM/m1c/m1c_scenarios.parquet` (inside the output archive).
- M1-F generated scenarios and manifests: `outputs/phase4/RTM/m1f/` (inside the output archive).
- The outputs are synthetic research evidence. They do not validate actual flood damage or geographical transfer.

## Restore and verify

1. Check out the tagged M1 code and verify the manifest's source hashes against the matching input datasets.
2. Extract the ZIP at repository root so it recreates `outputs/phase4/RTM/`.
3. Validate the ZIP SHA-256 against this manifest before extracting.
4. Run the targeted tests, e.g. `.\.venv\Scripts\python.exe -m pytest tests/test_m1c.py tests/test_m1d.py tests/test_m1d2.py tests/test_m1e.py tests/test_m1f.py -q`, in the pinned project environment.

## Storage and release conditions

- Local working archive: verified at the path above.
- Durable storage (Drive / research archive): **NOT YET RECORDED**. Add an immutable location and recheck the uploaded SHA-256 before claiming durable preservation.
- Source datasets: **NOT BACKED UP BY THIS MANIFEST**. Their hashes identify versions; preserve the bytes separately.
- Permanent tag `phase4-m1-complete-v1.0`: create only after the durable archive and required inputs are preserved and verified.

**What to test next:** HAM/DON feature portability and cross-city transfer are separate research checkpoints, not demonstrated by this archive.
