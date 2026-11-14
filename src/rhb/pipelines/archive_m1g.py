"""Archive completed local M1-G result bundles; never tags or publishes automatically.

    python -m rhb.pipelines.archive_m1g --city HAM --check
    python -m rhb.pipelines.archive_m1g --city HAM --archive-dir ../archives

An archive includes its own file inventory and is verified after writing. Raw
source elevation/Phase 3 assets are represented by fingerprints, not copied.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from rhb.experiments.m1f_population_sampling import sha256
from rhb.experiments.m1g_cross_city_sampling import CONFIG, ROOT, output_dir


def inventory(city: str) -> tuple[Path, list[Path]]:
    output = output_dir(city)
    summary_manifest = output / "full_summary/summary_manifest.json"
    run_manifest = output / "full_manifest.json"
    for p in (summary_manifest, run_manifest):
        if not p.is_file():
            raise FileNotFoundError(f"M1-G full run not ready for archive: {p}")
    summary = json.loads(summary_manifest.read_text(encoding="utf-8"))
    run = json.loads(run_manifest.read_text(encoding="utf-8"))
    if summary.get("cases") != 4 or summary.get("fits") != 8 or run.get("city") != city:
        raise ValueError("Not exactly four paired cases/eight fitted models")
    files = sorted(p for p in output.rglob("*") if p.is_file())
    if not files:
        raise ValueError("Empty experiment output")
    for p in files:
        if p.is_symlink():
            raise ValueError(f"Refuse symlink in research archive: {p}")
    return output, files


def archive_city(city: str, archive_dir: Path | None) -> None:
    base, files = inventory(city)
    records = [{"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size}
               for p in files]
    for p in (CONFIG, ROOT / "docs/phase4/M1_G_Protocol_and_Runbook.md"):
        records.append({"path": p.relative_to(ROOT).as_posix(), "sha256": sha256(p), "bytes": p.stat().st_size})
    print(f"M1-G {city} archive inventory PASS: {len(records)} hashed files")
    if archive_dir is None:
        return
    archive_dir.mkdir(parents=True, exist_ok=True)
    zip_path = archive_dir / f"Habnetic_M1G_{city}_synthetic_v0.1.zip"
    if zip_path.exists():
        raise FileExistsError(f"Refusing to replace existing frozen archive: {zip_path}")
    inventory_bytes = (json.dumps({
        "experiment": "M1-G", "city": city, "outcome_evidence": "synthetic",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
        "note": "Contains model artifacts and source fingerprints, not raw citywide input datasets. Not automatically scientifically approved.",
    }, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4,
                         allowZip64=True) as z:
        for record in records:
            z.write(ROOT / record["path"], arcname=record["path"])
        z.writestr("M1G_ARCHIVE_FILE_INVENTORY.json", inventory_bytes)
    with zipfile.ZipFile(zip_path) as z:
        if z.testzip() is not None:
            raise RuntimeError("ZIP CRC failure")
        for record in records:
            with z.open(record["path"]) as stream:
                import hashlib
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    h.update(block)
                if h.hexdigest() != record["sha256"]:
                    raise RuntimeError(f"Archived-file SHA-256 mismatch: {record['path']}")
    digest = sha256(zip_path)
    (Path(str(zip_path) + ".sha256")).write_text(f"{digest}  {zip_path.name}\n", encoding="utf-8")
    print("Archive:", zip_path)
    print("SHA256:", digest)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--city", choices=("HAM", "DON", "all"), required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--archive-dir", type=Path)
    args = p.parse_args()
    for city in (("HAM", "DON") if args.city == "all" else (args.city,)):
        archive_city(city, archive_dir=args.archive_dir)


if __name__ == "__main__":
    main()
