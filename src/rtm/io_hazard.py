from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import pandas as pd


@dataclass(frozen=True)
class HazardPaths:
    """Resolved paths to RTM pluvial hazard artifacts produced in Habnetic/data."""
    data_root: Path
    hazard_buildings: Path
    hazard_grid: Path


def _find_repo_root(start: Path) -> Path:
    """Walk upwards to locate repo root (pyproject/README/.git)."""
    for p in [start, *start.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "README.md").exists():
            return p
    return start


def resolve_habnetic_data_root() -> Path:
    """
    Resolve Habnetic/data repo root.

    Priority:
    1) HABNETIC_DATA env var (points directly to .../Habnetic/data)
    2) HABNETIC_ROOT env var (points to .../Habnetic) -> append /data
    3) Assume sibling repo: ../data relative to resilient-housing-bayes root
    """
    env_data = os.environ.get("HABNETIC_DATA")
    if env_data:
        return Path(env_data).expanduser().resolve()

    env_root = os.environ.get("HABNETIC_ROOT")
    if env_root:
        return (Path(env_root).expanduser().resolve() / "data").resolve()

    here = Path(__file__).resolve()
    r = _find_repo_root(here)
    # src/rtm/io_hazard.py -> repo root
    repo_root = r
    candidate = (repo_root.parent / "data").resolve()
    return candidate


def get_rtm_pluvial_v1_paths(data_root: Path | None = None) -> HazardPaths:
    """
    Build canonical paths for RTM pluvial hazard v1 artifacts.
    """
    data_root = (data_root or resolve_habnetic_data_root()).resolve()

    hazard_buildings = data_root / "processed" / "RTM" / "hazards" / "pluvial" / "H_pluvial_v1_buildings.parquet"
    hazard_grid = data_root / "processed" / "RTM" / "hazards" / "pluvial" / "H_pluvial_v1_grid.nc"

    return HazardPaths(
        data_root=data_root,
        hazard_buildings=hazard_buildings,
        hazard_grid=hazard_grid,
    )


def load_rtm_pluvial_v1_buildings(
    data_root: Path | None = None,
    *,
    expect_rows: int = 221_324,
    require_complete: bool = True,
) -> pd.DataFrame:
    """
    Load building-level deterministic pluvial hazard intensity H_pluvial_v1_mm.

    Expected schema:
    - bldg_id (stable building id)
    - H_pluvial_v1_mm (float, millimeters)

    Validation:
    - file exists
    - required columns present
    - bldg_id unique
    - (optional) row count matches expected_rows
    - (optional) no NaNs
    """
    paths = get_rtm_pluvial_v1_paths(data_root)
    p = paths.hazard_buildings

    if not p.exists():
        raise FileNotFoundError(
            f"Missing hazard parquet:\n  {p}\n"
            f"Resolved Habnetic/data root as:\n  {paths.data_root}\n"
            "Set HABNETIC_DATA or HABNETIC_ROOT env var if this is wrong."
        )

    df = pd.read_parquet(p)

    needed = {"bldg_id", "H_pluvial_v1_mm"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Hazard parquet missing columns: {sorted(missing)}")

    if df["bldg_id"].duplicated().any():
        ndup = int(df["bldg_id"].duplicated().sum())
        raise ValueError(f"Hazard parquet has duplicated bldg_id rows: {ndup}")

    if expect_rows is not None and len(df) != expect_rows:
        raise ValueError(f"Row count mismatch: got {len(df)}, expected {expect_rows}")

    if require_complete:
        n_nan = int(df["H_pluvial_v1_mm"].isna().sum())
        if n_nan:
            raise ValueError(f"H_pluvial_v1_mm contains NaNs: {n_nan}")

    return df[["bldg_id", "H_pluvial_v1_mm"]].copy()