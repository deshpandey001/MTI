"""
regrid.py
=========
Regrid CMIP6 model data onto the NOAA OISST grid using conservative
or bilinear interpolation (via xESMF).

Outputs
-------
- data/processed/cmip6_*_regridded.nc
"""

import logging
import logging.config
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def setup_logging(config: dict) -> None:
    level = config.get("logging", {}).get("level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str = "config/default.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config %s not found — using empty config", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg or {}


def build_target_grid(cfg: dict) -> None:
    """Construct the target (NOAA OISST) lat‑lon grid."""
    # TODO: read lat/lon from a sample OISST file; return xr.Dataset with grid
    logger.info("Target grid construction — placeholder")


def regrid_cmip6(cfg: dict, scenario: str = "historical") -> None:
    """Regrid a CMIP6 scenario dataset to the OISST grid."""
    # TODO: open OISST target grid (data/processed/noaa_oisst.nc)
    # TODO: open CMIP6 native grid data
    # TODO: create xESMF regridder (conservative_normed / bilinear)
    # TODO: apply regridder, write to data/processed/
    logger.info("Regridding CMIP6 %s — placeholder", scenario)


def main() -> None:
    cfg = load_config()
    setup_logging(cfg)

    logger.info("=" * 60)
    logger.info("START regrid.py")
    logger.info("=" * 60)

    build_target_grid(cfg)
    regrid_cmip6(cfg, scenario="historical")
    regrid_cmip6(cfg, scenario="future")

    logger.info("=" * 60)
    logger.info("END regrid.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
