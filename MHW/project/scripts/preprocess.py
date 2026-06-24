"""
preprocess.py
=============
Ingest and harmonize raw NOAA OISST and CMIP6 SST data.

Steps:
  1. Load raw NetCDF files from data/noaa/ and data/cmip6_*/
  2. Standardise time axes (daily / monthly), variable naming (sst -> tos),
     and units (Kelvin -> Celsius).
  3. Clip to common spatial domain and temporal overlap.
  4. Quality control / outlier rejection.
  5. Save cleaned, harmonised files to data/processed/.
"""

import logging
import logging.config
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def setup_logging(config: dict) -> None:
    """Configure logging from a dictionary (e.g. loaded from config)."""
    level = config.get("logging", {}).get("level", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("Logging initialised at %s level", level)


def load_config(config_path: str = "config/default.yaml") -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file %s not found — using empty config", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    logger.info("Configuration loaded from %s", path)
    return cfg or {}


def load_noaa_oisst(cfg: dict) -> None:
    """Load and clean NOAA OISST v2 data."""
    # TODO: use xarray to open NetCDF files from data/noaa/
    # TODO: rename dims/coords, convert to degC, select common period
    logger.info("NOAA OISST loading — placeholder")


def load_cmip6_data(cfg: dict, scenario: str = "historical") -> None:
    """Load CMIP6 model outputs for a given scenario.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    scenario : str
        One of 'historical' or 'future' (ssp245 / ssp585).
    """
    # TODO: glob data/cmip6_{scenario}/*.nc, open with xr.open_mfdataset
    # TODO: rename variable 'tos' -> 'sst', convert K -> degC
    logger.info("CMIP6 %s loading — placeholder", scenario)


def harmonise_datasets(cfg: dict) -> None:
    """Clip, regrid, and align all datasets to common grid & time period."""
    # TODO: interpolate CMIP6 to OISST grid (xr.interp / xesmf)
    # TODO: subset common time period
    # TODO: save to data/processed/{noaa_oisst, cmip6_historical, cmip6_future}.nc
    logger.info("Dataset harmonisation — placeholder")


def main() -> None:
    cfg = load_config()
    setup_logging(cfg)

    logger.info("=" * 60)
    logger.info("START preprocess.py")
    logger.info("=" * 60)

    load_noaa_oisst(cfg)
    load_cmip6_data(cfg, scenario="historical")
    load_cmip6_data(cfg, scenario="future")
    harmonise_datasets(cfg)

    logger.info("=" * 60)
    logger.info("END preprocess.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
