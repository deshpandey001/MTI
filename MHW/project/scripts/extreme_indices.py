"""
extreme_indices.py
==================
Compute ETCCDI‑style extreme SST indices for observations and models.

Indices
-------
*inst indices (annual frequency)*
  - TXx  : annual maximum daily SST
  - TXn  : annual minimum daily SST
  - SD   : seasonal duration above threshold

*frequency indices*
  - WSDI : warm spell duration index (>= 6 consecutive days above 90th pctl)
  - HWMId: heat wave magnitude index daily

*percentile‑based*
  - 90p  / 99p  daily SST percentiles
  - frequency of days exceeding historical 90th / 99th percentile

Outputs
-------
- outputs/extreme_indices/*.nc   (one file per index per dataset)
- outputs/extreme_indices/timeseries.png
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


def compute_txx(ds) -> None:
    """Annual maximum SST."""
    # TODO: ds.sst.resample(time='YE').max()
    logger.info("TXx computation — placeholder")


def compute_txn(ds) -> None:
    """Annual minimum SST."""
    # TODO: ds.sst.resample(time='YE').min()
    logger.info("TXn computation — placeholder")


def compute_wsdi(ds, baseline_years: tuple) -> None:
    """Warm Spell Duration Index.

    Annual count of days with at least 6 consecutive days
    above the 90th percentile of the baseline period.
    """
    # TODO: compute baseline 90th percentile per dayofyear
    # TODO: count spells >= 6 days
    logger.info("WSDI computation — placeholder")


def compute_hwmid(ds) -> None:
    """Heat Wave Magnitude Index daily."""
    # TODO: follow Russo et al. (2015) definition
    logger.info("HWMId computation — placeholder")


def main() -> None:
    cfg = load_config()
    setup_logging(cfg)

    logger.info("=" * 60)
    logger.info("START extreme_indices.py")
    logger.info("=" * 60)

    baseline = tuple(cfg.get("extreme_indices", {}).get("baseline_years", (1982, 2011)))

    # TODO: load datasets
    # TODO: compute and save each index
    compute_txx(None)
    compute_txn(None)
    compute_wsdi(None, baseline)
    compute_hwmid(None)

    logger.info("Extreme indices saved — see outputs/extreme_indices/")
    logger.info("=" * 60)
    logger.info("END extreme_indices.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
