"""
mhw_detection.py
================
Detect Marine Heatwaves using the Hobday et al. (2016) definition.

Algorithm
---------
1. Compute daily climatology (30‑year baseline, 11‑day window).
2. Compute threshold (90th percentile of the same window).
3. Identify events where SST > threshold for >= 5 consecutive days.
4. Categorise events (Moderate, Strong, Severe, Extreme).

Outputs
-------
- outputs/mhw/mhw_events_{obs,model,scenario}.csv
- outputs/mhw/mhw_summary_stats.nc
- outputs/mhw/category_maps/
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


def compute_climatology(ds, baseline_years: tuple) -> None:
    """Compute daily SST climatology and 90th‑percentile threshold.

    Parameters
    ----------
    ds : xr.Dataset
        SST data with a 'time' dimension.
    baseline_years : tuple of int
        (start, end) for the baseline period.
    """
    # TODO: select baseline period, group by dayofyear
    # TODO: apply 11‑day window smoothing (rolling)
    # TODO: compute mean (clim) and 90th percentile (threshold)
    logger.info("Climatology computation — placeholder")
    return None, None


def detect_events(ds, clim, threshold) -> None:
    """Label contiguous exceedances as MHW events."""
    # TODO: mask where sst > threshold
    # TODO: find contiguous blocks >= 5 days (xr.DataArray with scipy.ndimage)
    # TODO: compute duration, max intensity, cumulative intensity, category
    logger.info("MHW event detection — placeholder")


def summarise_events(events_ds) -> None:
    """Aggregate per‑grid‑cell event statistics over the full record."""
    # TODO: total events, mean duration, mean max intensity, etc.
    # TODO: save CSV summary and NetCDF with per‑cell stats
    logger.info("Event summary — placeholder")


def plot_category_maps(events_ds) -> None:
    """Map of maximum category reached per grid cell."""
    # TODO: create map figure, save to outputs/mhw/category_maps/
    logger.info("Category map plotting — placeholder")


def main() -> None:
    cfg = load_config()
    setup_logging(cfg)

    logger.info("=" * 60)
    logger.info("START mhw_detection.py")
    logger.info("=" * 60)

    baseline = tuple(cfg.get("mhw", {}).get("baseline_years", (1982, 2011)))

    # TODO: load each dataset (obs, hist_bc, future_bc) from data/processed/
    # TODO: loop over datasets:
    #   clim, thresh = compute_climatology(ds, baseline)
    #   events = detect_events(ds, clim, thresh)
    #   summarise_events(events)
    #   plot_category_maps(events)

    logger.info("MHW detection complete — see outputs/mhw/")
    logger.info("=" * 60)
    logger.info("END mhw_detection.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
