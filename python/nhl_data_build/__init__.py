"""Python port of the fastRhockey-nhl-data reshaper.

Reads the per-game ``final.json`` produced by the (Python) NHL scraper and reshapes it
into the season-level datasets published to the ``sportsdataverse-data`` ``nhl_*``
releases — the consumer side of NHL Python self-sufficiency (cfb-data-style dual-write).
"""

from nhl_data_build.build import build_season
from nhl_data_build.config import DATASETS
from nhl_data_build.extract import extract_all

__all__ = ["extract_all", "build_season", "DATASETS"]
