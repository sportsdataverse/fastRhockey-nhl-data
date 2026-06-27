"""Python port of the fastRhockey-nhl-data reshaper + xG model training.

Reads the per-game ``final.json`` produced by the (Python) NHL scraper and reshapes it
into the season-level datasets published to the ``sportsdataverse-data`` ``nhl_*``
releases (consumer half of NHL Python self-sufficiency, cfb-data-style dual-write), and
retrains the xG models (``xg_train``) that the scraper's inference path applies.
"""
from nhl_data_build.build import build_season, pbp_lite
from nhl_data_build.config import DATASETS
from nhl_data_build.extract import extract_all
from nhl_data_build.xg_train import prepare_training_frame, train_xg_models

__all__ = ["extract_all", "build_season", "pbp_lite", "DATASETS", "train_xg_models", "prepare_training_frame"]
