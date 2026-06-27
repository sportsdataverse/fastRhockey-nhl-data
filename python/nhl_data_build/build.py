"""Season-level dataset compile — Python port of ``nhl_data_creation.R``'s pivot+bind step.

``build_season`` extracts every game, then concatenates per dataset key (R's
``bind_rows |> distinct()``) into one frame per dataset.
"""

from __future__ import annotations

import polars as pl

from nhl_data_build.extract import extract_all


def pbp_lite(pbp: pl.DataFrame) -> pl.DataFrame:
    """Port of the ``pbp_lite`` derive — full pbp minus CHANGE (shift) events."""
    return pbp.filter(pl.col("event_type") != "CHANGE")


def build_season(game_jsons: list[dict]) -> dict[str, pl.DataFrame]:
    """Extract each game and concat per dataset key into season-level frames."""
    extracts = [extract_all(g) for g in game_jsons]
    keys: set[str] = set()
    for e in extracts:
        keys |= e.keys()

    out: dict[str, pl.DataFrame] = {}
    for key in keys:
        rows: list[dict] = []
        for e in extracts:
            rows += e.get(key) or []
        if rows:
            # diagonal-relaxed so player_box (skater+goalie column union) binds cleanly.
            frames = [pl.DataFrame([r], infer_schema_length=None) for r in rows] if key == "player_box" else None
            df = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame(rows, infer_schema_length=None)
            out[key] = df.unique(maintain_order=True)
    return out
