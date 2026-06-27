"""Season-level dataset compile — Python port of ``nhl_data_creation.R``'s pivot+bind step.

``build_season`` streams games in batches, converting each batch's extracted rows into compact
polars frames and concatenating per dataset key (R's ``bind_rows |> distinct()``). Streaming +
batching keeps peak memory bounded (≈ one batch of JSON + the accumulated columnar frames)
so a full season compiles on a standard CI runner rather than materializing every game's
row-dicts at once.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from nhl_data_build.extract import extract_all

_BATCH_SIZE = 250


def pbp_lite(pbp: pl.DataFrame) -> pl.DataFrame:
    """Port of the ``pbp_lite`` derive — full pbp minus CHANGE (shift) events."""
    return pbp.filter(pl.col("event_type") != "CHANGE")


def _rows_to_frame(rows: list[dict]) -> pl.DataFrame:
    # infer_schema_length=None scans the whole batch so heterogeneous dicts (e.g. player_box =
    # skater+goalie column union) bind cleanly without per-row frame construction.
    return pl.DataFrame(rows, infer_schema_length=None)


def build_season(game_jsons: Iterable[dict], *, batch_size: int = _BATCH_SIZE) -> dict[str, pl.DataFrame]:
    """Extract each game and concat per dataset key into season-level frames.

    Accepts any iterable of parsed ``final.json`` dicts (a list or a streaming generator). Games
    are processed in batches: each batch's rows are converted to one polars frame per key and the
    row-dicts freed, so peak memory tracks the compact columnar frames rather than every game's
    Python dicts at once. Cross-batch ``diagonal_relaxed`` concat reconciles column/type drift.
    """
    partial: dict[str, list[pl.DataFrame]] = {}
    batch: dict[str, list[dict]] = {}
    pending = 0

    def flush() -> None:
        for key, rows in batch.items():
            if rows:
                partial.setdefault(key, []).append(_rows_to_frame(rows))
        batch.clear()

    for g in game_jsons:
        for key, rows in extract_all(g).items():
            if rows:
                batch.setdefault(key, []).extend(rows)
        pending += 1
        if pending % batch_size == 0:
            flush()
    flush()

    out: dict[str, pl.DataFrame] = {}
    for key, frames in partial.items():
        df = frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")
        out[key] = df.unique(maintain_order=True)
    return out
