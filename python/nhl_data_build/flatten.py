"""Nested-column flattening for parquet — port of ``.flatten_struct_cols``.

Canonical R source: ``nhl_data_creation.R`` — ``.flatten_struct_cols`` + the per-dataset
``shots_by_period`` rename. Arrow can't write nested struct columns, so the nested NHL JSON
objects (localized ``{default, cs, ...}`` names, ``periodDescriptor``, ``committedByPlayer``)
are unfolded into dotted scalar columns; list columns (e.g. scoring ``assists``) are
JSON-stringified.
"""

from __future__ import annotations

import json

import polars as pl

# shots_by_period: periodDescriptor.* -> clean snake_case (R main-loop rename, keep all).
_SBP_RENAME = {
    "periodDescriptor.number": "period",
    "periodDescriptor.periodType": "period_type",
    "periodDescriptor.maxRegulationPeriods": "max_regulation_periods",
    "periodDescriptor.otPeriods": "ot_periods",
}


def flatten_struct_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Unnest struct columns into dotted-name scalar columns; JSON-stringify list columns."""
    for _ in range(5):
        structs = [c for c, t in df.schema.items() if isinstance(t, pl.Struct)]
        if not structs:
            break
        for c in structs:
            renamed = [f"{c}.{f.name}" for f in df.schema[c].fields]
            df = df.with_columns(pl.col(c).struct.rename_fields(renamed)).unnest(c)

    list_cols = [c for c, t in df.schema.items() if isinstance(t, pl.List)]
    for c in list_cols:
        df = df.with_columns(
            pl.col(c).map_elements(lambda v: None if v is None else json.dumps(v, default=str), return_dtype=pl.Utf8)
        )
    return df


def prepare_for_parquet(df: pl.DataFrame, key: str) -> pl.DataFrame:
    """Flatten + apply the per-dataset tidy step (the ``shots_by_period`` rename)."""
    df = flatten_struct_cols(df)
    if key == "shots_by_period":
        df = df.rename({k: v for k, v in _SBP_RENAME.items() if k in df.columns})
    return df
