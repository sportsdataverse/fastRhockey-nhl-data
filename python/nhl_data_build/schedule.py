"""Season schedule — fetch from fastRhockey-nhl-raw, enrich with per-dataset
completion flags, write + publish as ``nhl_schedules``.

Port of ``nhl_data_creation.R``'s STEP 1 (schedule fetch, lines 447-468) and
STEP 5 (completion flags, lines 644-694) -- the two pieces of the R producer
that never got a Python equivalent when this repo cut over to Python compile.
``build.py``/``season.py``'s DATASETS-driven extraction has no schedule source
of its own: the schedule is a fetch of the raw repo's own compiled file, not
something extracted from ``final/{gid}.json``.

Confirmed live 2026-09-09: without this, the release's ``nhl_schedule_{year}``
files for seasons 2014-2018 were never recompiled after fastRhockey-nhl-raw
renamed its own schedule files start-year -> end-year (2026-04-08,
afdddf9e6) -- each held the prior season's content, and 2013-14 was entirely
missing from the release.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import polars as pl
from sportsdataverse._rds import write_rds

from nhl_data_build.config import DATASETS
from nhl_data_build.fetch import RAW_BASE, _default_opener
from nhl_data_build.flatten import prepare_for_parquet
from nhl_data_build.season import RDS_CLASS as _RDS_CLASS

# Every family a schedule row can be flagged against: the 15 DATASETS keys
# plus the derived player_box. pbp_lite mirrors pbp's game_id set 1:1, so it
# gets no flag of its own -- matches the R tribble, which never had one either.
_FLAG_KEYS: list[str] = [key for key, *_ in DATASETS] + ["player_box"]


def _sanitize_utf8(table: "object") -> "object":
    """Strip schema/field metadata and replace invalid UTF-8 column bytes
    before polars sees any of it.

    Confirmed live 2026-09-09 on season 2010, in two rounds: (1) R/arrow can
    write genuinely non-UTF-8 bytes into a Utf8 COLUMN's data, and (2) even
    after fixing that, the SAME panic recurred from the schema/field-level
    metadata dict (the latin1 key metadata fetch_schedule's docstring already
    called out for the ``polars.read_parquet`` path -- it turns out
    ``pl.from_arrow`` hits the identical FFI schema-conversion code, not a
    separate, safer path). Both are pyarrow-tolerant, polars-Rust-fatal: a
    Rust ``panic!`` (``pyo3_runtime.PanicException``) that is NOT a subclass
    of ``Exception`` and killed the whole process straight through a bare
    ``except Exception`` wrapper in season.py, aborting every remaining
    season in a multi-season compile run. Metadata isn't needed for this
    dataset's columns, so it's simplest to drop it entirely rather than
    attempt to re-encode it; casting each string column to binary is a
    zero-copy reinterpret (no validation), so its bytes can be decoded
    leniently in plain Python instead of Rust's strict validator.
    """
    import pyarrow as pa

    cols = []
    fields = []
    for i, field in enumerate(table.schema):
        col = table.column(i)
        clean_field = field.remove_metadata()
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            raw_bytes = col.cast(pa.binary())
            cleaned = [b.decode("utf-8", "replace") if b is not None else None for b in raw_bytes.to_pylist()]
            cols.append(pa.array(cleaned, type=pa.string()))
        else:
            cols.append(col)
        fields.append(clean_field)
    clean_schema = pa.schema(fields)  # no table-level metadata either
    return pa.Table.from_arrays(cols, schema=clean_schema)


def fetch_schedule(
    season_end_year: int,
    *,
    raw_base: str = RAW_BASE,
    opener: Callable[[str], bytes | None] | None = None,
) -> pl.DataFrame:
    """The raw repo's own compiled schedule for one season (all columns).

    pyarrow, not polars.read_parquet: these files are R/arrow-written and can
    carry non-UTF-8 metadata/column bytes that trip polars' strict Rust-level
    ingestion (see fetch.py, and ``_sanitize_utf8`` above for the full story --
    it turns out pyarrow-then-``pl.from_arrow`` isn't actually a safe bypass on
    its own, contrary to what this docstring originally assumed).
    """
    import pyarrow.parquet as pq

    op = opener or _default_opener
    url = f"{raw_base}/nhl/schedules/parquet/nhl_schedule_{season_end_year}.parquet"
    raw = op(url)
    if raw is None:
        raise FileNotFoundError(f"schedule parquet not found: {url}")
    table = _sanitize_utf8(pq.read_table(io.BytesIO(raw)))
    return pl.from_arrow(table)


# R's tribble names every flag column after its DATASETS key verbatim except
# "pbp", which it stamps "PBP" (nhl_data_creation.R:661, and the same spelling
# is baked into fastRhockey::.nhl_attach_data_flags()'s flag_cols) -- match it
# so a consumer reading nhl_schedule_{year}.parquet across years sees one
# consistent schema, not a two-name drift for just the recompiled seasons.
_FLAG_COLUMN_NAME = {"pbp": "PBP"}


def add_completion_flags(sched: pl.DataFrame, season: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Port of ``nhl_data_creation.R`` STEP 5 -- one boolean column per dataset
    family, true where that game's data made it into this season's compile."""
    sched = sched.with_columns(pl.col("game_id").cast(pl.Int64))
    exprs = []
    for key in _FLAG_KEYS:
        df = season.get(key)
        ids: set[int] = set()
        if df is not None and "game_id" in df.columns:
            ids = {int(v) for v in df["game_id"].unique().to_list() if v is not None}
        col_name = _FLAG_COLUMN_NAME.get(key, key)
        exprs.append(pl.col("game_id").is_in(sorted(ids)).alias(col_name))
    return sched.with_columns(exprs).unique().sort("game_date", descending=True)


def write_schedule(df: pl.DataFrame, out_dir: str | Path, season_year: int) -> int:
    """Write to ``{out_dir}/schedules/{parquet,rds,csv}/nhl_schedule_{year}.*`` --
    the exact path ``fetch.py`` and the raw repo's own convention expect."""
    if df.height == 0:
        return 0
    flat = prepare_for_parquet(df, "schedules")
    out = Path(out_dir)
    for sub in ("parquet", "rds", "csv"):
        (out / "schedules" / sub).mkdir(parents=True, exist_ok=True)
    stem = f"nhl_schedule_{season_year}"
    flat.write_parquet(out / "schedules" / "parquet" / f"{stem}.parquet", compression="gzip")
    write_rds(
        flat,
        out / "schedules" / "rds" / f"{stem}.rds",
        cls=_RDS_CLASS,
        attributes={
            "fastRhockey_timestamp": datetime.now(timezone.utc),
            "fastRhockey_type": "NHL schedule",
        },
    )
    flat.write_csv(out / "schedules" / "csv" / f"{stem}.csv")
    return df.height


def build_and_write_schedule(
    season: dict[str, pl.DataFrame],
    out_dir: str | Path,
    season_year: int,
    *,
    raw_base: str = RAW_BASE,
) -> int:
    sched = fetch_schedule(season_year, raw_base=raw_base)
    enriched = add_completion_flags(sched, season)
    return write_schedule(enriched, out_dir, season_year)
