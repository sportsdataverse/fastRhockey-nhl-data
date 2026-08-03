"""Season compile driver — read the raw repo's ``final/{gid}.json`` → datasets → parquet.

Port of the per-season loop in ``nhl_data_creation.R``: read every per-game ``final.json``
(the output of the Python scraper in ``fastRhockey-nhl-raw``), build the season-level
datasets, flatten nested columns, and write parquet per dataset (+ ``pbp_lite`` and the
combined ``player_box``). Publish to the ``sportsdataverse-data`` ``nhl_*`` releases is a
follow-up (left to a thin ``gh release upload`` wrapper / CI).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from sportsdataverse._rds import write_rds

from nhl_data_build.build import build_season, pbp_lite
from nhl_data_build.config import DATASETS, TYPES
from nhl_data_build.flatten import prepare_for_parquet

# R writes these with a plain ``saveRDS(df)`` (R/nhl_data_creation.R:320) — no
# custom S3 stamp — so mirror a plain data.frame rather than a ``*_data`` class.
# The stamp ``fastRhockey:::make_fastRhockey_data()`` applies (fastRhockey/R/utils.R:364).
#
# R's own producer writes a bare ``saveRDS(df, compress = "xz")``, so the released files
# carry none of this — and ``rbindlist_with_attrs()`` (utils.R:400) reads
# fastRhockey_timestamp / fastRhockey_type off the last season's file to re-attach after
# the bind, so every load_nhl_*() call currently gets NULL for both and prints a blank
# header. Stamping here populates them for the first time.
RDS_CLASS: tuple[str, ...] = (
    "fastRhockey_data",
    "tbl_df",
    "tbl",
    "data.table",
    "data.frame",
)


def _season_start(game_id: int) -> int:
    return int(str(game_id)[:4])


def iter_final_dir(final_dir: str | Path, season_end_year: int | None = None) -> Iterator[dict]:
    """Yield each ``final/{gid}.json`` in ``final_dir`` one at a time (optionally one season).

    A generator (not a list) so a full season can be compiled without holding every game's
    parsed JSON in memory at once — the key to fitting a standard CI runner."""
    d = Path(final_dir)
    if not d.is_dir():  # fail fast on a bad --final-dir rather than silently compiling 0 games
        raise FileNotFoundError(f"final dir not found: {final_dir}")
    for p in sorted(d.glob("*.json")):
        try:
            gid = int(p.stem)
        except ValueError:
            continue
        if season_end_year is not None and _season_start(gid) != season_end_year - 1:
            continue
        try:
            yield json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue


def read_final_dir(final_dir: str | Path, season_end_year: int | None = None) -> list[dict]:
    """Eager list of every ``final/{gid}.json`` (thin wrapper over :func:`iter_final_dir`)."""
    return list(iter_final_dir(final_dir, season_end_year))


def build_season_from_dir(final_dir: str | Path, season_end_year: int | None = None) -> dict[str, pl.DataFrame]:
    """Stream final JSONs from a local dir and compile the season datasets (memory-bounded)."""
    return build_season(iter_final_dir(final_dir, season_end_year))


def write_datasets(
    season: dict[str, pl.DataFrame],
    out_dir: str | Path,
    season_year: int,
    timestamp: datetime | None = None,
) -> dict[str, int]:
    """Flatten + write each dataset to ``{out_dir}/{key}/{parquet,rds,csv}/{prefix}_{year}.*``.

    All three released formats are written from the same flattened frame, so they
    share one schema. ``rds``/``csv`` are release artifacts — they ship to the tag
    and are not committed to this repo.

    Each rds carries the ``make_fastRhockey_data`` stamp: the fastRhockey_data class
    chain plus fastRhockey_timestamp / fastRhockey_type. ``timestamp`` defaults to now
    and is injectable so tests can assert a fixed value.
    """
    out = Path(out_dir)
    written: dict[str, int] = {}
    stamped_at = timestamp or datetime.now(timezone.utc)

    def _write(df: pl.DataFrame, key: str, prefix: str) -> None:
        if df is None or df.height == 0:
            return
        flat = prepare_for_parquet(df, key)
        for sub in ("parquet", "rds", "csv"):
            (out / key / sub).mkdir(parents=True, exist_ok=True)
        stem = f"{prefix}_{season_year}"
        flat.write_parquet(out / key / "parquet" / f"{stem}.parquet", compression="gzip")
        write_rds(
            flat,
            out / key / "rds" / f"{stem}.rds",
            cls=RDS_CLASS,
            attributes={
                # one timestamp for the whole season run, as R's single Sys.time() would be
                "fastRhockey_timestamp": stamped_at,
                "fastRhockey_type": TYPES.get(key, key),
            },
        )
        flat.write_csv(out / key / "csv" / f"{stem}.csv")
        written[key] = df.height

    for key, _field, prefix, _tag, _desc in DATASETS:
        _write(season.get(key), key, prefix)
    if "pbp" in season:
        _write(pbp_lite(season["pbp"]), "pbp_lite", "play_by_play_lite")
    _write(season.get("player_box"), "player_box", "player_box")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m nhl_data_build.season",
        description="Compile NHL season datasets from the raw repo's final JSONs.",
    )
    ap.add_argument("-s", "--start", type=int, required=True, help="start season end-year (e.g. 2025 = 2024-25)")
    ap.add_argument("-e", "--end", type=int, help="end season end-year (default: --start)")
    ap.add_argument("--final-dir", required=True, help="dir of final/{gid}.json (fastRhockey-nhl-raw nhl/json/final)")
    ap.add_argument("--out-dir", default="nhl")
    args = ap.parse_args(argv)
    for year in range(args.start, (args.end or args.start) + 1):
        season = build_season_from_dir(args.final_dir, season_end_year=year)
        written = write_datasets(season, args.out_dir, year)
        print(f"season {year}: {sum(written.values())} rows across {len(written)} datasets -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
