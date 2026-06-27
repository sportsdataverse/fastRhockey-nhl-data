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
from pathlib import Path

import polars as pl

from nhl_data_build.build import build_season, pbp_lite
from nhl_data_build.config import DATASETS
from nhl_data_build.flatten import prepare_for_parquet


def _season_start(game_id: int) -> int:
    return int(str(game_id)[:4])


def read_final_dir(final_dir: str | Path, season_end_year: int | None = None) -> list[dict]:
    """Load every ``final/{gid}.json`` in ``final_dir`` (optionally filtered to one season)."""
    d = Path(final_dir)
    if not d.is_dir():  # fail fast on a bad --final-dir rather than silently compiling 0 games
        raise FileNotFoundError(f"final dir not found: {final_dir}")
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            gid = int(p.stem)
        except ValueError:
            continue
        if season_end_year is not None and _season_start(gid) != season_end_year - 1:
            continue
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def build_season_from_dir(final_dir: str | Path, season_end_year: int | None = None) -> dict[str, pl.DataFrame]:
    """Read final JSONs from a local dir and compile the season datasets."""
    return build_season(read_final_dir(final_dir, season_end_year))


def write_datasets(season: dict[str, pl.DataFrame], out_dir: str | Path, season_year: int) -> dict[str, int]:
    """Flatten + write each dataset to ``{out_dir}/{key}/parquet/{prefix}_{year}.parquet``."""
    out = Path(out_dir)
    written: dict[str, int] = {}

    def _write(df: pl.DataFrame, key: str, prefix: str) -> None:
        if df is None or df.height == 0:
            return
        d = out / key / "parquet"
        d.mkdir(parents=True, exist_ok=True)
        prepare_for_parquet(df, key).write_parquet(d / f"{prefix}_{season_year}.parquet", compression="gzip")
        written[key] = df.height

    for key, _field, prefix, _tag in DATASETS:
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
