"""Player bio (``shoots_catches``) from the ``nhl-raw`` bio index.

Handedness is the one player attribute that is NOT derivable from play-by-play,
and it is what the off-wing read needs: a left-shooting player on the right side
gets a different look at the net than a same-side shooter, and no coordinate
column implies it. Measured on the xG corpus it is worth +0.0021 LOSO AUC,
positive in 16 of 16 held-out seasons.

Source is ``nhl-raw``'s ``nhl/nhl_player_bio.parquet`` -- ONE file, ~30 KB,
rebuilt by that repo's stage 02 on every capture run. Deliberately not the 3,300+
per-player payloads it is derived from: this repo does not check ``nhl-raw`` out
(see ``fetch.RAW_BASE``), so reading the payloads would be 3,300+ HTTP requests
on every daily build.

Join contract: ``player_id`` is Utf8 on both sides, pinned here. The roster frame
stores it as an integer and the index keys on the payload's string id, so the
INTEGER is cast to Utf8 (never a float -- ``8478402.0`` matches nothing).
"""

from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

import polars as pl

from nhl_data_build.fetch import RAW_BASE, _default_opener

INDEX_NAME = "nhl_player_bio.parquet"
INDEX_URL = f"{RAW_BASE}/nhl/{INDEX_NAME}"

#: Sibling checkout, when there is one -- a dev box reads it locally and offline.
LOCAL_INDEX = Path(__file__).resolve().parents[3] / "fastRhockey-nhl-raw" / "nhl" / INDEX_NAME

SCHEMA: dict[str, pl.DataType] = {
    "player_id": pl.Utf8,
    "shoots_catches": pl.Utf8,
    "position_code_bio": pl.Utf8,
    "height_inches": pl.Int64,
    "weight_pounds": pl.Int64,
    "birth_date": pl.Utf8,
    "birth_country": pl.Utf8,
}

#: The index ships ``position_code``, which the roster frame already uses for the
#: player's position IN THAT GAME. Renamed so the join can never silently shadow it.
_RENAME = {"position_code": "position_code_bio"}


def resolve_source(source: str | Path | None = None) -> str:
    """Explicit arg > ``NHL_PLAYER_BIO`` env > sibling checkout > the raw repo URL."""
    if source:
        return str(source)
    if os.environ.get("NHL_PLAYER_BIO"):
        return os.environ["NHL_PLAYER_BIO"]
    return str(LOCAL_INDEX) if LOCAL_INDEX.is_file() else INDEX_URL


@lru_cache(maxsize=4)
def _load_cached(src: str) -> pl.DataFrame:
    """Read + normalise one source. Cached: a multi-season run
    (``-s 2010 -e 2026``) attaches per season and would otherwise re-download the
    same index 17 times. Keyed on the RESOLVED source so two spellings of the same
    default share an entry. Frames are only ever selected/joined downstream, never
    mutated in place, so one instance is safe to share.
    """
    try:
        if src.startswith("http://"):
            # Never read bio over cleartext. This frame decides a published
            # column, so a MITM could flip shoots_catches for every player in
            # the release; https is available for the only remote source we ship.
            return pl.DataFrame(schema=SCHEMA)
        if src.startswith("https://"):
            body = _default_opener(src)
            if not body:
                return pl.DataFrame(schema=SCHEMA)
            df = pl.read_parquet(io.BytesIO(body))
        else:
            if not Path(src).is_file():
                return pl.DataFrame(schema=SCHEMA)
            df = pl.read_parquet(src)

        df = df.rename({k: v for k, v in _RENAME.items() if k in df.columns})
        if "player_id" not in df.columns:
            # No join key = nothing joinable. Caught here rather than left to
            # raise on the cast below, which sits in the season build's path and
            # would take the whole run down over an optional enrichment.
            return pl.DataFrame(schema=SCHEMA)
        keep = [c for c in SCHEMA if c in df.columns]
        # One row per player, enforced here rather than trusted from upstream: a
        # duplicated key would fan the LEFT join out and multiply roster rows,
        # silently inflating the published dataset.
        return df.select(keep).with_columns(pl.col("player_id").cast(pl.Utf8)).unique(
            subset=["player_id"], keep="first"
        )
    except Exception:
        return pl.DataFrame(schema=SCHEMA)


def load_player_bio(source: str | Path | None = None) -> pl.DataFrame:
    """The bio index as a frame. An unreachable source is EMPTY, never an error.

    A missing index must degrade the roster dataset to null handedness, not fail
    the season build: this is an enrichment, and the 17 other families in the run
    do not depend on it.
    """
    return _load_cached(resolve_source(source))


def attach_player_bio(
    rosters: pl.DataFrame,
    bio: pl.DataFrame | None = None,
    *,
    source: str | Path | None = None,
    columns: tuple[str, ...] = ("shoots_catches",),
) -> pl.DataFrame:
    """Left-join bio columns onto a roster frame, keyed on ``player_id``.

    Left join on purpose: a roster row for a player with no captured bio keeps its
    row and gets a null. Dropping roster rows to satisfy a join would turn an
    enrichment into data loss.
    """
    want = [c for c in columns if c in SCHEMA and c != "player_id"]
    if not want or "player_id" not in rosters.columns:
        return rosters
    if bio is None:
        bio = load_player_bio(source)

    if bio.height == 0 or rosters.height == 0:
        # Still add the columns, so the dataset's schema does not change shape
        # depending on whether the index happened to be reachable.
        return rosters.with_columns([pl.lit(None, dtype=SCHEMA[c]).alias(c) for c in want if c not in rosters.columns])

    # A partial index must not change the OUTPUT schema. Without this, a bio frame
    # that happens to lack shoots_catches drops the column entirely, while the
    # empty-index path above adds a typed null -- so the dataset's columns would
    # depend on the shape of an optional input.
    missing = [c for c in want if c not in bio.columns]
    if missing:
        bio = bio.with_columns([pl.lit(None, dtype=SCHEMA[c]).alias(c) for c in missing])

    left = rosters.with_columns(_pid=pl.col("player_id").cast(pl.Int64, strict=False).cast(pl.Utf8))
    right = bio.select(["player_id", *want]).rename({"player_id": "_pid"})
    assert left.schema["_pid"] == right.schema["_pid"], "join-key dtype mismatch on _pid"
    # m:1 asserts what the loader already guarantees. Cheap, and it fails loudly
    # rather than quietly emitting duplicated roster rows if that ever breaks.
    return left.join(right, on="_pid", how="left", validate="m:1").drop("_pid")


def coverage(rosters: pl.DataFrame, column: str = "shoots_catches") -> dict:
    """How much of the frame the join actually reached -- reported, not assumed.

    A join that matches 3% looks identical to one that matches 97% unless someone
    counts, and a silently-degraded enrichment is the failure mode this guards.
    """
    if rosters.height == 0 or column not in rosters.columns:
        return {"rows": rosters.height, "matched": 0, "pct": 0.0}
    matched = int(rosters[column].is_not_null().sum())
    return {"rows": rosters.height, "matched": matched, "pct": round(100.0 * matched / rosters.height, 2)}
