"""Publish season dataset parquets to the ``sportsdataverse-data`` ``nhl_*`` releases.

Port of ``nhl_data_creation.R``'s ``.upload_to_release`` — a thin ``gh release upload``
wrapper (Python's equivalent of ``sportsdataversedata::sportsdataverse_save``). Each
dataset's ``{prefix}_{season}.parquet`` is uploaded (``--clobber``) to its release tag;
the release must already exist on ``sportsdataverse-data`` (created once, like the R side).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from nhl_data_build.config import DATASETS

_REPO = "sportsdataverse/sportsdataverse-data"

# (file_prefix, release_tag, dataset_key) for every published dataset incl. the derived ones.
_PUBLISH: list[tuple[str, str, str]] = [(prefix, tag, key) for key, _field, prefix, tag in DATASETS]
_PUBLISH += [
    ("play_by_play_lite", "nhl_pbp_lite", "pbp_lite"),
    ("player_box", "nhl_player_boxscores", "player_box"),
]


def publish_file(path: Path, release_tag: str, *, repo: str = _REPO) -> None:
    """Upload one file to a release tag (``gh release upload --clobber``)."""
    subprocess.run(
        ["gh", "release", "upload", release_tag, str(path), "--repo", repo, "--clobber"],
        check=True,
    )


def publish_season(
    out_dir: str | Path, season_year: int, *, repo: str = _REPO, dry_run: bool = False
) -> list[tuple[str, str]]:
    """Upload every present ``{prefix}_{season_year}.parquet`` to its ``nhl_*`` release."""
    out = Path(out_dir)
    done: list[tuple[str, str]] = []
    for prefix, tag, key in _PUBLISH:
        path = out / key / "parquet" / f"{prefix}_{season_year}.parquet"
        if not path.exists():
            continue
        if dry_run:
            print(f"[dry-run] {path.name} -> {repo}@{tag}")
        else:
            publish_file(path, tag, repo=repo)
        done.append((tag, path.name))
    return done
