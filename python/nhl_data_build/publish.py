"""Publish season dataset parquets to the ``sportsdataverse-data`` ``nhl_*`` releases.

Port of ``nhl_data_creation.R``'s ``.upload_to_release`` — a thin ``gh release upload``
wrapper (Python's equivalent of ``sportsdataversedata::sportsdataverse_save``). Each
dataset's ``{prefix}_{season}.parquet`` is uploaded (``--clobber``) to its release tag;
the release must already exist on ``sportsdataverse-data`` (created once, like the R side).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from sportsdataverse.release import upload_release_sidecars

from nhl_data_build.config import DATASETS

_REPO = "sportsdataverse/sportsdataverse-data"

# (file_prefix, release_tag, dataset_key) for every published dataset incl. the derived ones.
_PUBLISH: list[tuple[str, str, str]] = [(prefix, tag, key) for key, _field, prefix, tag, _desc in DATASETS]
_PUBLISH += [
    ("play_by_play_lite", "nhl_pbp_lite", "pbp_lite"),
    ("player_box", "nhl_player_boxscores", "player_box"),
]


#: Release sidecar metadata. Every published tag carries package_function.txt/.json
#: naming the loader a consumer reads it through -- the half of R's
#: sportsdataverse_save() this module's port dropped. fastRhockey names every one
#: of these loaders after its tag, and the derivation was checked against the
#: package_function.json already published to all 17 tags, so re-stamping from
#: Python does not change what a consumer sees.
PKG_FUNCTION: dict[str, str] = {tag: f"fastRhockey::load_{tag}()" for _p, tag, _k in _PUBLISH}


def _gh(args: list[str]) -> None:
    """Single ``gh`` chokepoint -- tests monkeypatch this to stay offline."""
    subprocess.run(
        ["gh", *args],
        check=True,
        timeout=600,  # fail fast instead of hanging the daily workflow on a stuck upload/auth prompt
    )


def publish_file(path: Path, release_tag: str, *, repo: str = _REPO) -> None:
    """Upload one file to a release tag (``gh release upload --clobber``)."""
    _gh(["release", "upload", release_tag, str(path), "--repo", repo, "--clobber"])


def publish_season(
    out_dir: str | Path, season_year: int, *, repo: str = _REPO, dry_run: bool = False
) -> list[tuple[str, str]]:
    """Upload every present ``{prefix}_{season_year}.{parquet,rds,csv}`` to its ``nhl_*`` release.

    All three released formats ship to the tag — the release is the distribution
    channel (rds/csv are build artifacts, not committed to this repo).
    """
    out = Path(out_dir)
    done: list[tuple[str, str]] = []
    for prefix, tag, key in _PUBLISH:
        uploaded = 0
        for sub, ext in (("parquet", "parquet"), ("rds", "rds"), ("csv", "csv")):
            path = out / key / sub / f"{prefix}_{season_year}.{ext}"
            if not path.exists():
                continue
            if dry_run:
                print(f"[dry-run] {path.name} -> {repo}@{tag}")
            else:
                publish_file(path, tag, repo=repo)
                uploaded += 1
            done.append((tag, path.name))
        # stamp LAST so the timestamp describes a finished upload, and only when
        # something actually uploaded -- a stamp on a no-op run would claim data
        # moved when it did not
        if uploaded:
            upload_release_sidecars(tag, runner=_gh, pkg_function=PKG_FUNCTION.get(tag), repo=repo)
    return done
