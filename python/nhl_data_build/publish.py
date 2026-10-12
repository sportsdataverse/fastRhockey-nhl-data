"""Publish season dataset parquets to the ``sportsdataverse-data`` ``nhl_*`` releases.

Port of ``nhl_data_creation.R``'s ``.upload_to_release`` — a thin ``gh release upload``
wrapper (Python's equivalent of ``sportsdataversedata::sportsdataverse_save``). Each
dataset's ``{prefix}_{season}.parquet`` is uploaded (``--clobber``) to its release tag;
the release must already exist on ``sportsdataverse-data`` (created once, like the R side).
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from sportsdataverse.release import upload_release_sidecars

from nhl_data_build.config import DATASETS

_REPO = "sportsdataverse/sportsdataverse-data"

# (file_prefix, release_tag, dataset_key) for every published dataset incl. the derived ones.
_PUBLISH: list[tuple[str, str, str]] = [(prefix, tag, key) for key, _field, prefix, tag, _desc in DATASETS]
_PUBLISH += [
    ("play_by_play_lite", "nhl_pbp_lite", "pbp_lite"),
    ("player_box", "nhl_player_boxscores", "player_box"),
    ("nhl_schedule", "nhl_schedules", "schedules"),
]


#: Release sidecar metadata. Every published tag carries package_function.txt/.json
#: naming the loader a consumer reads it through -- the half of R's
#: sportsdataverse_save() this module's port dropped. Each value is the exact
#: string the R producer already published to that tag (read back off the
#: release assets), so re-stamping from Python does not change what a consumer
#: sees. Spelled out rather than derived from the tag: these are consumer-facing
#: strings, and a future tag whose loader breaks the naming pattern would
#: otherwise ship a name that does not resolve.
PKG_FUNCTION: dict[str, str] = {
    "nhl_game_info": "fastRhockey::load_nhl_game_info()",
    "nhl_game_rosters": "fastRhockey::load_nhl_game_rosters()",
    "nhl_goalie_boxscores": "fastRhockey::load_nhl_goalie_boxscores()",
    "nhl_linescore": "fastRhockey::load_nhl_linescore()",
    "nhl_officials": "fastRhockey::load_nhl_officials()",
    "nhl_pbp_full": "fastRhockey::load_nhl_pbp_full()",
    "nhl_pbp_lite": "fastRhockey::load_nhl_pbp_lite()",
    "nhl_penalties": "fastRhockey::load_nhl_penalties()",
    "nhl_player_boxscores": "fastRhockey::load_nhl_player_boxscores()",
    "nhl_schedules": "fastRhockey::load_nhl_schedule()",
    "nhl_scoring": "fastRhockey::load_nhl_scoring()",
    "nhl_scratches": "fastRhockey::load_nhl_scratches()",
    "nhl_shifts": "fastRhockey::load_nhl_shifts()",
    "nhl_shootout": "fastRhockey::load_nhl_shootout()",
    "nhl_shots_by_period": "fastRhockey::load_nhl_shots_by_period()",
    "nhl_skater_boxscores": "fastRhockey::load_nhl_skater_boxscores()",
    "nhl_team_boxscores": "fastRhockey::load_nhl_team_boxscores()",
    "nhl_three_stars": "fastRhockey::load_nhl_three_stars()",
}


def _gh(args: list[str]) -> str:
    """Single ``gh`` chokepoint -- tests monkeypatch this to stay offline. Returns stdout."""
    try:
        return subprocess.run(
            ["gh", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,  # fail fast instead of hanging the daily workflow on a stuck upload/auth prompt
        ).stdout
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr or "")
        raise


def _release_digests(tag: str, repo: str) -> dict[str, str] | None:
    """``{asset name: "sha256:<hex>"}`` for *tag* in ONE API call; ``None`` = no release.

    The release object embeds every asset with a GitHub-computed ``digest``, so one
    listing per tag replaces a per-file probe. The droplet shares one 5,000/h quota
    and the old upload-everything run cost ~500 requests per season.
    """
    try:
        out = _gh(["api", f"repos/{repo}/releases/tags/{tag}", "--jq", r'.assets[] | "\(.name)\t\(.digest)"'])
    except subprocess.CalledProcessError as exc:
        if "404" in (exc.stderr or ""):
            return None  # first publish of this tag
        raise  # auth / rate limit must surface
    return dict(ln.split("\t", 1) for ln in (out or "").splitlines() if "\t" in ln)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()  # not hashlib.file_digest: requires-python is >=3.10
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def publish_file(path: Path, release_tag: str, *, repo: str = _REPO) -> None:
    """Upload one file to a release tag (``gh release upload --clobber``)."""
    _gh(["release", "upload", release_tag, str(path), "--repo", repo, "--clobber"])


def publish_season(
    out_dir: str | Path, season_year: int, *, repo: str = _REPO, dry_run: bool = False
) -> list[tuple[str, str]]:
    """Upload every changed ``{prefix}_{season_year}.{parquet,rds,csv}`` to its ``nhl_*`` release.

    All three released formats ship to the tag -- the release is the distribution
    channel (rds/csv are build artifacts, not committed to this repo). Returns the
    ``(tag, asset)`` pairs uploaded (all present files on a dry run).

    One release listing per tag; a file whose sha256 matches its asset is skipped.
    The rds embeds a write timestamp so it never matches itself: when the parquet
    matches and the rds/csv are already on the release, the whole dataset is
    skipped. The parquet uploads LAST, so a matching parquet means the previous
    upload of its siblings completed.
    """
    out = Path(out_dir)
    done: list[tuple[str, str]] = []
    for prefix, tag, key in _PUBLISH:
        paths = [
            p
            for sub in ("rds", "csv", "parquet")  # parquet last
            if (p := out / key / sub / f"{prefix}_{season_year}.{sub}").exists()
        ]
        if not paths:
            continue
        if dry_run:
            for path in paths:
                print(f"[dry-run] {path.name} -> {repo}@{tag}")
                done.append((tag, path.name))
            continue
        remote = _release_digests(tag, repo)
        if remote is None:
            _gh(["release", "create", tag, "--repo", repo, "--title", tag, "--notes", f"{tag} (fastRhockey-nhl-data)"])
            remote = {}
        pq = paths[-1] if paths[-1].suffix == ".parquet" else None
        if pq is not None and remote.get(pq.name) == _sha256(pq) and all(p.name in remote for p in paths):
            continue  # unchanged dataset: rds/csv are derived from this parquet
        uploaded = 0
        for path in paths:
            if path is not pq and remote.get(path.name) == _sha256(path):
                continue
            publish_file(path, tag, repo=repo)
            uploaded += 1
            done.append((tag, path.name))
        # stamp LAST so the timestamp describes a finished upload, and only when
        # something actually uploaded -- a stamp on a no-op run would claim data
        # moved when it did not
        if uploaded:
            upload_release_sidecars(tag, runner=_gh, pkg_function=PKG_FUNCTION.get(tag), repo=repo)
    return done
