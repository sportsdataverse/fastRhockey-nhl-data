"""Publish mapping + dry-run (no network) — every dataset maps to an nhl_* release tag."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import polars as pl
import pytest
from nhl_data_build import publish
from nhl_data_build.config import DATASETS
from nhl_data_build.publish import _PUBLISH, publish_season


def test_publish_covers_all_datasets() -> None:
    keys = {key for _prefix, _tag, key in _PUBLISH}
    assert {k for k, _f, _p, _t, _d in DATASETS} <= keys  # every registry dataset
    assert {"pbp_lite", "player_box"} <= keys  # + the derived ones
    assert all(tag.startswith("nhl_") for _p, tag, _k in _PUBLISH)


def test_publish_season_dry_run(tmp_path: Path) -> None:
    for prefix, key in (("skater_box", "skater_box"), ("play_by_play_lite", "pbp_lite")):
        d = tmp_path / key / "parquet"
        d.mkdir(parents=True)
        pl.DataFrame({"x": [1]}).write_parquet(d / f"{prefix}_2025.parquet")
    done = publish_season(tmp_path, 2025, dry_run=True)
    tags = {t for t, _name in done}
    assert "nhl_skater_boxscores" in tags
    assert "nhl_pbp_lite" in tags


def _digest(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def _season(tmp_path: Path, rds: bytes = b"rds") -> tuple[str, list[Path]]:
    prefix, tag, key = _PUBLISH[0]
    files = []
    for sub, data in (("parquet", b"pq"), ("rds", rds), ("csv", b"csv")):
        f = tmp_path / key / sub / f"{prefix}_2025.{sub}"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
        files.append(f)
    return tag, files


def _mock_gh(monkeypatch, listing: dict[str, str] | None) -> list[list[str]]:
    """Fake gh: `api releases/tags` answers *listing* (None = 404); records every call."""
    calls: list[list[str]] = []

    def gh(args: list[str]) -> str:
        calls.append(args)
        if args[0] == "api":
            if listing is None:
                raise subprocess.CalledProcessError(1, "gh", stderr="gh: Not Found (HTTP 404)")
            return "".join(f"{n}\t{d}\n" for n, d in listing.items())
        return ""

    monkeypatch.setattr(publish, "_gh", gh)
    return calls


def _uploaded(calls: list[list[str]]) -> list[str]:
    return [Path(c[3]).name for c in calls if c[:2] == ["release", "upload"]]


def test_unchanged_parquet_skips_the_whole_dataset_even_when_rds_restamped(tmp_path, monkeypatch):
    tag, (pq, rds, csv) = _season(tmp_path)
    listing = {pq.name: _digest(pq), rds.name: "sha256:old-timestamp", csv.name: _digest(csv)}
    calls = _mock_gh(monkeypatch, listing)
    assert publish_season(tmp_path, 2025) == []
    assert [c[0] for c in calls] == ["api"]  # one listing, no upload, no stamp


def test_changed_parquet_uploads_changed_files_parquet_last_then_stamps(tmp_path, monkeypatch):
    tag, (pq, rds, csv) = _season(tmp_path)
    listing = {pq.name: "sha256:stale", rds.name: "sha256:stale", csv.name: _digest(csv)}
    calls = _mock_gh(monkeypatch, listing)
    done = publish_season(tmp_path, 2025)
    names = _uploaded(calls)
    assert names[:2] == [rds.name, pq.name]  # csv unchanged -> skipped
    assert names[2:] == ["timestamp.txt", "timestamp.json", "package_function.txt", "package_function.json"]
    assert done == [(tag, rds.name), (tag, pq.name)]


def test_missing_sibling_is_uploaded_even_when_parquet_matches(tmp_path, monkeypatch):
    _tag, (pq, rds, csv) = _season(tmp_path)
    calls = _mock_gh(monkeypatch, {pq.name: _digest(pq), csv.name: _digest(csv)})
    publish_season(tmp_path, 2025)
    assert _uploaded(calls)[:2] == [rds.name, pq.name]


def test_missing_release_is_created(tmp_path, monkeypatch):
    tag, _files = _season(tmp_path)
    calls = _mock_gh(monkeypatch, None)
    publish_season(tmp_path, 2025)
    assert ["release", "create", tag] == calls[1][:3]
    assert len(_uploaded(calls)) == 3 + 4


def test_listing_failure_other_than_404_raises(tmp_path, monkeypatch):
    _season(tmp_path)

    def gh(args: list[str]) -> str:
        raise subprocess.CalledProcessError(1, "gh", stderr="API rate limit exceeded (HTTP 403)")

    monkeypatch.setattr(publish, "_gh", gh)
    with pytest.raises(subprocess.CalledProcessError):
        publish_season(tmp_path, 2025)
