"""The release sidecars R's sportsdataverse_save() attaches to every tag.

publish.py is a port of that function and dropped this half, which left the
nhl_* tags carrying a timestamp.json frozen at the last R run while the data
kept moving -- a consumer reading it to decide whether to re-download got a
confident wrong answer.
"""

import json
from pathlib import Path

import pytest
from nhl_data_build import publish

SIDECAR_NAMES = [
    "timestamp.txt",
    "timestamp.json",
    "package_function.txt",
    "package_function.json",
]


def test_every_published_tag_names_a_loader():
    missing = sorted({tag for _p, tag, _k in publish._PUBLISH} - set(publish.PKG_FUNCTION))
    assert missing == [], f"tags with no PKG_FUNCTION entry: {missing}"
    # every value is the string the R producer published to that tag
    assert publish.PKG_FUNCTION["nhl_pbp_full"] == "fastRhockey::load_nhl_pbp_full()"


def test_publish_season_stamps_each_tag_last(tmp_path, monkeypatch):
    """The four sidecars land after that tag's data assets, once per tag."""
    prefix, tag, key = publish._PUBLISH[0]
    for sub, ext in (("parquet", "parquet"), ("rds", "rds")):
        path = tmp_path / key / sub / f"{prefix}_2025.{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    calls: list[list[str]] = []
    monkeypatch.setattr(publish, "_gh", calls.append)

    publish.publish_season(tmp_path, 2025)

    names = [Path(c[3]).name for c in calls if c[:2] == ["release", "upload"]]
    assert names == [f"{prefix}_2025.parquet", f"{prefix}_2025.rds", *SIDECAR_NAMES]
    assert all(c[2] == tag and c[-1] == "--clobber" for c in calls)


def test_no_files_means_no_stamp(tmp_path, monkeypatch):
    """A season with nothing on disk must not move any timestamp."""
    calls: list[list[str]] = []
    monkeypatch.setattr(publish, "_gh", calls.append)

    publish.publish_season(tmp_path, 2025)

    assert calls == []


def test_stamped_sidecars_carry_the_loader_and_a_timestamp(tmp_path, monkeypatch):
    seen: dict[str, str] = {}
    prefix, tag, key = publish._PUBLISH[0]
    path = tmp_path / key / "parquet" / f"{prefix}_2025.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")

    def _capture(argv: list[str]) -> None:
        # read inside the runner: the temp dir is cleaned up behind the upload
        p = Path(argv[3])
        if p.name.startswith(("timestamp.", "package_function.")):
            seen[p.name] = p.read_text()

    monkeypatch.setattr(publish, "_gh", _capture)
    publish.publish_season(tmp_path, 2025)

    assert seen["package_function.txt"].strip() == publish.PKG_FUNCTION[tag]
    assert json.loads(seen["package_function.json"])["package_function"] == publish.PKG_FUNCTION[tag]
    assert json.loads(seen["timestamp.json"])["last_updated"].strip()


@pytest.mark.parametrize(("tag", "expected"), sorted(publish.PKG_FUNCTION.items()))
def test_every_mapping_reaches_the_sidecar_verbatim(tmp_path, monkeypatch, tag, expected):
    """Pin every tag's loader name, not just the one the happy path uses.

    These strings ship to consumers as the canonical way to read the tag, so
    each one is asserted on the bytes that actually land.
    """
    seen: dict[str, str] = {}

    def _capture(argv: list[str]) -> None:
        path = Path(argv[3])
        if path.name.startswith(("timestamp.", "package_function.")):
            seen[path.name] = path.read_text()

    monkeypatch.setattr(publish, "_gh", _capture)
    publish.upload_release_sidecars(
        tag, runner=publish._gh, pkg_function=publish.PKG_FUNCTION[tag], repo="r/r"
    )

    assert seen["package_function.txt"].strip() == expected
    assert json.loads(seen["package_function.json"])["package_function"] == expected
