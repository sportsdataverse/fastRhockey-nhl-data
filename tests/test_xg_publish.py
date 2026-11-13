"""The nhl_xg_models release is built from the committed champion and carries its own manifest.

Consumers (fastRhockey, sportsdataverse-py) check cached files against the
manifest's sha256s instead of hashes pinned in their source, so the manifest
must describe exactly the bytes uploaded, and it must land after them.
"""

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from nhl_data_build import publish, xg_publish

MODELS = Path(__file__).resolve().parents[1] / "models"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_manifest_describes_exactly_the_built_assets(tmp_path):
    m = xg_publish.build(MODELS, tmp_path, source="test@0")
    assert sorted(m["files"]) == ["feats_5v5.txt", "feats_st.txt", "xg_model_5v5.json", "xg_model_st.json"]
    for name, digest in m["files"].items():
        assert _sha(tmp_path / name) == digest, name
    assert _sha(tmp_path / "xg_model_5v5.json") == _sha(MODELS / "xg_model_5v5.json")
    on_disk = json.loads((tmp_path / "xg_model_meta.json").read_text(encoding="utf-8"))
    assert on_disk == m
    assert m["xg_model_ps"] == json.loads((MODELS / "xg_model_meta.json").read_text(encoding="utf-8"))["xg_model_ps"]
    assert m["version"] == max(m["trained_at"].values())[:10]
    # the R scorer selects columns in feats order: it must be the booster's own order, LF-only
    for v in ("5v5", "st"):
        raw = (tmp_path / f"feats_{v}.txt").read_bytes()
        assert b"\r" not in raw
        names = json.loads((MODELS / f"xg_model_{v}.json").read_bytes())["learner"]["feature_names"]
        assert raw.decode().splitlines() == names


def test_a_sidecar_from_another_fit_is_refused(tmp_path):
    models = tmp_path / "models"
    shutil.copytree(MODELS, models, ignore=shutil.ignore_patterns("ledger.jsonl", "*.md", "*.yaml", "*.rds"))
    meta = json.loads((models / "xg_model_meta.json").read_text(encoding="utf-8"))
    meta["xg_feature_names_st"] = list(reversed(meta["xg_feature_names_st"]))
    (models / "xg_model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="st: booster feature names differ"):
        xg_publish.build(models, tmp_path / "out", source="test@0")


def _fake_release(monkeypatch, *, corrupt: str | None = None):
    calls, store = [], {}

    def gh(args):
        calls.append(args)
        if args[:2] == ["release", "upload"]:
            p = Path(args[3])
            store[p.name] = "sha256:" + ("0" * 64 if p.name == corrupt else _sha(p))
        if args[0] == "api":
            return "".join(f"{n}\t{d}\n" for n, d in store.items())
        return ""

    monkeypatch.setattr(publish, "_gh", gh)
    return calls


def test_upload_sends_the_manifest_after_every_file_it_names(tmp_path, monkeypatch):
    m = xg_publish.build(MODELS, tmp_path, source="test@0")
    calls = _fake_release(monkeypatch)
    xg_publish.upload(tmp_path, m)
    uploaded = [Path(c[3]).name for c in calls if c[:2] == ["release", "upload"]]
    assert uploaded.index("xg_model_meta.json") > max(uploaded.index(n) for n in m["files"])
    assert all(c[2] == "nhl_xg_models" for c in calls if c[:2] == ["release", "upload"])
    assert "--clobber" in calls[0]


def test_upload_fails_when_the_release_does_not_hold_the_uploaded_bytes(tmp_path, monkeypatch):
    m = xg_publish.build(MODELS, tmp_path, source="test@0")
    _fake_release(monkeypatch, corrupt="xg_model_st.json")
    with pytest.raises(RuntimeError, match="xg_model_st.json"):
        xg_publish.upload(tmp_path, m)
