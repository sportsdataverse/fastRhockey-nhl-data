"""A failing gate must NOT replace the committed champion artifact.

Regression for the 2026-09-02 incident: the stage-01 5v5 retrain scored
cv_auc 0.7786 with ``gate_pass: false`` and still overwrote
``models/xg_model_5v5.json`` in place, because ``train_xg_models`` wrote
straight to the champion path and ``run_stage`` evaluated the gate afterwards.
``run_stage`` now trains into ``models/.candidate/<stage>/`` and copies files
across only on a pass.
"""

import hashlib
import json
from pathlib import Path

from nhl_data_build._stage import CANDIDATE_DIR, run_stage


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _champion(tmp_path: Path) -> tuple[Path, list[Path]]:
    models = tmp_path / "models"
    models.mkdir()
    (models / "xg_model_5v5.json").write_text("CHAMPION-BOOSTER", encoding="utf-8")
    (models / "xg_model_meta.json").write_text(
        json.dumps({"info_5v5": {"cv_auc": 0.8322}, "xg_feature_names_st": ["keep_me"]}), encoding="utf-8"
    )
    return models, [models / "xg_model_5v5.json", models / "xg_model_meta.json"]


def _write_candidate(out_dir: Path, tag: str) -> None:
    """Stand-in for train_xg_models: writes the artifacts into the dir it is handed."""
    (out_dir / "xg_model_5v5.json").write_text(f"{tag}-BOOSTER", encoding="utf-8")
    meta = json.loads((out_dir / "xg_model_meta.json").read_text(encoding="utf-8"))
    meta["info_5v5"] = {"cv_auc": 0.7786 if tag == "FAILED" else 0.84}
    (out_dir / "xg_model_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_failed_gate_leaves_the_champion_byte_identical(tmp_path, monkeypatch):
    models, artifacts = _champion(tmp_path)
    before = {a.name: _sha(a) for a in artifacts}
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr("nhl_data_build._stage.LEDGER", ledger)

    def train(out_dir: Path):
        _write_candidate(out_dir, "FAILED")
        return {"cv_auc": 0.7786, "gate_pass": False}

    rc = run_stage(
        name="xg_5v5",
        suite="nhl_data_build",
        config={"model": "xg_5v5"},
        artifacts=artifacts,
        train=train,
        force=True,
    )

    assert rc == 1
    assert {a.name: _sha(a) for a in artifacts} == before, "a failed gate replaced the champion"

    # the run is still RECORDED (ledger row with the failing gates) ...
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1 and rows[0]["gates"]["gate_pass"] is False
    assert rows[0]["gates"]["cv_auc"] == 0.7786

    # ... and the failed artifact is kept for inspection, not silently discarded.
    cand = models / CANDIDATE_DIR / "xg_5v5" / "xg_model_5v5.json"
    assert cand.is_file() and cand.read_text(encoding="utf-8") == "FAILED-BOOSTER"

    # the fingerprint is NOT recorded, so the next run retrains rather than skipping
    store = models / ".fingerprints.json"
    assert not store.is_file() or "xg_5v5" not in json.loads(store.read_text(encoding="utf-8"))


def test_passing_gate_promotes_and_preserves_the_sibling_variant(tmp_path, monkeypatch):
    models, artifacts = _champion(tmp_path)
    monkeypatch.setattr("nhl_data_build._stage.LEDGER", tmp_path / "ledger.jsonl")

    def train(out_dir: Path):
        # the candidate is seeded from the champion, so the single-variant sidecar
        # merge still sees the sibling's entries
        assert json.loads((out_dir / "xg_model_meta.json").read_text(encoding="utf-8"))["xg_feature_names_st"] == [
            "keep_me"
        ]
        _write_candidate(out_dir, "NEW")
        return {"cv_auc": 0.84, "gate_pass": True}

    rc = run_stage(
        name="xg_5v5",
        suite="nhl_data_build",
        config={"model": "xg_5v5"},
        artifacts=artifacts,
        train=train,
        force=True,
    )

    assert rc == 0
    assert (models / "xg_model_5v5.json").read_text(encoding="utf-8") == "NEW-BOOSTER"
    meta = json.loads((models / "xg_model_meta.json").read_text(encoding="utf-8"))
    assert meta["info_5v5"]["cv_auc"] == 0.84
    assert meta["xg_feature_names_st"] == ["keep_me"]  # sibling variant survived
    assert not (models / CANDIDATE_DIR / "xg_5v5").exists()  # cleaned on promotion
    assert "xg_5v5" in json.loads((models / ".fingerprints.json").read_text(encoding="utf-8"))


def test_missing_candidate_artifact_is_not_promoted(tmp_path, monkeypatch):
    models, artifacts = _champion(tmp_path)
    before = {a.name: _sha(a) for a in artifacts}
    monkeypatch.setattr("nhl_data_build._stage.LEDGER", tmp_path / "ledger.jsonl")

    def train(out_dir: Path):
        (out_dir / "xg_model_5v5.json").unlink()  # trainer crashed mid-write
        return {"gate_pass": True}

    rc = run_stage(
        name="xg_5v5",
        suite="nhl_data_build",
        config={"model": "xg_5v5"},
        artifacts=artifacts,
        train=train,
        force=True,
    )
    assert rc == 1
    assert {a.name: _sha(a) for a in artifacts} == before
