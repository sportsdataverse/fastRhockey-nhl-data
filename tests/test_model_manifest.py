"""models/manifest.yaml is the single home for the model/stage list (Track C step 2).

Per-row biting guards: manifest ↔ numbered stage scripts ↔ models/REGISTRY.md.
Deleting a model from any one of them goes red here.
"""

from importlib import import_module
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "manifest.yaml"
REGISTRY = ROOT / "models" / "REGISTRY.md"
STAGES_DIR = ROOT / "python"


def _doc() -> dict:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_parses():
    doc = _doc()
    assert set(doc["suites"]) == {"xg"}
    assert doc["driver"] == "scripts/nhl_models.sh"
    assert (ROOT / doc["driver"]).is_file()


def test_stages_and_manifest_agree_bidirectionally():
    files = {p.stem for p in STAGES_DIR.glob("nhl_model_[0-9][0-9]_*.py")}
    manifest = {Path(m["stage"]).stem for spec in _doc()["suites"].values() for m in spec["models"].values()}
    assert files == manifest, f"files-only={files - manifest}, manifest-only={manifest - files}"
    for spec in _doc()["suites"].values():
        for name, m in spec["models"].items():
            assert (ROOT / m["stage"]).is_file(), f"{name} stage missing"
            assert (ROOT / m["artifact"]).is_file(), f"{name} artifact missing on disk"
            assert m["gate"], f"{name} has no gate"


def test_stage_modules_import_and_expose_main():
    for p in sorted(STAGES_DIR.glob("nhl_model_[0-9][0-9]_*.py")):
        mod = import_module(p.stem)
        assert callable(getattr(mod, "main", None)), f"{p.stem} has no main()"


def test_registry_names_every_manifest_artifact_and_vendored_row():
    registry = REGISTRY.read_text(encoding="utf-8")
    doc = _doc()
    for spec in doc["suites"].values():
        for name, m in spec["models"].items():
            assert Path(m["artifact"]).name in registry, f"{name} not in REGISTRY.md"
        for sc in spec.get("sidecars", []):
            assert Path(sc).name in registry, f"sidecar {sc} not in REGISTRY.md"
    for v in doc["vendored"]:
        assert Path(v).name in registry, f"vendored {v} not in REGISTRY.md"
