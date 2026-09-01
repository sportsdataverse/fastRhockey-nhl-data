"""models/REGISTRY.md carries one row per committed model artifact.

Bites per-artifact: the required set is read from DISK (models/*.json|rds and
the vendored hockeyR artifacts), so a new artifact without a row — or a
deleted row — fails. Package-name matching (the cfbfastR-cfb-data registry
test's weakness) is exactly what this avoids.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "REGISTRY.md"

HOCKEYR_ARTIFACTS = [
    "hockeyR/inst/extdata/xg_model_5v5.json",
    "hockeyR/inst/extdata/xg_model_st.json",
    "hockeyR/R/xg_model_ps.rds",
]


def _table_rows() -> list[str]:
    text = REGISTRY.read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if ln.startswith("|") and "---" not in ln]


def _required() -> list[str]:
    on_disk = sorted(f"models/{p.name}" for p in (ROOT / "models").iterdir() if p.suffix in {".json", ".rds", ".ubj"})
    vendored = [a for a in HOCKEYR_ARTIFACTS if (ROOT / a).exists()]
    return on_disk + vendored


def test_registry_exists():
    assert REGISTRY.is_file(), "models/REGISTRY.md is missing"


def test_every_artifact_on_disk_has_a_row():
    rows = _table_rows()
    missing = [a for a in _required() if not any(a in r for r in rows)]
    assert not missing, f"artifacts with no registry TABLE ROW: {missing}"


def test_rows_state_taglessness():
    """These models publish no release tag; every row must say so explicitly."""
    for row in _table_rows():
        if "xg_model" in row:
            assert "—" in row, f"row must state its (lack of) release tag: {row[:70]}"
