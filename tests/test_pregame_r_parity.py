"""R twin parity: ``R/nhl_pregame_snapshots.R`` and ``nhl_data_build.pregame`` on the real captured day.

Both compile ``fixtures/pregame_2026-11-25.json`` (nine real ESPN probes, see ``fixtures/PREGAME.md``)
against the same schedule and roster rows; the two parquets must be equal row for row, dtypes
included. Skipped where ``Rscript`` (with dplyr, jsonlite, arrow, stringi, optparse, glue) is absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import polars as pl
import pytest
from nhl_data_build import pregame

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures"
RSCRIPT = shutil.which("Rscript")
_PKGS = "c('dplyr','jsonlite','arrow','stringi','optparse','glue','cli','tibble')"


def _r_ready() -> bool:
    if RSCRIPT is None:
        return False
    probe = f"quit(status = !all(vapply({_PKGS}, requireNamespace, NA, quietly = TRUE)))"
    return subprocess.run([RSCRIPT, "-e", probe], capture_output=True).returncode == 0


pytestmark = pytest.mark.skipif(not _r_ready(), reason="Rscript with dplyr/jsonlite/arrow/stringi not available")


def _layout(root: Path) -> tuple[Path, Path]:
    raw = root / "raw" / "2027"
    raw.mkdir(parents=True)
    doc = json.loads((FIX / "pregame_2026-11-25.json").read_text(encoding="utf-8"))
    (raw / "2026-11-25.json").write_text(json.dumps(doc), encoding="utf-8")
    out = root / "nhl"
    (out / "schedules" / "parquet").mkdir(parents=True)
    (out / "rosters" / "parquet").mkdir(parents=True)
    shutil.copy(
        FIX / "pregame_schedule_2026-11-25.parquet", out / "schedules" / "parquet" / "nhl_schedule_2027.parquet"
    )
    shutil.copy(FIX / "pregame_rosters.parquet", out / "rosters" / "parquet" / "rosters_2027.parquet")
    return root / "raw", out


def test_r_twin_writes_the_same_table(tmp_path):
    raw, py_out = _layout(tmp_path / "py")
    assert pregame.main(["-s", "2027", "--pregame-dir", str(raw), "--out-dir", str(py_out)]) == 0
    raw_r, r_out = _layout(tmp_path / "r")
    run = subprocess.run(
        [RSCRIPT, "R/nhl_pregame_snapshots.R", "-s", "2027", "--pregame-dir", str(raw_r), "--out-dir", str(r_out)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    rel = Path("pregame_snapshots") / "parquet" / "nhl_pregame_snapshots_2027.parquet"
    py = pl.read_parquet(py_out / rel)
    r = pl.read_parquet(r_out / rel)
    assert dict(r.schema) == dict(py.schema) == pregame.SCHEMA
    assert r.height == py.height == 128
    assert r.equals(py), r.join(py, on=list(py.columns), how="anti")
