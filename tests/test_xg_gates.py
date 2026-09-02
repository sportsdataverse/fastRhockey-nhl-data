"""The committed xG sidecars satisfy the frozen gates and describe one consistent fit.

Bites on the COMMITTED artifacts (no data, no training): a retrain whose meta
misses a floor, or a split file that disagrees with its meta, cannot be merged
quietly. Floors are the stage constants (frozen just below the values observed
at gate time; never lowered) and the ST drift ceiling (derived from the
observed exact-holdout max |z|; never raised) — see models/REGISTRY.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nhl_model_02_xg_st import st_gates

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "models" / "xg_model_meta.json"
SPLIT = ROOT / "models" / "xg_model_split.json"

# The python sidecars land with the first python retrain that PASSES the frozen floors (the
# 2026-09-02 retrain did not: ST cv AUC 0.7569 < 0.81 -- see models/REGISTRY.md). Until then the
# committed boosters are the 2026-04 R fit, which has no python sidecar to check.
pytestmark = pytest.mark.skipif(not META.is_file(), reason="no python xg_model_meta.json committed yet")

# The python sidecars land with the first python retrain that PASSES the frozen floors (the
# 2026-09-02 retrain did not: ST cv AUC 0.7569 < 0.81 -- see models/REGISTRY.md). Until then the
# committed boosters are the 2026-04 R fit, which has no python sidecar to check.
pytestmark = pytest.mark.skipif(not META.is_file(), reason="no python xg_model_meta.json committed yet")


def _meta() -> dict:
    return json.loads(META.read_text(encoding="utf-8"))


def _split() -> dict:
    return json.loads(SPLIT.read_text(encoding="utf-8"))


def test_sidecars_travel_together() -> None:
    assert SPLIT.is_file(), "xg_model_meta.json is committed without xg_model_split.json"


def test_frozen_cv_auc_floors_hold() -> None:
    meta = _meta()
    assert meta["info_5v5"]["cv_auc"] >= 0.82, meta["info_5v5"]["cv_auc"]  # 5v5 floor (nhl_model_01_xg_5v5)
    assert meta["info_st"]["cv_auc"] >= 0.81, meta["info_st"]["cv_auc"]  # ST floor (nhl_model_02_xg_st)


def test_st_drift_gate_holds_on_committed_meta() -> None:
    verdict = st_gates(_meta()["info_st"])
    assert verdict["st_drift_ceiling"] is not None, (
        "ST drift ceiling not derived — set _ST_DRIFT_MAX_ABS_Z from the observed value"
    )
    assert verdict["st_drift_pass"] is True, verdict
    assert verdict["gate_pass"] is True, verdict


def test_split_matches_meta_and_is_a_partition() -> None:
    meta, split = _meta(), _split()
    assert split["seed"] == meta["seed"] == 37
    for v in ("5v5", "st"):
        train, test = set(split[v]["train_game_ids"]), set(split[v]["test_game_ids"])
        assert train and test and not (train & test), f"{v}: train/test game ids must be disjoint and non-empty"
        assert len(test) == meta[f"info_{v}"]["n_games_test"], f"{v}: split file and meta disagree on the holdout"
        assert len(train) == meta[f"info_{v}"]["n_games_train"]
        assert len(meta[f"xg_feature_names_{v}"]) == (36 if v == "5v5" else 38)
