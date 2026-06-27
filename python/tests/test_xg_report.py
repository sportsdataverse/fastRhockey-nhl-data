"""Hermetic check of the xG training report rendering (full training is smoke-tested live)."""

from __future__ import annotations

from pathlib import Path

from nhl_data_build.xg_report import build_report, write_xg_report


def _meta() -> dict:
    def info(nt: int, nv: int, gr: float) -> dict:
        return {
            "n_train": nt,
            "n_test": nv,
            "min_child_weight": 3,
            "nrounds": 420,
            "goal_rate": gr,
            "cv_logloss": 0.21,
            "cv_auc": 0.78,
            "test_logloss": 0.22,
            "test_auc": 0.77,
            "importance": [{"feature": "shot_distance", "gain": 1200.5}, {"feature": "shot_angle", "gain": 800.1}],
        }

    return {
        "xg_model_ps": 0.3327,
        "xg_feature_names_5v5": ["f"] * 36,
        "xg_feature_names_st": ["f"] * 38,
        "info_5v5": info(90000, 20000, 0.07),
        "info_st": info(18000, 4000, 0.11),
    }


def test_build_report_structure() -> None:
    md = build_report(_meta())
    assert "# fastRhockey xG model training report" in md
    assert "| 5v5 |" in md and "| Special teams |" in md
    assert "Penalty shot" in md and "0.3327" in md
    assert "Test AUC" in md and "era_2011_2013" in md
    assert "shot_distance" in md  # feature importance rendered


def test_write_xg_report(tmp_path: Path) -> None:
    path = write_xg_report(_meta(), tmp_path)
    assert path.exists() and path.read_text(encoding="utf-8").startswith("# fastRhockey xG model")
