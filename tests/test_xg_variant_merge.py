"""Single-variant retrains merge into the existing meta (numbered stages).

The per-model stages train ONE booster; xg_model_meta.json must keep the
sibling variant's entries rather than clobbering them.
"""

import json
from pathlib import Path

import nhl_data_build.xg_train as xt
import polars as pl


class _FakeBooster:
    def save_model(self, path):
        Path(path).write_text("booster", encoding="utf-8")


def test_single_variant_merges_meta(tmp_path, monkeypatch):
    calls = []

    def fake_variant(pbp, variant, rng, quick=False):
        calls.append(variant)
        return _FakeBooster(), [f"feat_{variant}"], {"cv_auc": 0.9}

    monkeypatch.setattr(xt, "_train_variant", fake_variant)
    monkeypatch.setattr(xt, "penalty_shot_xg", lambda pbp: 0.3)
    pbp = pl.DataFrame({"a": [1]})

    xt.train_xg_models(pbp, tmp_path, report=False, variants=("5v5",))
    meta1 = json.loads((tmp_path / "xg_model_meta.json").read_text(encoding="utf-8"))
    assert "xg_feature_names_5v5" in meta1
    assert "xg_feature_names_st" not in meta1

    xt.train_xg_models(pbp, tmp_path, report=False, variants=("st",))
    meta2 = json.loads((tmp_path / "xg_model_meta.json").read_text(encoding="utf-8"))
    assert meta2["xg_feature_names_5v5"] == ["feat_5v5"]  # preserved, not clobbered
    assert meta2["xg_feature_names_st"] == ["feat_st"]
    assert meta2["xg_model_ps"] == 0.3
    assert calls == ["5v5", "st"]


def test_both_variants_fresh_meta(tmp_path, monkeypatch):
    monkeypatch.setattr(
        xt,
        "_train_variant",
        lambda pbp, v, rng, quick=False: (_FakeBooster(), [f"feat_{v}"], {}),
    )
    monkeypatch.setattr(xt, "penalty_shot_xg", lambda pbp: 0.25)
    xt.train_xg_models(pl.DataFrame({"a": [1]}), tmp_path, report=False)
    meta = json.loads((tmp_path / "xg_model_meta.json").read_text(encoding="utf-8"))
    assert set(meta) >= {"xg_model_ps", "xg_feature_names_5v5", "xg_feature_names_st"}
