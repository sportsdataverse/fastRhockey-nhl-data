"""Is a COMMITTED booster level-calibrated on the frame the trainer builds today?

A booster and a feature recipe are two artifacts, and until now nothing checked that
they agree. The promotion gates cannot see a disagreement: ``cv_auc`` is scale-free,
and the ST drift ceiling reads a *retrain's own* meta — so a champion fit on a
different corpus can over-price every shot forever and every gate stays green.
``artifact_calibration`` closes that hole: it scores an existing booster on the frame
``prepare_training_frame`` builds NOW and reports goals / ΣxG overall and per season.

Measured 2026-09-02 on the committed 2026-04 boosters over ``nhl/pbp/full/parquet``
(17 files, 1,829,710 5v5 rows / 359,730 ST): overall goals/ΣxG **0.7676** (5v5) and
**0.7616** (ST) — a 25-30% over-prediction on every season through 2023-24. Dropping
``MISSED_SHOT`` from the same corpus moves those to **1.0556** / **1.0663**, and to
0.99-1.02 for each individual season 2009-10 … 2023-24 — which is what identifies the
champions' training corpus as one that carried no missed shots for those seasons.
The R and python feature recipes themselves were proved identical on real games
(0 row difference, 0 value difference on 32 of 33 shared features), so this is an
artifact/corpus defect, not a recipe port defect. Full evidence:
``ClaudeCowork/ledgers/2026-09-02-next-ten/reports/nhl-xg-frame.md``.
"""

from __future__ import annotations

import polars as pl

from nhl_data_build.xg_train import _rank_auc, per_season_calibration, prepare_training_frame


def artifact_calibration(pbp: pl.DataFrame, booster, *, variant: str) -> dict:
    """Score ``booster`` on the frame built from ``pbp`` now; report the level it prices at.

    Args:
        pbp: raw play-by-play (the same shape ``prepare_training_frame`` consumes).
        booster: a loaded ``xgboost.Booster`` whose ``feature_names`` name the columns to use.
            Features the current frame does not carry are supplied as 0, exactly as the R
            inference path back-fills an unseen one-hot level.
        variant: ``"5v5"`` or ``"st"``.

    Returns:
        dict: ``n``, ``goals``, ``sum_xg``, ``ratio`` (goals / ΣxG — 1.0 is level-correct),
        ``auc``, and ``per_season`` (the ``per_season_calibration`` table, so the same
        binomial z the ST drift gate uses is available per season). ``ratio``/``auc`` are
        ``None`` on an empty frame rather than raising.
    """
    import xgboost as xgb

    frame = prepare_training_frame(pbp, variant=variant)
    feats = list(booster.feature_names)
    missing = [f for f in feats if f not in frame.columns]
    frame = frame.with_columns([pl.lit(0).cast(pl.Int64).alias(f) for f in missing])
    if not frame.height:
        return {
            "n": 0,
            "goals": 0,
            "sum_xg": 0.0,
            "ratio": None,
            "auc": None,
            "per_season": [],
            "missing_features": missing,
        }
    x = frame.select(feats).to_numpy().astype(float)
    p = booster.predict(xgb.DMatrix(x, feature_names=feats))
    y = frame["goal"].to_numpy()
    gids = frame["game_id"].cast(pl.Int64).to_numpy()
    total = float(p.sum())
    return {
        "n": int(frame.height),
        "goals": int(y.sum()),
        "sum_xg": total,
        "ratio": (float(y.sum()) / total) if total > 0 else None,
        "auc": _rank_auc(p, y),
        "per_season": per_season_calibration(gids, y, p),
        "missing_features": missing,
    }
