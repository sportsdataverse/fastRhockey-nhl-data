"""Stage 02 — fastRhockey xG st model.

Thin numbered pipeline for ONE booster: training logic lives in
nhl_data_build.xg_train (shared with the R twin recipe); this file owns
operability — fingerprint skip/--force, gate rc, ledger append. Two gates:

* discrimination floor ``cv_auc >= 0.81`` — frozen just below the 2026-04
  observed 0.8213; never lowered.
* per-season calibration drift ceiling ``max |z| <= _ST_DRIFT_MAX_ABS_Z`` over
  the exact holdout seasons, where ``z = (goals - sum xG) / sqrt(sum xG(1-xG))``
  (``xg_train.per_season_calibration``). The ST sample is ~5x smaller than
  5v5, so a per-season level drift is the failure the AUC floor cannot see.
  Derived from the observed value at derivation time (models/REGISTRY.md);
  a ceiling is never raised. ``None`` = not yet derived: the statistic is
  recorded in the ledger, not enforced.

Usage::

    python -m nhl_model_02_xg_st [--force] [--quick]
    scripts/nhl_models.sh 02
"""

from __future__ import annotations

import argparse
import glob as _glob
from pathlib import Path

# Derived 2026-09-02 from the observed exact-holdout max |z| = 2.008 (season 2018, 15 holdout
# seasons) of the python ST retrain (models/ledger.jsonl fingerprint f0f3b5aafa088bd8; that run
# failed the cv-AUC floor and was NOT promoted, but its per-season calibration is the trusted
# observation for THIS statistic). Ceiling = observed + ~1 z-unit = the per-season 99.7% band; a
# calibrated model's max over 15 seasons exceeds it ~4% of the time. Never raised (see REGISTRY.md).
_ST_DRIFT_MAX_ABS_Z: float | None = 3.0


def st_gates(info: dict) -> dict:
    """Gate verdicts for one ST retrain from its ``info_st`` meta block (pure; re-runnable on the committed meta)."""
    auc = info.get("cv_auc")
    drift = info.get("max_abs_season_z")
    drift_pass = None if _ST_DRIFT_MAX_ABS_Z is None or drift is None else drift <= _ST_DRIFT_MAX_ABS_Z
    return {
        "cv_auc": auc,
        "cv_logloss": info.get("cv_logloss"),
        "test_auc": info.get("test_auc"),
        "st_max_abs_season_z": drift,
        "st_drift_ceiling": _ST_DRIFT_MAX_ABS_Z,
        "st_drift_pass": drift_pass,
        "gate_pass": (auc is not None and auc >= 0.81) and drift_pass is not False,
    }


def main(argv: list[str] | None = None) -> int:
    from nhl_data_build._stage import run_stage

    ap = argparse.ArgumentParser(prog="python -m nhl_model_02_xg_st")
    ap.add_argument("--pbp", default="nhl/pbp/parquet/*.parquet", metavar="GLOB")
    ap.add_argument("--out", default="models", metavar="DIR")
    ap.add_argument("--quick", action="store_true", help="reduced grid/rounds (smoke; gate tolerated)")
    ap.add_argument("--force", action="store_true", help="retrain even when the fingerprint is unchanged")
    args = ap.parse_args(argv)

    def train(out_dir: Path):
        import polars as pl
        from nhl_data_build.xg_train import train_xg_models

        files = sorted(_glob.glob(args.pbp))
        if not files:
            raise SystemExit(f"no pbp parquet matched {args.pbp!r}")
        pbp = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
        meta = train_xg_models(pbp, out_dir, quick=args.quick, variants=("st",))
        return st_gates(meta["info_st"])

    return run_stage(
        name="xg_st",
        suite="nhl_data_build",
        force=args.force,
        config={"model": "xg_st", "pbp": args.pbp, "quick": args.quick},
        # CHAMPION paths -- run_stage trains into a candidate dir and copies these
        # names across only when the gate passes. Both sidecars are stage artifacts
        # too: train_xg_models writes them, and the fingerprint skip must not declare
        # the stage done when one has been removed.
        artifacts=[
            Path(args.out) / "xg_model_st.json",
            Path(args.out) / "xg_model_meta.json",
            Path(args.out) / "xg_model_split.json",
        ],
        train=train,
        smoke=args.quick,
    )


if __name__ == "__main__":
    raise SystemExit(main())
