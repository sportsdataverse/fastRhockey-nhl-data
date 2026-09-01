"""Stage 02 — fastRhockey xG st model.

Thin numbered pipeline for ONE booster: training logic lives in
nhl_data_build.xg_train (shared with the R twin recipe); this file owns
operability — fingerprint skip/--force, gate rc, ledger append. The gate floor
(cv_auc >= 0.81) is frozen just below the 2026-04 observed 0.8213 and is
never lowered.

Usage::

    python -m nhl_model_build.nhl_model_02_xg_st [--force] [--quick]
    scripts/nhl_models.sh 02
"""
from __future__ import annotations

import argparse
import glob as _glob
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from nhl_model_build._stage import run_stage

    ap = argparse.ArgumentParser(prog="nhl_model_build.nhl_model_02_xg_st")
    ap.add_argument("--pbp", default="nhl/pbp/parquet/*.parquet", metavar="GLOB")
    ap.add_argument("--out", default="models", metavar="DIR")
    ap.add_argument("--quick", action="store_true", help="reduced grid/rounds (smoke; gate tolerated)")
    ap.add_argument("--force", action="store_true", help="retrain even when the fingerprint is unchanged")
    args = ap.parse_args(argv)

    def train():
        import polars as pl
        from nhl_data_build.xg_train import train_xg_models

        files = sorted(_glob.glob(args.pbp))
        if not files:
            raise SystemExit(f"no pbp parquet matched {args.pbp!r}")
        pbp = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
        meta = train_xg_models(pbp, args.out, quick=args.quick, variants=("st",))
        info = meta["info_st"]
        auc = info.get("cv_auc")
        return {
            "cv_auc": auc,
            "cv_logloss": info.get("cv_logloss"),
            "test_auc": info.get("test_auc"),
            "gate_pass": auc is not None and auc >= 0.81,
        }

    return run_stage(
        name="xg_st", suite="nhl_data_build", force=args.force,
        config={"model": "xg_st", "pbp": args.pbp, "quick": args.quick},
        artifacts=[Path(args.out) / "xg_model_st.json"],
        train=train, smoke=args.quick,
    )


if __name__ == "__main__":
    raise SystemExit(main())
