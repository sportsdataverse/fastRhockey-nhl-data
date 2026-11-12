"""Stage 01 — fastRhockey xG 5v5 model.

Thin numbered pipeline for ONE booster: training logic lives in
nhl_data_build.xg_train (shared with the R twin recipe); this file owns
operability — fingerprint skip/--force, gate rc, ledger append. The gate floor
``cv_auc >= _CV_AUC_FLOOR`` sits just below an observed fit and is never lowered;
it is re-derived only from a new fit on a corrected corpus (models/REGISTRY.md).

Usage::

    python -m nhl_model_01_xg_5v5 [--force] [--quick]
    scripts/nhl_models.sh 01
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Re-derived 2026-11-12 from run 43165388160: cv AUC 0.779 on the complete corpus (2009-10 ... 2026-27,
# one season per file), every holdout season calibrated (goals/xG 0.96-1.03, max |z| 1.73). The old
# 0.82 came from the 2026-04 fit (0.8322) on a corpus with no missed shots, which no reconstruction of
# today's frame reproduces (PR #11); that champion scores 0.778 on today's frame. Never lowered.
_CV_AUC_FLOOR = 0.77


def main(argv: list[str] | None = None) -> int:
    from nhl_data_build._stage import run_stage

    ap = argparse.ArgumentParser(prog="python -m nhl_model_01_xg_5v5")
    ap.add_argument("--pbp", default="nhl/pbp/parquet/*.parquet", metavar="GLOB")
    ap.add_argument("--out", default="models", metavar="DIR")
    ap.add_argument("--quick", action="store_true", help="reduced grid/rounds (smoke; gate tolerated)")
    ap.add_argument("--force", action="store_true", help="retrain even when the fingerprint is unchanged")
    args = ap.parse_args(argv)

    def train(out_dir: Path):
        from nhl_data_build.xg_train import load_pbp, train_xg_models

        pbp = load_pbp(args.pbp)
        meta = train_xg_models(pbp, out_dir, quick=args.quick, variants=("5v5",))
        info = meta["info_5v5"]
        auc = info.get("cv_auc")
        return {
            "cv_auc": auc,
            "cv_logloss": info.get("cv_logloss"),
            "test_auc": info.get("test_auc"),
            "gate_pass": auc is not None and auc >= _CV_AUC_FLOOR,
        }

    return run_stage(
        name="xg_5v5",
        suite="nhl_data_build",
        force=args.force,
        config={"model": "xg_5v5", "pbp": args.pbp, "quick": args.quick},
        # CHAMPION paths -- run_stage trains into a candidate dir and copies these
        # names across only when the gate passes. Both sidecars are stage artifacts
        # too: train_xg_models writes them, and the fingerprint skip must not declare
        # the stage done when one has been removed.
        artifacts=[
            Path(args.out) / "xg_model_5v5.json",
            Path(args.out) / "xg_model_meta.json",
            Path(args.out) / "xg_model_split.json",
        ],
        train=train,
        smoke=args.quick,
    )


if __name__ == "__main__":
    raise SystemExit(main())
