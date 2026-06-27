"""Port of ``build_xg_model.R`` — train the fastRhockey xG models (5v5 + ST + PS constant).

Canonical R source: ``fastRhockey-nhl-data/R/build_xg_model.R``. The feature recipe
(secondary_type normalization, lag features, zones, era dummies, tactical flags, one-hot
shot_type + last_event_type) is the SAME recipe the inference path uses
(``nhl_raw.xg.prepare_xg_data``); this module keeps the ``goal`` target + all one-hot
columns and adds the XGBoost training loop (grouped 80/20 split + 5-fold grouped CV +
min_child_weight grid + final fit), then saves ``xg_model_5v5.json`` / ``xg_model_st.json``
/ ``xg_model_meta.json``.

The "5v5" model is trained on ALL non-shootout, non-penalty-shot unblocked shots (no
strength filter, per the R script); the "st" model is trained only on the special-teams
strength states and adds ``total_skaters_on`` / ``event_team_advantage`` features.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import polars as pl

_FENWICK = ["SHOT", "MISSED_SHOT", "GOAL"]
_VALID_LAST = [
    "FACEOFF",
    "GIVEAWAY",
    "TAKEAWAY",
    "BLOCKED_SHOT",
    "HIT",
    "MISSED_SHOT",
    "SHOT",
    "STOP",
    "PENALTY",
    "GOAL",
]
_ST_STRENGTHS = ["5v4", "5v3", "6v5", "6v4", "4v4", "4v3", "3v3", "4v5", "3v5", "5v6", "4v6", "3v4"]
_SEED = 37

_SHOT_TYPE_NORM = {
    "wrist": "Wrist Shot",
    "wrist shot": "Wrist Shot",
    "snap": "Snap Shot",
    "snap shot": "Snap Shot",
    "slap": "Slap Shot",
    "slap shot": "Slap Shot",
    "backhand": "Backhand",
    "deflected": "Deflected",
    "tip-in": "Tip-In",
    "wrap-around": "Wrap-around",
    "bat": "Batted",
    "batted": "Batted",
    "poke": "Poke",
    "between-legs": "Between Legs",
    "between legs": "Between Legs",
    "cradle": "Cradle",
    "penalty shot": "Penalty Shot",
}
_SHOT_TYPE_COL = {
    "Wrist Shot": "wrist_shot",
    "Snap Shot": "snap_shot",
    "Slap Shot": "slap_shot",
    "Backhand": "backhand",
    "Wrap-around": "wrap_around",
    "Tip-In": "tip_in",
    "Deflected": "deflected",
    "Poke": "poke",
    "Batted": "batted",
    "Between Legs": "between_legs",
    "Cradle": "cradle",
}
_LAST_EVENT_COL = {
    "FACEOFF": "last_faceoff",
    "GIVEAWAY": "last_giveaway",
    "TAKEAWAY": "last_takeaway",
    "BLOCKED_SHOT": "last_blocked_shot",
    "HIT": "last_hit",
    "MISSED_SHOT": "last_missed_shot",
    "SHOT": "last_shot",
    "STOP": "last_stop",
    "PENALTY": "last_penalty",
    "GOAL": "last_goal",
}


def _harmonize(pbp: pl.DataFrame) -> pl.DataFrame:
    """Standardize to event_team / home_team / away_team (R column harmonisation)."""
    ren = {}
    if "event_team" not in pbp.columns and "event_team_abbr" in pbp.columns:
        ren["event_team_abbr"] = "event_team"
    if "home_team" not in pbp.columns and "home_abbr" in pbp.columns:
        ren["home_abbr"] = "home_team"
    if "away_team" not in pbp.columns and "away_abbr" in pbp.columns:
        ren["away_abbr"] = "away_team"
    return pbp.rename(ren) if ren else pbp


def _norm_secondary() -> pl.Expr:
    e = pl.col("secondary_type")
    expr = pl.when(e.is_null()).then(None)
    for raw, canon in _SHOT_TYPE_NORM.items():
        expr = expr.when(e.str.to_lowercase() == raw).then(pl.lit(canon))
    return expr.otherwise(e)


def _event_zone() -> pl.Expr:
    x, xf, eta = pl.col("x"), pl.col("x_fixed"), pl.col("event_team")
    home, away = pl.col("home_team"), pl.col("away_team")
    return (
        pl.when((x >= -25) & (x <= 25))
        .then(pl.lit("NZ"))
        .when(((xf < -25) & (eta == home)) | ((xf > 25) & (eta == away)))
        .then(pl.lit("DZ"))
        .when(((xf > 25) & (eta == home)) | ((xf < -25) & (eta == away)))
        .then(pl.lit("OZ"))
        .otherwise(None)
    )


def prepare_training_frame(pbp: pl.DataFrame, *, variant: str) -> pl.DataFrame:
    """Build the model matrix (features + ``goal`` target) for one variant.

    variant ``"5v5"`` = all non-shootout/non-PS unblocked shots; ``"st"`` = special-teams
    strengths only (+ total_skaters_on / event_team_advantage).
    """
    df = _harmonize(pbp).with_columns(secondary_type=_norm_secondary())
    df = df.filter(
        (pl.col("period_type") != "SHOOTOUT")
        & ((pl.col("secondary_type") != "Penalty Shot") | pl.col("secondary_type").is_null())
    )
    if variant == "st":
        df = df.filter(pl.col("strength_state").is_in(_ST_STRENGTHS))

    grp = "game_id"
    df = (
        df.with_columns(event_zone=_event_zone())
        .with_columns(
            last_event_type=pl.col("event_type").shift(1).over(grp),
            time_since_last=pl.col("game_seconds") - pl.col("game_seconds").shift(1).over(grp),
            last_x=pl.col("x").shift(1).over(grp),
            last_y=pl.col("y").shift(1).over(grp),
            last_event_zone=pl.col("event_zone").shift(1).over(grp),
        )
        .with_columns(
            distance_from_last=((pl.col("y") - pl.col("last_y")) ** 2 + (pl.col("x") - pl.col("last_x")) ** 2)
            .sqrt()
            .round(1),
        )
    )
    df = df.filter(pl.col("event_type").is_in(_FENWICK) & pl.col("last_event_type").is_in(_VALID_LAST))
    if df.height == 0:
        return df

    season = pl.col("season").cast(pl.Utf8)
    eta, home = pl.col("event_team"), pl.col("home_team")
    ets = pl.when(eta == home).then(pl.col("home_skaters")).otherwise(pl.col("away_skaters"))
    ots = pl.when(eta == home).then(pl.col("away_skaters")).otherwise(pl.col("home_skaters"))
    lz, lt, tsl = pl.col("last_event_zone"), pl.col("last_event_type"), pl.col("time_since_last")
    df = df.with_columns(
        era_2011_2013=season.is_in(["20102011", "20112012", "20122013"]).cast(pl.Int64),
        era_2014_2018=season.is_in(["20132014", "20142015", "20152016", "20162017", "20172018"]).cast(pl.Int64),
        era_2019_2021=season.is_in(["20182019", "20192020", "20202021"]).cast(pl.Int64),
        era_2022_2024=season.is_in(["20212022", "20222023", "20232024"]).cast(pl.Int64),
        era_2025_on=(season.cast(pl.Float64) > 20232024).cast(pl.Int64),
        total_skaters_on=ets + ots,
        event_team_advantage=ets - ots,
        rebound=(lt.is_in(_FENWICK) & (tsl <= 2)).cast(pl.Int64),
        rush=(lz.is_in(["NZ", "DZ"]) & (tsl <= 4)).cast(pl.Int64),
        cross_ice_event=(
            (lz == "OZ")
            & (((pl.col("last_y") > 3) & (pl.col("y") < -3)) | ((pl.col("last_y") < -3) & (pl.col("y") > 3)))
            & (tsl <= 2)
        ).cast(pl.Int64),
        empty_net=pl.col("empty_net").cast(pl.Boolean).fill_null(False).cast(pl.Int64),
        goal=(pl.col("event_type") == "GOAL").cast(pl.Int64),
    )
    onehots = {col: (pl.col("secondary_type") == canon).cast(pl.Int64) for canon, col in _SHOT_TYPE_COL.items()}
    onehots |= {col: (pl.col("last_event_type") == raw).cast(pl.Int64) for raw, col in _LAST_EVENT_COL.items()}

    base = [
        "shot_distance",
        "shot_angle",
        "rebound",
        "rush",
        "time_since_last",
        "distance_from_last",
        "cross_ice_event",
        "empty_net",
        "last_x",
        "last_y",
        "era_2011_2013",
        "era_2014_2018",
        "era_2019_2021",
        "era_2022_2024",
        "era_2025_on",
    ]
    if variant == "st":
        base += ["total_skaters_on", "event_team_advantage"]
    keep = ["game_id", *base, *_SHOT_TYPE_COL.values(), *_LAST_EVENT_COL.values(), "goal"]
    return df.with_columns(**onehots).select(keep).drop_nulls()


def penalty_shot_xg(pbp: pl.DataFrame) -> float:
    """xG constant for penalty shots/shootouts = historical conversion rate."""
    df = _harmonize(pbp).with_columns(secondary_type=_norm_secondary())
    pens = df.filter(
        ((pl.col("period_type") == "SHOOTOUT") | (pl.col("secondary_type") == "Penalty Shot"))
        & pl.col("event_type").is_in(_FENWICK)
    )
    return float((pens["event_type"] == "GOAL").mean()) if pens.height else 0.326


def _grouped_split(df: pl.DataFrame, rng: np.random.Generator, frac: float = 0.8) -> tuple[pl.DataFrame, pl.DataFrame]:
    gids = df["game_id"].unique().to_list()
    train_ids = set(rng.choice(gids, size=int(frac * len(gids)), replace=False).tolist())
    in_train = pl.col("game_id").is_in(list(train_ids))
    return df.filter(in_train), df.filter(~in_train)


def _rank_auc(pred: np.ndarray, actual: np.ndarray) -> float:
    """Rank-based AUC (port of build_xg_model.R's .calc_auc; exact for continuous preds)."""
    actual = np.asarray(actual)
    n1, n0 = float((actual == 1).sum()), float((actual == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.empty(len(pred), dtype=float)
    ranks[np.argsort(pred, kind="mergesort")] = np.arange(1, len(pred) + 1)
    return float((ranks[actual == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _train_variant(pbp: pl.DataFrame, variant: str, rng: np.random.Generator, *, quick: bool) -> tuple:
    import xgboost as xgb

    frame = prepare_training_frame(pbp, variant=variant)
    train, test = _grouped_split(frame, rng)
    feats = [c for c in train.columns if c not in ("game_id", "goal")]
    gids = train["game_id"].unique().to_list()
    folds = {g: i % 5 for i, g in enumerate(rng.permutation(gids))}
    fold_vec = np.array([folds[g] for g in train["game_id"].to_list()])

    Xtr, ytr = train.select(feats).to_numpy().astype(float), train["goal"].to_numpy()
    dtrain = xgb.DMatrix(Xtr, label=ytr, feature_names=feats)
    base = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 4,
        "eta": 0.06,
        "gamma": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    grid = [1] if quick else list(range(1, 11))
    nrounds = 60 if quick else 1500

    best = min(
        (
            (
                xgb.cv(
                    {**base, "min_child_weight": m},
                    dtrain,
                    num_boost_round=nrounds,
                    folds=[(np.where(fold_vec != k)[0], np.where(fold_vec == k)[0]) for k in range(5)],
                    early_stopping_rounds=30,
                    verbose_eval=False,
                ),
                m,
            )
            for m in grid
        ),
        key=lambda cm: cm[0]["test-logloss-mean"].min(),
    )
    cv_log, mcw = best[0], best[1]
    rounds = int(cv_log["test-logloss-mean"].idxmin()) + 1
    booster = xgb.train({**base, "min_child_weight": mcw}, dtrain, num_boost_round=rounds)

    # ----- holdout evaluation + feature importance (the reporting) -----
    Xte, yte = test.select(feats).to_numpy().astype(float), test["goal"].to_numpy()
    preds = booster.predict(xgb.DMatrix(Xte, feature_names=feats))
    gain = booster.get_score(importance_type="gain")
    top_imp = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:15]
    info = {
        "min_child_weight": mcw,
        "nrounds": rounds,
        "n_train": train.height,
        "n_test": test.height,
        "goal_rate": round(float(frame["goal"].mean()), 4),
        "cv_logloss": round(float(cv_log["test-logloss-mean"].min()), 4),
        "cv_auc": round(float(cv_log["test-auc-mean"].iloc[rounds - 1]), 4) if "test-auc-mean" in cv_log else None,
        "test_logloss": round(_logloss(yte, preds), 4),
        "test_auc": round(_rank_auc(preds, yte), 4),
        "importance": [{"feature": f, "gain": round(g, 2)} for f, g in top_imp],
    }
    return booster, feats, info


def train_xg_models(
    pbp: pl.DataFrame, out_dir: str | Path, *, quick: bool = False, report: bool = True, figures: bool = False
) -> dict:
    """Train + save the 5v5 / ST boosters (JSON) + ``xg_model_meta.json`` + a training report.

    With ``report`` (default), writes ``xg_model_report.md`` (per-model CV/test logloss + AUC,
    PS constant, era groupings, top feature importance). With ``figures``, also writes
    feature-importance PNGs (best-effort; needs matplotlib).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(_SEED)
    ps = penalty_shot_xg(pbp)

    meta: dict = {"xg_model_ps": round(ps, 7)}
    for variant in ("5v5", "st"):
        booster, feats, info = _train_variant(pbp, variant, rng, quick=quick)
        booster.save_model(str(out / f"xg_model_{variant}.json"))
        meta[f"xg_feature_names_{variant}"] = feats
        meta[f"info_{variant}"] = info
    (out / "xg_model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if report:
        from nhl_data_build.xg_report import write_xg_report

        write_xg_report(meta, out)
    if figures:
        from nhl_data_build.xg_report import write_importance_figures

        write_importance_figures(meta, out / "figures")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m nhl_data_build.xg_train",
        description="Train the fastRhockey xG models (5v5 / ST / PS) from pbp parquet(s).",
    )
    ap.add_argument("--pbp", required=True, help="pbp parquet path or glob (e.g. 'nhl/pbp/parquet/*.parquet')")
    ap.add_argument("--out", default="models", help="output dir for the .json models + meta + report")
    ap.add_argument("--quick", action="store_true", help="reduced grid/rounds (smoke run)")
    ap.add_argument("--figures", action="store_true", help="also write feature-importance PNGs (needs matplotlib)")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(args.pbp))
    if not files:
        ap.error(f"no pbp parquet matched {args.pbp!r}")
    pbp = pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    meta = train_xg_models(pbp, args.out, quick=args.quick, figures=args.figures)
    print(
        f"trained xG models -> {args.out} "
        f"(5v5 {len(meta['xg_feature_names_5v5'])} feats, st {len(meta['xg_feature_names_st'])} feats); "
        f"report: {args.out}/xg_model_report.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
