# Model registry

One row per model artifact this repo owns (moved here from the README —
Track C: a table a test parses is repository data, and `docs/`-adjacent
locations get regenerated). `tests/test_model_registry.py` fails the build
when an artifact on disk has no row here, and bites per-artifact.

These models publish **no release tag of their own**: the artifacts are
committed under `models/` (and vendored `hockeyR/`), and their OUTPUT reaches
consumers as the xG columns inside the published `nhl_pbp_full` datasets
(`in_published_data` = every season compile since the 2026-04 retrain).

## fastRhockey xG suite (`models/`)

Trained on this repo's full NHL play-by-play (16 seasons 2009-10 → 2025-26, no
2014-15; five era groupings). Two trainers, one recipe: `R/build_xg_model.R`
(canonical; the COMMITTED 2026-04 boosters) and its port
`python/nhl_data_build/xg_train.py` (grouped 80/20 split by `game_id`, seed 37,
logloss+auc grouped 5-fold CV). The python trainer writes two sidecars beside the
boosters — `xg_model_meta.json` (feature lists, CV + exact-holdout metrics, per-season
calibration) and `xg_model_split.json` (the train/test `game_id` partition) — which
are committed with the first python retrain that passes the frozen gates (see
"Retrain attempts" below: the 2026-09-02 attempt did not).

| model | artifact(s) | release tag | training data | fitting script | gates at publish | last retrain | cadence |
|---|---|---|---|---|---|---|---|
| xG 5v5 | `models/xg_model_5v5.json` | — (committed; output ships in `nhl_pbp_full`) | nhl pbp 2011–2026, era-grouped | `R/build_xg_model.R` / `nhl_data_build/xg_train.py` | CV AUC **0.8322**, log-loss **0.2053** (README table; grouped split — no game leakage) | 2026-04 | **annual cron** — `nhl_model_pipeline.yml`, `0 6 15 7 *` (Jul 15); **has never fired**, so no run exists yet |
| xG special teams | `models/xg_model_st.json` | — (committed; output ships in `nhl_pbp_full`) | nhl pbp 2011–2026, era-grouped | `R/build_xg_model.R` / `nhl_data_build/xg_train.py` (stage `nhl_model_02_xg_st`) | CV AUC **0.8213**, log-loss **0.2567** (2026-04 fit); floors: cv AUC ≥ 0.81 **and** per-season calibration drift max holdout-season \|z\| ≤ 3.0 (ceiling derived 2026-09-02 from the observed 2.008 in season 2018; never raised) | 2026-04 | annual July cron + dispatch (`nhl_model_pipeline.yml`) |
| xG meta (incl. penalty-shot constant) | `models/xg_model_meta.rds` (R era, committed); python sidecars `models/xg_model_meta.json` + `models/xg_model_split.json` (written by every python retrain, committed with the first gate-passing one — none yet) | — (committed) | derived at train time | `R/build_xg_model.R` / `nhl_data_build/xg_train.py` | rides the suite's gates; `tests/test_xg_gates.py` bites on the committed python meta/split once present | 2026-04 | **annual cron** — `nhl_model_pipeline.yml`, `0 6 15 7 *` (Jul 15); **has never fired**, so no run exists yet |

## Vendored hockeyR models (`hockeyR/`)

Reproduction lineage (Morse 2022); retrained locally, **no CI trigger by
design** (needs the `hockeyR-data` sibling checkout).

| model | artifact(s) | release tag | training data | fitting script | gates at publish | last retrain | cadence |
|---|---|---|---|---|---|---|---|
| hockeyR xG 5v5 | `hockeyR/inst/extdata/xg_model_5v5.json` (+ `hockeyR/R/sysdata.rda`) | — (vendored package data) | hockeyR-data corpus | `hockeyR/retrain_xg_models.R` | reproduction of Morse 2022 recipe | 2026-03 | manual — local runbook |
| hockeyR xG special teams | `hockeyR/inst/extdata/xg_model_st.json` (+ `hockeyR/R/sysdata.rda`) | — (vendored package data) | hockeyR-data corpus | `hockeyR/retrain_xg_models.R` | reproduction of Morse 2022 recipe | 2026-03 | manual — local runbook |
| hockeyR penalty-shot constant | `hockeyR/R/xg_model_ps.rds` (bundled into `sysdata.rda`) | — (vendored package data) | hockeyR-data corpus | `hockeyR/retrain_xg_models.R` (restore step) | constant — no fit gate | 2026-03 | manual — local runbook |

## Operability (Track C steps 2–6)

- `models/manifest.yaml` — single home for the model/stage list (guarded by `tests/test_model_manifest.py`).
- One model = one numbered pipeline: `python/nhl_model_01_xg_5v5.py` / `nhl_model_02_xg_st.py`; run subsets with `scripts/nhl_models.sh`; retrain CI = `nhl_model_pipeline.yml` (dispatch + annual July cron; previously the cadence had NO workflow behind it).
- Gate floors frozen just below the 2026-04 observed values (5v5 cv AUC ≥ 0.82, ST ≥ 0.81); never lowered; `--quick` smoke runs tolerate misses. ST per-season calibration drift ceiling `max |z| ≤ 3.0` over the exact-holdout seasons (`nhl_model_02_xg_st._ST_DRIFT_MAX_ABS_Z`; z = (goals − ΣxG)/√ΣxG(1−xG) per season, `xg_train.per_season_calibration`, recorded in `info_st.per_season_calibration`), derived 2026-09-02 from the observed max |z| = 2.008 (season 2018, 15 holdout seasons) of the python ST retrain — observed + ~1 z-unit = the per-season 99.7% band, exceeded by a calibrated model's 15-season max ~4% of the time; a ceiling is never raised.
- Fingerprints: stages skip when `hash(code subtree, config)` is unchanged (`--force` to retrain); every trained model appends a `models/ledger.jsonl` line; each retrain rewrites the `xg_model_meta.json` + `xg_model_split.json` + `xg_model_report.md` sidecars (single-variant runs merge, never clobber — but two single-variant runs must not overlap in time: each reads the sidecars at start and writes at end).
- **Candidate/promote (2026-09-02).** A stage trains into `models/.candidate/<stage>/` (gitignored, seeded from the current champion so a single-variant sidecar merge still sees its sibling) and `nhl_data_build._stage.run_stage` copies the files onto the champion paths **only when `gate_pass` is not `False`**. A failing run leaves every champion file byte-identical, still appends its `models/ledger.jsonl` row, keeps the candidate for inspection, and does NOT record its fingerprint (so the next run retrains rather than skipping). Before this, the trainer wrote straight to `models/` and the gate was checked afterwards: the 2026-09-02 stage-01 retrain scored cv_auc **0.7786** with `gate_pass: false` and overwrote `models/xg_model_5v5.json` in place. Promotion itself is **all-or-nothing**: every champion file about to be overwritten is snapshotted to `models/.candidate/<stage>.rollback/` first and restored (rc 1) if any copy raises, so a partial sweep can never leave a new booster beside stale `xg_model_meta.json` / `xg_model_split.json`. Guarded by `tests/test_stage_promotion.py`.
- Promotion = committing the retrained `models/*.json` (this repo's step-6 convention); CI only uploads run artifacts, it never pushes.
- **The 2026-04 champions do not price the current frame (measured 2026-09-02).** `nhl_data_build.xg_parity.artifact_calibration` scores an EXISTING booster on the frame `prepare_training_frame` builds now — the check no promotion gate can make, because `cv_auc` is scale-free and the ST drift ceiling reads a *retrain's own* meta. On `nhl/pbp/full/parquet` (17 files, 1,829,710 5v5 / 359,730 ST rows) the committed boosters score goals/ΣxG **0.7676** (5v5) and **0.7616** (ST) — a 25-30% over-prediction on every season through 2023-24. Dropping `MISSED_SHOT` from the same corpus moves them to **1.0556** / **1.0663**, and to 0.99-1.02 for each individual season 2009-10 … 2023-24, which identifies the 2026-04 training corpus as one carrying no missed shots for those seasons. The R (`R/build_xg_model.R`) and python (`xg_train.prepare_training_frame`) recipes were proved identical on real games first — same rows, same values on 32 of 33 shared features — so this is an artifact/corpus defect, not a port defect.
- **The frozen AUC floors are therefore not achievable and must be RE-DERIVED, not lowered.** Re-fitting the R script's own params (max_depth 4, eta .06, gamma 1, subsample/colsample .8, min_child_weight 8, 1500 rounds, seed-37 grouped 80/20) on today's corpus gives holdout AUC **0.7781** 5v5 / **0.7576** ST with goals/ΣxG **1.0094** / **1.0029** — well calibrated, and matching the two python retrains recorded below (0.7786 / 0.7569) that were read as failures. The same fit on a SHOT|GOAL-only frame gives 0.7731 / 0.7573, so the recorded 0.8322 / 0.8213 is not reproducible from any reconstruction of the current corpus. Floors stay where they are until a fit on a CORRECTED corpus sets new ones from its own observation.
- **Corpus integrity blocks that re-derivation.** In `nhl/pbp/full/{rds,parquet}` the file→season mapping is broken for 2014-2018: each of those files holds the season one year LATER than its name, season `20132014` is absent from the corpus entirely, and `20182019` appears in two files (1,359 and 1,358 games), so any retrain double-weights 2018-19 and never sees 2013-14. `nhl/pbp/parquet/` (the python stages' default glob) additionally has no `play_by_play_2015.parquet`. Fix the producer output before re-deriving any floor.


## Retrain attempts (python trainer, this repo's parquet frame)

- **2026-09-02** — `scripts/nhl_models.sh --force 01` / `02` on the committed corpus (16 seasons). ST: cv AUC **0.7569** vs floor 0.81 → **FAIL**, not promoted (holdout AUC 0.7556, per-season calibration excellent: max |z| 2.008, goals/ΣxG 0.93–1.08). 5v5 result recorded in `models/ledger.jsonl`. The committed 2026-04 R boosters scored on the same python frame reach only 0.761 / 0.778 (ST / 5v5) and over-predict goals by 25–30% on every season through 2023-24 (|z| up to 9.2 / 14.8), so the deficit is in the python FRAME vs the R training frame for pre-2024 seasons — an R↔Python parity question (`sdv-parity-reviewer` follow-up) that blocks promoting any python retrain. Ledger lines are committed; the run's sidecars were not (they would describe a non-promoted model).