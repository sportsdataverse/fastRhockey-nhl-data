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

Trained on this repo's full NHL play-by-play, five era groupings. `nhl/pbp/parquet/` (the python
stages' default glob) now holds **18 seasons 2009-10 → 2026-27, one season per file**
(`play_by_play_2010` … `_2027`, named by END year; `corpus_seasons` in the committed
`xg_model_meta.json` lists them). The 2026-09-02 state — 2014-15 absent here, 2013-14 absent and
2018-19 doubled in `nhl/pbp/full/` from a start-year/end-year naming split — was repaired by the
producer before the 2026-11-12 retrain; `nhl/pbp/full/parquet/` now names 2010 … 2026 by end year.

Two trainers, one recipe: `R/build_xg_model.R`
(the 2026-04 boosters) and its port
`python/nhl_data_build/xg_train.py` (grouped 80/20 split by `game_id`, seed 37,
logloss+auc grouped 5-fold CV). The python trainer writes two sidecars beside the
boosters — `xg_model_meta.json` (feature lists, CV + exact-holdout metrics, per-season
calibration) and `xg_model_split.json` (the train/test `game_id` partition) — which
are committed with the boosters. The COMMITTED boosters are the python 2026-11-12 fit
(`nhl_model_pipeline.yml` run 43181159544); `xg_model_meta.rds` is rewritten from its JSON meta so the
R-era sidecar lists the same feature order.

| model | artifact(s) | release tag | training data | fitting script | gates at publish | last retrain | cadence |
|---|---|---|---|---|---|---|---|
| xG 5v5 | `models/xg_model_5v5.json` | — (committed; output ships in `nhl_pbp_full` once rescored) | nhl pbp 2009-10 … 2026-27, era-grouped | `nhl_data_build/xg_train.py` (stage `nhl_model_01_xg_5v5`) | CV AUC **0.7787**, log-loss **0.2116**; exact holdout AUC 0.7797, goals/ΣxG 0.96–1.03 per season (max \|z\| 1.84); floor cv AUC ≥ 0.77 (re-derived 2026-11-12) | 2026-11-12 (run 43181159544) | annual July cron + dispatch (`nhl_model_pipeline.yml`) |
| xG special teams | `models/xg_model_st.json` | — (committed; output ships in `nhl_pbp_full` once rescored) | nhl pbp 2009-10 … 2026-27, era-grouped | `nhl_data_build/xg_train.py` (stage `nhl_model_02_xg_st`) | CV AUC **0.7570**, log-loss **0.2825**; exact holdout AUC 0.7557, max holdout-season \|z\| **2.846** (2010-11, goals/ΣxG 0.885); floors: cv AUC ≥ 0.75 (re-derived 2026-11-12) **and** drift max \|z\| ≤ 3.0 (derived 2026-09-02; never raised) | 2026-11-12 (run 43181159544) | annual July cron + dispatch (`nhl_model_pipeline.yml`) |
| xG meta (incl. penalty-shot constant) | python sidecars `models/xg_model_meta.json` + `models/xg_model_split.json` + `models/xg_model_report.md` (committed with the 2026-11-12 boosters); `models/xg_model_meta.rds` rewritten from the JSON meta | — (committed) | derived at train time | `nhl_data_build/xg_train.py` | rides the suite's gates; `tests/test_xg_gates.py` bites on the committed meta/split; penalty-shot constant **0.3184** | 2026-11-12 (run 43181159544) | annual July cron + dispatch (`nhl_model_pipeline.yml`) |

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
- Gate floors (`_CV_AUC_FLOOR` in each stage) re-derived 2026-11-12 just below the complete-corpus fit of run 43165388160 (cv AUC 0.779 / 0.757 → floors **0.77** 5v5, **0.75** ST); never lowered, re-derived only from a fit on a corrected corpus. The 2026-04 floors (0.82 / 0.81) were fitted on a corpus with no missed shots (bullets below); `--quick` smoke runs tolerate misses. ST per-season calibration drift ceiling `max |z| ≤ 3.0` over the exact-holdout seasons (`nhl_model_02_xg_st._ST_DRIFT_MAX_ABS_Z`; z = (goals − ΣxG)/√ΣxG(1−xG) per season, `xg_train.per_season_calibration`, recorded in `info_st.per_season_calibration`), derived 2026-09-02 from the observed max |z| = 2.008 (season 2018, 15 holdout seasons) of the python ST retrain — observed + ~1 z-unit = the per-season 99.7% band, exceeded by a calibrated model's 15-season max ~4% of the time; a ceiling is never raised.
- Fingerprints: stages skip when `hash(code subtree, config)` is unchanged (`--force` to retrain); every trained model appends a `models/ledger.jsonl` line; each retrain rewrites the `xg_model_meta.json` + `xg_model_split.json` + `xg_model_report.md` sidecars (single-variant runs merge, never clobber — but two single-variant runs must not overlap in time: each reads the sidecars at start and writes at end).
- **Candidate/promote (2026-09-02).** A stage trains into `models/.candidate/<stage>/` (gitignored, seeded from the current champion so a single-variant sidecar merge still sees its sibling) and `nhl_data_build._stage.run_stage` copies the files onto the champion paths **only when `gate_pass` is not `False`**. A failing run leaves every champion file byte-identical, still appends its `models/ledger.jsonl` row, keeps the candidate for inspection, and does NOT record its fingerprint (so the next run retrains rather than skipping). Before this, the trainer wrote straight to `models/` and the gate was checked afterwards: the 2026-09-02 stage-01 retrain scored cv_auc **0.7786** with `gate_pass: false` and overwrote `models/xg_model_5v5.json` in place. Promotion itself is **all-or-nothing**: every champion file about to be overwritten is snapshotted to `models/.candidate/<stage>.rollback/` first and restored (rc 1) if any copy raises, so a partial sweep can never leave a new booster beside stale `xg_model_meta.json` / `xg_model_split.json`. Guarded by `tests/test_stage_promotion.py`. The candidate is seeded WITHOUT `models/ledger.jsonl`: a seeded copy was promoted back over the row the run had just appended, so both promoting stages of run 43181159544 left no ledger trace (fixed 2026-11-12; the two rows were reconstructed from the run's meta and carry `"run"`).
- Promotion = committing the retrained `models/*.json` (this repo's step-6 convention); CI only uploads run artifacts, it never pushes.
- **The 2026-04 champions do not price the current frame (measured 2026-09-02).** `nhl_data_build.xg_parity.artifact_calibration` scores an EXISTING booster on the frame `prepare_training_frame` builds now — the check no promotion gate can make, because `cv_auc` is scale-free and the ST drift ceiling reads a *retrain's own* meta. On `nhl/pbp/parquet` (the stages' own glob; 1,724,290 5v5 / 349,232 ST rows) the committed boosters score goals/ΣxG **0.7711** (5v5) and **0.7679** (ST) — a 25-30% over-prediction on every season through 2023-24. Dropping `MISSED_SHOT` from the same corpus moves them to **1.0618** / **1.0777**, and to 0.99-1.02 for each individual season 2009-10 … 2023-24 (2024-25 and 2025-26, the only two already near-calibrated on the Fenwick frame at 0.949 / 0.913, blow out to 1.365 / 1.327). That identifies the 2026-04 training corpus as one carrying no missed shots for those seasons. The R (`R/build_xg_model.R`) and python (`xg_train.prepare_training_frame`) recipes were proved identical on real games first — same rows, same values on 32 of 33 shared features — so this is an artifact/corpus defect, not a port defect.
- **The frozen AUC floors are therefore not achievable and must be RE-DERIVED, not lowered.** Re-fitting the R script's own params (max_depth 4, eta .06, gamma 1, subsample/colsample .8, min_child_weight 8, 1500 rounds, seed-37 grouped 80/20) on `nhl/pbp/parquet` gives holdout AUC **0.7794** 5v5 / **0.7542** ST with goals/ΣxG **1.0042** / **1.0064** — well calibrated, and matching the two python retrains recorded below (0.7786 / 0.7569) that were read as failures. The same fit on a SHOT|GOAL-only frame gives 0.7738 / **0.7611**, so the recorded 0.8322 / 0.8213 is not reproducible from any reconstruction of the current corpus. Floors stay where they are until a fit on a CORRECTED corpus sets new ones from its own observation — done 2026-11-12 (see Retrain attempts).
- **Corpus integrity (resolved before the re-derivation).** On 2026-09-02 neither pbp directory was complete (2014-15 missing from `nhl/pbp/parquet`; 2013-14 missing and 2018-19 doubled in `nhl/pbp/full`). The producer was fixed first; the 2026-11-12 fits read 18 seasons, one per file.


## Retrain attempts (python trainer, this repo's parquet frame)

- **2026-11-12 — promoted.** `nhl_model_pipeline.yml` run 43165388160 (old floors) on the complete corpus (18 seasons, 1,485,612 / 302,894 training shots) scored cv AUC **0.779** 5v5 / **0.757** ST with every holdout season calibrated (goals/ΣxG 0.96–1.03 / 0.95–1.06) and failed only the unreproducible 0.82 / 0.81 floors. Floors re-derived from it (0.77 / 0.75) and run **43181159544** retrained on the same corpus plus that week's 2026-27 games: 5v5 cv AUC **0.7787** (holdout 0.7797, 1,327 rounds, max |z| 1.84), ST **0.7570** (holdout 0.7557, 537 rounds, max |z| **2.846** in 2010-11, under the 3.0 ceiling; run 43165388160 saw 1.51 on a slightly smaller corpus, so ST drift is the margin to watch). Both promoted; boosters + sidecars committed from the two per-stage run artifacts (each job carries only its own variant, merged by hand). `in_published_data` stays false until the pbp `xg` column is rescored.
- **2026-09-02** — `scripts/nhl_models.sh --force 01` / `02` on the committed corpus (16 seasons). ST: cv AUC **0.7569** vs floor 0.81 → **FAIL**, not promoted (holdout AUC 0.7556, per-season calibration excellent: max |z| 2.008, goals/ΣxG 0.93–1.08). 5v5 result recorded in `models/ledger.jsonl`. The committed 2026-04 R boosters scored on the same python frame reach only 0.761 / 0.778 (ST / 5v5) and over-predict goals by 25–30% on every season through 2023-24 (|z| up to 9.2 / 14.8). **This was originally read as a python-FRAME-vs-R-frame parity problem. It is not** (settled 2026-09-02, see the champion-calibration bullet above): the two recipes were run on byte-identical input rows and returned the same rows and the same values on 32 of 33 shared features, so 0.7569 is an honest fit measured against a floor the current corpus cannot reach — not a frame deficit. The blocker is the champion artifact and the corpus, not the port. Ledger lines are committed; the run's sidecars were not (they would describe a non-promoted model).