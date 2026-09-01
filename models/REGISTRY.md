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

Trained on this repo's full NHL play-by-play (`nhl/pbp/full/rds/`, 16 seasons
2010-11 → 2025-26, five era groupings). Two trainers, one recipe:
`R/build_xg_model.R` (canonical) and its port
`python/nhl_data_build/xg_train.py` (grouped 80/20 split, logloss+auc CV).

| model | artifact(s) | release tag | training data | fitting script | gates at publish | last retrain | cadence |
|---|---|---|---|---|---|---|---|
| xG 5v5 | `models/xg_model_5v5.json` | — (committed; output ships in `nhl_pbp_full`) | nhl pbp 2011–2026, era-grouped | `R/build_xg_model.R` / `nhl_data_build/xg_train.py` | CV AUC **0.8322**, log-loss **0.2053** (README table; grouped split — no game leakage) | 2026-04 | as-needed / manual |
| xG special teams | `models/xg_model_st.json` | — (committed; output ships in `nhl_pbp_full`) | nhl pbp 2011–2026, era-grouped | `R/build_xg_model.R` / `nhl_data_build/xg_train.py` | CV AUC **0.8213**, log-loss **0.2567** | 2026-04 | as-needed / manual |
| xG meta (incl. penalty-shot constant) | `models/xg_model_meta.rds` | — (committed) | derived at train time | `R/build_xg_model.R` (python port emits `xg_model_meta.json`) | rides the suite's gates | 2026-04 | as-needed / manual |

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
- Gate floors frozen just below the 2026-04 observed values (5v5 cv AUC ≥ 0.82, ST ≥ 0.81); never lowered; `--quick` smoke runs tolerate misses.
- Fingerprints: stages skip when `hash(code subtree, config)` is unchanged (`--force` to retrain); every trained model appends a `models/ledger.jsonl` line; each retrain rewrites the `xg_model_meta.json` + `xg_model_report.md` sidecars (single-variant runs merge, never clobber).
- Promotion = committing the retrained `models/*.json` (this repo's step-6 convention); CI only uploads run artifacts, it never pushes.
