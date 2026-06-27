# nhl-data-build (Python)

Python port of the `fastRhockey-nhl-data` reshaper + xG model training, so the NHL data
pipeline produces the `sportsdataverse-data` `nhl_*` releases **without R** — the consumer
half of NHL Python self-sufficiency (the producer half is the scraper in
[`fastRhockey-nhl-raw/python`](https://github.com/sportsdataverse/fastRhockey-nhl-raw)).

The (Python) scraper writes `final/{game_id}.json`; this reads it and reshapes.

## Pipeline

| step | module | R source |
|---|---|---|
| extract per-game datasets | `extract.py` (`extract_all`) | `nhl_data_creation.R::.extract_all` |
| compile a season | `build.py` (`build_season`, `pbp_lite`) | the per-key bind + distinct |
| flatten nested cols for parquet | `flatten.py` (`flatten_struct_cols`) | `.flatten_struct_cols` |
| read final dir → write parquet | `season.py` (`build_season_from_dir`, `write_datasets`) | the per-season loop |
| publish to releases | `publish.py` (`publish_season`) | `.upload_to_release` |
| retrain xG models | `xg_train.py` (`train_xg_models`) | `build_xg_model.R` |

## Run

```sh
# compile a season's datasets from the raw repo's final JSONs
uv run python -m nhl_data_build.season -s 2025 --final-dir ../../fastRhockey-nhl-raw/nhl/json/final --out-dir nhl
# retrain xG models from a pbp parquet
uv run python -c "import polars as pl; from nhl_data_build.xg_train import train_xg_models; train_xg_models(pl.read_parquet('nhl/pbp/parquet/play_by_play_2025.parquet'), 'models')"
```

## Parity (hermetic, vs R's local `nhl/{key}/parquet`, sliced to game 2024020001)

**13 datasets** GREEN — skater/goalie/team box, game_info, game_rosters, officials,
scratches, linescore, shifts (flat) + scoring, penalties, three_stars, shots_by_period
(nested → `flatten_struct_cols`). pbp shape verified (850 full / 349 lite); its event
columns are validated upstream in `fastRhockey-nhl-raw`. xG training live-validated on the
2025 season (36/38 feats matching the canonical boosters). **20 tests, ruff clean.**

CI: `.github/workflows/daily_nhl_python.yml` (workflow_dispatch + repository_dispatch;
add the cron and retire `daily_nhl.yml` to cut over).
