# fastRhockey xG model training report

Three models keyed off `strength_state`: a 5v5 model (all non-shootout/non-PS unblocked
shots), a special-teams model, and a penalty-shot xG constant. Grouped 80/20 split + 5-fold
grouped CV + min_child_weight grid (XGBoost `binary:logistic`).

| Model | train / test | min_child_weight | nrounds | CV logloss | CV AUC | Test logloss | Test AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| 5v5 | 1158143 / 290218 | 6 | 956 | 0.1906 | 0.7820 | 0.1908 | 0.7819 |
| Special teams | 327917 / 81808 | 1 | 521 | 0.2806 | 0.7574 | 0.2772 | 0.7607 |
| Penalty shot | - | - | - | - | - | - | constant **0.3184** |

## Era groupings

- `era_2011_2013` - 2010-11 through 2012-13
- `era_2014_2018` - 2013-14 through 2017-18
- `era_2019_2021` - 2018-19 through 2020-21
- `era_2022_2024` - 2021-22 through 2023-24
- `era_2025_on` - 2024-25 and beyond

## 5v5 feature importance (top 15, by gain)

Goal rate 0.0579 | 36 features.

| feature | gain |
|---|---:|
| `shot_distance` | 130.86 |
| `cross_ice_event` | 68.18 |
| `snap_shot` | 60.58 |
| `last_hit` | 55.63 |
| `empty_net` | 54.13 |
| `rebound` | 52.82 |
| `backhand` | 46.31 |
| `wrap_around` | 43.43 |
| `wrist_shot` | 37.78 |
| `shot_angle` | 37.41 |
| `tip_in` | 36.74 |
| `last_shot` | 36.28 |
| `era_2025_on` | 35.21 |
| `last_faceoff` | 34.14 |
| `slap_shot` | 32.7 |

## Special teams feature importance (top 15, by gain)

Goal rate 0.1016 | 38 features.

| feature | gain |
|---|---:|
| `empty_net` | 351.92 |
| `shot_distance` | 70.36 |
| `event_team_advantage` | 57.07 |
| `total_skaters_on` | 51.12 |
| `snap_shot` | 39.43 |
| `cross_ice_event` | 37.43 |
| `time_since_last` | 26.73 |
| `last_faceoff` | 26.23 |
| `wrist_shot` | 22.35 |
| `backhand` | 22.05 |
| `era_2025_on` | 21.28 |
| `slap_shot` | 20.41 |
| `tip_in` | 17.5 |
| `last_hit` | 16.95 |
| `rebound` | 16.14 |
