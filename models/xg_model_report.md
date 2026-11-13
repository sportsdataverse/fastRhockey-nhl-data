# fastRhockey xG model training report

Three models keyed off `strength_state`: a 5v5 model (all non-shootout/non-PS unblocked
shots), a special-teams model, and a penalty-shot xG constant. Grouped 80/20 split + 5-fold
grouped CV + min_child_weight grid (XGBoost `binary:logistic`).

| Model | train / test | min_child_weight | nrounds | CV logloss | CV AUC | Test logloss | Test AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| 5v5 | 1485915 / 372192 | 7 | 1327 | 0.2116 | 0.7787 | 0.2112 | 0.7797 |
| Special teams | 303626 / 75335 | 8 | 537 | 0.2825 | 0.7570 | 0.2806 | 0.7557 |
| Penalty shot | - | - | - | - | - | - | constant **0.3184** |

## Era groupings

- `era_2011_2013` - 2010-11 through 2012-13
- `era_2014_2018` - 2013-14 through 2017-18
- `era_2019_2021` - 2018-19 through 2020-21
- `era_2022_2024` - 2021-22 through 2023-24
- `era_2025_on` - 2024-25 and beyond

## 5v5 feature importance (top 15, by gain)

Goal rate 0.0677 | 36 features.

| feature | gain |
|---|---:|
| `empty_net` | 559.39 |
| `rebound` | 131.61 |
| `shot_distance` | 123.6 |
| `snap_shot` | 67.04 |
| `cross_ice_event` | 55.96 |
| `backhand` | 51.91 |
| `wrap_around` | 47.38 |
| `slap_shot` | 46.56 |
| `last_hit` | 46.0 |
| `wrist_shot` | 44.7 |
| `era_2025_on` | 39.72 |
| `shot_angle` | 33.17 |
| `tip_in` | 26.73 |
| `time_since_last` | 23.75 |
| `era_2022_2024` | 23.61 |

## Special teams feature importance (top 15, by gain)

Goal rate 0.1024 | 38 features.

| feature | gain |
|---|---:|
| `empty_net` | 379.47 |
| `rebound` | 83.58 |
| `shot_distance` | 57.48 |
| `event_team_advantage` | 45.58 |
| `snap_shot` | 41.04 |
| `total_skaters_on` | 35.84 |
| `last_faceoff` | 33.6 |
| `cross_ice_event` | 27.81 |
| `era_2025_on` | 24.03 |
| `wrist_shot` | 20.99 |
| `time_since_last` | 20.21 |
| `backhand` | 19.88 |
| `slap_shot` | 17.87 |
| `era_2022_2024` | 17.13 |
| `tip_in` | 15.8 |
