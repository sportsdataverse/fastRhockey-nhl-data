# NHL Expected Goals (fastRhockey xG suite)


The fastRhockey xG suite estimates the probability that an unblocked
shot becomes a goal, in two strength regimes: **5v5** (all non-shootout,
non-penalty-shot unblocked shots) and **special teams** (power play /
short-handed). A third component, the **penalty-shot constant**, is the
historical conversion rate computed at train time. The suite’s output
ships as the xG columns inside every published `nhl_pbp_full` season
asset — the model never ships alone; it reaches consumers embedded in
the play-by-play.

The models are XGBoost binary classifiers (logloss objective), a
faithful reproduction lineage of the hockeyR xG recipe (Morse 2022),
re-fit on this repository’s own committed corpus by
`python/nhl_data_build/xg_train.py` (canonical R twin
`R/build_xg_model.R`). The fitted boosters are **committed in this
repo** (`models/xg_model_{5v5,st}.json`) — that is the promotion step —
and every figure and number below is computed at render time from those
committed artifacts plus the committed play-by-play. The python trainer
also writes two sidecars beside the boosters, `xg_model_meta.json`
(feature lists, training-time CV and holdout metrics, per-season
calibration) and `xg_model_split.json` (the train/test `game_id`
partition); when they are committed the evaluation below scores the
*exact* held-out games, otherwise it replays the trainer’s split recipe
on today’s corpus and labels the result a near-holdout.

## Training data

<div id="ihmezeabms" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#ihmezeabms table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#ihmezeabms thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#ihmezeabms p { margin: 0; padding: 0; }
 #ihmezeabms .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #ihmezeabms .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #ihmezeabms .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #ihmezeabms .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #ihmezeabms .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #ihmezeabms .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ihmezeabms .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #ihmezeabms .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #ihmezeabms .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #ihmezeabms .gt_column_spanner_outer:first-child { padding-left: 0; }
 #ihmezeabms .gt_column_spanner_outer:last-child { padding-right: 0; }
 #ihmezeabms .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #ihmezeabms .gt_spanner_row { border-bottom-style: hidden; }
 #ihmezeabms .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #ihmezeabms .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #ihmezeabms .gt_from_md> :first-child { margin-top: 0; }
 #ihmezeabms .gt_from_md> :last-child { margin-bottom: 0; }
 #ihmezeabms .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #ihmezeabms .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #ihmezeabms .gt_indent_1 { text-indent: 5px; }
 #ihmezeabms .gt_indent_2 { text-indent: calc(5px * 2); }
 #ihmezeabms .gt_indent_3 { text-indent: calc(5px * 3); }
 #ihmezeabms .gt_indent_4 { text-indent: calc(5px * 4); }
 #ihmezeabms .gt_indent_5 { text-indent: calc(5px * 5); }
 #ihmezeabms .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #ihmezeabms .gt_row_group_first td { border-top-width: 2px; }
 #ihmezeabms .gt_row_group_first th { border-top-width: 2px; }
 #ihmezeabms .gt_striped { color: #333333; background-color: #F4F4F4; }
 #ihmezeabms .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ihmezeabms .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #ihmezeabms .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #ihmezeabms .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ihmezeabms .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #ihmezeabms .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #ihmezeabms .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #ihmezeabms .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ihmezeabms .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #ihmezeabms .gt_left { text-align: left; }
 #ihmezeabms .gt_center { text-align: center; }
 #ihmezeabms .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #ihmezeabms .gt_font_normal { font-weight: normal; }
 #ihmezeabms .gt_font_bold { font-weight: bold; }
 #ihmezeabms .gt_font_italic { font-style: italic; }
 #ihmezeabms .gt_super { font-size: 65%; }
 #ihmezeabms .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ihmezeabms .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #ihmezeabms .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ihmezeabms .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #ihmezeabms .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #ihmezeabms .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Committed NHL play-by-play corpus, by season |  |  |  |
|----|----|----|----|
| unblocked shot events (SHOT / MISSED_SHOT / GOAL); computed at render time |  |  |  |
| season | games | fenwick_events | goal_share |
| 20092010 | 1317 | 111,805 | 6.9% |
| 20102011 | 1318 | 112,525 | 6.7% |
| 20112012 | 1316 | 110,029 | 6.7% |
| 20122013 | 806 | 66,752 | 6.7% |
| 20132014 | 1323 | 112,057 | 6.7% |
| 20142015 | 1319 | 110,923 | 6.7% |
| 20152016 | 1321 | 110,264 | 6.6% |
| 20162017 | 1317 | 111,712 | 6.6% |
| 20172018 | 1355 | 120,553 | 6.8% |
| 20182019 | 1358 | 118,441 | 7.0% |
| 20192020 | 1212 | 105,036 | 7.0% |
| 20202021 | 952 | 79,119 | 7.1% |
| 20212022 | 1401 | 122,350 | 7.3% |
| 20222023 | 1400 | 122,711 | 7.4% |
| 20232024 | 1400 | 123,136 | 7.1% |
| 20242025 | 1398 | 120,462 | 7.1% |
| 20252026 | 1394 | 120,154 | 7.4% |
| 20262027 | 309 | 26,047 | 7.6% |

&#10;</div>

Training corpus: this repository’s committed play-by-play
(`nhl/pbp/parquet/`, seasons listed above, ~13.6M events). Seasons are
grouped into **five era buckets** (2011–2013, 2014–2018, 2019–2021,
2022–2024, 2025-on) that enter as one-hot features, so rule- and
tracking-era shifts are modeled rather than silently biasing the fit.
The 5v5/ST split is a modeling decision, not a convenience: shooting
talent and shot-quality distributions differ enough across strength
states that a single pooled model miscalibrates the power play.

## Exploratory data analysis

<img src="nhl_xg_files/figure-commonmark/cell-5-output-1.png"
width="420" height="300"
alt="Goal rate by shot distance, 5v5 model frame — the dominant feature." />

<div id="rqoqgwbigg" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#rqoqgwbigg table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#rqoqgwbigg thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#rqoqgwbigg p { margin: 0; padding: 0; }
 #rqoqgwbigg .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #rqoqgwbigg .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #rqoqgwbigg .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #rqoqgwbigg .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #rqoqgwbigg .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #rqoqgwbigg .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #rqoqgwbigg .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #rqoqgwbigg .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #rqoqgwbigg .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #rqoqgwbigg .gt_column_spanner_outer:first-child { padding-left: 0; }
 #rqoqgwbigg .gt_column_spanner_outer:last-child { padding-right: 0; }
 #rqoqgwbigg .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #rqoqgwbigg .gt_spanner_row { border-bottom-style: hidden; }
 #rqoqgwbigg .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #rqoqgwbigg .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #rqoqgwbigg .gt_from_md> :first-child { margin-top: 0; }
 #rqoqgwbigg .gt_from_md> :last-child { margin-bottom: 0; }
 #rqoqgwbigg .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #rqoqgwbigg .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #rqoqgwbigg .gt_indent_1 { text-indent: 5px; }
 #rqoqgwbigg .gt_indent_2 { text-indent: calc(5px * 2); }
 #rqoqgwbigg .gt_indent_3 { text-indent: calc(5px * 3); }
 #rqoqgwbigg .gt_indent_4 { text-indent: calc(5px * 4); }
 #rqoqgwbigg .gt_indent_5 { text-indent: calc(5px * 5); }
 #rqoqgwbigg .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #rqoqgwbigg .gt_row_group_first td { border-top-width: 2px; }
 #rqoqgwbigg .gt_row_group_first th { border-top-width: 2px; }
 #rqoqgwbigg .gt_striped { color: #333333; background-color: #F4F4F4; }
 #rqoqgwbigg .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #rqoqgwbigg .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #rqoqgwbigg .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #rqoqgwbigg .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #rqoqgwbigg .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #rqoqgwbigg .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #rqoqgwbigg .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #rqoqgwbigg .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #rqoqgwbigg .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #rqoqgwbigg .gt_left { text-align: left; }
 #rqoqgwbigg .gt_center { text-align: center; }
 #rqoqgwbigg .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #rqoqgwbigg .gt_font_normal { font-weight: normal; }
 #rqoqgwbigg .gt_font_bold { font-weight: bold; }
 #rqoqgwbigg .gt_font_italic { font-style: italic; }
 #rqoqgwbigg .gt_super { font-size: 65%; }
 #rqoqgwbigg .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #rqoqgwbigg .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #rqoqgwbigg .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #rqoqgwbigg .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #rqoqgwbigg .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #rqoqgwbigg .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Shot-type mix and conversion (5v5 frame) |         |           |
|------------------------------------------|---------|-----------|
| shot_type                                | shots   | goal_rate |
| wrist_shot                               | 742,874 | 5.5%      |
| slap_shot                                | 225,068 | 3.4%      |
| snap_shot                                | 221,139 | 6.6%      |
| backhand                                 | 113,994 | 7.6%      |
| tip_in                                   | 96,556  | 8.4%      |
| deflected                                | 29,518  | 9.0%      |
| wrap_around                              | 15,593  | 4.7%      |
| batted                                   | 1,472   | 11.2%     |
| poke                                     | 1,421   | 11.9%     |
| between_legs                             | 216     | 7.4%      |
| cradle                                   | 33      | 15.2%     |

&#10;</div>

Distance dominates conversion — from ~25% at the crease to ~2% beyond 60
ft — and the shot-type table explains the one-hot block: tips and
deflections convert well above wrist shots from similar range, and the
model needs the type to separate them. The special-teams frame adds
`total_skaters_on` and `event_team_advantage`, the two features that
encode the man-advantage state.

## Feature importance

<div id="ethnpmmqmh" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#ethnpmmqmh table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#ethnpmmqmh thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#ethnpmmqmh p { margin: 0; padding: 0; }
 #ethnpmmqmh .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #ethnpmmqmh .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #ethnpmmqmh .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #ethnpmmqmh .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #ethnpmmqmh .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #ethnpmmqmh .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ethnpmmqmh .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #ethnpmmqmh .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #ethnpmmqmh .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #ethnpmmqmh .gt_column_spanner_outer:first-child { padding-left: 0; }
 #ethnpmmqmh .gt_column_spanner_outer:last-child { padding-right: 0; }
 #ethnpmmqmh .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #ethnpmmqmh .gt_spanner_row { border-bottom-style: hidden; }
 #ethnpmmqmh .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #ethnpmmqmh .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #ethnpmmqmh .gt_from_md> :first-child { margin-top: 0; }
 #ethnpmmqmh .gt_from_md> :last-child { margin-bottom: 0; }
 #ethnpmmqmh .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #ethnpmmqmh .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #ethnpmmqmh .gt_indent_1 { text-indent: 5px; }
 #ethnpmmqmh .gt_indent_2 { text-indent: calc(5px * 2); }
 #ethnpmmqmh .gt_indent_3 { text-indent: calc(5px * 3); }
 #ethnpmmqmh .gt_indent_4 { text-indent: calc(5px * 4); }
 #ethnpmmqmh .gt_indent_5 { text-indent: calc(5px * 5); }
 #ethnpmmqmh .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #ethnpmmqmh .gt_row_group_first td { border-top-width: 2px; }
 #ethnpmmqmh .gt_row_group_first th { border-top-width: 2px; }
 #ethnpmmqmh .gt_striped { color: #333333; background-color: #F4F4F4; }
 #ethnpmmqmh .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ethnpmmqmh .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #ethnpmmqmh .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #ethnpmmqmh .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ethnpmmqmh .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #ethnpmmqmh .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #ethnpmmqmh .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #ethnpmmqmh .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ethnpmmqmh .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #ethnpmmqmh .gt_left { text-align: left; }
 #ethnpmmqmh .gt_center { text-align: center; }
 #ethnpmmqmh .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #ethnpmmqmh .gt_font_normal { font-weight: normal; }
 #ethnpmmqmh .gt_font_bold { font-weight: bold; }
 #ethnpmmqmh .gt_font_italic { font-style: italic; }
 #ethnpmmqmh .gt_super { font-size: 65%; }
 #ethnpmmqmh .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ethnpmmqmh .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #ethnpmmqmh .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ethnpmmqmh .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #ethnpmmqmh .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #ethnpmmqmh .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Top 12 features by gain — 5v5 booster (committed artifact) |          |
|------------------------------------------------------------|----------|
| feature                                                    | gain_5v5 |
| shot_distance                                              | 130.9    |
| cross_ice_event                                            | 68.2     |
| snap_shot                                                  | 60.6     |
| last_hit                                                   | 55.6     |
| empty_net                                                  | 54.1     |
| rebound                                                    | 52.8     |
| backhand                                                   | 46.3     |
| wrap_around                                                | 43.4     |
| wrist_shot                                                 | 37.8     |
| shot_angle                                                 | 37.4     |
| tip_in                                                     | 36.7     |
| last_shot                                                  | 36.3     |

&#10;</div>

<img src="nhl_xg_files/figure-commonmark/cell-8-output-1.png"
width="420" height="300"
alt="Gain importance, both committed boosters (top 12 each)." />

## SHAP

TreeSHAP attributions come directly from the committed boosters
(`pred_contribs=True` — exact for tree ensembles, no extra dependency).
The distribution of per-shot attributions for the top features shows how
each one moves individual shots on the log-odds scale:

<img src="nhl_xg_files/figure-commonmark/cell-9-output-1.png"
width="420" height="300"
alt="TreeSHAP per-shot attributions, 5v5 booster, top 8 features by mean |SHAP|; 4,000-shot sample." />

<div id="paloyhsifh" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#paloyhsifh table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#paloyhsifh thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#paloyhsifh p { margin: 0; padding: 0; }
 #paloyhsifh .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #paloyhsifh .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #paloyhsifh .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #paloyhsifh .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #paloyhsifh .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #paloyhsifh .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #paloyhsifh .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #paloyhsifh .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #paloyhsifh .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #paloyhsifh .gt_column_spanner_outer:first-child { padding-left: 0; }
 #paloyhsifh .gt_column_spanner_outer:last-child { padding-right: 0; }
 #paloyhsifh .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #paloyhsifh .gt_spanner_row { border-bottom-style: hidden; }
 #paloyhsifh .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #paloyhsifh .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #paloyhsifh .gt_from_md> :first-child { margin-top: 0; }
 #paloyhsifh .gt_from_md> :last-child { margin-bottom: 0; }
 #paloyhsifh .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #paloyhsifh .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #paloyhsifh .gt_indent_1 { text-indent: 5px; }
 #paloyhsifh .gt_indent_2 { text-indent: calc(5px * 2); }
 #paloyhsifh .gt_indent_3 { text-indent: calc(5px * 3); }
 #paloyhsifh .gt_indent_4 { text-indent: calc(5px * 4); }
 #paloyhsifh .gt_indent_5 { text-indent: calc(5px * 5); }
 #paloyhsifh .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #paloyhsifh .gt_row_group_first td { border-top-width: 2px; }
 #paloyhsifh .gt_row_group_first th { border-top-width: 2px; }
 #paloyhsifh .gt_striped { color: #333333; background-color: #F4F4F4; }
 #paloyhsifh .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #paloyhsifh .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #paloyhsifh .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #paloyhsifh .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #paloyhsifh .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #paloyhsifh .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #paloyhsifh .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #paloyhsifh .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #paloyhsifh .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #paloyhsifh .gt_left { text-align: left; }
 #paloyhsifh .gt_center { text-align: center; }
 #paloyhsifh .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #paloyhsifh .gt_font_normal { font-weight: normal; }
 #paloyhsifh .gt_font_bold { font-weight: bold; }
 #paloyhsifh .gt_font_italic { font-style: italic; }
 #paloyhsifh .gt_super { font-size: 65%; }
 #paloyhsifh .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #paloyhsifh .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #paloyhsifh .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #paloyhsifh .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #paloyhsifh .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #paloyhsifh .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Mean \|SHAP\| — 5v5 booster, 4,000-shot sample |               |
|------------------------------------------------|---------------|
| feature                                        | mean_abs_shap |
| shot_distance                                  | 0.9478        |
| shot_angle                                     | 0.2725        |
| time_since_last                                | 0.1667        |
| slap_shot                                      | 0.0838        |
| snap_shot                                      | 0.0678        |
| backhand                                       | 0.0676        |
| distance_from_last                             | 0.0637        |
| era_2025_on                                    | 0.0514        |

&#10;</div>

## Evaluation

Two views, honestly labeled. First, the **training-time metrics**:
grouped 5-fold CV on the 80% side (the gated number) and the
game-grouped 20% holdout the fit never saw. They come from
`models/xg_model_meta.json` when a python sidecar is committed,
otherwise from the 2026-04 R fit’s console output in the README (CV only
— that run persisted no holdout metrics and no partition, which is
exactly what the sidecars fix). The gate floors (`_CV_AUC_FLOOR` per
stage) sit just below an observed fit — re-derived 2026-11-12 from the
complete-corpus fit (5v5 cv AUC ≥ 0.77, ST ≥ 0.75) — and are never
lowered; a real retrain below them fails (`--quick` smoke runs tolerate
misses).

**Sidecar state:** `xg_model_meta.json` + `xg_model_split.json` are
committed — the holdout below is the exact training-time partition (fit
2026-11-13).

<div id="fbbqnjssed" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#fbbqnjssed table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#fbbqnjssed thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#fbbqnjssed p { margin: 0; padding: 0; }
 #fbbqnjssed .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #fbbqnjssed .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #fbbqnjssed .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #fbbqnjssed .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #fbbqnjssed .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #fbbqnjssed .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #fbbqnjssed .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #fbbqnjssed .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #fbbqnjssed .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #fbbqnjssed .gt_column_spanner_outer:first-child { padding-left: 0; }
 #fbbqnjssed .gt_column_spanner_outer:last-child { padding-right: 0; }
 #fbbqnjssed .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #fbbqnjssed .gt_spanner_row { border-bottom-style: hidden; }
 #fbbqnjssed .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #fbbqnjssed .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #fbbqnjssed .gt_from_md> :first-child { margin-top: 0; }
 #fbbqnjssed .gt_from_md> :last-child { margin-bottom: 0; }
 #fbbqnjssed .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #fbbqnjssed .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #fbbqnjssed .gt_indent_1 { text-indent: 5px; }
 #fbbqnjssed .gt_indent_2 { text-indent: calc(5px * 2); }
 #fbbqnjssed .gt_indent_3 { text-indent: calc(5px * 3); }
 #fbbqnjssed .gt_indent_4 { text-indent: calc(5px * 4); }
 #fbbqnjssed .gt_indent_5 { text-indent: calc(5px * 5); }
 #fbbqnjssed .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #fbbqnjssed .gt_row_group_first td { border-top-width: 2px; }
 #fbbqnjssed .gt_row_group_first th { border-top-width: 2px; }
 #fbbqnjssed .gt_striped { color: #333333; background-color: #F4F4F4; }
 #fbbqnjssed .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #fbbqnjssed .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #fbbqnjssed .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #fbbqnjssed .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #fbbqnjssed .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #fbbqnjssed .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #fbbqnjssed .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #fbbqnjssed .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #fbbqnjssed .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #fbbqnjssed .gt_left { text-align: left; }
 #fbbqnjssed .gt_center { text-align: center; }
 #fbbqnjssed .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #fbbqnjssed .gt_font_normal { font-weight: normal; }
 #fbbqnjssed .gt_font_bold { font-weight: bold; }
 #fbbqnjssed .gt_font_italic { font-style: italic; }
 #fbbqnjssed .gt_super { font-size: 65%; }
 #fbbqnjssed .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #fbbqnjssed .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #fbbqnjssed .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #fbbqnjssed .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #fbbqnjssed .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #fbbqnjssed .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Training-time metrics |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|
| grouped 5-fold CV on the 80% side; game-grouped 20% holdout; floors re-derived from an observed fit, never lowered |  |  |  |  |  |  |  |  |  |  |  |
| variant | source | train shots | holdout shots | holdout games | CV AUC | CV log-loss | holdout AUC | holdout log-loss | baseline log-loss | gate | passes |
| 5v5 | xg_model_meta.json (fit 2026-11-13) | 1,158,143 | 290,218 | 4,443 | 0.7820 | 0.1906 | 0.7819 | 0.1908 | 0.2213 | cv AUC ≥ 0.77 | True |
| st | xg_model_meta.json (fit 2026-11-13) | 327,917 | 81,808 | 4,440 | 0.7574 | 0.2806 | 0.7607 | 0.2772 | 0.3272 | cv AUC ≥ 0.75 | True |

&#10;</div>

Second, the **held-out games re-scored at render time** by the committed
boosters. With a committed split file this is today’s corpus filtered to
the persisted `test_game_ids`; games that landed after the fit are in
neither list and are reported separately as **out-of-time** shots — an
honest forward test that grows as seasons land — and `Δ vs meta`
(render-time holdout AUC minus the meta’s) is exactly 0 unless the
committed play-by-play was repaired after the fit. Without a split file
the set is the seed-37 replay described above.

<div id="wmrfeysuqd" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#wmrfeysuqd table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#wmrfeysuqd thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#wmrfeysuqd p { margin: 0; padding: 0; }
 #wmrfeysuqd .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #wmrfeysuqd .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #wmrfeysuqd .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #wmrfeysuqd .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #wmrfeysuqd .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #wmrfeysuqd .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wmrfeysuqd .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #wmrfeysuqd .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #wmrfeysuqd .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #wmrfeysuqd .gt_column_spanner_outer:first-child { padding-left: 0; }
 #wmrfeysuqd .gt_column_spanner_outer:last-child { padding-right: 0; }
 #wmrfeysuqd .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #wmrfeysuqd .gt_spanner_row { border-bottom-style: hidden; }
 #wmrfeysuqd .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #wmrfeysuqd .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #wmrfeysuqd .gt_from_md> :first-child { margin-top: 0; }
 #wmrfeysuqd .gt_from_md> :last-child { margin-bottom: 0; }
 #wmrfeysuqd .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #wmrfeysuqd .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #wmrfeysuqd .gt_indent_1 { text-indent: 5px; }
 #wmrfeysuqd .gt_indent_2 { text-indent: calc(5px * 2); }
 #wmrfeysuqd .gt_indent_3 { text-indent: calc(5px * 3); }
 #wmrfeysuqd .gt_indent_4 { text-indent: calc(5px * 4); }
 #wmrfeysuqd .gt_indent_5 { text-indent: calc(5px * 5); }
 #wmrfeysuqd .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #wmrfeysuqd .gt_row_group_first td { border-top-width: 2px; }
 #wmrfeysuqd .gt_row_group_first th { border-top-width: 2px; }
 #wmrfeysuqd .gt_striped { color: #333333; background-color: #F4F4F4; }
 #wmrfeysuqd .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wmrfeysuqd .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #wmrfeysuqd .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #wmrfeysuqd .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wmrfeysuqd .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #wmrfeysuqd .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #wmrfeysuqd .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #wmrfeysuqd .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wmrfeysuqd .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #wmrfeysuqd .gt_left { text-align: left; }
 #wmrfeysuqd .gt_center { text-align: center; }
 #wmrfeysuqd .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #wmrfeysuqd .gt_font_normal { font-weight: normal; }
 #wmrfeysuqd .gt_font_bold { font-weight: bold; }
 #wmrfeysuqd .gt_font_italic { font-style: italic; }
 #wmrfeysuqd .gt_super { font-size: 65%; }
 #wmrfeysuqd .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wmrfeysuqd .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #wmrfeysuqd .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wmrfeysuqd .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #wmrfeysuqd .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #wmrfeysuqd .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Held-out evaluation — exact holdout (render-time, current corpus) |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
| test_game_ids from models/xg_model_split.json; 0 post-fit games in the out-of-time set |  |  |  |  |  |  |  |  |
| variant | set | shots | games | goal_rate | logloss | baseline_logloss | rank_AUC | Δ vs meta |
| 5v5 | exact holdout | 290,218 | 4,443 | 5.8% | 0.1908 | 0.2213 | 0.7819 | −0.0000 |
| st | exact holdout | 81,808 | 4,440 | 10.1% | 0.2772 | 0.3272 | 0.7607 | −0.0000 |
| 5v5 | out-of-time (post-fit games) | 0 | 0 | — | — | — | — | — |
| st | out-of-time (post-fit games) | 0 | 0 | — | — | — | — | — |

&#10;</div>

<img src="nhl_xg_files/figure-commonmark/cell-14-output-1.png"
width="420" height="300"
alt="Calibration by xG decile, 5v5, held-out shots." />

<img src="nhl_xg_files/figure-commonmark/cell-15-output-1.png"
width="420" height="300"
alt="Per-era discrimination: rank AUC of the committed 5v5 booster by era bucket (held-out shots)." />

The penalty-shot constant computed from the current corpus:

<div id="lczudklmyi" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#lczudklmyi table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#lczudklmyi thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#lczudklmyi p { margin: 0; padding: 0; }
 #lczudklmyi .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #lczudklmyi .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #lczudklmyi .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #lczudklmyi .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #lczudklmyi .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #lczudklmyi .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #lczudklmyi .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #lczudklmyi .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #lczudklmyi .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #lczudklmyi .gt_column_spanner_outer:first-child { padding-left: 0; }
 #lczudklmyi .gt_column_spanner_outer:last-child { padding-right: 0; }
 #lczudklmyi .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #lczudklmyi .gt_spanner_row { border-bottom-style: hidden; }
 #lczudklmyi .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #lczudklmyi .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #lczudklmyi .gt_from_md> :first-child { margin-top: 0; }
 #lczudklmyi .gt_from_md> :last-child { margin-bottom: 0; }
 #lczudklmyi .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #lczudklmyi .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #lczudklmyi .gt_indent_1 { text-indent: 5px; }
 #lczudklmyi .gt_indent_2 { text-indent: calc(5px * 2); }
 #lczudklmyi .gt_indent_3 { text-indent: calc(5px * 3); }
 #lczudklmyi .gt_indent_4 { text-indent: calc(5px * 4); }
 #lczudklmyi .gt_indent_5 { text-indent: calc(5px * 5); }
 #lczudklmyi .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #lczudklmyi .gt_row_group_first td { border-top-width: 2px; }
 #lczudklmyi .gt_row_group_first th { border-top-width: 2px; }
 #lczudklmyi .gt_striped { color: #333333; background-color: #F4F4F4; }
 #lczudklmyi .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #lczudklmyi .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #lczudklmyi .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #lczudklmyi .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #lczudklmyi .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #lczudklmyi .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #lczudklmyi .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #lczudklmyi .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #lczudklmyi .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #lczudklmyi .gt_left { text-align: left; }
 #lczudklmyi .gt_center { text-align: center; }
 #lczudklmyi .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #lczudklmyi .gt_font_normal { font-weight: normal; }
 #lczudklmyi .gt_font_bold { font-weight: bold; }
 #lczudklmyi .gt_font_italic { font-style: italic; }
 #lczudklmyi .gt_super { font-size: 65%; }
 #lczudklmyi .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #lczudklmyi .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #lczudklmyi .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #lczudklmyi .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #lczudklmyi .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #lczudklmyi .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Penalty-shot xG constant (historical conversion rate) |        |
|-------------------------------------------------------|--------|
| component                                             | xG     |
| penalty-shot / shootout constant                      | 0.3184 |

&#10;</div>

## Per-season calibration drift

The special-teams sample is ~5× smaller than 5v5, so its AUC floor alone
cannot see a season whose *level* drifts — a rule or tracking change
that shifts conversion without reordering shots. The monitor is the
binomial calibration z-score per season,
`z = (goals − ΣxG) / sqrt(ΣxG · (1 − xG))`: a season the model prices
correctly has \|z\| ~ N(0, 1) whatever its shot count, which is what
makes `max |z|` comparable across seasons of unequal size. The trainer
records this table in `info_st.per_season_calibration` at every retrain
(`xg_train.per_season_calibration`), and stage `nhl_model_02_xg_st`
gates on `max |z| ≤ _ST_DRIFT_MAX_ABS_Z` — a ceiling derived from the
value observed on the exact holdout of the trusted retrain and never
raised (see `models/REGISTRY.md`). The same statistic is recomputed here
at render time on the held-out shots, with post-fit seasons appended as
out-of-time rows.

<div id="tdjimrygds" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#tdjimrygds table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#tdjimrygds thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#tdjimrygds p { margin: 0; padding: 0; }
 #tdjimrygds .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #tdjimrygds .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #tdjimrygds .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #tdjimrygds .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #tdjimrygds .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #tdjimrygds .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #tdjimrygds .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #tdjimrygds .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #tdjimrygds .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #tdjimrygds .gt_column_spanner_outer:first-child { padding-left: 0; }
 #tdjimrygds .gt_column_spanner_outer:last-child { padding-right: 0; }
 #tdjimrygds .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #tdjimrygds .gt_spanner_row { border-bottom-style: hidden; }
 #tdjimrygds .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #tdjimrygds .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #tdjimrygds .gt_from_md> :first-child { margin-top: 0; }
 #tdjimrygds .gt_from_md> :last-child { margin-bottom: 0; }
 #tdjimrygds .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #tdjimrygds .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #tdjimrygds .gt_indent_1 { text-indent: 5px; }
 #tdjimrygds .gt_indent_2 { text-indent: calc(5px * 2); }
 #tdjimrygds .gt_indent_3 { text-indent: calc(5px * 3); }
 #tdjimrygds .gt_indent_4 { text-indent: calc(5px * 4); }
 #tdjimrygds .gt_indent_5 { text-indent: calc(5px * 5); }
 #tdjimrygds .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #tdjimrygds .gt_row_group_first td { border-top-width: 2px; }
 #tdjimrygds .gt_row_group_first th { border-top-width: 2px; }
 #tdjimrygds .gt_striped { color: #333333; background-color: #F4F4F4; }
 #tdjimrygds .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #tdjimrygds .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #tdjimrygds .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #tdjimrygds .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #tdjimrygds .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #tdjimrygds .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #tdjimrygds .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #tdjimrygds .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #tdjimrygds .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #tdjimrygds .gt_left { text-align: left; }
 #tdjimrygds .gt_center { text-align: center; }
 #tdjimrygds .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #tdjimrygds .gt_font_normal { font-weight: normal; }
 #tdjimrygds .gt_font_bold { font-weight: bold; }
 #tdjimrygds .gt_font_italic { font-style: italic; }
 #tdjimrygds .gt_super { font-size: 65%; }
 #tdjimrygds .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #tdjimrygds .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #tdjimrygds .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #tdjimrygds .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #tdjimrygds .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #tdjimrygds .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Special-teams calibration by season — exact holdout (+ out-of-time) |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| ratio = goals / ΣxG; z = binomial calibration z-score; auc = per-season rank AUC |  |  |  |  |  |  |  |
| set | season | n | goals | xg_sum | ratio | z | auc |
| holdout | 2009 | 5,692 | 525 | 521.6 | 1.006 | 0.17 | 0.759 |
| holdout | 2010 | 5,164 | 439 | 462.6 | 0.949 | −1.22 | 0.751 |
| holdout | 2011 | 4,155 | 355 | 380.3 | 0.933 | −1.45 | 0.782 |
| holdout | 2012 | 3,191 | 315 | 294.0 | 1.071 | 1.37 | 0.759 |
| holdout | 2013 | 5,330 | 472 | 495.8 | 0.952 | −1.19 | 0.752 |
| holdout | 2014 | 4,712 | 426 | 455.6 | 0.935 | −1.57 | 0.744 |
| holdout | 2015 | 4,762 | 471 | 465.5 | 1.012 | 0.29 | 0.762 |
| holdout | 2016 | 4,688 | 492 | 464.9 | 1.058 | 1.43 | 0.765 |
| holdout | 2017 | 5,484 | 562 | 560.5 | 1.003 | 0.07 | 0.769 |
| holdout | 2018 | 4,458 | 476 | 478.1 | 0.996 | −0.11 | 0.759 |
| holdout | 2019 | 4,366 | 468 | 455.7 | 1.027 | 0.66 | 0.740 |
| holdout | 2020 | 3,439 | 353 | 363.8 | 0.970 | −0.65 | 0.775 |
| holdout | 2021 | 5,039 | 563 | 542.2 | 1.038 | 1.03 | 0.772 |
| holdout | 2022 | 5,461 | 578 | 622.6 | 0.928 | −2.04 | 0.746 |
| holdout | 2023 | 5,224 | 572 | 578.2 | 0.989 | −0.29 | 0.764 |
| holdout | 2024 | 4,523 | 515 | 479.8 | 1.073 | 1.84 | 0.771 |
| holdout | 2025 | 4,899 | 530 | 560.9 | 0.945 | −1.50 | 0.748 |
| holdout | 2026 | 1,221 | 146 | 139.3 | 1.048 | 0.64 | 0.751 |

&#10;</div>

<div id="yztwlznswa" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#yztwlznswa table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#yztwlznswa thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#yztwlznswa p { margin: 0; padding: 0; }
 #yztwlznswa .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #yztwlznswa .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #yztwlznswa .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #yztwlznswa .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #yztwlznswa .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #yztwlznswa .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #yztwlznswa .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #yztwlznswa .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #yztwlznswa .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #yztwlznswa .gt_column_spanner_outer:first-child { padding-left: 0; }
 #yztwlznswa .gt_column_spanner_outer:last-child { padding-right: 0; }
 #yztwlznswa .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #yztwlznswa .gt_spanner_row { border-bottom-style: hidden; }
 #yztwlznswa .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #yztwlznswa .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #yztwlznswa .gt_from_md> :first-child { margin-top: 0; }
 #yztwlznswa .gt_from_md> :last-child { margin-bottom: 0; }
 #yztwlznswa .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #yztwlznswa .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #yztwlznswa .gt_indent_1 { text-indent: 5px; }
 #yztwlznswa .gt_indent_2 { text-indent: calc(5px * 2); }
 #yztwlznswa .gt_indent_3 { text-indent: calc(5px * 3); }
 #yztwlznswa .gt_indent_4 { text-indent: calc(5px * 4); }
 #yztwlznswa .gt_indent_5 { text-indent: calc(5px * 5); }
 #yztwlznswa .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #yztwlznswa .gt_row_group_first td { border-top-width: 2px; }
 #yztwlznswa .gt_row_group_first th { border-top-width: 2px; }
 #yztwlznswa .gt_striped { color: #333333; background-color: #F4F4F4; }
 #yztwlznswa .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #yztwlznswa .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #yztwlznswa .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #yztwlznswa .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #yztwlznswa .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #yztwlznswa .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #yztwlznswa .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #yztwlznswa .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #yztwlznswa .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #yztwlznswa .gt_left { text-align: left; }
 #yztwlznswa .gt_center { text-align: center; }
 #yztwlznswa .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #yztwlznswa .gt_font_normal { font-weight: normal; }
 #yztwlznswa .gt_font_bold { font-weight: bold; }
 #yztwlznswa .gt_font_italic { font-style: italic; }
 #yztwlznswa .gt_super { font-size: 65%; }
 #yztwlznswa .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #yztwlznswa .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #yztwlznswa .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #yztwlznswa .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #yztwlznswa .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #yztwlznswa .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Stage gates (nhl_model_02_xg_st.st_gates)           |          |           |      |
|-----------------------------------------------------|----------|-----------|------|
| evaluated on the committed xg_model_meta.json       |          |           |      |
| check                                               | observed | threshold | pass |
| cv AUC ≥ 0.75                                       | 0.7574   | 0.75      | True |
| ST drift: max \|z\| over held-out seasons ≤ ceiling | 2.043    | 3.0       | True |
| stage verdict                                       |          |           | True |

&#10;</div>

<img src="nhl_xg_files/figure-commonmark/cell-19-output-1.png"
width="420" height="300"
alt="Per-season calibration: observed goals / ΣxG (points) against the ±2 SE band a calibrated model occupies (bars); the second point shape marks out-of-time seasons." />

Both boosters carry era one-hots, so a booster fit on this frame sits
inside the band in every season it was trained on; the informative rows
are the out-of-time ones, which is where a level drift first shows.

## Results — players and teams

<div id="ppcfllzwxc" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#ppcfllzwxc table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#ppcfllzwxc thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#ppcfllzwxc p { margin: 0; padding: 0; }
 #ppcfllzwxc .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #ppcfllzwxc .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #ppcfllzwxc .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #ppcfllzwxc .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #ppcfllzwxc .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #ppcfllzwxc .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ppcfllzwxc .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #ppcfllzwxc .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #ppcfllzwxc .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #ppcfllzwxc .gt_column_spanner_outer:first-child { padding-left: 0; }
 #ppcfllzwxc .gt_column_spanner_outer:last-child { padding-right: 0; }
 #ppcfllzwxc .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #ppcfllzwxc .gt_spanner_row { border-bottom-style: hidden; }
 #ppcfllzwxc .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #ppcfllzwxc .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #ppcfllzwxc .gt_from_md> :first-child { margin-top: 0; }
 #ppcfllzwxc .gt_from_md> :last-child { margin-bottom: 0; }
 #ppcfllzwxc .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #ppcfllzwxc .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #ppcfllzwxc .gt_indent_1 { text-indent: 5px; }
 #ppcfllzwxc .gt_indent_2 { text-indent: calc(5px * 2); }
 #ppcfllzwxc .gt_indent_3 { text-indent: calc(5px * 3); }
 #ppcfllzwxc .gt_indent_4 { text-indent: calc(5px * 4); }
 #ppcfllzwxc .gt_indent_5 { text-indent: calc(5px * 5); }
 #ppcfllzwxc .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #ppcfllzwxc .gt_row_group_first td { border-top-width: 2px; }
 #ppcfllzwxc .gt_row_group_first th { border-top-width: 2px; }
 #ppcfllzwxc .gt_striped { color: #333333; background-color: #F4F4F4; }
 #ppcfllzwxc .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ppcfllzwxc .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #ppcfllzwxc .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #ppcfllzwxc .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ppcfllzwxc .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #ppcfllzwxc .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #ppcfllzwxc .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #ppcfllzwxc .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ppcfllzwxc .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #ppcfllzwxc .gt_left { text-align: left; }
 #ppcfllzwxc .gt_center { text-align: center; }
 #ppcfllzwxc .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #ppcfllzwxc .gt_font_normal { font-weight: normal; }
 #ppcfllzwxc .gt_font_bold { font-weight: bold; }
 #ppcfllzwxc .gt_font_italic { font-style: italic; }
 #ppcfllzwxc .gt_super { font-size: 65%; }
 #ppcfllzwxc .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ppcfllzwxc .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #ppcfllzwxc .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ppcfllzwxc .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #ppcfllzwxc .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #ppcfllzwxc .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Top 15 skaters by 5v5 expected goals — season 2026-2027 |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
| unblocked 5v5 shots scored by the committed booster; GAX = goals above expected |  |  |  |  |  |  |
|  | Player | Team | Shots | xG | G | GAX |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8479542.png"
height="42" /> | Brandon Hagel | TBL | 83 | 6.80 | 10 | 3.20 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8481559.png"
height="42" /> | Jack Hughes | NJD | 79 | 6.43 | 3 | −3.43 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8482702.png"
height="42" /> | Logan Stankoven | CAR | 72 | 6.13 | 8 | 1.87 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8482745.png"
height="42" /> | Mason McTavish | STL | 52 | 6.02 | 6 | −0.02 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8478398.png"
height="42" /> | Kyle Connor | WPG | 79 | 5.98 | 6 | 0.02 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8479420.png"
height="42" /> | Tage Thompson | BUF | 70 | 5.37 | 9 | 3.63 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8478493.png"
height="42" /> | Joel Eriksson Ek | MIN | 57 | 5.37 | 4 | −1.37 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8478414.png"
height="42" /> | Timo Meier | NJD | 86 | 5.32 | 2 | −3.32 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8477956.png"
height="42" /> | David Pastrnak | BOS | 70 | 5.14 | 5 | −0.14 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8479318.png"
height="42" /> | Auston Matthews | TOR | 70 | 5.07 | 6 | 0.93 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8484210.png"
height="42" /> | Gabe Perreault | NYR | 50 | 4.93 | 6 | 1.07 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8476468.png"
height="42" /> | J.T. Miller | NYR | 43 | 4.93 | 3 | −1.93 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8484794.png"
height="42" /> | Michael Brandsegg-Nygård | DET | 52 | 4.87 | 3 | −1.87 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8475314.png"
height="42" /> | Anders Lee | UTA | 57 | 4.80 | 3 | −1.80 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8481604.png"
height="42" /> | Pavel Dorofeyev | NYR | 57 | 4.75 | 3 | −1.75 |

&#10;</div>

<div id="qpohanawpc" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#qpohanawpc table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#qpohanawpc thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#qpohanawpc p { margin: 0; padding: 0; }
 #qpohanawpc .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #qpohanawpc .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #qpohanawpc .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #qpohanawpc .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #qpohanawpc .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #qpohanawpc .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #qpohanawpc .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #qpohanawpc .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #qpohanawpc .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #qpohanawpc .gt_column_spanner_outer:first-child { padding-left: 0; }
 #qpohanawpc .gt_column_spanner_outer:last-child { padding-right: 0; }
 #qpohanawpc .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #qpohanawpc .gt_spanner_row { border-bottom-style: hidden; }
 #qpohanawpc .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #qpohanawpc .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #qpohanawpc .gt_from_md> :first-child { margin-top: 0; }
 #qpohanawpc .gt_from_md> :last-child { margin-bottom: 0; }
 #qpohanawpc .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #qpohanawpc .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #qpohanawpc .gt_indent_1 { text-indent: 5px; }
 #qpohanawpc .gt_indent_2 { text-indent: calc(5px * 2); }
 #qpohanawpc .gt_indent_3 { text-indent: calc(5px * 3); }
 #qpohanawpc .gt_indent_4 { text-indent: calc(5px * 4); }
 #qpohanawpc .gt_indent_5 { text-indent: calc(5px * 5); }
 #qpohanawpc .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #qpohanawpc .gt_row_group_first td { border-top-width: 2px; }
 #qpohanawpc .gt_row_group_first th { border-top-width: 2px; }
 #qpohanawpc .gt_striped { color: #333333; background-color: #F4F4F4; }
 #qpohanawpc .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #qpohanawpc .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #qpohanawpc .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #qpohanawpc .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #qpohanawpc .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #qpohanawpc .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #qpohanawpc .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #qpohanawpc .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #qpohanawpc .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #qpohanawpc .gt_left { text-align: left; }
 #qpohanawpc .gt_center { text-align: center; }
 #qpohanawpc .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #qpohanawpc .gt_font_normal { font-weight: normal; }
 #qpohanawpc .gt_font_bold { font-weight: bold; }
 #qpohanawpc .gt_font_italic { font-style: italic; }
 #qpohanawpc .gt_super { font-size: 65%; }
 #qpohanawpc .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #qpohanawpc .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #qpohanawpc .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #qpohanawpc .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #qpohanawpc .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #qpohanawpc .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Team 5v5 shot generation — season 2026-2027 |       |        |           |       |
|---------------------------------------------|-------|--------|-----------|-------|
| team                                        | shots | xG_for | goals_for | GAX   |
| CAR                                         | 789   | 50.4   | 58        | 7.6   |
| UTA                                         | 722   | 46.8   | 37        | −9.8  |
| NJD                                         | 742   | 45.2   | 45        | −0.2  |
| COL                                         | 699   | 42.5   | 59        | 16.5  |
| PIT                                         | 644   | 42.5   | 39        | −3.5  |
| TBL                                         | 644   | 42.4   | 51        | 8.6   |
| VGK                                         | 667   | 42.2   | 42        | −0.2  |
| MIN                                         | 686   | 42.1   | 43        | 0.9   |
| BOS                                         | 603   | 41.8   | 48        | 6.2   |
| NYR                                         | 591   | 41.2   | 34        | −7.2  |
| DET                                         | 656   | 40.7   | 35        | −5.7  |
| BUF                                         | 610   | 39.4   | 37        | −2.4  |
| STL                                         | 596   | 38.9   | 34        | −4.9  |
| MTL                                         | 619   | 38.4   | 45        | 6.6   |
| OTT                                         | 578   | 37.8   | 39        | 1.2   |
| TOR                                         | 683   | 37.6   | 42        | 4.4   |
| WSH                                         | 547   | 37.6   | 41        | 3.4   |
| CGY                                         | 672   | 37.3   | 27        | −10.3 |
| WPG                                         | 639   | 37.2   | 40        | 2.8   |
| NSH                                         | 576   | 37.0   | 33        | −4.0  |
| NYI                                         | 549   | 36.6   | 35        | −1.6  |
| FLA                                         | 600   | 35.4   | 33        | −2.4  |
| SJS                                         | 632   | 35.3   | 27        | −8.3  |
| PHI                                         | 570   | 34.2   | 28        | −6.2  |
| ANA                                         | 546   | 33.9   | 36        | 2.1   |
| DAL                                         | 528   | 33.7   | 32        | −1.7  |
| EDM                                         | 568   | 33.4   | 41        | 7.6   |
| CBJ                                         | 543   | 33.0   | 25        | −8.0  |
| LAK                                         | 535   | 31.7   | 25        | −6.7  |
| SEA                                         | 494   | 30.3   | 41        | 10.7  |
| CHI                                         | 525   | 30.1   | 27        | −3.1  |
| VAN                                         | 515   | 30.0   | 34        | 4.0   |

&#10;</div>

Positive GAX marks finishing above shot quality. At the team level,
xG-for minus goals-for separates shot-generation strength from shooting
luck — the spread between the two is exactly what the embedded xG
columns in `nhl_pbp_full` exist to expose.

## Provenance & reproducibility

- **Trained on:** this repository’s committed play-by-play
  (`nhl/pbp/parquet/`, seasons in the corpus table above); committed
  boosters = the python 2026-11-12 fit (`nhl_model_pipeline.yml` run
  43181159544); grouped 80/20 split by `game_id`, grouped 5-fold CV for
  `min_child_weight`, seed 37 (the python trainer persists the partition
  in `models/xg_model_split.json`).
- **Artifacts:** `models/xg_model_5v5.json`, `models/xg_model_st.json`
  (committed — the promotion step); python sidecars
  `models/xg_model_meta.json` (feature lists, CV + exact-holdout
  metrics, per-season calibration, PS constant) and
  `models/xg_model_split.json`, committed with the boosters; every stage
  run appends a `models/ledger.jsonl` line (including failed ones).
  Output ships inside `nhl_pbp_full`.
- **Retrain:** `scripts/nhl_models.sh` (stages `nhl_model_01_xg_5v5` /
  `nhl_model_02_xg_st`; fingerprint-skipped when unchanged, `--force` to
  retrain); gates recorded in `models/manifest.yaml` (cv AUC floors +
  the ST per-season drift ceiling), registry row in
  `models/REGISTRY.md`.
- **Rebuild this document:** `scripts/render_model_docs.sh` (Quarto →
  GFM; uses this repo’s `.venv` via `QUARTO_PYTHON`;
  `uv sync --group docs --group train`).

## Avenues for improvement & open issues

- **Pre-shot context** — rush/rebound flags exist, but passing sequences
  and screens do not; public-feed models plateau here, so the honest
  gain is better rebound/rush definitions, not more trees.
- **Resolved (2026-11-12):** *Commit the python sidecars / exact
  holdout* — the 2026-11-12 retrain passed the re-derived floors and
  committed `xg_model_meta.json` + `xg_model_split.json`; the evaluation
  above is the exact training-time holdout.
- **Resolved (2026-09-01, PR \#7):** *Per-season ST calibration drift* —
  monitored by the per-season table and figure above; the trainer
  records the statistic in the meta and stage `nhl_model_02_xg_st` gates
  on the observed-value-derived ceiling `max |z| ≤ _ST_DRIFT_MAX_ABS_Z`.
- **Resolved (2026-11-12):** the 2026-04 R boosters over-predicted goals
  by 25–30% through 2023-24 on this frame because they were fitted on a
  corpus carrying no missed shots (settled 2026-09-02; the R and python
  recipes match on real games). They are replaced by the python fit on
  the complete corpus, calibrated per season (5v5 max \|z\| 1.84, ST
  2.85 under the 3.0 ceiling). The published pbp `xg` column keeps the
  2026-04 scores until it is rescored.
