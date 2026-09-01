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
committed artifacts plus the committed play-by-play.

## Training data

<div id="nhetasxkhf" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#nhetasxkhf table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#nhetasxkhf thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#nhetasxkhf p { margin: 0; padding: 0; }
 #nhetasxkhf .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #nhetasxkhf .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #nhetasxkhf .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #nhetasxkhf .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #nhetasxkhf .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #nhetasxkhf .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #nhetasxkhf .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #nhetasxkhf .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #nhetasxkhf .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #nhetasxkhf .gt_column_spanner_outer:first-child { padding-left: 0; }
 #nhetasxkhf .gt_column_spanner_outer:last-child { padding-right: 0; }
 #nhetasxkhf .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #nhetasxkhf .gt_spanner_row { border-bottom-style: hidden; }
 #nhetasxkhf .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #nhetasxkhf .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #nhetasxkhf .gt_from_md> :first-child { margin-top: 0; }
 #nhetasxkhf .gt_from_md> :last-child { margin-bottom: 0; }
 #nhetasxkhf .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #nhetasxkhf .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #nhetasxkhf .gt_indent_1 { text-indent: 5px; }
 #nhetasxkhf .gt_indent_2 { text-indent: calc(5px * 2); }
 #nhetasxkhf .gt_indent_3 { text-indent: calc(5px * 3); }
 #nhetasxkhf .gt_indent_4 { text-indent: calc(5px * 4); }
 #nhetasxkhf .gt_indent_5 { text-indent: calc(5px * 5); }
 #nhetasxkhf .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #nhetasxkhf .gt_row_group_first td { border-top-width: 2px; }
 #nhetasxkhf .gt_row_group_first th { border-top-width: 2px; }
 #nhetasxkhf .gt_striped { color: #333333; background-color: #F4F4F4; }
 #nhetasxkhf .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #nhetasxkhf .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #nhetasxkhf .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #nhetasxkhf .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #nhetasxkhf .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #nhetasxkhf .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #nhetasxkhf .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #nhetasxkhf .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #nhetasxkhf .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #nhetasxkhf .gt_left { text-align: left; }
 #nhetasxkhf .gt_center { text-align: center; }
 #nhetasxkhf .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #nhetasxkhf .gt_font_normal { font-weight: normal; }
 #nhetasxkhf .gt_font_bold { font-weight: bold; }
 #nhetasxkhf .gt_font_italic { font-style: italic; }
 #nhetasxkhf .gt_super { font-size: 65%; }
 #nhetasxkhf .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #nhetasxkhf .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #nhetasxkhf .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #nhetasxkhf .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #nhetasxkhf .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #nhetasxkhf .gt_asterisk { font-size: 100%; vertical-align: 0; }
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
| 20152016 | 1321 | 110,264 | 6.6% |
| 20162017 | 1317 | 111,712 | 6.6% |
| 20172018 | 1355 | 120,553 | 6.8% |
| 20182019 | 1358 | 118,441 | 7.0% |
| 20192020 | 1200 | 104,050 | 7.0% |
| 20202021 | 952 | 79,119 | 7.1% |
| 20212022 | 1401 | 122,350 | 7.3% |
| 20222023 | 1400 | 122,711 | 7.4% |
| 20232024 | 1400 | 123,136 | 7.1% |
| 20242025 | 1398 | 120,462 | 7.1% |
| 20252026 | 1394 | 120,154 | 7.4% |

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

<div id="fmhoxeeavb" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#fmhoxeeavb table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#fmhoxeeavb thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#fmhoxeeavb p { margin: 0; padding: 0; }
 #fmhoxeeavb .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #fmhoxeeavb .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #fmhoxeeavb .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #fmhoxeeavb .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #fmhoxeeavb .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #fmhoxeeavb .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #fmhoxeeavb .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #fmhoxeeavb .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #fmhoxeeavb .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #fmhoxeeavb .gt_column_spanner_outer:first-child { padding-left: 0; }
 #fmhoxeeavb .gt_column_spanner_outer:last-child { padding-right: 0; }
 #fmhoxeeavb .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #fmhoxeeavb .gt_spanner_row { border-bottom-style: hidden; }
 #fmhoxeeavb .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #fmhoxeeavb .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #fmhoxeeavb .gt_from_md> :first-child { margin-top: 0; }
 #fmhoxeeavb .gt_from_md> :last-child { margin-bottom: 0; }
 #fmhoxeeavb .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #fmhoxeeavb .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #fmhoxeeavb .gt_indent_1 { text-indent: 5px; }
 #fmhoxeeavb .gt_indent_2 { text-indent: calc(5px * 2); }
 #fmhoxeeavb .gt_indent_3 { text-indent: calc(5px * 3); }
 #fmhoxeeavb .gt_indent_4 { text-indent: calc(5px * 4); }
 #fmhoxeeavb .gt_indent_5 { text-indent: calc(5px * 5); }
 #fmhoxeeavb .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #fmhoxeeavb .gt_row_group_first td { border-top-width: 2px; }
 #fmhoxeeavb .gt_row_group_first th { border-top-width: 2px; }
 #fmhoxeeavb .gt_striped { color: #333333; background-color: #F4F4F4; }
 #fmhoxeeavb .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #fmhoxeeavb .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #fmhoxeeavb .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #fmhoxeeavb .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #fmhoxeeavb .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #fmhoxeeavb .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #fmhoxeeavb .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #fmhoxeeavb .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #fmhoxeeavb .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #fmhoxeeavb .gt_left { text-align: left; }
 #fmhoxeeavb .gt_center { text-align: center; }
 #fmhoxeeavb .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #fmhoxeeavb .gt_font_normal { font-weight: normal; }
 #fmhoxeeavb .gt_font_bold { font-weight: bold; }
 #fmhoxeeavb .gt_font_italic { font-style: italic; }
 #fmhoxeeavb .gt_super { font-size: 65%; }
 #fmhoxeeavb .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #fmhoxeeavb .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #fmhoxeeavb .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #fmhoxeeavb .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #fmhoxeeavb .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #fmhoxeeavb .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Shot-type mix and conversion (5v5 frame) |         |           |
|------------------------------------------|---------|-----------|
| shot_type                                | shots   | goal_rate |
| wrist_shot                               | 864,283 | 6.6%      |
| slap_shot                                | 295,807 | 4.2%      |
| snap_shot                                | 259,008 | 7.7%      |
| backhand                                 | 129,096 | 8.6%      |
| tip_in                                   | 119,157 | 9.5%      |
| deflected                                | 36,344  | 9.6%      |
| wrap_around                              | 15,935  | 5.1%      |
| batted                                   | 1,824   | 12.1%     |
| poke                                     | 1,781   | 12.8%     |
| between_legs                             | 305     | 9.5%      |
| cradle                                   | 35      | 14.3%     |

&#10;</div>

Distance dominates conversion — from ~25% at the crease to ~2% beyond 60
ft — and the shot-type table explains the one-hot block: tips and
deflections convert well above wrist shots from similar range, and the
model needs the type to separate them. The special-teams frame adds
`total_skaters_on` and `event_team_advantage`, the two features that
encode the man-advantage state.

## Feature importance

<div id="wisozeqbfu" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#wisozeqbfu table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#wisozeqbfu thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#wisozeqbfu p { margin: 0; padding: 0; }
 #wisozeqbfu .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #wisozeqbfu .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #wisozeqbfu .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #wisozeqbfu .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #wisozeqbfu .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #wisozeqbfu .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wisozeqbfu .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #wisozeqbfu .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #wisozeqbfu .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #wisozeqbfu .gt_column_spanner_outer:first-child { padding-left: 0; }
 #wisozeqbfu .gt_column_spanner_outer:last-child { padding-right: 0; }
 #wisozeqbfu .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #wisozeqbfu .gt_spanner_row { border-bottom-style: hidden; }
 #wisozeqbfu .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #wisozeqbfu .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #wisozeqbfu .gt_from_md> :first-child { margin-top: 0; }
 #wisozeqbfu .gt_from_md> :last-child { margin-bottom: 0; }
 #wisozeqbfu .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #wisozeqbfu .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #wisozeqbfu .gt_indent_1 { text-indent: 5px; }
 #wisozeqbfu .gt_indent_2 { text-indent: calc(5px * 2); }
 #wisozeqbfu .gt_indent_3 { text-indent: calc(5px * 3); }
 #wisozeqbfu .gt_indent_4 { text-indent: calc(5px * 4); }
 #wisozeqbfu .gt_indent_5 { text-indent: calc(5px * 5); }
 #wisozeqbfu .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #wisozeqbfu .gt_row_group_first td { border-top-width: 2px; }
 #wisozeqbfu .gt_row_group_first th { border-top-width: 2px; }
 #wisozeqbfu .gt_striped { color: #333333; background-color: #F4F4F4; }
 #wisozeqbfu .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wisozeqbfu .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #wisozeqbfu .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #wisozeqbfu .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wisozeqbfu .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #wisozeqbfu .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #wisozeqbfu .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #wisozeqbfu .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wisozeqbfu .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #wisozeqbfu .gt_left { text-align: left; }
 #wisozeqbfu .gt_center { text-align: center; }
 #wisozeqbfu .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #wisozeqbfu .gt_font_normal { font-weight: normal; }
 #wisozeqbfu .gt_font_bold { font-weight: bold; }
 #wisozeqbfu .gt_font_italic { font-style: italic; }
 #wisozeqbfu .gt_super { font-size: 65%; }
 #wisozeqbfu .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wisozeqbfu .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #wisozeqbfu .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wisozeqbfu .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #wisozeqbfu .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #wisozeqbfu .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Top 12 features by gain — 5v5 booster (committed artifact) |          |
|------------------------------------------------------------|----------|
| feature                                                    | gain_5v5 |
| empty_net                                                  | 568.4    |
| rebound                                                    | 113.9    |
| snap_shot                                                  | 94.6     |
| wrist_shot                                                 | 84.9     |
| shot_distance                                              | 78.8     |
| slap_shot                                                  | 77.0     |
| tip_in                                                     | 54.2     |
| last_hit                                                   | 42.6     |
| backhand                                                   | 40.2     |
| era_2025_on                                                | 39.3     |
| deflected                                                  | 38.3     |
| last_faceoff                                               | 27.0     |

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

<div id="kbxvkmpncy" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#kbxvkmpncy table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#kbxvkmpncy thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#kbxvkmpncy p { margin: 0; padding: 0; }
 #kbxvkmpncy .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #kbxvkmpncy .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #kbxvkmpncy .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #kbxvkmpncy .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #kbxvkmpncy .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #kbxvkmpncy .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #kbxvkmpncy .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #kbxvkmpncy .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #kbxvkmpncy .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #kbxvkmpncy .gt_column_spanner_outer:first-child { padding-left: 0; }
 #kbxvkmpncy .gt_column_spanner_outer:last-child { padding-right: 0; }
 #kbxvkmpncy .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #kbxvkmpncy .gt_spanner_row { border-bottom-style: hidden; }
 #kbxvkmpncy .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #kbxvkmpncy .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #kbxvkmpncy .gt_from_md> :first-child { margin-top: 0; }
 #kbxvkmpncy .gt_from_md> :last-child { margin-bottom: 0; }
 #kbxvkmpncy .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #kbxvkmpncy .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #kbxvkmpncy .gt_indent_1 { text-indent: 5px; }
 #kbxvkmpncy .gt_indent_2 { text-indent: calc(5px * 2); }
 #kbxvkmpncy .gt_indent_3 { text-indent: calc(5px * 3); }
 #kbxvkmpncy .gt_indent_4 { text-indent: calc(5px * 4); }
 #kbxvkmpncy .gt_indent_5 { text-indent: calc(5px * 5); }
 #kbxvkmpncy .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #kbxvkmpncy .gt_row_group_first td { border-top-width: 2px; }
 #kbxvkmpncy .gt_row_group_first th { border-top-width: 2px; }
 #kbxvkmpncy .gt_striped { color: #333333; background-color: #F4F4F4; }
 #kbxvkmpncy .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #kbxvkmpncy .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #kbxvkmpncy .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #kbxvkmpncy .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #kbxvkmpncy .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #kbxvkmpncy .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #kbxvkmpncy .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #kbxvkmpncy .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #kbxvkmpncy .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #kbxvkmpncy .gt_left { text-align: left; }
 #kbxvkmpncy .gt_center { text-align: center; }
 #kbxvkmpncy .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #kbxvkmpncy .gt_font_normal { font-weight: normal; }
 #kbxvkmpncy .gt_font_bold { font-weight: bold; }
 #kbxvkmpncy .gt_font_italic { font-style: italic; }
 #kbxvkmpncy .gt_super { font-size: 65%; }
 #kbxvkmpncy .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #kbxvkmpncy .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #kbxvkmpncy .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #kbxvkmpncy .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #kbxvkmpncy .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #kbxvkmpncy .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Mean \|SHAP\| — 5v5 booster, 4,000-shot sample |               |
|------------------------------------------------|---------------|
| feature                                        | mean_abs_shap |
| wrist_shot                                     | 1.0464        |
| shot_distance                                  | 0.8045        |
| slap_shot                                      | 0.6813        |
| snap_shot                                      | 0.6197        |
| tip_in                                         | 0.3785        |
| backhand                                       | 0.3012        |
| shot_angle                                     | 0.2365        |
| time_since_last                                | 0.1493        |

&#10;</div>

## Evaluation

Two views, honestly labeled. First, the **frozen training-time gates**:
the grouped 5-fold CV metrics observed at the 2026-04 fit, below which a
retrain fails (`--quick` smoke runs tolerate misses; a real retrain does
not):

| variant       | CV AUC (observed at fit) | CV log-loss | frozen floor  |
|---------------|--------------------------|-------------|---------------|
| 5v5           | **0.8322**               | 0.2053      | cv AUC ≥ 0.82 |
| special teams | **0.8213**               | 0.2567      | cv AUC ≥ 0.81 |

Second, a **reproduced grouped split evaluated at render time**: the
same seed-37, game-grouped 80/20 partition recipe the trainer uses,
applied to *today’s* committed corpus, scoring the committed boosters on
the 20% side. Because the corpus has grown since the fit (new games,
repairs), this is a *near*-holdout — some evaluation games were in the
boosters’ training data — so read it as a stability check on live data,
not a pristine test score.

<div id="eqzbwkrque" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#eqzbwkrque table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#eqzbwkrque thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#eqzbwkrque p { margin: 0; padding: 0; }
 #eqzbwkrque .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #eqzbwkrque .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #eqzbwkrque .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #eqzbwkrque .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #eqzbwkrque .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #eqzbwkrque .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #eqzbwkrque .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #eqzbwkrque .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #eqzbwkrque .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #eqzbwkrque .gt_column_spanner_outer:first-child { padding-left: 0; }
 #eqzbwkrque .gt_column_spanner_outer:last-child { padding-right: 0; }
 #eqzbwkrque .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #eqzbwkrque .gt_spanner_row { border-bottom-style: hidden; }
 #eqzbwkrque .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #eqzbwkrque .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #eqzbwkrque .gt_from_md> :first-child { margin-top: 0; }
 #eqzbwkrque .gt_from_md> :last-child { margin-bottom: 0; }
 #eqzbwkrque .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #eqzbwkrque .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #eqzbwkrque .gt_indent_1 { text-indent: 5px; }
 #eqzbwkrque .gt_indent_2 { text-indent: calc(5px * 2); }
 #eqzbwkrque .gt_indent_3 { text-indent: calc(5px * 3); }
 #eqzbwkrque .gt_indent_4 { text-indent: calc(5px * 4); }
 #eqzbwkrque .gt_indent_5 { text-indent: calc(5px * 5); }
 #eqzbwkrque .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #eqzbwkrque .gt_row_group_first td { border-top-width: 2px; }
 #eqzbwkrque .gt_row_group_first th { border-top-width: 2px; }
 #eqzbwkrque .gt_striped { color: #333333; background-color: #F4F4F4; }
 #eqzbwkrque .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #eqzbwkrque .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #eqzbwkrque .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #eqzbwkrque .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #eqzbwkrque .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #eqzbwkrque .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #eqzbwkrque .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #eqzbwkrque .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #eqzbwkrque .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #eqzbwkrque .gt_left { text-align: left; }
 #eqzbwkrque .gt_center { text-align: center; }
 #eqzbwkrque .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #eqzbwkrque .gt_font_normal { font-weight: normal; }
 #eqzbwkrque .gt_font_bold { font-weight: bold; }
 #eqzbwkrque .gt_font_italic { font-style: italic; }
 #eqzbwkrque .gt_super { font-size: 65%; }
 #eqzbwkrque .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #eqzbwkrque .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #eqzbwkrque .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #eqzbwkrque .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #eqzbwkrque .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #eqzbwkrque .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Reproduced grouped-split evaluation (render-time, current corpus) |  |  |  |  |  |
|----|----|----|----|----|----|
| seed-37 game-grouped 80/20 recipe; committed boosters scored on the 20% side |  |  |  |  |  |
| variant | eval_shots | goal_rate | logloss | baseline_logloss | rank_AUC |
| 5v5 | 345,108 | 6.8% | 0.2189 | 0.2488 | 0.7783 |
| st | 69,967 | 10.2% | 0.2999 | 0.3293 | 0.7605 |

&#10;</div>

<img src="nhl_xg_files/figure-commonmark/cell-12-output-1.png"
width="420" height="300"
alt="Calibration by xG decile, 5v5, reproduced-split evaluation shots." />

<img src="nhl_xg_files/figure-commonmark/cell-13-output-1.png"
width="420" height="300"
alt="Per-era discrimination: rank AUC of the committed 5v5 booster by era bucket (evaluation shots)." />

The penalty-shot constant computed from the current corpus:

<div id="hmtvehfhxn" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#hmtvehfhxn table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#hmtvehfhxn thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#hmtvehfhxn p { margin: 0; padding: 0; }
 #hmtvehfhxn .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #hmtvehfhxn .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #hmtvehfhxn .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #hmtvehfhxn .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #hmtvehfhxn .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #hmtvehfhxn .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #hmtvehfhxn .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #hmtvehfhxn .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #hmtvehfhxn .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #hmtvehfhxn .gt_column_spanner_outer:first-child { padding-left: 0; }
 #hmtvehfhxn .gt_column_spanner_outer:last-child { padding-right: 0; }
 #hmtvehfhxn .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #hmtvehfhxn .gt_spanner_row { border-bottom-style: hidden; }
 #hmtvehfhxn .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #hmtvehfhxn .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #hmtvehfhxn .gt_from_md> :first-child { margin-top: 0; }
 #hmtvehfhxn .gt_from_md> :last-child { margin-bottom: 0; }
 #hmtvehfhxn .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #hmtvehfhxn .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #hmtvehfhxn .gt_indent_1 { text-indent: 5px; }
 #hmtvehfhxn .gt_indent_2 { text-indent: calc(5px * 2); }
 #hmtvehfhxn .gt_indent_3 { text-indent: calc(5px * 3); }
 #hmtvehfhxn .gt_indent_4 { text-indent: calc(5px * 4); }
 #hmtvehfhxn .gt_indent_5 { text-indent: calc(5px * 5); }
 #hmtvehfhxn .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #hmtvehfhxn .gt_row_group_first td { border-top-width: 2px; }
 #hmtvehfhxn .gt_row_group_first th { border-top-width: 2px; }
 #hmtvehfhxn .gt_striped { color: #333333; background-color: #F4F4F4; }
 #hmtvehfhxn .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #hmtvehfhxn .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #hmtvehfhxn .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #hmtvehfhxn .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #hmtvehfhxn .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #hmtvehfhxn .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #hmtvehfhxn .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #hmtvehfhxn .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #hmtvehfhxn .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #hmtvehfhxn .gt_left { text-align: left; }
 #hmtvehfhxn .gt_center { text-align: center; }
 #hmtvehfhxn .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #hmtvehfhxn .gt_font_normal { font-weight: normal; }
 #hmtvehfhxn .gt_font_bold { font-weight: bold; }
 #hmtvehfhxn .gt_font_italic { font-style: italic; }
 #hmtvehfhxn .gt_super { font-size: 65%; }
 #hmtvehfhxn .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #hmtvehfhxn .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #hmtvehfhxn .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #hmtvehfhxn .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #hmtvehfhxn .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #hmtvehfhxn .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Penalty-shot xG constant (historical conversion rate) |        |
|-------------------------------------------------------|--------|
| component                                             | xG     |
| penalty-shot / shootout constant                      | 0.3203 |

&#10;</div>

## Results — players and teams

<div id="xpauyoqlxd" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#xpauyoqlxd table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#xpauyoqlxd thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#xpauyoqlxd p { margin: 0; padding: 0; }
 #xpauyoqlxd .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #xpauyoqlxd .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #xpauyoqlxd .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #xpauyoqlxd .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #xpauyoqlxd .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #xpauyoqlxd .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #xpauyoqlxd .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #xpauyoqlxd .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #xpauyoqlxd .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #xpauyoqlxd .gt_column_spanner_outer:first-child { padding-left: 0; }
 #xpauyoqlxd .gt_column_spanner_outer:last-child { padding-right: 0; }
 #xpauyoqlxd .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #xpauyoqlxd .gt_spanner_row { border-bottom-style: hidden; }
 #xpauyoqlxd .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #xpauyoqlxd .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #xpauyoqlxd .gt_from_md> :first-child { margin-top: 0; }
 #xpauyoqlxd .gt_from_md> :last-child { margin-bottom: 0; }
 #xpauyoqlxd .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #xpauyoqlxd .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #xpauyoqlxd .gt_indent_1 { text-indent: 5px; }
 #xpauyoqlxd .gt_indent_2 { text-indent: calc(5px * 2); }
 #xpauyoqlxd .gt_indent_3 { text-indent: calc(5px * 3); }
 #xpauyoqlxd .gt_indent_4 { text-indent: calc(5px * 4); }
 #xpauyoqlxd .gt_indent_5 { text-indent: calc(5px * 5); }
 #xpauyoqlxd .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #xpauyoqlxd .gt_row_group_first td { border-top-width: 2px; }
 #xpauyoqlxd .gt_row_group_first th { border-top-width: 2px; }
 #xpauyoqlxd .gt_striped { color: #333333; background-color: #F4F4F4; }
 #xpauyoqlxd .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #xpauyoqlxd .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #xpauyoqlxd .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #xpauyoqlxd .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #xpauyoqlxd .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #xpauyoqlxd .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #xpauyoqlxd .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #xpauyoqlxd .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #xpauyoqlxd .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #xpauyoqlxd .gt_left { text-align: left; }
 #xpauyoqlxd .gt_center { text-align: center; }
 #xpauyoqlxd .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #xpauyoqlxd .gt_font_normal { font-weight: normal; }
 #xpauyoqlxd .gt_font_bold { font-weight: bold; }
 #xpauyoqlxd .gt_font_italic { font-style: italic; }
 #xpauyoqlxd .gt_super { font-size: 65%; }
 #xpauyoqlxd .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #xpauyoqlxd .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #xpauyoqlxd .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #xpauyoqlxd .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #xpauyoqlxd .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #xpauyoqlxd .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Top 15 skaters by 5v5 expected goals — season 2025-2026 |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
| unblocked 5v5 shots scored by the committed booster; GAX = goals above expected |  |  |  |  |  |  |
|  | Player | Team | Shots | xG | G | GAX |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8477492.png"
height="42" /> | Nathan MacKinnon | COL | 535 | 53.76 | 60 | 6.24 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8478402.png"
height="42" /> | Connor McDavid | EDM | 437 | 48.75 | 46 | −2.75 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8482093.png"
height="42" /> | Seth Jarvis | CAR | 382 | 44.53 | 35 | −9.53 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8481540.png"
height="42" /> | Cole Caufield | MTL | 434 | 43.90 | 56 | 12.10 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8478864.png"
height="42" /> | Kirill Kaprizov | MIN | 455 | 42.48 | 47 | 4.52 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8479420.png"
height="42" /> | Tage Thompson | BUF | 472 | 42.28 | 45 | 2.72 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8479542.png"
height="42" /> | Brandon Hagel | TBL | 343 | 42.01 | 41 | −1.01 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8483515.png"
height="42" /> | Juraj Slafkovský | MTL | 376 | 42.01 | 35 | −7.01 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8480027.png"
height="42" /> | Jason Robertson | DAL | 445 | 41.87 | 50 | 8.13 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8481557.png"
height="42" /> | Matt Boldy | MIN | 458 | 40.98 | 49 | 8.02 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8482740.png"
height="42" /> | Wyatt Johnston | DAL | 325 | 40.60 | 49 | 8.40 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8479337.png"
height="42" /> | Alex DeBrincat | DET | 431 | 40.24 | 40 | −0.24 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8476881.png"
height="42" /> | Tomas Hertl | VGK | 383 | 40.10 | 29 | −11.10 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8481604.png"
height="42" /> | Pavel Dorofeyev | VGK | 422 | 38.65 | 49 | 10.35 |
| <img src="https://assets.nhle.com/mugs/nhl/latest/8476453.png"
height="42" /> | Nikita Kucherov | TBL | 421 | 38.48 | 43 | 4.52 |

&#10;</div>

<div id="cmvqeebemx" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>
#cmvqeebemx table {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
&#10;#cmvqeebemx thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#cmvqeebemx p { margin: 0; padding: 0; }
 #cmvqeebemx .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #cmvqeebemx .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #cmvqeebemx .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #cmvqeebemx .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #cmvqeebemx .gt_heading { background-color: #FFFFFF; text-align: center; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #cmvqeebemx .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #cmvqeebemx .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #cmvqeebemx .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #cmvqeebemx .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #cmvqeebemx .gt_column_spanner_outer:first-child { padding-left: 0; }
 #cmvqeebemx .gt_column_spanner_outer:last-child { padding-right: 0; }
 #cmvqeebemx .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #cmvqeebemx .gt_spanner_row { border-bottom-style: hidden; }
 #cmvqeebemx .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #cmvqeebemx .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #cmvqeebemx .gt_from_md> :first-child { margin-top: 0; }
 #cmvqeebemx .gt_from_md> :last-child { margin-bottom: 0; }
 #cmvqeebemx .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #cmvqeebemx .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #cmvqeebemx .gt_indent_1 { text-indent: 5px; }
 #cmvqeebemx .gt_indent_2 { text-indent: calc(5px * 2); }
 #cmvqeebemx .gt_indent_3 { text-indent: calc(5px * 3); }
 #cmvqeebemx .gt_indent_4 { text-indent: calc(5px * 4); }
 #cmvqeebemx .gt_indent_5 { text-indent: calc(5px * 5); }
 #cmvqeebemx .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #cmvqeebemx .gt_row_group_first td { border-top-width: 2px; }
 #cmvqeebemx .gt_row_group_first th { border-top-width: 2px; }
 #cmvqeebemx .gt_striped { color: #333333; background-color: #F4F4F4; }
 #cmvqeebemx .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #cmvqeebemx .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #cmvqeebemx .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #cmvqeebemx .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #cmvqeebemx .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #cmvqeebemx .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #cmvqeebemx .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #cmvqeebemx .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #cmvqeebemx .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #cmvqeebemx .gt_left { text-align: left; }
 #cmvqeebemx .gt_center { text-align: center; }
 #cmvqeebemx .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #cmvqeebemx .gt_font_normal { font-weight: normal; }
 #cmvqeebemx .gt_font_bold { font-weight: bold; }
 #cmvqeebemx .gt_font_italic { font-style: italic; }
 #cmvqeebemx .gt_super { font-size: 65%; }
 #cmvqeebemx .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #cmvqeebemx .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #cmvqeebemx .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #cmvqeebemx .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #cmvqeebemx .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #cmvqeebemx .gt_asterisk { font-size: 100%; vertical-align: 0; }
 &#10;</style>

| Team 5v5 shot generation — season 2025-2026 |       |        |           |       |
|---------------------------------------------|-------|--------|-----------|-------|
| team                                        | shots | xG_for | goals_for | GAX   |
| CAR                                         | 5,070 | 401.3  | 347       | −54.3 |
| COL                                         | 4,700 | 364.2  | 331       | −33.2 |
| VGK                                         | 4,452 | 358.0  | 331       | −27.0 |
| MTL                                         | 3,995 | 344.4  | 323       | −21.4 |
| BUF                                         | 4,090 | 331.3  | 322       | −9.3  |
| ANA                                         | 4,429 | 325.8  | 295       | −30.8 |
| TBL                                         | 3,815 | 315.4  | 291       | −24.4 |
| EDM                                         | 3,775 | 311.6  | 295       | −16.6 |
| MIN                                         | 4,061 | 309.9  | 304       | −5.9  |
| OTT                                         | 3,739 | 301.8  | 274       | −27.8 |
| PIT                                         | 3,823 | 296.9  | 295       | −1.9  |
| UTA                                         | 3,753 | 291.5  | 277       | −14.5 |
| PHI                                         | 3,483 | 289.9  | 250       | −39.9 |
| DAL                                         | 3,280 | 287.2  | 285       | −2.2  |
| WSH                                         | 3,496 | 280.4  | 254       | −26.4 |
| NJD                                         | 3,587 | 275.8  | 223       | −52.8 |
| DET                                         | 3,515 | 273.5  | 235       | −38.5 |
| CBJ                                         | 3,547 | 273.2  | 241       | −32.2 |
| LAK                                         | 3,694 | 272.4  | 221       | −51.4 |
| BOS                                         | 3,646 | 272.3  | 274       | 1.7   |
| FLA                                         | 3,577 | 267.3  | 233       | −34.3 |
| NYI                                         | 3,475 | 265.4  | 224       | −41.4 |
| NSH                                         | 3,444 | 258.6  | 239       | −19.6 |
| WPG                                         | 3,296 | 252.4  | 225       | −27.4 |
| TOR                                         | 3,216 | 252.3  | 245       | −7.3  |
| STL                                         | 3,049 | 249.3  | 223       | −26.3 |
| SJS                                         | 3,077 | 247.6  | 244       | −3.6  |
| VAN                                         | 3,291 | 243.4  | 206       | −37.4 |
| NYR                                         | 3,251 | 241.1  | 232       | −9.1  |
| SEA                                         | 3,194 | 241.0  | 218       | −23.0 |
| CGY                                         | 3,354 | 237.4  | 203       | −34.4 |
| CHI                                         | 3,063 | 231.3  | 206       | −25.3 |

&#10;</div>

Positive GAX marks finishing above shot quality. At the team level,
xG-for minus goals-for separates shot-generation strength from shooting
luck — the spread between the two is exactly what the embedded xG
columns in `nhl_pbp_full` exist to expose.

## Provenance & reproducibility

- **Trained on:** this repository’s committed play-by-play
  (`nhl/pbp/parquet/`, seasons in the corpus table above), 2026-04 fit;
  grouped 80/20 split by `game_id`, grouped 5-fold CV for
  `min_child_weight`, seed 37.
- **Artifacts:** `models/xg_model_5v5.json`, `models/xg_model_st.json`
  (committed — the promotion step); output ships inside `nhl_pbp_full`.
- **Retrain:** `scripts/nhl_models.sh` (stages `nhl_model_01_xg_5v5` /
  `nhl_model_02_xg_st`; fingerprint-skipped when unchanged; gates frozen
  in `models/manifest.yaml`, registry row in `models/REGISTRY.md`).
- **Rebuild this document:** `scripts/render_model_docs.sh` (Quarto →
  GFM; uses this repo’s `.venv` via `QUARTO_PYTHON`;
  `uv sync --group docs`).

## Avenues for improvement & open issues

- **Pre-shot context** — rush/rebound flags exist, but passing sequences
  and screens do not; public-feed models plateau here, so the honest
  gain is better rebound/rush definitions, not more trees.
- **Commit the python meta sidecar** — `xg_model_meta.json` (feature
  lists + per-retrain metrics) is regenerated per run but only the R-era
  `.rds` is committed; committing the json would let this document quote
  the exact training-time holdout metrics instead of the CV gates.
- **Exact holdout reproduction** — the render-time evaluation is a
  near-holdout because the corpus has grown since the fit; persisting
  the training-time game-id partition alongside the meta would make the
  exact test set reproducible forever.
- **Known issue:** the ST sample is ~5× smaller than 5v5 — its gate
  floor is lower for that reason, and per-season ST calibration drift is
  unmonitored.
