"""Box-score player names keep the space after the initial.

The NHL api-web boxscore sends ``name.default`` as ``"X. Surname"`` for every player except a
few, which come through as ``"J.Kulich"`` / ``"D.Jiricek"`` / ``"D.Misyul"`` (187 of 891,579
skater+goalie rows across the 22,292 raw games, 2009-10 to 2026-27; no other irregular form).
Those rows broke name joins against other providers.

Fixture ``final_2026020305_box.json``: the ``game_info``, ``skater_stats`` and ``goalie_stats``
blocks of fastRhockey-nhl-raw ``nhl/json/final/2026020305.json`` (commit 2b84026c), unmodified;
BUF's Jiri Kulich arrives as ``"J.Kulich"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from nhl_data_build.build import build_season
from nhl_data_build.extract import normalize_player_name

FIX = Path(__file__).parent / "fixtures"


def test_real_game_names_all_have_the_space() -> None:
    final = json.loads((FIX / "final_2026020305_box.json").read_text(encoding="utf-8"))
    assert "J.Kulich" in [r["player_name"] for r in final["skater_stats"]]  # the raw quirk is present
    season = build_season([final])
    for key in ("skater_box", "goalie_box", "player_box"):
        names = season[key]["player_name"].to_list()
        assert all(". " in n for n in names), [n for n in names if ". " not in n]
    assert "J. Kulich" in season["skater_box"]["player_name"].to_list()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("J.Kulich", "J. Kulich"),
        ("D.Jiricek", "D. Jiricek"),
        ("C. McDavid", "C. McDavid"),  # already correct: unchanged
        ("J.T. Compher", "J.T. Compher"),  # has a space: unchanged
        (None, None),
    ],
)
def test_normalize_player_name(raw: str | None, expected: str | None) -> None:
    assert normalize_player_name(raw) == expected
