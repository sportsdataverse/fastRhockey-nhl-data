"""Per-game dataset extraction — Python port of ``nhl_data_creation.R``'s ``.extract_all``.

Reads one parsed ``final.json`` and returns ``{dataset_key: list[row-dict]}``. The
standard case is a direct block with ``game_id``/``season``/``game_date`` attached when
absent; the special cases (scoring/penalties period-unroll, linescore single-row,
three_stars decisions, scratches, officials/shots_by_period/shootout) mirror R's branches.

In Python's parsed JSON every block is already a list[dict] / dict, so R's
``is.data.frame`` vs ``is.list`` branching collapses to the list path.
"""

from __future__ import annotations

from nhl_data_build.config import DATASETS, STANDARD_KEYS

_FIELD = {key: field for key, field, _, _, _ in DATASETS}


def _game_info(game_json: dict) -> dict:
    gi = game_json.get("game_info")
    if isinstance(gi, list):
        gi = gi[0] if gi else {}
    return gi or {}


def _attach_ids(rows: list[dict], gi: dict) -> list[dict]:
    """Attach game_id/season/game_date when absent (R's standard-case fill)."""
    out = []
    for r in rows:
        r = dict(r)
        r.setdefault("game_id", gi.get("game_id"))
        r.setdefault("season", gi.get("season"))
        r.setdefault("game_date", gi.get("game_date"))
        out.append(r)
    return out


def _extract_scoring_like(val: object, items_key: str, gid: int | None) -> list[dict]:
    """scoring/penalties: array of period-blocks -> unrolled goal/penalty rows."""
    rows = []
    for block in val or []:
        pd = block.get("periodDescriptor") or {}
        for item in block.get(items_key) or []:
            r = dict(item)
            r["game_id"] = gid
            r["period_number"] = pd.get("number")
            r["period_type"] = pd.get("periodType")
            rows.append(r)
    return rows


def _extract_linescore(val: dict, gid: int | None) -> list[dict]:
    teams = (val or {}).get("teams") or {}
    home, away = teams.get("home") or {}, teams.get("away") or {}
    ht, at = home.get("team") or {}, away.get("team") or {}
    return [
        {
            "game_id": gid,
            "home_team_id": ht.get("id"),
            "home_team_abbr": ht.get("abbreviation"),
            "home_goals": home.get("goals"),
            "home_shots": home.get("shotsOnGoal"),
            "away_team_id": at.get("id"),
            "away_team_abbr": at.get("abbreviation"),
            "away_goals": away.get("goals"),
            "away_shots": away.get("shotsOnGoal"),
            "has_shootout": ((val or {}).get("shootout") or {}).get("hasShootout", False),
        }
    ]


def _extract_three_stars(val: dict, gid: int | None) -> list[dict]:
    stars = (val or {}).get("threeStars")
    if not stars:
        return []
    winner, loser = (val.get("winner") or {}), (val.get("loser") or {})
    rows = []
    for s in stars:
        r = dict(s)
        r["game_id"] = gid
        r["winner_id"], r["winner_name"] = winner.get("id"), winner.get("name")
        r["loser_id"], r["loser_name"] = loser.get("id"), loser.get("name")
        rows.append(r)
    return rows


def _extract_shootout(val: object, gid: object, season: object, gdate: object) -> list[dict]:
    """Port of R's ``bind_rows(shootout)`` on the ``{liveScore, events}`` object. R binds the
    1-row ``liveScore`` (home/away only) and the ``events`` (attempts) into one frame, so the
    released ``shootout_summary`` carries, per game, a leading liveScore-only row (home/away
    set, attempt fields null) followed by one row per attempt (attempt fields set, home/away
    null). Reproduced faithfully so the dataset is a drop-in replacement."""
    if not isinstance(val, dict):
        return []
    ids = {"game_id": gid, "season": season, "game_date": gdate}
    out = []
    live = val.get("liveScore")
    if isinstance(live, dict) and live:
        out.append({"home": live.get("home"), "away": live.get("away"), **ids})
    for ev in val.get("events") or []:
        if isinstance(ev, dict):
            out.append({**ev, **ids})
    return out


def extract_all(game_json: dict | None) -> dict[str, list[dict]]:
    """Port of ``.extract_all`` — one final.json -> {dataset_key: list[row-dict]}."""
    if not game_json:
        return {}
    gi = _game_info(game_json)
    gid, season, gdate = gi.get("game_id"), gi.get("season"), gi.get("game_date")
    out: dict[str, list[dict]] = {}

    for key, field, _, _, _ in DATASETS:
        val = game_json.get(field)
        if key in STANDARD_KEYS:
            # A standard block is normally a list of rows, but a singleton dict (R's
            # data.frame-vs-list duality) becomes a one-row dataset rather than being dropped.
            rows = val if isinstance(val, list) else ([val] if isinstance(val, dict) else None)
            if rows:
                out[key] = _attach_ids(rows, gi)
        elif key in ("scoring", "penalties"):
            rows = _extract_scoring_like(val, "goals" if key == "scoring" else "penalties", gid)
            if rows:
                out[key] = rows
        elif key == "linescore":
            if val:
                out[key] = _extract_linescore(val, gid)
        elif key == "three_stars":
            rows = _extract_three_stars(val or {}, gid)
            if rows:
                out[key] = rows
        elif key == "scratches":
            if isinstance(val, dict):  # singleton -> one-row list
                val = [val]
            if val:
                out[key] = [{**dict(s), "game_id": gid} for s in val]
        elif key == "shootout":
            rows = _extract_shootout(val, gid, season, gdate)
            if rows:
                out[key] = rows
        elif key in ("officials", "shots_by_period"):
            if isinstance(val, dict):  # singleton -> one-row list
                val = [val]
            if val:
                out[key] = [{**dict(s), "game_id": gid, "season": season, "game_date": gdate} for s in val]

    # player_box = skater_box + goalie_box (column union handled at build time).
    pb = (out.get("skater_box") or []) + (out.get("goalie_box") or [])
    if pb:
        out["player_box"] = pb
    return out
