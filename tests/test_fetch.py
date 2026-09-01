"""Offline tests for nhl_data_build.fetch (opener injected; no network)."""

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from nhl_data_build import fetch


def _schedule_bytes(game_ids):
    buf = io.BytesIO()
    pq.write_table(pa.table({"game_id": pa.array(game_ids, pa.int64())}), buf)
    return buf.getvalue()


def _opener(mapping):
    def op(url):
        return mapping.get(url)

    return op


BASE = "https://example.test/raw"
SCHED = f"{BASE}/nhl/schedules/parquet/nhl_schedule_2026.parquet"


def test_season_game_ids_reads_schedule_parquet():
    op = _opener({SCHED: _schedule_bytes([2025020002, 2025020001, 2025020001, None])})
    ids = fetch.season_game_ids(2026, raw_base=BASE, opener=op)
    assert ids == [2025020001, 2025020002]  # deduped + sorted, nulls dropped


def test_season_game_ids_missing_schedule_raises():
    with pytest.raises(FileNotFoundError):
        fetch.season_game_ids(2026, raw_base=BASE, opener=_opener({}))


def test_fetch_finals_fail_soft_and_counters(tmp_path):
    good = json.dumps({"game_id": 2025020001}).encode()
    op = _opener(
        {
            SCHED: _schedule_bytes([2025020001, 2025020002, 2025020003]),
            f"{BASE}/nhl/json/final/2025020001.json": good,
            # 2025020002 -> None (no final yet): counted missing, never fatal
            f"{BASE}/nhl/json/final/2025020003.json": b"{not json",  # unparseable: never persisted
        }
    )
    out = fetch.fetch_finals(2026, tmp_path, raw_base=BASE, opener=op)
    assert out == {"fetched": 1, "skipped": 0, "missing": 2, "total": 3}
    assert (tmp_path / "2025020001.json").exists()
    assert not (tmp_path / "2025020003.json").exists()


def test_fetch_finals_corrupt_cache_refetches(tmp_path):
    (tmp_path / "2025020001.json").write_text("{trunc", encoding="utf-8")  # half-written entry
    good = json.dumps({"game_id": 2025020001}).encode()
    op = _opener(
        {
            SCHED: _schedule_bytes([2025020001]),
            f"{BASE}/nhl/json/final/2025020001.json": good,
        }
    )
    out = fetch.fetch_finals(2026, tmp_path, raw_base=BASE, opener=op)
    assert out["fetched"] == 1 and out["skipped"] == 0
    assert json.loads((tmp_path / "2025020001.json").read_text(encoding="utf-8"))


def test_fetch_finals_cache_hit_skips(tmp_path):
    (tmp_path / "2025020001.json").write_text('{"game_id": 2025020001}', encoding="utf-8")
    op = _opener({SCHED: _schedule_bytes([2025020001])})  # final URL absent: must not be needed
    out = fetch.fetch_finals(2026, tmp_path, raw_base=BASE, opener=op)
    assert out == {"fetched": 0, "skipped": 1, "missing": 0, "total": 1}
