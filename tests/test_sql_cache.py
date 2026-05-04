"""Unit tests for the content-addressed disk cache."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from buildstock_query.sql_cache import SqlCache, hash_sql, normalize_sql


def _df(values):
    return pd.DataFrame({"x": list(values)})


def test_normalize_collapses_whitespace():
    assert normalize_sql("SELECT  *\nFROM   t") == "SELECT * FROM t"
    assert normalize_sql("  SELECT 1  ") == "SELECT 1"


def test_hash_is_normalized():
    assert hash_sql("SELECT  1") == hash_sql("SELECT 1")
    assert hash_sql("SELECT 1") != hash_sql("SELECT 2")
    h = hash_sql("SELECT 1")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_round_trip_put_get(tmp_path: Path):
    cache = SqlCache(tmp_path)
    sql = "SELECT a, b FROM t WHERE x = 1"
    df = _df([1, 2, 3])

    assert sql not in cache
    assert cache.get(sql) is None

    cache.put(sql, df)
    assert sql in cache
    pd.testing.assert_frame_equal(cache.get(sql), df)


def test_whitespace_variations_share_slot(tmp_path: Path):
    cache = SqlCache(tmp_path)
    raw = "SELECT *\n  FROM t"
    formatted = "SELECT * FROM t"
    df = _df([42])

    cache.put(raw, df)
    pd.testing.assert_frame_equal(cache.get(formatted), df)
    assert formatted in cache


def test_delete(tmp_path: Path):
    cache = SqlCache(tmp_path)
    sql = "SELECT 1"
    cache.put(sql, _df([1]))
    assert sql in cache
    cache.delete(sql)
    assert sql not in cache
    assert cache.get(sql) is None
    cache.delete(sql)  # idempotent


def test_rename_moves_parquet_and_rewrites_sql_sidecar(tmp_path: Path):
    cache = SqlCache(tmp_path)
    old_sql = "SELECT 1"
    new_sql = "SELECT 1 + 0"  # different SQL, same data
    df = _df([1])
    cache.put(old_sql, df)

    cache.rename(old_sql, new_sql)

    assert old_sql not in cache
    assert new_sql in cache
    pd.testing.assert_frame_equal(cache.get(new_sql), df)
    assert cache.read_sql(new_sql) == normalize_sql(new_sql)


def test_rename_no_op_when_hashes_match(tmp_path: Path):
    cache = SqlCache(tmp_path)
    old_sql = "SELECT  1"
    new_sql = "SELECT 1"  # whitespace-only diff → same hash
    cache.put(old_sql, _df([1]))

    cache.rename(old_sql, new_sql)

    assert new_sql in cache
    pd.testing.assert_frame_equal(cache.get(new_sql), _df([1]))


def test_read_sql_by_hash(tmp_path: Path):
    cache = SqlCache(tmp_path)
    sql = "SELECT col FROM t"
    cache.put(sql, _df([0]))

    h = hash_sql(sql)
    assert cache.read_sql(h) == normalize_sql(sql)
    assert cache.read_sql("a" * 64) is None  # missing


def test_hashes_lists_only_parquet_files(tmp_path: Path):
    cache = SqlCache(tmp_path)
    cache.put("SELECT 1", _df([1]))
    cache.put("SELECT 2", _df([2]))

    hashes = sorted(cache.hashes())
    assert hashes == sorted([hash_sql("SELECT 1"), hash_sql("SELECT 2")])


def test_concurrent_same_sql_writes_idempotent(tmp_path: Path):
    """N threads writing the same SQL → same path. All must succeed; result is the
    DataFrame they all agreed on. This is the batch_query race that the disk cache
    fixes by construction."""
    cache = SqlCache(tmp_path)
    sql = "SELECT shared FROM t"
    df = _df(range(100))

    errors: list[Exception] = []

    def writer():
        try:
            cache.put(sql, df)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    pd.testing.assert_frame_equal(cache.get(sql), df)
    assert sum(1 for _ in cache.hashes()) == 1


def test_concurrent_distinct_sql_writes_independent(tmp_path: Path):
    """N threads writing N distinct SQLs → N distinct paths. No contention."""
    cache = SqlCache(tmp_path)
    sqls = [f"SELECT {i} FROM t" for i in range(20)]
    dfs = {sql: _df([i]) for i, sql in enumerate(sqls)}

    with ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(lambda sql: cache.put(sql, dfs[sql]), sqls))

    assert sum(1 for _ in cache.hashes()) == 20
    for sql in sqls:
        pd.testing.assert_frame_equal(cache.get(sql), dfs[sql])


def test_torn_parquet_raises_on_get(tmp_path: Path):
    """A reader hitting a half-written parquet should fail loudly. Callers (execute())
    will treat that as a cache miss and re-run."""
    cache = SqlCache(tmp_path)
    sql = "SELECT 1"
    cache.put(sql, _df([1]))

    parquet_path = tmp_path / f"{hash_sql(sql)}.parquet"
    parquet_path.write_bytes(b"not a parquet")

    with pytest.raises(Exception):
        cache.get(sql)
