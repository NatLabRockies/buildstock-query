"""Content-addressed on-disk cache for SQL query results.

Each cached query is up to three files in the cache directory:
    <sha256(normalized_sql)>.sql      — the normalized SQL text
    <sha256(normalized_sql)>.parquet  — the query result DataFrame
    <sha256(normalized_sql)>.json     — Athena execution metadata (cost, runtime,
                                        engine version, etc.) — optional;
                                        present when the query came from a real
                                        Athena execution (not a stale-cache hit).

Lookups, writes, and existence checks all key on the hash. No in-memory state —
every get() reads from disk, every put() writes to disk. Concurrency-safe by
construction: same SQL → same path (idempotent), different SQLs → different
paths (no contention).
"""
from __future__ import annotations

import hashlib
import json as _json
import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_sql(sql: str) -> str:
    """Collapse all whitespace runs to single spaces and strip ends.

    Two SQL strings that differ only in whitespace (indentation, trailing
    newlines, line breaks inside expressions) normalize to the same text and
    therefore share a cache slot.
    """
    return _WHITESPACE_RE.sub(" ", sql).strip()


def hash_sql(sql: str) -> str:
    """Return sha256 of the normalized SQL as 64 hex chars.

    Single source of truth for cache addressing. Every SqlCache method routes
    through this — never recompute the hash inline.
    """
    return hashlib.sha256(normalize_sql(sql).encode()).hexdigest()


class SqlCache:
    """Content-addressed on-disk SQL→DataFrame cache."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _parquet_path(self, sql: str) -> Path:
        return self.root / f"{hash_sql(sql)}.parquet"

    def _sql_path(self, sql: str) -> Path:
        return self.root / f"{hash_sql(sql)}.sql"

    def _meta_path(self, sql: str) -> Path:
        return self.root / f"{hash_sql(sql)}.json"

    def __contains__(self, sql: str) -> bool:
        return self._parquet_path(sql).exists()

    def get(self, sql: str) -> pd.DataFrame | None:
        """Return the cached DataFrame for `sql`, or None if not cached.

        A torn parquet (writer crashed mid-write) raises from pd.read_parquet;
        callers should treat that as a miss and re-run the query.
        """
        path = self._parquet_path(sql)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def put(self, sql: str, df: pd.DataFrame) -> None:
        """Write `df` and the normalized SQL sidecar atomically per file."""
        df.to_parquet(self._parquet_path(sql), index=False)
        self._sql_path(sql).write_text(normalize_sql(sql))

    def put_metadata(self, sql: str, metadata: dict[str, Any]) -> None:
        """Write Athena execution metadata for `sql` as a JSON sidecar.

        `metadata` is the full GetQueryExecution response (or any subset the
        caller chose) — stored as-is so future analyses can pull whatever
        Athena reports (DataScannedInBytes, EngineExecutionTimeInMillis,
        ResultReuseInformation, EngineVersion, etc.) without needing to
        re-fetch from Athena history.
        """
        self._meta_path(sql).write_text(_json.dumps(metadata, indent=2, default=str))

    def get_metadata(self, sql: str) -> dict[str, Any] | None:
        """Return the metadata JSON for `sql`, or None if not present."""
        path = self._meta_path(sql)
        if not path.exists():
            return None
        try:
            return _json.loads(path.read_text())
        except _json.JSONDecodeError:
            return None

    def delete(self, sql: str) -> None:
        self._sql_path(sql).unlink(missing_ok=True)
        self._parquet_path(sql).unlink(missing_ok=True)
        self._meta_path(sql).unlink(missing_ok=True)

    def rename(self, old_sql: str, new_sql: str) -> None:
        """Move an entry from old_sql's hash to new_sql's hash.

        Used by --update-snapshot when the SQL changed cosmetically (sqlglot
        considers it equivalent) but the data is unchanged — saves an Athena
        re-run by reusing the existing parquet.
        """
        old_parquet = self._parquet_path(old_sql)
        new_parquet = self._parquet_path(new_sql)
        if old_parquet != new_parquet:
            old_parquet.rename(new_parquet)
        self._sql_path(new_sql).write_text(normalize_sql(new_sql))
        old_sql_path = self._sql_path(old_sql)
        if old_sql_path != self._sql_path(new_sql):
            old_sql_path.unlink(missing_ok=True)
        # Carry the metadata along too if it exists. Cost is associated with
        # the data, and the data didn't change — so the cost numbers carry.
        old_meta_path = self._meta_path(old_sql)
        new_meta_path = self._meta_path(new_sql)
        if old_meta_path != new_meta_path and old_meta_path.exists():
            old_meta_path.rename(new_meta_path)

    def read_sql(self, sql_or_hash: str) -> str | None:
        """Read the stored .sql sidecar text, by either SQL or raw hash.

        Used by --update-snapshot to compare the previously-stored SQL against
        a freshly-generated one (sqlglot equivalence check).
        """
        if len(sql_or_hash) == 64 and all(c in "0123456789abcdef" for c in sql_or_hash):
            path = self.root / f"{sql_or_hash}.sql"
        else:
            path = self._sql_path(sql_or_hash)
        if not path.exists():
            return None
        return path.read_text()

    def hashes(self) -> Iterator[str]:
        """Yield every cached entry's hash (one per .parquet file present)."""
        for parquet in self.root.glob("*.parquet"):
            yield parquet.stem
