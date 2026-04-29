"""Find cache hashes used by tests but not owned by any JSON snapshot entry.

Reads:
  - `tests/query_snapshots/<schema>_cache/.cache_usage_log` — hashes touched
    in the most recent test session (snapshot + invariants).
  - `tests/query_snapshots/*.json` — hashes claimed by snapshot entries via
    their `sql_hash` field.

Reports per-schema lists of hashes in (a) but not (b). For each unowned
hash, prints the SQL from the sibling `<hash>.sql` sidecar — that's the
basis for adding a new JSON entry.

Read-only; safe to run anytime. Does not modify caches or JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SNAPSHOTS_ROOT = Path(__file__).resolve().parent / "query_snapshots"
USAGE_LOG_NAME = ".cache_usage_log"
SCHEMAS = ("resstock_oedi", "comstock_oedi", "comstock_oedi_agg")


def _collect_json_hashes() -> dict[str, set[str]]:
    """Return {schema: {hashes}} aggregated across every flavor JSON file."""
    out: dict[str, set[str]] = {s: set() for s in SCHEMAS}
    for json_path in sorted(SNAPSHOTS_ROOT.glob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            if not isinstance(entry, dict):
                continue
            raw = entry.get("sql_hash")
            if isinstance(raw, dict):
                for schema, h in raw.items():
                    if schema in out and isinstance(h, str) and h:
                        out[schema].add(h)
            elif isinstance(raw, str) and raw:
                for schema in out:
                    out[schema].add(raw)
    return out


def _collect_log_hashes(cache_dir: Path) -> set[str]:
    log = cache_dir / USAGE_LOG_NAME
    if not log.exists():
        return set()
    out: set[str] = set()
    for line in log.read_text().splitlines():
        line = line.strip()
        if len(line) == 64 and all(c in "0123456789abcdef" for c in line):
            out.add(line)
    return out


def main() -> int:
    json_hashes = _collect_json_hashes()
    for schema in SCHEMAS:
        cache_dir = SNAPSHOTS_ROOT / f"{schema}_cache"
        if not cache_dir.is_dir():
            continue
        used = _collect_log_hashes(cache_dir)
        owned = json_hashes.get(schema, set())
        uncovered = sorted(used - owned)
        print(f"[{schema}] used={len(used)}  owned={len(owned)}  uncovered={len(uncovered)}")
        for h in uncovered:
            sql_path = cache_dir / f"{h}.sql"
            sql = sql_path.read_text().strip() if sql_path.exists() else "(no sql sidecar)"
            print(f"\n  {h}")
            print(f"    {sql}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
