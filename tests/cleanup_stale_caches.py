"""Remove stale cache entries from `tests/query_snapshots/<schema>_cache/`.

A cache entry (`<hash>.sql`, `<hash>.parquet`, optional `<hash>.json`) is
"needed" if either:

  1. Its hash appears as the `sql_hash` for some entry in any of the
     top-level `*.json` flavor files under `tests/query_snapshots/`, OR

  2. Its hash appears in the per-cache `.cache_usage_log` file, written by
     `SqlCache._record_usage` whenever the hash was hit (cache read) or
     freshly written during the most recent test session.

Anything else is stale: a leftover from a previous SQL shape that no
current snapshot or invariant test references. By default the script runs
in dry-run mode and only reports what it would remove. Pass `--delete` to
actually unlink the files.

When to run
-----------
After a test session that exercised the invariant tests (and ran any
queries those invariants emit which aren't tracked in the snapshot JSONs),
the `.cache_usage_log` will accurately reflect what the suite needs. That
is the right time to run this script — running it before that risks
deleting cache entries the invariants depend on, forcing an Athena
re-execution next time around.

Typical workflow:

    pytest -s -v tests/test_query_snapshots.py --check-data
    pytest -s -v tests/test_invariants.py
    python tests/cleanup_stale_caches.py            # dry-run report
    python tests/cleanup_stale_caches.py --delete   # actually remove

Both pytest commands must run in the same session-fixture lifetime as the
cleanup expectation: each fresh BSQ construction truncates
`.cache_usage_log`, so if you run the snapshot suite, then run something
else that constructs a new BSQ on the same cache folder, the log resets
and the snapshot hashes drop out of the "used" set. Run cleanup
immediately after the test sessions that should populate the log.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


SNAPSHOTS_ROOT = Path(__file__).resolve().parent / "query_snapshots"
USAGE_LOG_NAME = ".cache_usage_log"
SCHEMAS = ("resstock_oedi", "comstock_oedi", "comstock_oedi_agg")
SIDECAR_SUFFIXES = (".sql", ".parquet", ".json")


def _collect_json_hashes(snapshots_root: Path) -> dict[str, set[str]]:
    """Return {schema: {hashes}} aggregated across every flavor JSON file."""
    out: dict[str, set[str]] = {s: set() for s in SCHEMAS}
    for json_path in sorted(snapshots_root.glob("*.json")):
        try:
            data = json.loads(json_path.read_text())
        except json.JSONDecodeError as e:
            print(f"warn: skipping {json_path.name}: {e}", file=sys.stderr)
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
                # Schema-agnostic string: applies to every schema's cache.
                for schema in out:
                    out[schema].add(raw)
    return out


def _collect_log_hashes(cache_dir: Path) -> set[str]:
    """Read hashes from the cache's `.cache_usage_log` (one per line)."""
    log = cache_dir / USAGE_LOG_NAME
    if not log.exists():
        return set()
    out: set[str] = set()
    for line in log.read_text().splitlines():
        line = line.strip()
        if len(line) == 64 and all(c in "0123456789abcdef" for c in line):
            out.add(line)
    return out


def _collect_disk_hashes(cache_dir: Path) -> set[str]:
    """Hashes physically present on disk (one per `<hash>.parquet`)."""
    return {p.stem for p in cache_dir.glob("*.parquet")}


def _stale_files_for(cache_dir: Path, stale_hashes: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for h in stale_hashes:
        for suffix in SIDECAR_SUFFIXES:
            p = cache_dir / f"{h}{suffix}"
            if p.exists():
                paths.append(p)
    return paths


def _human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually unlink stale files. Without this, the script only reports.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help=(
            "Truncate `.cache_usage_log` in every schema cache and exit "
            "without computing stale entries. Run this before the test "
            "session(s) you want to track."
        ),
    )
    parser.add_argument(
        "--snapshots-root",
        type=Path,
        default=SNAPSHOTS_ROOT,
        help=f"Override snapshots root (default: {SNAPSHOTS_ROOT}).",
    )
    args = parser.parse_args()

    if args.clear:
        for schema in SCHEMAS:
            cache_dir = args.snapshots_root / f"{schema}_cache"
            log = cache_dir / USAGE_LOG_NAME
            if cache_dir.is_dir():
                log.write_text("")
                print(f"[{schema}] cleared {log.relative_to(args.snapshots_root.parent)}")
            else:
                print(f"[{schema}] cache dir missing, skipping: {cache_dir}")
        return 0

    json_hashes = _collect_json_hashes(args.snapshots_root)
    total_stale = 0
    total_bytes = 0

    for schema in SCHEMAS:
        cache_dir = args.snapshots_root / f"{schema}_cache"
        if not cache_dir.is_dir():
            print(f"[{schema}] cache dir missing, skipping: {cache_dir}")
            continue
        log_hashes = _collect_log_hashes(cache_dir)
        disk_hashes = _collect_disk_hashes(cache_dir)
        needed = json_hashes[schema] | log_hashes
        stale = sorted(disk_hashes - needed)

        print(
            f"[{schema}] disk={len(disk_hashes)} "
            f"json={len(json_hashes[schema])} "
            f"log={len(log_hashes)} "
            f"needed={len(needed)} "
            f"stale={len(stale)}"
        )
        if not stale:
            continue

        stale_files = _stale_files_for(cache_dir, stale)
        size = sum(p.stat().st_size for p in stale_files if p.exists())
        total_stale += len(stale)
        total_bytes += size
        print(f"  would free {_human_size(size)} across {len(stale_files)} files")
        for h in stale[:10]:
            print(f"    {h}")
        if len(stale) > 10:
            print(f"    ... and {len(stale) - 10} more")

        if args.delete:
            for p in stale_files:
                p.unlink(missing_ok=True)
            print(f"  deleted {len(stale_files)} files")

    summary = f"total stale hashes: {total_stale} ({_human_size(total_bytes)})"
    if args.delete:
        print(f"\nDONE — {summary}")
    else:
        print(f"\nDRY RUN — {summary}; pass --delete to remove")
    return 0


if __name__ == "__main__":
    sys.exit(main())
