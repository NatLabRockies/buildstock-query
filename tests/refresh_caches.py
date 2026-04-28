"""End-to-end cache refresh: clear usage logs → run tests → drop stale entries.

This is the convenience wrapper around `cleanup_stale_caches.py`. It does:

  1. Clear `.cache_usage_log` in every schema cache.
  2. Run the snapshot suite (default: SQL-hash check; pass `--check-data`
     to also fall through to Athena/cache and compare DataFrames).
  3. Run the invariants suite (so its hashes get added to the log).
  4. Run `cleanup_stale_caches.py --delete` (or `--dry-run` if `-n`).

Each step can be skipped individually if you only want a partial run, but
the order matters: clear must come before any test that should populate
the log, and cleanup must come after every such test. The defaults run
all steps in order.

Examples
--------
    # Full refresh, deletes stale entries.
    python tests/refresh_caches.py

    # Same flow, but only report stale (don't delete).
    python tests/refresh_caches.py -n

    # Compare DataFrames against parquet during the snapshot run.
    python tests/refresh_caches.py --check-data

    # Skip the snapshot run (e.g. you only care about invariants).
    python tests/refresh_caches.py --skip-snapshot
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def _run(cmd: list[str]) -> int:
    print(f"\n→ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Pass --delete=False to cleanup (report stale, don't remove).",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Pass --check-data to the snapshot suite (DataFrame comparison).",
    )
    parser.add_argument(
        "--skip-clear",
        action="store_true",
        help="Skip step 1 (don't truncate `.cache_usage_log`).",
    )
    parser.add_argument(
        "--skip-snapshot",
        action="store_true",
        help="Skip step 2 (don't run the snapshot suite).",
    )
    parser.add_argument(
        "--skip-invariants",
        action="store_true",
        help="Skip step 3 (don't run the invariants suite).",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip step 4 (don't run cleanup_stale_caches.py).",
    )
    args = parser.parse_args()

    cleanup = ["python", str(HERE / "cleanup_stale_caches.py")]

    if not args.skip_clear:
        rc = _run(cleanup + ["--clear"])
        if rc != 0:
            return rc

    pytest = [sys.executable, "-m", "pytest", "-s", "-v"]

    if not args.skip_snapshot:
        cmd = pytest + ["tests/test_query_snapshots.py"]
        if args.check_data:
            cmd.append("--check-data")
        rc = _run(cmd)
        if rc != 0:
            print("snapshot suite failed; aborting before cleanup", file=sys.stderr)
            return rc

    if not args.skip_invariants:
        rc = _run(pytest + ["tests/test_invariants.py"])
        if rc != 0:
            print("invariants suite failed; aborting before cleanup", file=sys.stderr)
            return rc

    if not args.skip_cleanup:
        cmd = list(cleanup)
        if not args.dry_run:
            cmd.append("--delete")
        rc = _run(cmd)
        if rc != 0:
            return rc

    print("\nrefresh complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
