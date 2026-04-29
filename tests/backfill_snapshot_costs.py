"""Backfill `<hash>.json` Athena execution metadata sidecars into the snapshot
cache directories.

Walks each schema's snapshot cache (`tests/query_snapshots/<schema>_cache/`),
finds `<hash>.parquet` entries that lack an `<hash>.json` companion, and
populates them by looking up the past Athena execution via the workgroup's
query history. Each metadata file is the full GetQueryExecution response —
DataScannedInBytes, EngineExecutionTimeInMillis, ResultReuseInformation,
EngineVersion, etc. — stored verbatim so future analyses can pull whatever
they need without re-fetching.

Usage:
    python tests/backfill_snapshot_costs.py [--dry-run]

Idempotent: re-running on already-populated entries is a no-op (the script
only fills in missing files). Entries whose original execution is older than
Athena's ~45-day history retention are reported as "skipped" and remain
without a metadata sidecar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from buildstock_query import BuildStockQuery


SNAPSHOTS_ROOT = Path(__file__).parent / "query_snapshots"


def _make_bsq(schema: str) -> BuildStockQuery:
    """Construct a BSQ for the given snapshot fixture schema."""
    if schema == "resstock_oedi":
        return BuildStockQuery(
            "rescore", "buildstock_sdr", "resstock_2024_amy2018_release_2",
            buildstock_type="resstock", db_schema="resstock_oedi_vu",
            skip_reports=True,
            cache_folder=str(SNAPSHOTS_ROOT / "resstock_oedi_cache"),
        )
    if schema == "comstock_oedi":
        return BuildStockQuery(
            "rescore", "buildstock_sdr", "comstock_amy2018_r2_2025",
            buildstock_type="comstock", db_schema="comstock_oedi_state_and_county",
            skip_reports=True,
            cache_folder=str(SNAPSHOTS_ROOT / "comstock_oedi_cache"),
        )
    if schema == "comstock_oedi_agg":
        return BuildStockQuery(
            "rescore", "buildstock_sdr", "comstock_amy2018_r2_2025",
            buildstock_type="comstock", db_schema="comstock_oedi_agg_state_and_county",
            skip_reports=True,
            cache_folder=str(SNAPSHOTS_ROOT / "comstock_oedi_agg_cache"),
        )
    raise ValueError(f"Unknown schema: {schema}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes; just report.")
    args = parser.parse_args()

    schemas = ("resstock_oedi", "comstock_oedi", "comstock_oedi_agg")
    for schema in schemas:
        cache_dir = SNAPSHOTS_ROOT / f"{schema}_cache"
        # Find cached entries lacking metadata sidecars BEFORE we walk history
        # so we can report what we're about to do.
        missing = [p.stem for p in cache_dir.glob("*.parquet") if not (cache_dir / f"{p.stem}.json").exists()]
        if not missing:
            print(f"{schema}: all cache entries already have metadata; skipping")
            continue
        print(f"{schema}: {len(missing)} cache entries missing metadata; querying history...")
        bsq = _make_bsq(schema)
        if args.dry_run:
            index = bsq.build_query_metadata_index()
            found = sum(1 for h in missing if h in index)
            print(f"  would fill {found}, skip {len(missing) - found}")
        else:
            filled, skipped = bsq.backfill_cache_metadata()
            print(f"  filled {filled}, skipped {skipped} (older than Athena history)")


if __name__ == "__main__":
    main()
