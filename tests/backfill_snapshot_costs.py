"""Backfill `cost` metrics into snapshot JSON entries from Athena history.

For each entry in tests/query_snapshots/*.json that has a `sql_hash` populated
but no `cost` field (or one with null values), look up the past Athena
execution via `bsq.build_query_cost_index()` and fill in `data_scanned_mb`,
`query_time_ms`, etc. Entries whose original execution is older than Athena's
~45-day history retention are left as `cost: null` per schema.

Usage:
    python tests/backfill_snapshot_costs.py [--dry-run]

The script builds ONE index per schema (resstock_oedi, comstock_oedi) by
walking that workgroup's full Athena history once, then dictionary-looks-up
each snapshot's hash. Dry-run prints what would change without writing.
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
    raise ValueError(f"Unknown schema: {schema}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes; just report.")
    args = parser.parse_args()

    schemas = ("resstock_oedi", "comstock_oedi")
    indices: dict[str, dict] = {}
    for schema in schemas:
        print(f"Building cost index for {schema}...")
        bsq = _make_bsq(schema)
        idx = bsq.build_query_cost_index()
        indices[schema] = idx
        print(f"  {len(idx)} entries indexed")

    json_files = sorted(p for p in SNAPSHOTS_ROOT.glob("*.json"))
    print(f"\nProcessing {len(json_files)} snapshot JSON files...")

    total_filled = 0
    total_missing = 0
    for jf in json_files:
        data = json.loads(jf.read_text())
        if not isinstance(data, list):
            continue
        modified = False
        per_file_filled = 0
        per_file_missing = 0
        for entry in data:
            sql_hash_dict = entry.get("sql_hash", {})
            if not isinstance(sql_hash_dict, dict):
                continue
            cost_dict = entry.get("cost") or {}
            if not isinstance(cost_dict, dict):
                cost_dict = {}
            entry_modified = False
            for schema, hash_val in sql_hash_dict.items():
                if not hash_val:
                    continue
                if schema not in indices:
                    continue
                if cost_dict.get(schema):  # already filled
                    continue
                cost_entry = indices[schema].get(hash_val)
                if cost_entry is None:
                    per_file_missing += 1
                    cost_dict[schema] = None  # explicit null marker
                    entry_modified = True
                    continue
                cost_dict[schema] = {
                    "data_scanned_mb": cost_entry["data_scanned_mb"],
                    "query_time_ms": cost_entry["query_time_ms"],
                }
                per_file_filled += 1
                entry_modified = True
            if entry_modified:
                entry["cost"] = cost_dict
                modified = True
        if modified:
            total_filled += per_file_filled
            total_missing += per_file_missing
            print(f"  {jf.name}: filled={per_file_filled} missing={per_file_missing}")
            if not args.dry_run:
                jf.write_text(json.dumps(data, indent=2) + "\n")

    print(f"\nDone. Filled {total_filled} cost entries; {total_missing} not found in history.")
    if args.dry_run:
        print("(dry-run; no files written)")


if __name__ == "__main__":
    main()
