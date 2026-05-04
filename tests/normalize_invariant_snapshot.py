"""Normalize the per-session JSONL log into `from_invariants.json`.

Reads `tests/query_snapshots/.from_invariants_log.jsonl` (written by
`tests/snapshot_recorder.py::record_query` during invariant test runs)
and merges into `tests/query_snapshots/from_invariants.json`.

Default behavior:
  - For each (schema, method, args) seen in the JSONL log:
      * if not already in `from_invariants.json`: add a new entry with
        `sql_hash: ""`. The user populates the hash via
        `pytest --update-snapshot` next.
      * if already in `from_invariants.json`: leave it alone. The
        snapshot harness handles SQL drift via its hash check; we
        don't want to silently overwrite a real drift here.

With `--prune`:
  - Any entry currently in `from_invariants.json` whose key is NOT
    in the JSONL is removed AND its sibling cache files
    (`<hash>.parquet`, `<hash>.sql`, `<hash>.json`) are deleted.
  - Use after a FULL invariant suite run (`pytest tests/test_invariants.py`).
    Pruning after a partial run will wipe legitimate entries.

Usage:
    python tests/normalize_invariant_snapshot.py            # add-only
    python tests/normalize_invariant_snapshot.py --prune    # add + delete-stale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from buildstock_query import BuildStockQuery
from buildstock_query.sql_cache import hash_sql
from tests.test_utility import _rehydrate_args, run_query_sql

SNAPSHOTS_ROOT = Path(__file__).resolve().parent / "query_snapshots"
JSONL_PATH = SNAPSHOTS_ROOT / ".from_invariants_log.jsonl"
JSON_PATH = SNAPSHOTS_ROOT / "from_invariants.json"


def _canonicalize_args(value: Any) -> Any:
    """Normalize legacy alias-qualified column refs to logical names."""
    if isinstance(value, str) and (value.startswith("bs.") or value.startswith("up.")):
        return value.split(".", 1)[1]
    if isinstance(value, list):
        return [_canonicalize_args(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_canonicalize_args(item) for item in value)
    if isinstance(value, dict):
        return {k: _canonicalize_args(v) for k, v in value.items()}
    return value


def _group_by_display_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, dict) and value.get("_bsq_col_ref") == "md_bldgid_column":
        return "bldg_id"
    if isinstance(value, dict) and value.get("_bsq_cols_ref") == "md_key_cols":
        return "md_key_cols"
    return None


def _stable_key(schema: str, method: str, args: dict[str, Any]) -> str:
    """Deterministic key for grouping: schema + dotted method + canonical-JSON args."""
    args = _canonicalize_args(args)
    return schema + "::" + method + "::" + json.dumps(args, sort_keys=True, default=str)


def _entry_name_for(schema: str, method: str, args: dict[str, Any], counter: int) -> str:
    """Generate a stable, human-readable entry name from (schema, method, args)."""
    parts = [method.replace(".", "_"), schema]
    if args.get("applied_only"):
        parts.append("applied")
    upgrade_id = args.get("upgrade_id")
    if upgrade_id and upgrade_id != "0":
        parts.append(f"u{upgrade_id}")
    elif upgrade_id == "0":
        parts.append("baseline")
    if args.get("annual_only") is False:
        tg = args.get("timestamp_grouping_func")
        parts.append(f"ts_{tg}" if tg else "ts")
    if args.get("group_by"):
        gb_names = [name for g in args["group_by"] if (name := _group_by_display_name(g))]
        if gb_names:
            parts.append("by_" + "_".join(gb_names))
    parts.append(f"v{counter}")
    return "__".join(parts)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"warn: skipping malformed JSONL line: {e}", file=sys.stderr)
    return out


def _read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"warn: existing {path.name} is malformed: {e}", file=sys.stderr)
        return []
    return data if isinstance(data, list) else []


def _entry_signature(entry: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    """Reverse the recorder's structure for an existing entry. Returns
    (schema, method, args) or None if the entry isn't in our format."""
    sql_hash = entry.get("sql_hash")
    if not isinstance(sql_hash, dict) or len(sql_hash) != 1:
        return None
    schema = next(iter(sql_hash))
    args = dict(entry.get("args", {}))
    method = args.pop("_method", "query")
    return schema, method, args


def _make_bsq(schema: str) -> BuildStockQuery:
    if schema == "resstock_oedi":
        return BuildStockQuery(
            "rescore",
            "buildstock_sdr",
            "resstock_2024_amy2018_release_2",
            buildstock_type="resstock",
            db_schema="resstock_oedi_vu",
            skip_reports=True,
            cache_folder=str(SNAPSHOTS_ROOT / "resstock_oedi_cache"),
        )
    if schema == "comstock_oedi":
        return BuildStockQuery(
            "rescore",
            "buildstock_sdr",
            "comstock_amy2018_r2_2025",
            buildstock_type="comstock",
            db_schema="comstock_oedi_state_and_county",
            skip_reports=True,
            cache_folder=str(SNAPSHOTS_ROOT / "comstock_oedi_cache"),
        )
    if schema == "comstock_oedi_agg":
        return BuildStockQuery(
            "rescore",
            "buildstock_sdr",
            "comstock_amy2018_r2_2025",
            buildstock_type="comstock",
            db_schema="comstock_oedi_agg_state_and_county",
            skip_reports=True,
            cache_folder=str(SNAPSHOTS_ROOT / "comstock_oedi_agg_cache"),
        )
    raise ValueError(f"unsupported schema {schema!r}")


def _cache_artifact_exists(schema: str, sha: str) -> bool:
    if not sha:
        return False
    cache_dir = SNAPSHOTS_ROOT / f"{schema}_cache"
    return any((cache_dir / f"{sha}{suffix}").exists() for suffix in (".parquet", ".sql", ".json"))


def _compute_cached_hash(
    *,
    schema: str,
    method: str,
    args_dict: dict[str, Any],
    bsq_by_schema: dict[str, BuildStockQuery],
) -> str:
    """Return the live SQL hash when the exact artifact already exists locally."""
    bsq = bsq_by_schema.get(schema)
    if bsq is None:
        bsq = bsq_by_schema[schema] = _make_bsq(schema)

    replay_args = dict(args_dict)
    if method != "query":
        replay_args["_method"] = method
    replay_args = _rehydrate_args(replay_args)
    actual_sql = run_query_sql(bsq, replay_args)
    actual_hash = hash_sql(actual_sql)
    return actual_hash if _cache_artifact_exists(schema, actual_hash) else ""


def _fill_blank_hashes(existing_by_key: dict[str, dict[str, Any]]) -> int:
    filled = 0
    bsq_by_schema: dict[str, BuildStockQuery] = {}
    warned_schemas: set[str] = set()
    try:
        for entry in existing_by_key.values():
            sig = _entry_signature(entry)
            if sig is None:
                continue
            schema, method, args_dict = sig
            if entry["sql_hash"].get(schema):
                continue
            try:
                sha = _compute_cached_hash(
                    schema=schema,
                    method=method,
                    args_dict=args_dict,
                    bsq_by_schema=bsq_by_schema,
                )
            except Exception as exc:
                if schema not in warned_schemas:
                    print(
                        f"warn: could not auto-fill blank hashes for {schema}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    warned_schemas.add(schema)
                continue
            if sha:
                entry["sql_hash"][schema] = sha
                filled += 1
    finally:
        bsq_by_schema.clear()
    return filled


def _delete_cache_triple(schema: str, sha: str) -> int:
    """Unlink `<hash>.parquet/.sql/.json` for `sha` in the schema's cache dir.
    Returns the count of files actually removed."""
    if not sha:
        return 0
    cache_dir = SNAPSHOTS_ROOT / f"{schema}_cache"
    removed = 0
    for suffix in (".parquet", ".sql", ".json"):
        path = cache_dir / f"{sha}{suffix}"
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Remove entries in from_invariants.json whose key isn't in the "
        "JSONL log, and delete their sibling cache files. Use only after a "
        "FULL invariant test run.",
    )
    args = parser.parse_args()

    log_entries = _read_jsonl(JSONL_PATH)
    if not log_entries and not args.prune:
        print(f"no log entries at {JSONL_PATH.name}; nothing to normalize")
        return 0

    existing = _read_existing(JSON_PATH)
    existing_by_key: dict[str, dict[str, Any]] = {}
    other_entries: list[dict[str, Any]] = []
    for entry in existing:
        sig = _entry_signature(entry)
        if sig is None:
            other_entries.append(entry)
            continue
        existing_by_key[_stable_key(*sig)] = entry

    seen_keys: set[str] = set()
    new_count = 0

    for log in log_entries:
        method = log["method"]
        args_dict = _canonicalize_args(log["args"])
        schema = log["schema"]
        key = _stable_key(schema, method, args_dict)
        seen_keys.add(key)

        if key in existing_by_key:
            continue  # add-only: don't touch existing entries

        # New entry — placeholder hash, user fills via --update-snapshot.
        json_args: dict[str, Any] = (
            {"_method": method, **args_dict} if method != "query" else dict(args_dict)
        )
        new_entry = {
            "name": _entry_name_for(schema, method, args_dict, len(existing_by_key) + 1),
            "schemas": [schema],
            "sql_hash": {schema: ""},
            "description": f"Auto-recorded from test_invariants ({method}, {schema}).",
            "args": json_args,
        }
        existing_by_key[key] = new_entry
        new_count += 1

    pruned_keys: list[str] = []
    pruned_files = 0
    if args.prune:
        for key in list(existing_by_key.keys()):
            if key in seen_keys:
                continue
            entry = existing_by_key[key]
            schema = next(iter(entry["sql_hash"]))
            sha = entry["sql_hash"].get(schema, "")
            pruned_files += _delete_cache_triple(schema, sha)
            del existing_by_key[key]
            pruned_keys.append(entry["name"])

    filled_hashes = _fill_blank_hashes(existing_by_key)

    # Rebuild list, preserving any non-recorder entries first, then
    # recorder entries in insertion order (Python dicts are ordered).
    out_list = list(other_entries) + list(existing_by_key.values())
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(out_list, indent=2) + "\n")

    parts = [f"normalized {len(log_entries)} log entries"]
    if new_count:
        unresolved_new = max(new_count - filled_hashes, 0)
        if unresolved_new:
            parts.append(f"+{new_count} new ({unresolved_new} still blank — run --update-snapshot to fill)")
        else:
            parts.append(f"+{new_count} new")
    if filled_hashes:
        parts.append(f"{filled_hashes} blank hashes filled from local cache")
    if args.prune:
        parts.append(f"-{len(pruned_keys)} pruned, {pruned_files} cache files deleted")
    parts.append(f"{len(out_list)} total in {JSON_PATH.name}")
    print("; ".join(parts))
    if pruned_keys:
        print("pruned entries:", file=sys.stderr)
        for name in pruned_keys:
            print(f"  - {name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
