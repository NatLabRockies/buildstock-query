"""Auto-record `bsq.query()` / `bsq.get_building_ids()` call shapes from
invariant tests into a session JSONL log so the existing snapshot harness
can pin them.

Use from invariant tests by adding a sibling `record_query(...)` call
next to the live invocation, with the same args:

    overall_df = bsq.query(enduses=enduses, restrict=restrict)
    record_query(bsq, {"enduses": enduses, "restrict": restrict})

The args dict you pass IS what gets serialized — no placeholder
reverse-encoding. Resolved column names like
`"out.electricity.total.energy_consumption"` are stored as-is, which
means each schema gets its own entry (the column name differs across
schemas). That's fine: the snapshot value is the (SQL, data)
fingerprint of one specific query, not a portable args description.

Only special case: `get_applied_buildings_filter()` returns a tuple
that contains a SA Subquery, which can't be JSON-encoded. For those
cases pass the marker form `{"_applied_filter": {"all_of": [...]}}`
inside `restrict` (same shape used by other JSON entries).

The recorder logs ONLY the (schema, method, args) shape — never a
SQL hash. Hashes belong to the snapshot harness; auto-writing them
would defeat the whole purpose of having a drift check. New entries
land in `from_invariants.json` with `sql_hash: ""` and the user runs
`--update-snapshot` to populate.

Concurrent pytest workers append safely: each line is one entry,
written with a single `f.write()` call (atomic on POSIX up to PIPE_BUF,
comfortably above our line sizes). Dedup happens at normalize time.

Normalize via `python tests/normalize_invariant_snapshot.py`.

Set `BSQ_DISABLE_RECORD=1` in the env to disable recording during
runs that shouldn't accumulate log entries (e.g. the snapshot suite
itself).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tests.test_utility import SNAPSHOTS_ROOT


JSONL_PATH = SNAPSHOTS_ROOT / ".from_invariants_log.jsonl"


# Map from `BuildStockQuery.run_params.db_schema` (TOML name) to the
# snapshot-key name used in JSON `sql_hash` dicts.
_DB_SCHEMA_TO_SNAPSHOT_KEY = {
    "resstock_oedi_vu": "resstock_oedi",
    "comstock_oedi_state_and_county": "comstock_oedi",
    "comstock_oedi_agg_state_and_county": "comstock_oedi_agg",
}


def _schema_key_from_bsq(bsq) -> str:
    """Return the snapshot-key name (`resstock_oedi`, `comstock_oedi`,
    `comstock_oedi_agg`) matching the BSQ's TOML schema."""
    db_schema_name = bsq.run_params.db_schema
    try:
        return _DB_SCHEMA_TO_SNAPSHOT_KEY[db_schema_name]
    except KeyError:
        raise ValueError(
            f"Unknown db_schema {db_schema_name!r} — extend "
            f"_DB_SCHEMA_TO_SNAPSHOT_KEY in snapshot_recorder.py."
        )


def record_query(bsq, args: dict[str, Any], *, method: str = "query") -> None:
    """Append one snapshot-shape entry to the session JSONL log.

    `args` is the live args dict you'd pass to `bsq.<method>(...)`.
    Resolved values are stored verbatim — there's no placeholder
    reverse-encoding. Marker shapes (`{"_applied_filter": {...}}`,
    `"rate_map_flat": <rate>`) are preserved as-is for the snapshot
    harness to resolve at run time.

    `method` defaults to `"query"`. For other public methods (e.g.
    `"get_building_ids"`), pass that name. Dotted paths
    (`"utility.get_eiaids"`) are resolved by getattr-walking from `bsq`.

    No SQL hash is computed here — hashing/drift detection is the
    snapshot harness's job, and short-circuiting it would defeat the
    purpose of the drift check.
    """
    if os.environ.get("BSQ_DISABLE_RECORD"):
        return

    schema = _schema_key_from_bsq(bsq)
    entry = {
        "schema": schema,
        "method": method,
        "args": args,
    }
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(entry, default=str, sort_keys=True) + "\n")
