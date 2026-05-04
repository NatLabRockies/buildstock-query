"""Auto-record `bsq.query()` / `bsq.get_building_ids()` call shapes from
invariant tests into a session JSONL log so the existing snapshot harness
can pin them.

Use from invariant tests by adding a sibling `record_query(...)` call
next to the live invocation, with the same args:

    overall_df = bsq.query(enduses=enduses, restrict=restrict)
    record_query(bsq, {"enduses": enduses, "restrict": restrict})

The args dict you pass is stored mostly as-is. Resolved column names
like `"out.electricity.total.energy_consumption"` stay concrete, which
means each schema gets its own entry (the column name differs across
schemas). That's fine: the snapshot value is the (SQL, data)
fingerprint of one specific query, not a portable args description.

The one reverse-encoding exception is BSQ's live metadata-column
handles (`md_bldgid_column`, `md_key_cols`): those are preserved as
markers so snapshot replay can rebuild the same live objects instead
of drifting to a different SQL shape.

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
from typing import Any

from buildstock_query.schema.utilities import MappedColumn, SACol, SALabel
from tests.test_utility import SNAPSHOTS_ROOT


JSONL_PATH = SNAPSHOTS_ROOT / ".from_invariants_log.jsonl"

_BSQ_COL_REF = "_bsq_col_ref"
_BSQ_COLS_REF = "_bsq_cols_ref"


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


def _is_bsq_md_bldgid_column(bsq, value: Any) -> bool:
    return bsq is not None and value is getattr(bsq, "md_bldgid_column", None)


def _is_bsq_md_key_cols(bsq, value: Any) -> bool:
    if bsq is None or not isinstance(value, tuple):
        return False
    md_key_cols = tuple(getattr(bsq, "md_key_cols", ()))
    return len(value) == len(md_key_cols) and all(item is key for item, key in zip(value, md_key_cols))


def _jsonable_arg(value: Any, *, bsq=None) -> Any:
    """Convert live BSQ args into a JSON-stable shape the snapshot harness can replay.

    Important normalization:
      - SQLAlchemy columns like `bs.c.bldg_id` serialize to `column.name`
        (`"bldg_id"`), not `str(column)` (`"bs.bldg_id"`). The latter bakes in
        an alias that `_get_column()` cannot resolve when the snapshot harness
        replays the entry later.
      - BSQ's live metadata-column handles (`md_bldgid_column`, `md_key_cols`)
        are preserved as markers instead of flattened to strings. Some TS
        shapes hash differently when replayed from plain string refs.
      - Marker dicts (`_applied_filter`, `_calc_column`, `_mapped_column`) are
        preserved structurally and recursed into.
    """
    if _is_bsq_md_bldgid_column(bsq, value):
        return {_BSQ_COL_REF: "md_bldgid_column"}
    if _is_bsq_md_key_cols(bsq, value):
        return {_BSQ_COLS_REF: "md_key_cols"}
    if isinstance(value, SACol):
        return value.name
    if isinstance(value, SALabel):
        return value.name
    if isinstance(value, MappedColumn):
        key = value.key.name if isinstance(value.key, SACol) else str(value.key)
        return {
            "_mapped_column": {
                "name": value.name,
                "key_column": key,
                "mapping_dict": value.mapping_dict,
            }
        }
    if isinstance(value, tuple):
        return [_jsonable_arg(item, bsq=bsq) for item in value]
    if isinstance(value, list):
        return [_jsonable_arg(item, bsq=bsq) for item in value]
    if isinstance(value, dict):
        return {k: _jsonable_arg(v, bsq=bsq) for k, v in value.items()}
    return value


def record_query(bsq, args: dict[str, Any], *, method: str = "query") -> None:
    """Append one snapshot-shape entry to the session JSONL log.

    `args` is the live args dict you'd pass to `bsq.<method>(...)`.
    Resolved values are stored mostly verbatim. Marker shapes
    (`{"_applied_filter": {...}}`, `{"_bsq_col_ref": ...}`,
    `{"_bsq_cols_ref": ...}`, `"rate_map_flat": <rate>`) are
    preserved for the snapshot harness to resolve at run time.

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
        "args": _jsonable_arg(args, bsq=bsq),
    }
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(JSONL_PATH, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
