"""Shared helpers for the query snapshot tests.

Snapshot entries are content-addressed: each JSON entry carries an `sql_hash`
field, and the corresponding `<sql_hash>.sql` + `<sql_hash>.parquet` pair lives
in `<schema>_cache/`. Lookups happen automatically through the bsq instance's
SqlCache (set up in conftest via `cache_folder`).

`--update-snapshot` flow:
- Hash matches stored value → no-op (already correct).
- Hash differs but sqlglot says equivalent → rename the parquet to the new hash,
  rewrite the .sql sidecar, patch the JSON. No Athena re-run.
- Hash differs and sqlglot says they really differ → run query, compare to old
  parquet. Write new pair + delete old pair if data matches/is equivalent.
  Leave both alone on real data drift unless --overwrite-snapshot.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sqlglot

from buildstock_query.sql_cache import SqlCache, hash_sql, normalize_sql


SNAPSHOTS_ROOT = Path(__file__).parent / "query_snapshots"

DATA_RTOL = 1e-4
DATA_ATOL = 1e-6


# Session-wide tally appended by evaluate_entries(); pytest_terminal_summary in
# conftest.py reads this at the end of the run to print real entry/variant counts
# (which exceed the test-function count because each pytest node processes a
# whole JSON file of entries × variants).
SESSION_TOTALS: dict[str, int] = {
    "entries": 0,
    "variants": 0,
    "passed": 0,
    "skipped": 0,
    "errored": 0,
    "updated": 0,
    "failed": 0,
}


# Per-entry status precedence (worst → best). `match` = hash equal; `sqlglot_only`
# = hash differs but sqlglot considers equivalent; `mismatch` = real SQL drift.
# `skipped` sits outside the precedence: schema doesn't support this query shape
# (raises UnsupportedQueryShape during SQL generation).
STATUS_PRECEDENCE = ["mismatch", "sqlglot_only", "match"]


# --- per-schema placeholder substitution -------------------------------------
#
# Test JSON fixtures and invariant tests reference schema-specific values
# (column names, building-type values, mapping dicts) by a placeholder name like
# `"$ELECTRICITY_TOTAL"`. `resolve_placeholder(schema, name, annual=...)`
# returns the corresponding concrete value for the requested schema.
#
# `annual` controls the comstock annual/timeseries column-suffix split:
# annual energy columns end in `..kwh`, timeseries ones don't. Other
# placeholders ignore `annual`.
#
# Two callers:
#   1. Invariant tests (`test_invariants.py`) call `resolve_placeholder(...)`
#      directly, one value at a time.
#   2. The JSON fixture loader walks the args tree and calls it for each
#      `"$"`-prefixed string it encounters (see `_resolve_placeholders` below).
# Strings without a `$` prefix pass through unchanged. Names are case-
# insensitive, accepted with or without the leading `$`.

_SUPPORTED_SCHEMAS = ("resstock_oedi", "comstock_oedi")


def _resolve_resstock_placeholder(name: str, *, annual: bool) -> Any:
    table: dict[str, Any] = {
        "ELECTRICITY_TOTAL": "out.electricity.total.energy_consumption",
        "NATURAL_GAS_TOTAL": "out.natural_gas.total.energy_consumption",
        "BUILDING_TYPE_COL": "geometry_building_type_recs",
        # Raw column name as it appears on the SA-reflected baseline table.
        # `restrict=[("state", ...)]` works without the prefix because the
        # restrict resolver auto-tries `in.` variants; methods that index
        # `tbl.c[...]` directly (get_distinct_vals/get_distinct_count) need
        # the raw name.
        "STATE_COL_BASELINE": "in.state",
        # Real Athena table name for the resstock_oedi_vu baseline view.
        # Some BSQ entry points (`get_distinct_vals`, `get_distinct_count`)
        # require a literal table name when the implicit `bs_table.name` is
        # the SA alias `"baseline"` rather than a real Athena table.
        "BS_TABLE_NAME": "resstock_2024_amy2018_release_2_metadata",
        # Per-schema sqft column for the `weights` snapshot — the `..ft2`
        # suffix on comstock annual columns mirrors the `..kwh` enduse suffix.
        "SQFT_COL": "in.sqft",
        # Pair of states used by test_multi_state_savings_equals_sum_of_per_state.
        # Resstock's bldg_id namespace happens to be globally unique across
        # states, so any two states give a clean disjoint additivity test —
        # CO + WY chosen for the small data volume.
        "MULTI_STATE_PAIR": ["CO", "WY"],
        "AVOID_BUILDING_TYPE": "Mobile Home",
        "AVOID_BUILDING_TYPES_MULTI": ["Mobile Home", "Multi-Family with 5+ Units"],
        "VINTAGE_BUCKET": "1980s",
        "BUILDING_TYPE_MAPPING": {
            "Mobile Home": "MH",
            "Single-Family Detached": "SF",
            "Single-Family Attached": "SF",
            "Multi-Family with 2 - 4 Units": "MF",
            "Multi-Family with 5+ Units": "MF",
        },
        "BUILDING_TYPE_LOAD_FACTOR": {
            "Mobile Home": 0.5,
            "Single-Family Detached": 1.0,
            "Single-Family Attached": 0.8,
            "Multi-Family with 2 - 4 Units": 0.6,
            "Multi-Family with 5+ Units": 0.4,
        },
    }
    return table[name]


def _resolve_comstock_placeholder(name: str, *, annual: bool) -> Any:
    suffix = "..kwh" if annual else ""
    table: dict[str, Any] = {
        "ELECTRICITY_TOTAL": f"out.electricity.total.energy_consumption{suffix}",
        "NATURAL_GAS_TOTAL": f"out.natural_gas.total.energy_consumption{suffix}",
        "BUILDING_TYPE_COL": "comstock_building_type",
        # comstock_oedi_state_and_county TOML places state as a top-level
        # partition column on the baseline table — no `in.` prefix.
        "STATE_COL_BASELINE": "state",
        "BS_TABLE_NAME": "comstock_amy2018_r2_2025_md_by_state_and_county_parquet",
        "SQFT_COL": "in.sqft..ft2",
        # Pair of states used by test_multi_state_savings_equals_sum_of_per_state.
        # Comstock's bldg_id namespace IS reused across states (the schema's
        # composite key (bldg_id, state) is what disambiguates them), so this
        # pair is chosen specifically for confirmed bldg_id collision: CO ∩ NM
        # contains 413 overlapping bldg_id values. CO is already cached from
        # other tests, so this pair only needs one fresh Athena query (NM-only)
        # plus the multi-state aggregate. Additivity holding under this pair
        # confirms the join logic correctly treats (bldg_id, state) as the
        # composite identity rather than bldg_id alone — a bldg_id-only join
        # would double-count overlapping buildings and break the sum check.
        "MULTI_STATE_PAIR": ["CO", "NM"],
        "AVOID_BUILDING_TYPE": "Warehouse",
        "AVOID_BUILDING_TYPES_MULTI": ["Warehouse", "SmallOffice"],
        "VINTAGE_BUCKET": "1980 to 1989",
        "BUILDING_TYPE_MAPPING": {
            "FullServiceRestaurant": "Food",
            "QuickServiceRestaurant": "Food",
            "RetailStandalone": "Retail",
            "RetailStripmall": "Retail",
            "PrimarySchool": "School",
            "SecondarySchool": "School",
            "SmallOffice": "Office",
            "MediumOffice": "Office",
            "LargeOffice": "Office",
            "SmallHotel": "Lodging",
            "LargeHotel": "Lodging",
            "Hospital": "Health",
            "Outpatient": "Health",
            "Warehouse": "Warehouse",
        },
        "BUILDING_TYPE_LOAD_FACTOR": {
            "FullServiceRestaurant": 1.0,
            "QuickServiceRestaurant": 0.8,
            "RetailStandalone": 0.6,
            "RetailStripmall": 0.5,
            "PrimarySchool": 0.7,
            "SecondarySchool": 0.7,
            "SmallOffice": 0.4,
            "MediumOffice": 0.6,
            "LargeOffice": 0.9,
            "SmallHotel": 0.5,
            "LargeHotel": 0.8,
            "Hospital": 1.0,
            "Outpatient": 0.7,
            "Warehouse": 0.3,
        },
    }
    return table[name]


def resolve_placeholder(schema: str, placeholder: str, *, annual: bool = True) -> Any:
    """Return the schema-specific value for `placeholder`.

    `placeholder` is accepted with or without a leading `$` and is case-
    insensitive (`"$ELECTRICITY_TOTAL"`, `"electricity_total"` both work).
    Raises `ValueError` for unknown schemas, and `KeyError` (with schema
    context) for unknown placeholder names.
    """
    name = placeholder.lstrip("$").upper()
    if schema == "resstock_oedi":
        resolver = _resolve_resstock_placeholder
    elif schema == "comstock_oedi":
        resolver = _resolve_comstock_placeholder
    else:
        raise ValueError(f"unknown schema '{schema}'; expected one of {list(_SUPPORTED_SCHEMAS)}")
    try:
        return resolver(name, annual=annual)
    except KeyError:
        raise KeyError(f"unknown {schema} placeholder: ${name}") from None


def _resolve_placeholders(value: Any, schema: str, *, annual: bool) -> Any:
    """Recursively walk a JSON-loaded structure, resolving `"$"`-prefixed strings.

    Only strings that start with `$` are treated as placeholders; everything
    else passes through unchanged. Dicts and lists are walked into. Dict-
    valued placeholders (`$BUILDING_TYPE_MAPPING`) plug in a whole sub-tree.
    """
    if isinstance(value, str):
        # Only treat as a placeholder if the WHOLE string is a single $TOKEN.
        # Multi-token strings like "$ELECTRICITY_TOTAL - $NATURAL_GAS_TOTAL"
        # (used by calculated_column expressions) are left alone so a
        # specialized test function can substitute them with proper context.
        if value.startswith("$") and re.fullmatch(r"\$[A-Z_][A-Z0-9_]*", value):
            return resolve_placeholder(schema, value, annual=annual)
        return value
    if isinstance(value, list):
        return [_resolve_placeholders(item, schema, annual=annual) for item in value]
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v, schema, annual=annual) for k, v in value.items()}
    return value


@dataclass
class SnapshotEntry:
    name: str
    description: str
    sql_hash: str  # empty string for new entries; --update-snapshot fills it
    args: list[dict[str, Any]]  # list of equivalent arg-variants (≥1)
    source_path: Path
    schema: str  # e.g. "resstock_oedi" — picks the cache directory

    @property
    def cache_dir(self) -> Path:
        return SNAPSHOTS_ROOT / f"{self.schema}_cache"

    @property
    def stored_sql_path(self) -> Path | None:
        if not self.sql_hash:
            return None
        return self.cache_dir / f"{self.sql_hash}.sql"

    @property
    def stored_parquet_path(self) -> Path | None:
        if not self.sql_hash:
            return None
        return self.cache_dir / f"{self.sql_hash}.parquet"


@dataclass
class VariantResult:
    index: int
    args: dict[str, Any]
    actual_sql: str
    actual_hash: str
    status: str  # match, sqlglot_only, mismatch


@dataclass
class EntryOutcome:
    entry: SnapshotEntry
    status: str  # worst variant status (precedence above)
    variants: list[VariantResult]
    expected_sql: str | None = None  # stored .sql sidecar (same for all variants)
    data_status: str | None = None  # match, equivalent, mismatch, missing, error, skipped
    data_error: str | None = None
    updated: bool = False  # True if the cache/JSON was updated this run
    update_note: str = ""  # human-readable description of what was updated

    @property
    def primary_sql(self) -> str:
        return self.variants[0].actual_sql

    @property
    def primary_hash(self) -> str:
        return self.variants[0].actual_hash


def load_entries(json_path: Path, *, schema: str) -> list[SnapshotEntry]:
    """Load every entry from `json_path`, resolving placeholders for `schema`.

    Each variant's `annual_only` flag picks the leg-aware resolution
    (comstock annual columns get the `..kwh` suffix, ts columns don't), so the
    same JSON entry can declare a single `$ELECTRICITY_TOTAL` placeholder and
    still produce the right column name for both legs.

    Entries may declare an optional top-level `"schemas": ["resstock_oedi", ...]`
    field to restrict which schemas they apply to. This is for methods that
    only exist on a subset of schemas (e.g. utility queries need an `eiaid`
    mapping configured, which only resstock has). Entries without this field
    apply to all schemas.
    """
    if schema not in _SUPPORTED_SCHEMAS:
        raise ValueError(f"unknown schema '{schema}'; expected one of {list(_SUPPORTED_SCHEMAS)}")
    raw = json.loads(json_path.read_text())

    entries = []
    for item in raw:
        allowed_schemas = item.get("schemas")
        if allowed_schemas is not None and schema not in allowed_schemas:
            continue
        args_field = item["args"]
        # Accept either a single dict (legacy) or a list of dicts (new).
        if isinstance(args_field, dict):
            raw_variants = [args_field]
        elif isinstance(args_field, list):
            if not args_field:
                raise ValueError(f"{json_path.name}: entry '{item['name']}' has empty args list")
            raw_variants = list(args_field)
        else:
            raise ValueError(
                f"{json_path.name}: entry '{item['name']}' args must be dict or list of dicts, "
                f"got {type(args_field).__name__}"
            )

        variants = []
        for raw_variant in raw_variants:
            # `annual_only` is a `bsq.query()` kwarg that signals the leg-aware
            # column-suffix resolution (`out.electricity.total.energy_consumption..kwh`
            # for comstock annual; bare for TS). For methods that don't accept
            # `annual_only` but operate on the TS table (agg.get_building_average_kws_at,
            # utility.aggregate_ts_by_eiaid, utility.calculate_tou_bill), the
            # entry can opt out of annual resolution by setting
            # `_annual_resolve: false`. This marker is consumed at load time and
            # not passed through to the method.
            annual_resolve_override = raw_variant.pop("_annual_resolve", None)
            if annual_resolve_override is not None:
                annual_only = bool(annual_resolve_override)
            else:
                annual_only = bool(raw_variant.get("annual_only", True))
            resolved = _resolve_placeholders(raw_variant, schema, annual=annual_only)
            variants.append(_rehydrate_args(resolved))

        # sql_hash in the JSON is a per-schema dict: {"resstock_oedi": "...", "comstock_oedi": "..."}.
        # Resolve to the current schema so the rest of the harness stays unaware of the dict shape.
        raw_hash = item.get("sql_hash", {})
        if isinstance(raw_hash, str):
            # Legacy single-string shape (pre-multi-schema migration). Treat as empty.
            sql_hash = ""
        else:
            sql_hash = raw_hash.get(schema, "")

        entries.append(
            SnapshotEntry(
                name=item["name"],
                description=item.get("description", ""),
                sql_hash=sql_hash,
                args=variants,
                source_path=json_path,
                schema=schema,
            )
        )
    return entries


def _rehydrate_args(args: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON list-of-two entries in restrict/avoid back into tuples."""
    out = dict(args)
    for key in ("restrict", "avoid"):
        if key in out and out[key] is not None:
            out[key] = [tuple(item) if isinstance(item, list) and len(item) == 2 else item for item in out[key]]
    return out


def compare_sql(expected: str, actual: str) -> str:
    """Return one of: match, sqlglot_only, mismatch.

    Hashes are computed on normalized SQL, so whitespace-only differences never
    reach this function. We're either looking at semantic equivalence
    (sqlglot_only — cosmetic refactor like `a + b` → `b + a` on a commutative op,
    parens reshuffles, etc.) or real drift (mismatch).

    Multi-statement results (joined with MULTI_SQL_SEPARATOR by `_flatten_sql_result`)
    are compared statement-by-statement: must have the same count and each pair must
    match or be sqlglot-equivalent. Mismatched count → mismatch."""
    if normalize_sql(expected) == normalize_sql(actual):
        return "match"
    expected_parts = expected.split(MULTI_SQL_SEPARATOR)
    actual_parts = actual.split(MULTI_SQL_SEPARATOR)
    if len(expected_parts) != len(actual_parts):
        return "mismatch"
    overall = "match"
    for exp_part, act_part in zip(expected_parts, actual_parts):
        if normalize_sql(exp_part) == normalize_sql(act_part):
            continue
        try:
            diff = sqlglot.diff(sqlglot.parse_one(exp_part), sqlglot.parse_one(act_part), delta_only=True)
        except Exception:
            return "mismatch"
        if diff:
            return "mismatch"
        overall = "sqlglot_only"
    return overall


ARRAY_RTOL = 0.05  # approx_percentile sampling noise; scaled to max|array| (see _compare_array_column).


def compare_data(expected: pd.DataFrame, actual: pd.DataFrame) -> tuple[bool, str | None]:
    """Sort both frames by all columns then compare with pd.testing.assert_frame_equal.

    Returns (matched, error_message). `matched=True` covers exact match plus tolerance-
    bounded numeric drift and dtype differences (Decimal/float64/int interop). For the
    "looser" notion of equivalence — added/removed columns where the intersection still
    matches — see `is_data_equivalent_but_different`.

    Columns containing array values (e.g. quartile percentile arrays from Athena's
    `approx_percentile`) are excluded from the sort key (numpy arrays aren't hashable)
    and compared element-wise with a looser `ARRAY_RTOL` to absorb the per-run sampling
    noise that approx_percentile introduces.
    """
    if set(expected.columns) != set(actual.columns):
        return False, f"column mismatch: expected={sorted(expected.columns)} actual={sorted(actual.columns)}"
    actual = actual[list(expected.columns)]
    array_cols = [c for c in expected.columns if _has_array_values(expected[c])]
    sort_cols = [c for c in expected.columns if c not in array_cols]
    if not sort_cols:
        # Pathological case: all columns are arrays. Compare in source order.
        exp_sorted = expected.reset_index(drop=True)
        act_sorted = actual.reset_index(drop=True)
    else:
        try:
            exp_sorted = expected.sort_values(sort_cols, kind="stable").reset_index(drop=True)
            act_sorted = actual.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        except TypeError as exc:
            return False, f"sort failed: {exc}"
    if array_cols:
        scalar_cols = [c for c in expected.columns if c not in array_cols]
        try:
            pd.testing.assert_frame_equal(
                exp_sorted[scalar_cols], act_sorted[scalar_cols],
                rtol=DATA_RTOL, atol=DATA_ATOL, check_dtype=False,
            )
        except (AssertionError, TypeError) as exc:
            # TypeError fires when assert_frame_equal can't even attempt the
            # element-wise check — e.g. one parquet's column is dtype=object
            # holding Decimal/python-int values while the other is native int64.
            # That's a real shape difference; treat it as a mismatch so the
            # overwrite path writes the new parquet rather than crashing.
            return False, str(exc)
        for col in array_cols:
            err = _compare_array_column(col, exp_sorted[col], act_sorted[col])
            if err is not None:
                return False, err
        return True, None
    try:
        pd.testing.assert_frame_equal(
            exp_sorted, act_sorted, rtol=DATA_RTOL, atol=DATA_ATOL, check_dtype=False
        )
    except (AssertionError, TypeError) as exc:
        return False, str(exc)
    return True, None


def _has_array_values(series: pd.Series) -> bool:
    if series.dtype != object or series.empty:
        return False
    sample = series.dropna()
    if sample.empty:
        return False
    first = sample.iloc[0]
    return isinstance(first, (np.ndarray, list, tuple))


def _compare_array_column(name: str, expected: pd.Series, actual: pd.Series) -> str | None:
    """Element-wise array comparison scaled to the array's own magnitude.

    Per-element tolerance: `atol = ARRAY_RTOL * max(|expected_values|)`. This means
    "differ by no more than X% of the largest value in the array" — the right metric
    for percentile arrays from `approx_percentile`, where sampling noise is bounded
    relative to the array's range, not relative to each individual value (a
    near-zero quartile shouldn't fail the comparison just because two near-zero
    samples happen to differ by a factor of 2).
    """
    if len(expected) != len(actual):
        return f"column '{name}' length differs: expected={len(expected)} actual={len(actual)}"
    for idx, (exp_val, act_val) in enumerate(zip(expected, actual)):
        exp_arr = np.asarray(exp_val, dtype=float)
        act_arr = np.asarray(act_val, dtype=float)
        if exp_arr.shape != act_arr.shape:
            return f"column '{name}' row {idx} shape differs: {exp_arr.shape} vs {act_arr.shape}"
        scale = float(np.nanmax(np.abs(exp_arr))) if exp_arr.size else 0.0
        atol = max(DATA_ATOL, ARRAY_RTOL * scale)
        if not np.allclose(exp_arr, act_arr, rtol=0.0, atol=atol, equal_nan=True):
            diff = np.nanmax(np.abs(exp_arr - act_arr))
            return (
                f"column '{name}' row {idx} differs by {diff:.4e} (tolerance "
                f"{atol:.4e} = ARRAY_RTOL * max|expected| = {ARRAY_RTOL} * {scale:.4e})"
            )
    return None


def is_data_equivalent_but_different(
    expected: pd.DataFrame, actual: pd.DataFrame
) -> tuple[bool, str | None]:
    """Same answer in a different shape.

    True when the column sets differ but every shared column matches within tolerance
    — typically a refactor that adds or drops a metadata column without changing the
    underlying query semantics. This is the data analog of `sqlglot_only` for SQL.

    Same-column-set differences are not equivalent: `compare_data` is the authority
    for those (match within tolerance or real mismatch).
    """
    exp_cols = set(expected.columns)
    act_cols = set(actual.columns)
    if exp_cols == act_cols:
        return False, None
    common = sorted(exp_cols & act_cols)
    if not common:
        return False, "no common columns to compare"
    only_expected = sorted(exp_cols - act_cols)
    only_actual = sorted(act_cols - exp_cols)
    matched, err = compare_data(expected[common], actual[common])
    if not matched:
        return False, f"shared columns differ: {err}"
    return True, (
        f"columns shifted but shared values match: "
        f"removed={only_expected!r} added={only_actual!r}"
    )


# Separator inserted between SQL strings when a method (e.g. report.get_success_report,
# utility.aggregate_ts_by_eiaid, agg.get_building_average_kws_at) returns multiple
# SQL strings from get_query_only=True. The separator is a SQL comment so it
# survives `normalize_sql` (whitespace-only) but gets stripped/ignored by sqlglot
# when we split for diff. Stable text → stable hash across runs.
MULTI_SQL_SEPARATOR = "\n-- next snapshot sql --\n"


def _dispatch_method(bsq, args: dict[str, Any]):
    """Pick which BSQ method to call based on the entry's `_method` marker.

    `_method` may be a dotted path like "utility.aggregate_ts_by_eiaid" — each
    segment is followed via getattr. Defaults to `bsq.query` when absent.
    Returns (method, args_without_method).
    """
    args = {k: v for k, v in args.items() if k != "get_query_only"}
    method_name = args.pop("_method", None)
    if method_name is None:
        return bsq.query, args
    target = bsq
    for segment in method_name.split("."):
        target = getattr(target, segment)
    return target, args


def _flatten_sql_result(result: Any) -> str:
    """Normalize get_query_only return into a single string for hashing.

    Some methods (report.get_success_report, utility.aggregate_ts_by_eiaid,
    agg.get_building_average_kws_at) return list[str] or tuple[str|list[str], ...]
    instead of a bare string. Concatenating with a stable separator lets the
    snapshot harness hash the whole call's SQL as one unit while staying agnostic
    to the per-method return shape.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        parts = [_flatten_sql_result(item) for item in result]
        return MULTI_SQL_SEPARATOR.join(parts)
    raise TypeError(f"get_query_only returned unexpected type {type(result).__name__}: {result!r}")


def run_query_sql(bsq, args: dict[str, Any]) -> str:
    """Generate SQL without execution for a single args-variant."""
    method, args = _dispatch_method(bsq, args)
    return _flatten_sql_result(method(**args, get_query_only=True))


def run_query_data(bsq, args: dict[str, Any]):
    """Execute the query. Return type varies by method: usually a DataFrame, but
    `get_distinct_vals` returns a pd.Series, `report.get_buildings_by_change`
    returns a list[int], `report.get_success_report` returns a tuple of frames.
    Normalize Series and list into single-column DataFrames so the harness's
    parquet round-trip works uniformly."""
    method, args = _dispatch_method(bsq, args)
    result = method(**args, get_query_only=False)
    if isinstance(result, pd.Series):
        return result.to_frame()
    if isinstance(result, list):
        return pd.DataFrame({"value": result})
    return result


def _worst_status(statuses: list[str]) -> str:
    """Return the worst-precedence status among the inputs."""
    for candidate in STATUS_PRECEDENCE:
        if candidate in statuses:
            return candidate
    raise ValueError("no statuses given")


def evaluate_entries(
    bsq,
    entries: list[SnapshotEntry],
    *,
    check_data: bool,
    update_snapshot: bool,
    overwrite_snapshot: bool,
) -> list[EntryOutcome]:
    from buildstock_query.aggregate_query import UnsupportedQueryShape

    cache = SqlCache(SNAPSHOTS_ROOT / f"{entries[0].schema}_cache") if entries else None

    outcomes: list[EntryOutcome] = []
    total = len(entries)
    for idx, entry in enumerate(entries, start=1):
        n_variants = len(entry.args)
        stored_sql = entry.stored_sql_path.read_text() if entry.stored_sql_path and entry.stored_sql_path.exists() else None
        _log(f"[{idx}/{total}] {entry.name} :: {n_variants} variant(s)")

        variants: list[VariantResult] = []
        skip_reason: str | None = None
        sql_error: str | None = None
        for v_idx, variant_args in enumerate(entry.args):
            _log(
                f"[{idx}/{total}] {entry.name} :: variant {v_idx + 1}/{n_variants} :: "
                f"calling {_format_call(variant_args, get_query_only=True)}"
            )
            t0 = time.time()
            try:
                actual_sql = run_query_sql(bsq, variant_args)
            except UnsupportedQueryShape as exc:
                skip_reason = str(exc)
                _log(
                    f"[{idx}/{total}] {entry.name} :: variant {v_idx + 1}/{n_variants} :: "
                    f"SKIPPED: {exc}"
                )
                break
            except Exception as exc:
                sql_error = f"{type(exc).__name__}: {exc}"
                _log(
                    f"[{idx}/{total}] {entry.name} :: variant {v_idx + 1}/{n_variants} :: "
                    f"SQL generation FAILED: {sql_error}"
                )
                break
            _log(
                f"[{idx}/{total}] {entry.name} :: variant {v_idx + 1}/{n_variants} :: "
                f"SQL generated in {time.time()-t0:.2f}s"
            )

            actual_hash = hash_sql(actual_sql)
            if not entry.sql_hash:
                v_status = "mismatch"  # new entry — must populate the hash
            elif actual_hash == entry.sql_hash:
                v_status = "match"
            elif stored_sql is not None:
                v_status = compare_sql(stored_sql, actual_sql)
            else:
                v_status = "mismatch"  # hash differs and we have no stored SQL to sqlglot-compare
            _log(
                f"[{idx}/{total}] {entry.name} :: variant {v_idx + 1}/{n_variants} :: status={v_status}"
            )
            variants.append(
                VariantResult(
                    index=v_idx, args=variant_args, actual_sql=actual_sql,
                    actual_hash=actual_hash, status=v_status,
                )
            )

        if skip_reason is not None:
            outcomes.append(EntryOutcome(
                entry=entry, status="skipped", variants=variants,
                expected_sql=stored_sql, data_status="skipped", data_error=skip_reason,
            ))
            continue

        if sql_error is not None:
            outcomes.append(EntryOutcome(
                entry=entry, status="error", variants=variants,
                expected_sql=stored_sql, data_status="skipped", data_error=sql_error,
            ))
            continue

        overall_status = _worst_status([v.status for v in variants])
        outcome = EntryOutcome(
            entry=entry, status=overall_status, variants=variants, expected_sql=stored_sql,
        )

        # Data check fires when --check-data is set (regardless of SQL status), or when
        # --update-snapshot needs to know whether real SQL drift is safe to write through.
        # sqlglot_only skips the data check — sqlglot already proved equivalence.
        sql_drift_real = outcome.status == "mismatch"
        needs_data_check = (
            (check_data and outcome.status != "match")
            or ((update_snapshot or overwrite_snapshot) and sql_drift_real)
        )
        if needs_data_check:
            _log(
                f"[{idx}/{total}] {entry.name} :: about to submit query (variant 1) for data check\n"
                f"  call: {_format_call(entry.args[0], get_query_only=False)}\n"
                f"  SQL:\n{_indent(outcome.primary_sql, '    ')}"
            )
            t0 = time.time()
            _run_and_compare_data(bsq, outcome, cache=cache)
            _log(
                f"[{idx}/{total}] {entry.name} :: data check finished in {time.time()-t0:.2f}s "
                f"(data_status={outcome.data_status})"
            )

        _maybe_update(
            outcome, cache=cache,
            update_snapshot=update_snapshot, overwrite_snapshot=overwrite_snapshot,
        )
        if outcome.updated:
            _log(f"[{idx}/{total}] {entry.name} :: {outcome.update_note}")

        outcomes.append(outcome)

    SESSION_TOTALS["entries"] += len(outcomes)
    SESSION_TOTALS["variants"] += sum(len(o.variants) for o in outcomes)
    for o in outcomes:
        if o.status == "skipped":
            SESSION_TOTALS["skipped"] += 1
        elif o.status == "error":
            SESSION_TOTALS["errored"] += 1
        elif o.updated:
            SESSION_TOTALS["updated"] += 1
        elif o.status == "match":
            SESSION_TOTALS["passed"] += 1
        else:
            SESSION_TOTALS["failed"] += 1
    return outcomes


def resolve_update_flags(config) -> tuple[bool, bool]:
    """--overwrite-snapshot implies --update-snapshot."""
    update_snapshot = bool(config.getoption("--update-snapshot"))
    overwrite_snapshot = bool(config.getoption("--overwrite-snapshot"))
    if overwrite_snapshot:
        update_snapshot = True
    return update_snapshot, overwrite_snapshot


def _log(msg: str) -> None:
    print(msg, flush=True)


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _format_call(args: dict[str, Any], *, get_query_only: bool) -> str:
    """Render the bsq.query(...) call the entry is about to make, for live logging."""
    parts = [f"{k}={v!r}" for k, v in args.items()]
    parts.append(f"get_query_only={get_query_only}")
    return "bsq.query(" + ", ".join(parts) + ")"


def _run_and_compare_data(bsq, outcome: EntryOutcome, *, cache: SqlCache) -> None:
    """Run data check using variant[0]'s args — all variants are declared equivalent.

    Sets outcome.data_status to one of:
      - match: exact match within tolerance
      - equivalent: column set differs but shared columns match
      - mismatch: shared columns disagree beyond tolerance
      - missing: no parquet on disk yet (new entry, or stored hash points nowhere)
      - error: Athena query itself failed
    """
    # Bypass cache for the freshly-generated SQL — we want the real Athena answer
    # to compare against the stored parquet. We do this by deleting the new-hash
    # parquet (if any) just before the call, then letting execute() repopulate.
    new_hash_parquet = cache.root / f"{outcome.primary_hash}.parquet"
    if new_hash_parquet.exists():
        # The user might have a stale parquet under the new hash. Trust it: skip
        # the call, treat it as fresh data.
        actual_df = pd.read_parquet(new_hash_parquet)
    else:
        try:
            actual_df = run_query_data(bsq, outcome.entry.args[0])
        except Exception as exc:
            outcome.data_status = "error"
            outcome.data_error = f"query execution failed: {exc}"
            outcome._actual_df = None
            return

    outcome._actual_df = actual_df
    old_parquet = outcome.entry.stored_parquet_path
    if old_parquet is None or not old_parquet.exists():
        outcome.data_status = "missing"
        return
    expected_df = pd.read_parquet(old_parquet)
    matched, err = compare_data(expected_df, actual_df)
    if matched:
        outcome.data_status = "match"
        return
    equivalent, eq_reason = is_data_equivalent_but_different(expected_df, actual_df)
    if equivalent:
        outcome.data_status = "equivalent"
        outcome.data_error = eq_reason
        return
    outcome.data_status = "mismatch"
    outcome.data_error = err


def _maybe_update(
    outcome: EntryOutcome,
    *,
    cache: SqlCache,
    update_snapshot: bool,
    overwrite_snapshot: bool,
) -> None:
    """Update the cache (and patch the JSON's sql_hash) per the decision table.

    Decisions (rows are SQL status, cols are data status):

                          update-snapshot      overwrite-snapshot
    SQL match             no-op                no-op
    SQL sqlglot_only      rename parquet,      rename parquet,
                          patch JSON           patch JSON
    SQL mismatch + data:
        match             rename parquet,      rename parquet,
                          patch JSON           patch JSON
        equivalent        write new pair,      write new pair,
                          delete old, patch    delete old, patch
        missing           write new pair,      write new pair,
                          patch JSON           patch JSON
        mismatch          leave alone          write new pair,
                                               delete old, patch
        error             leave alone          leave alone
    """
    if not (update_snapshot or overwrite_snapshot):
        return
    if outcome.status == "match":
        return

    new_hash = outcome.primary_hash
    new_sql = outcome.primary_sql
    old_hash = outcome.entry.sql_hash

    if outcome.status == "sqlglot_only":
        # SQL refactored cosmetically; data unchanged. Rename the parquet to the
        # new hash and rewrite the .sql sidecar.
        if old_hash:
            old_parquet = cache.root / f"{old_hash}.parquet"
            old_sql_path = cache.root / f"{old_hash}.sql"
            new_parquet = cache.root / f"{new_hash}.parquet"
            if old_parquet.exists() and old_parquet != new_parquet:
                old_parquet.rename(new_parquet)
            (cache.root / f"{new_hash}.sql").write_text(normalize_sql(new_sql))
            if old_sql_path.exists() and old_sql_path != cache.root / f"{new_hash}.sql":
                old_sql_path.unlink()
        else:
            cache.put(new_sql, getattr(outcome, "_actual_df", None) or pd.DataFrame())
        _patch_hash(outcome.entry.source_path, outcome.entry.name, outcome.entry.schema, new_hash)
        outcome.updated = True
        outcome.update_note = f"renamed parquet {old_hash[:12] if old_hash else '<empty>'} → {new_hash[:12]}; patched JSON"
        return

    # outcome.status == "mismatch" — real SQL drift.
    actual_df = getattr(outcome, "_actual_df", None)
    ds = outcome.data_status

    if ds == "match":
        # SQL changed but data is identical — same as sqlglot_only path.
        if old_hash:
            old_parquet = cache.root / f"{old_hash}.parquet"
            old_sql_path = cache.root / f"{old_hash}.sql"
            new_parquet = cache.root / f"{new_hash}.parquet"
            if old_parquet.exists() and old_parquet != new_parquet:
                old_parquet.rename(new_parquet)
            (cache.root / f"{new_hash}.sql").write_text(normalize_sql(new_sql))
            if old_sql_path.exists() and old_sql_path != cache.root / f"{new_hash}.sql":
                old_sql_path.unlink()
        _patch_hash(outcome.entry.source_path, outcome.entry.name, outcome.entry.schema, new_hash)
        outcome.updated = True
        outcome.update_note = f"data matched; renamed parquet {old_hash[:12] if old_hash else '<empty>'} → {new_hash[:12]}; patched JSON"
        return

    if ds in {"equivalent", "missing"} and actual_df is not None:
        cache.put(new_sql, actual_df)
        if old_hash and old_hash != new_hash:
            (cache.root / f"{old_hash}.parquet").unlink(missing_ok=True)
            (cache.root / f"{old_hash}.sql").unlink(missing_ok=True)
        _patch_hash(outcome.entry.source_path, outcome.entry.name, outcome.entry.schema, new_hash)
        outcome.updated = True
        outcome.update_note = f"wrote new pair {new_hash[:12]}; deleted old {old_hash[:12] if old_hash else '<empty>'}; patched JSON"
        return

    if ds == "mismatch" and overwrite_snapshot and actual_df is not None:
        cache.put(new_sql, actual_df)
        if old_hash and old_hash != new_hash:
            (cache.root / f"{old_hash}.parquet").unlink(missing_ok=True)
            (cache.root / f"{old_hash}.sql").unlink(missing_ok=True)
        _patch_hash(outcome.entry.source_path, outcome.entry.name, outcome.entry.schema, new_hash)
        outcome.updated = True
        outcome.update_note = f"OVERWROTE: new pair {new_hash[:12]}; deleted old {old_hash[:12] if old_hash else '<empty>'}; patched JSON"
        return


def _patch_hash(json_path: Path, entry_name: str, schema: str, new_hash: str) -> None:
    """Replace the per-schema sql_hash for `entry_name` by exact text substitution.

    Locates `"name": "<entry_name>"`, then `"sql_hash": { ... "<schema>": "<old>" ... }`
    within that entry's window, and rewrites just the `<old>` value. Preserves all
    other formatting in the file."""
    text = json_path.read_text()
    # Anchor on entry name, then on sql_hash, then on the schema key.
    pattern = re.compile(
        r'("name"\s*:\s*"' + re.escape(entry_name)
        + r'".*?"sql_hash"\s*:\s*\{[^{}]*?"' + re.escape(schema) + r'"\s*:\s*")[^"]*(")',
        re.DOTALL,
    )
    new_text, n = pattern.subn(rf'\g<1>{new_hash}\g<2>', text, count=1)
    if n == 0:
        raise RuntimeError(
            f"Could not patch sql_hash[{schema!r}] for entry {entry_name!r} in {json_path} — "
            f"entry not found or sql_hash dict is missing the {schema!r} key."
        )
    json_path.write_text(new_text)


def format_failures(outcomes: list[EntryOutcome], *, check_data: bool) -> str:
    lines: list[str] = []
    failing = [o for o in outcomes if _is_failure(o, check_data=check_data)]
    for outcome in failing:
        lines.append(_format_one(outcome))
    if failing:
        lines.insert(0, f"{len(failing)}/{len(outcomes)} entries failed.")
    return "\n".join(lines)


def _is_failure(outcome: EntryOutcome, *, check_data: bool) -> bool:
    if outcome.status == "skipped":
        return False
    if outcome.updated:
        # Entry was auto-updated; treat as pass for reporting purposes.
        return False
    if check_data and outcome.data_status == "match":
        # Data check ran and the returned DataFrame matches — the query is
        # functionally correct, so SQL-level drift is not a test failure.
        return False
    if outcome.status != "match":
        return True
    if check_data and outcome.data_status in {"mismatch", "missing", "error"}:
        return True
    return False


def _format_one(outcome: EntryOutcome) -> str:
    entry = outcome.entry
    header = f"\n--- {entry.name} [{outcome.status}] ---"
    parts = [header, f"stored sql_hash: {entry.sql_hash or '<empty>'}"]
    if outcome.expected_sql is None:
        parts.append("(no stored SQL on disk)")
    else:
        parts.append("EXPECTED SQL:")
        parts.append(outcome.expected_sql)

    # One block per variant that is not `match`.
    for variant in outcome.variants:
        if variant.status == "match":
            continue
        parts.append(
            f"\n  variant {variant.index + 1}/{len(outcome.variants)} "
            f"[{variant.status}] hash={variant.actual_hash[:12]}"
        )
        parts.append(f"  args: {variant.args}")
        parts.append("  ACTUAL SQL:")
        parts.append(_indent(variant.actual_sql, "    "))

    if outcome.data_status:
        parts.append(f"\ndata check: {outcome.data_status}")
        if outcome.data_error:
            parts.append(outcome.data_error)
    return "\n".join(parts)


def run_snapshot_file(json_path: Path, bsq, config, *, schema: str) -> None:
    import pytest

    if not json_path.exists():
        pytest.skip(f"snapshot file not found: {json_path}")
    entries = load_entries(json_path, schema=schema)
    if not entries:
        pytest.skip(f"no entries in {json_path}")

    check_data = config.getoption("--check-data")
    update_snapshot, overwrite_snapshot = resolve_update_flags(config)

    total_variants = sum(len(e.args) for e in entries)
    _log(
        f"\n=== {json_path.relative_to(SNAPSHOTS_ROOT.parent)}: {len(entries)} entries "
        f"({total_variants} variants, check_data={check_data}, "
        f"update_snapshot={update_snapshot}, overwrite_snapshot={overwrite_snapshot}) ==="
    )

    outcomes = evaluate_entries(
        bsq,
        entries,
        check_data=check_data,
        update_snapshot=update_snapshot,
        overwrite_snapshot=overwrite_snapshot,
    )

    failure_report = format_failures(outcomes, check_data=check_data)
    if failure_report:
        pytest.fail(failure_report, pytrace=False)
