"""Shared helpers for the query snapshot tests."""
from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sqlglot


SNAPSHOTS_ROOT = Path(__file__).parent / "query_snapshots"

DATA_RTOL = 1e-4
DATA_ATOL = 1e-6


# Status values for individual SQL comparisons.
# Precedence (worst → best): mismatch > missing_sql > whitespace_only > sqlglot_only > match.
# `skipped` sits outside the precedence: it means the schema doesn't support this query
# shape yet (raises UnsupportedQueryShape during SQL generation), so no comparison or
# write is meaningful and the entry is treated as a non-failure.
STATUS_PRECEDENCE = ["mismatch", "missing_sql", "whitespace_only", "sqlglot_only", "match"]


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
        if value.startswith("$"):
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
    sql_file: str
    data_file: str
    args: list[dict[str, Any]]  # list of equivalent arg-variants (≥1)
    source_path: Path
    schema: str  # e.g. "resstock_oedi" — picks the artifact subdirectory

    @property
    def sql_path(self) -> Path:
        return SNAPSHOTS_ROOT / f"{self.schema}_sql" / self.sql_file

    @property
    def data_path(self) -> Path:
        return SNAPSHOTS_ROOT / f"{self.schema}_data" / self.data_file


@dataclass
class VariantResult:
    index: int
    args: dict[str, Any]
    actual_sql: str
    status: str  # match, whitespace_only, sqlglot_only, mismatch, missing_sql


@dataclass
class EntryOutcome:
    entry: SnapshotEntry
    status: str  # worst variant status (precedence above)
    variants: list[VariantResult]
    expected_sql: str | None = None  # stored SQL (same for all variants)
    data_status: str | None = None  # match, mismatch, missing, error
    data_error: str | None = None
    updated_paths: list[Path] = field(default_factory=list)

    @property
    def primary_sql(self) -> str:
        """variant[0]'s SQL — the one used for updates and data check."""
        return self.variants[0].actual_sql


def load_entries(json_path: Path, *, schema: str) -> list[SnapshotEntry]:
    """Load every entry from `json_path`, resolving placeholders for `schema`.

    Each variant's `annual_only` flag picks the leg-aware resolution
    (comstock annual columns get the `..kwh` suffix, ts columns don't), so the
    same JSON entry can declare a single `$ELECTRICITY_TOTAL` placeholder and
    still produce the right column name for both legs.
    """
    if schema not in _SUPPORTED_SCHEMAS:
        raise ValueError(f"unknown schema '{schema}'; expected one of {list(_SUPPORTED_SCHEMAS)}")
    raw = json.loads(json_path.read_text())

    entries = []
    for item in raw:
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
            annual_only = bool(raw_variant.get("annual_only", True))
            resolved = _resolve_placeholders(raw_variant, schema, annual=annual_only)
            variants.append(_rehydrate_args(resolved))

        entries.append(
            SnapshotEntry(
                name=item["name"],
                description=item.get("description", ""),
                sql_file=item["sql_file"],
                data_file=item["data_file"],
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


def normalize_whitespace(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


def compare_sql(expected: str, actual: str) -> str:
    """Return one of: match, whitespace_only, sqlglot_only, mismatch."""
    if expected == actual:
        return "match"
    if normalize_whitespace(expected) == normalize_whitespace(actual):
        return "whitespace_only"
    try:
        diff = sqlglot.diff(sqlglot.parse_one(expected), sqlglot.parse_one(actual), delta_only=True)
    except Exception:
        return "mismatch"
    if not diff:
        return "sqlglot_only"
    return "mismatch"


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
        except AssertionError as exc:
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
    except AssertionError as exc:
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


def _dispatch_method(bsq, args: dict[str, Any]):
    """Pick which BSQ method to call based on the entry's `_method` marker.

    Defaults to `bsq.query` when `_method` is absent. Returns (method, args_without_method).
    """
    args = {k: v for k, v in args.items() if k != "get_query_only"}
    method_name = args.pop("_method", None)
    method = getattr(bsq, method_name) if method_name else bsq.query
    return method, args


def run_query_sql(bsq, args: dict[str, Any]) -> str:
    """Generate SQL without execution for a single args-variant."""
    method, args = _dispatch_method(bsq, args)
    return method(**args, get_query_only=True)


def run_query_data(bsq, args: dict[str, Any]) -> pd.DataFrame:
    method, args = _dispatch_method(bsq, args)
    return method(**args, get_query_only=False)


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

    outcomes: list[EntryOutcome] = []
    total = len(entries)
    for idx, entry in enumerate(entries, start=1):
        n_variants = len(entry.args)
        stored_sql = entry.sql_path.read_text() if entry.sql_path.exists() else None
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

            if stored_sql is None:
                v_status = "missing_sql"
            else:
                v_status = compare_sql(stored_sql, actual_sql)
            _log(
                f"[{idx}/{total}] {entry.name} :: variant {v_idx + 1}/{n_variants} :: status={v_status}"
            )
            if v_status == "sqlglot_only":
                warnings.warn(
                    f"{entry.name} variant {v_idx + 1}: SQL differs in whitespace/structure but "
                    "sqlglot considers it equivalent.",
                    stacklevel=2,
                )
            variants.append(
                VariantResult(index=v_idx, args=variant_args, actual_sql=actual_sql, status=v_status)
            )

        if skip_reason is not None:
            outcome = EntryOutcome(
                entry=entry,
                status="skipped",
                variants=variants,
                expected_sql=stored_sql,
                data_status="skipped",
                data_error=skip_reason,
            )
            outcomes.append(outcome)
            continue

        if sql_error is not None:
            outcome = EntryOutcome(
                entry=entry,
                status="error",
                variants=variants,
                expected_sql=stored_sql,
                data_status="skipped",
                data_error=sql_error,
            )
            outcomes.append(outcome)
            continue

        overall_status = _worst_status([v.status for v in variants])
        outcome = EntryOutcome(
            entry=entry, status=overall_status, variants=variants, expected_sql=stored_sql
        )

        # Data check fires when --check-data was passed (and SQL didn't match), or when
        # --update-snapshot/--overwrite-snapshot needs the data result to decide whether
        # the SQL change is safe to write. Cosmetic SQL drift skips the data check —
        # sqlglot already proved the queries are equivalent.
        sql_drift_needs_validation = outcome.status in {"mismatch", "missing_sql"}
        needs_data_check = (
            (check_data and outcome.status != "match")
            or ((update_snapshot or overwrite_snapshot) and sql_drift_needs_validation)
        )
        if needs_data_check:
            _log(
                f"[{idx}/{total}] {entry.name} :: about to submit query (variant 1) for data check\n"
                f"  call: {_format_call(entry.args[0], get_query_only=False)}\n"
                f"  SQL:\n{_indent(outcome.primary_sql, '    ')}"
            )
            t0 = time.time()
            _run_and_compare_data(bsq, outcome)
            _log(
                f"[{idx}/{total}] {entry.name} :: data check finished in {time.time()-t0:.2f}s "
                f"(data_status={outcome.data_status})"
            )

        _maybe_update(outcome, update_snapshot=update_snapshot, overwrite_snapshot=overwrite_snapshot)
        if outcome.updated_paths:
            _log(
                f"[{idx}/{total}] {entry.name} :: wrote {[str(p.name) for p in outcome.updated_paths]}"
            )

        outcomes.append(outcome)
    return outcomes


def resolve_update_flags(config) -> tuple[bool, bool]:
    """Pull --update-snapshot / --overwrite-snapshot from pytest config.

    --overwrite-snapshot implies --update-snapshot — it does everything --update-snapshot
    does plus also writes through real data mismatches.
    """
    update_snapshot = bool(config.getoption("--update-snapshot"))
    overwrite_snapshot = bool(config.getoption("--overwrite-snapshot"))
    if overwrite_snapshot:
        update_snapshot = True
    return update_snapshot, overwrite_snapshot


def _log(msg: str) -> None:
    print(msg, flush=True)


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


# --- invariant-test helpers --------------------------------------------------

def build_snapshot_index(schema: str) -> dict[str, SnapshotEntry]:
    """Index every snapshot entry for `schema` by normalized stored SQL.

    Invariant tests use this to look up an entry whose SQL matches a freshly
    generated one. Entries whose `.sql` file is missing are skipped (they'll be
    bootstrapped by --update-snapshot later).
    """
    index: dict[str, SnapshotEntry] = {}
    for json_path in sorted(SNAPSHOTS_ROOT.glob("*.json")):
        for entry in load_entries(json_path, schema=schema):
            if not entry.sql_path.exists():
                continue
            key = normalize_whitespace(entry.sql_path.read_text())
            if key in index:
                existing = index[key]
                _log(
                    f"[invariant-index] duplicate SQL between {existing.name} and "
                    f"{entry.name} — keeping first"
                )
                continue
            index[key] = entry
    return index


def find_data_for_sql(schema: str, sql: str) -> pd.DataFrame:
    """Look up a snapshot entry whose stored SQL matches `sql` (normalized), return
    the reference DataFrame loaded from that entry's parquet.

    The caller is responsible for generating `sql`; this lets invariants use any
    SQL source (a single `bsq.query()` call, a hand-composed query, etc.).

    Raises AssertionError if no entry matches or the matching entry's parquet is
    missing, with guidance on how to fix it.
    """
    key = normalize_whitespace(sql)
    index = _cached_index(schema)
    entry = index.get(key)
    if entry is None:
        raise AssertionError(
            f"Invariant needs a snapshot entry for {schema} that stores this SQL:\n"
            f"  normalized SQL:\n    {key}\n"
            f"No such entry exists. Add it to a shared JSON file and re-run --update-snapshot."
        )
    if not entry.data_path.exists():
        raise AssertionError(
            f"Invariant matched entry '{entry.name}' but its parquet is missing at {entry.data_path}. "
            f"Run `pytest --update-snapshot` to bootstrap."
        )
    return pd.read_parquet(entry.data_path)


_INDEX_CACHE: dict[str, dict[str, SnapshotEntry]] = {}


def _cached_index(schema: str) -> dict[str, SnapshotEntry]:
    if schema not in _INDEX_CACHE:
        _INDEX_CACHE[schema] = build_snapshot_index(schema)
    return _INDEX_CACHE[schema]


def _format_call(args: dict[str, Any], *, get_query_only: bool) -> str:
    """Render the bsq.query(...) call the entry is about to make, for live logging."""
    parts = [f"{k}={v!r}" for k, v in args.items()]
    parts.append(f"get_query_only={get_query_only}")
    return "bsq.query(" + ", ".join(parts) + ")"


def _run_and_compare_data(bsq, outcome: EntryOutcome) -> None:
    """Run data check using variant[0]'s args — all variants are declared equivalent.

    Sets outcome.data_status to one of:
      - match: exact match within tolerance
      - equivalent: column set differs but shared columns match (a refactor adding/
        removing a metadata column, etc.)
      - mismatch: shared columns disagree beyond tolerance
      - missing: no parquet on disk yet
      - error: Athena query itself failed
    """
    try:
        actual_df = run_query_data(bsq, outcome.entry.args[0])
    except Exception as exc:
        outcome.data_status = "error"
        outcome.data_error = f"query execution failed: {exc}"
        outcome._actual_df = None
        return

    outcome._actual_df = actual_df
    if not outcome.entry.data_path.exists():
        outcome.data_status = "missing"
        return
    expected_df = pd.read_parquet(outcome.entry.data_path)
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


def _maybe_update(outcome: EntryOutcome, *, update_snapshot: bool, overwrite_snapshot: bool) -> None:
    """Decide what (if anything) to write to disk for this entry.

    `update_snapshot` is the routine refresh: write anything the test can prove is
    a safe refactor. `overwrite_snapshot` adds one extra power: also write through
    a real data mismatch, on the user's assertion that the new value is correct.

    Rules
    -----
                              update-snapshot    overwrite-snapshot
    SQL match                 no-op              no-op
    SQL whitespace_only       write SQL          write SQL
    SQL sqlglot_only          write SQL          write SQL
    SQL real-drift + data:
        match                 write SQL          write SQL
        equivalent            write SQL+data     write SQL+data
        missing               write SQL+data     write SQL+data
        mismatch              leave both         write SQL+data
        error                 leave both         leave both
    """
    sql_cosmetic_only = outcome.status in {"whitespace_only", "sqlglot_only"}
    sql_drift_real = outcome.status in {"mismatch", "missing_sql"}

    if not (update_snapshot or overwrite_snapshot):
        return
    if outcome.status == "match":
        return

    if sql_cosmetic_only:
        _write_sql(outcome)
        return

    if sql_drift_real:
        actual_df = getattr(outcome, "_actual_df", None)
        ds = outcome.data_status

        if ds == "match":
            _write_sql(outcome)
        elif ds in {"equivalent", "missing"} and actual_df is not None:
            _write_sql(outcome)
            _write_data(outcome, actual_df)
        elif ds == "mismatch" and overwrite_snapshot and actual_df is not None:
            _write_sql(outcome)
            _write_data(outcome, actual_df)


def _write_sql(outcome: EntryOutcome) -> None:
    # variant[0]'s SQL is canonical; differing variants surface as mismatches on the next run.
    if outcome.entry.sql_path in outcome.updated_paths:
        return
    outcome.entry.sql_path.parent.mkdir(parents=True, exist_ok=True)
    outcome.entry.sql_path.write_text(outcome.primary_sql)
    outcome.updated_paths.append(outcome.entry.sql_path)


def _write_data(outcome: EntryOutcome, actual_df: pd.DataFrame) -> None:
    if outcome.entry.data_path in outcome.updated_paths:
        return
    outcome.entry.data_path.parent.mkdir(parents=True, exist_ok=True)
    actual_df.to_parquet(outcome.entry.data_path, index=False)
    outcome.updated_paths.append(outcome.entry.data_path)


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
    if outcome.updated_paths:
        # Entry was auto-updated; treat as pass for reporting purposes.
        return False
    if check_data and outcome.data_status == "match":
        # Data check ran and the returned DataFrame matches the snapshot — the
        # query is functionally correct, so SQL-level drift (whitespace_only,
        # sqlglot_only, or even mismatch) is not a test failure.
        return False
    if outcome.status != "match":
        return True
    if check_data and outcome.data_status in {"mismatch", "missing", "error"}:
        return True
    return False


def _format_one(outcome: EntryOutcome) -> str:
    entry = outcome.entry
    header = f"\n--- {entry.name} [{outcome.status}] ---"
    parts = [header, f"stored SQL file: {entry.sql_path}"]
    if outcome.expected_sql is None:
        parts.append(f"(missing expected SQL at {entry.sql_path})")
    else:
        parts.append("EXPECTED SQL:")
        parts.append(outcome.expected_sql)

    # One block per variant that is not `match`.
    for variant in outcome.variants:
        if variant.status == "match":
            continue
        parts.append(
            f"\n  variant {variant.index + 1}/{len(outcome.variants)} [{variant.status}]"
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
