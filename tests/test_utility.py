"""Shared helpers for the query snapshot tests."""
from __future__ import annotations

import json
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import sqlglot


SNAPSHOTS_ROOT = Path(__file__).parent / "query_snapshots"

DATA_RTOL = 1e-4
DATA_ATOL = 1e-6


# Status values for individual SQL comparisons.
# Precedence (worst → best): mismatch > missing_sql > whitespace_only > sqlglot_only > match
STATUS_PRECEDENCE = ["mismatch", "missing_sql", "whitespace_only", "sqlglot_only", "match"]


@dataclass
class SnapshotEntry:
    name: str
    description: str
    sql_file: str
    data_file: str
    args: list[dict[str, Any]]  # list of equivalent arg-variants (≥1)
    source_path: Path

    @property
    def schema_dir(self) -> Path:
        return self.source_path.parent

    @property
    def sql_path(self) -> Path:
        return self.schema_dir / "sql" / self.sql_file

    @property
    def data_path(self) -> Path:
        return self.schema_dir / "data" / self.data_file


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


def load_entries(json_path: Path) -> list[SnapshotEntry]:
    raw = json.loads(json_path.read_text())
    entries = []
    for item in raw:
        args_field = item["args"]
        # Accept either a single dict (legacy) or a list of dicts (new).
        if isinstance(args_field, dict):
            variants = [_rehydrate_args(args_field)]
        elif isinstance(args_field, list):
            if not args_field:
                raise ValueError(f"{json_path.name}: entry '{item['name']}' has empty args list")
            variants = [_rehydrate_args(v) for v in args_field]
        else:
            raise ValueError(
                f"{json_path.name}: entry '{item['name']}' args must be dict or list of dicts, "
                f"got {type(args_field).__name__}"
            )
        entries.append(
            SnapshotEntry(
                name=item["name"],
                description=item.get("description", ""),
                sql_file=item["sql_file"],
                data_file=item["data_file"],
                args=variants,
                source_path=json_path,
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


def compare_data(expected: pd.DataFrame, actual: pd.DataFrame) -> tuple[bool, str | None]:
    """Sort both frames by all columns then compare with pd.testing.assert_frame_equal."""
    if set(expected.columns) != set(actual.columns):
        return False, f"column mismatch: expected={sorted(expected.columns)} actual={sorted(actual.columns)}"
    actual = actual[list(expected.columns)]
    try:
        exp_sorted = expected.sort_values(list(expected.columns), kind="stable").reset_index(drop=True)
        act_sorted = actual.sort_values(list(actual.columns), kind="stable").reset_index(drop=True)
    except TypeError as exc:
        return False, f"sort failed: {exc}"
    try:
        pd.testing.assert_frame_equal(
            exp_sorted, act_sorted, rtol=DATA_RTOL, atol=DATA_ATOL, check_dtype=False
        )
    except AssertionError as exc:
        return False, str(exc)
    return True, None


def run_query_sql(bsq, args: dict[str, Any]) -> str:
    """Generate SQL without execution for a single args-variant."""
    return bsq.query(**args, get_query_only=True)


def run_query_data(bsq, args: dict[str, Any]) -> pd.DataFrame:
    args = {k: v for k, v in args.items() if k != "get_query_only"}
    return bsq.query(**args, get_query_only=False)


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
    update_sql: bool,
    update_data: bool,
) -> list[EntryOutcome]:
    outcomes: list[EntryOutcome] = []
    total = len(entries)
    for idx, entry in enumerate(entries, start=1):
        n_variants = len(entry.args)
        stored_sql = entry.sql_path.read_text() if entry.sql_path.exists() else None
        _log(f"[{idx}/{total}] {entry.name} :: {n_variants} variant(s)")

        variants: list[VariantResult] = []
        for v_idx, variant_args in enumerate(entry.args):
            _log(
                f"[{idx}/{total}] {entry.name} :: variant {v_idx + 1}/{n_variants} :: "
                f"calling {_format_call(variant_args, get_query_only=True)}"
            )
            t0 = time.time()
            try:
                actual_sql = run_query_sql(bsq, variant_args)
            except Exception as exc:
                _log(
                    f"[{idx}/{total}] {entry.name} :: variant {v_idx + 1}/{n_variants} :: "
                    f"SQL generation FAILED: {exc}"
                )
                raise
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

        overall_status = _worst_status([v.status for v in variants])
        outcome = EntryOutcome(
            entry=entry, status=overall_status, variants=variants, expected_sql=stored_sql
        )

        needs_data_check = check_data and outcome.status != "match"
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

        _maybe_update(outcome, update_sql=update_sql, update_data=update_data)
        if outcome.updated_paths:
            _log(
                f"[{idx}/{total}] {entry.name} :: wrote {[str(p.name) for p in outcome.updated_paths]}"
            )

        outcomes.append(outcome)
    return outcomes


def _log(msg: str) -> None:
    print(msg, flush=True)


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


# --- invariant-test helpers --------------------------------------------------

def build_snapshot_index(schema_dir: Path) -> dict[str, SnapshotEntry]:
    """Index every snapshot entry under `schema_dir` by normalized stored SQL.

    Invariant tests use this to look up an entry whose SQL matches a freshly
    generated one. Entries whose `.sql` file is missing are skipped (they'll be
    bootstrapped by --update-sql later).
    """
    index: dict[str, SnapshotEntry] = {}
    for json_path in sorted(schema_dir.glob("*.json")):
        for entry in load_entries(json_path):
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


def find_data_for_sql(schema_dir: Path, sql: str) -> pd.DataFrame:
    """Look up a snapshot entry whose stored SQL matches `sql` (normalized), return
    the reference DataFrame loaded from that entry's parquet.

    The caller is responsible for generating `sql`; this lets invariants use any
    SQL source (a single `bsq.query()` call, a hand-composed query, etc.).

    Raises AssertionError if no entry matches or the matching entry's parquet is
    missing, with guidance on how to fix it.
    """
    key = normalize_whitespace(sql)
    index = _cached_index(schema_dir)
    entry = index.get(key)
    if entry is None:
        raise AssertionError(
            f"Invariant needs a snapshot entry under {schema_dir.name} that stores this SQL:\n"
            f"  normalized SQL:\n    {key}\n"
            f"No such entry exists. Add it to the relevant JSON file and re-run --update-sql-and-data."
        )
    if not entry.data_path.exists():
        raise AssertionError(
            f"Invariant matched entry '{entry.name}' but its parquet is missing at {entry.data_path}. "
            f"Run `pytest --update-sql-and-data` to bootstrap."
        )
    return pd.read_parquet(entry.data_path)


_INDEX_CACHE: dict[Path, dict[str, SnapshotEntry]] = {}


def _cached_index(schema_dir: Path) -> dict[str, SnapshotEntry]:
    schema_dir = schema_dir.resolve()
    if schema_dir not in _INDEX_CACHE:
        _INDEX_CACHE[schema_dir] = build_snapshot_index(schema_dir)
    return _INDEX_CACHE[schema_dir]


def _format_call(args: dict[str, Any], *, get_query_only: bool) -> str:
    """Render the bsq.query(...) call the entry is about to make, for live logging."""
    parts = [f"{k}={v!r}" for k, v in args.items()]
    parts.append(f"get_query_only={get_query_only}")
    return "bsq.query(" + ", ".join(parts) + ")"


def _run_and_compare_data(bsq, outcome: EntryOutcome) -> None:
    """Run data check using variant[0]'s args — all variants are declared equivalent."""
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
    outcome.data_status = "match" if matched else "mismatch"
    if not matched:
        outcome.data_error = err


def _maybe_update(outcome: EntryOutcome, *, update_sql: bool, update_data: bool) -> None:
    sql_needs_write = outcome.status != "match"
    if update_sql and sql_needs_write:
        outcome.entry.sql_path.parent.mkdir(parents=True, exist_ok=True)
        # Write variant[0]'s SQL. If variant[1..n] differ from variant[0], they'll surface
        # as mismatches on the next test run.
        outcome.entry.sql_path.write_text(outcome.primary_sql)
        outcome.updated_paths.append(outcome.entry.sql_path)

    # --update-data implies --update-sql. Write data only for entries that fell through
    # both SQL and data checks (or had missing files treated as mismatches).
    data_failed = outcome.data_status in {"mismatch", "missing"}
    if update_data and sql_needs_write and data_failed:
        actual_df = getattr(outcome, "_actual_df", None)
        if actual_df is not None:
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


def run_snapshot_file(json_path: Path, bsq, config) -> None:
    import pytest

    if not json_path.exists():
        pytest.skip(f"snapshot file not found: {json_path}")
    entries = load_entries(json_path)
    if not entries:
        pytest.skip(f"no entries in {json_path}")

    check_data = config.getoption("--check-data")
    update_sql = config.getoption("--update-sql")
    update_data = config.getoption("--update-sql-and-data")
    if update_data:
        update_sql = True  # --update-sql-and-data implies --update-sql
        check_data = True  # data writes gated on data check

    total_variants = sum(len(e.args) for e in entries)
    _log(
        f"\n=== {json_path.relative_to(SNAPSHOTS_ROOT.parent)}: {len(entries)} entries "
        f"({total_variants} variants, check_data={check_data}, update_sql={update_sql}, "
        f"update_data={update_data}) ==="
    )

    outcomes = evaluate_entries(
        bsq,
        entries,
        check_data=check_data,
        update_sql=update_sql,
        update_data=update_data,
    )

    failure_report = format_failures(outcomes, check_data=check_data)
    if failure_report:
        pytest.fail(failure_report, pytrace=False)
