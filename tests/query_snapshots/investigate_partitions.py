"""Read-only Athena probe: why is ComStock metadata so much slower than ResStock?

The cached snapshot stats show ComStock metadata-only queries take ~42 s on
average vs ResStock's ~3 s, even when scanning <50 MB. The smoking gun is a
trivial `SELECT count(*) ... WHERE upgrade=0 AND applicability=true` that
takes 75 s on ComStock and 0.5 MB scanned. This script collects the
partition/file-layout evidence we need to confirm or refute the hypothesis
that ComStock's `_md_*_by_state_and_county_parquet` tables suffer from
small-files / over-partitioning pathology.

Run it once. It executes ~5 small probes per table × 3 tables = 15 Athena
queries, all metadata-only, all <50 MB scanned. Combined cost is well under
$0.05. Probe results are cached in `_partition_probe_cache/` so repeated
runs are free.

Output: a per-table summary plus a verdict block at the end recommending the
next step (parquet repartition vs. Glue overhead investigation).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from buildstock_query import BuildStockQuery  # noqa: E402
from buildstock_query.sql_cache import SqlCache, hash_sql  # noqa: E402

PROBE_CACHE_DIR = Path(__file__).resolve().parent / "_partition_probe_cache"
RESULT_JSON = Path(__file__).resolve().parent / "partition_investigation.json"

# Cost guardrail: every probe SQL must match the metadata regex AND must NOT
# touch the TS table. CLAUDE.md documents a 4.87 TB / $24 incident from a
# single unbounded TS scan; this guardrail makes it impossible for a typo
# in this script to trigger a similar accident.
METADATA_TABLE_RE = re.compile(
    r"\bFROM\s+\w*_md_\w*_parquet\b|\bFROM\s+\w+_metadata\b|\bSHOW\s+PARTITIONS\b",
    re.IGNORECASE,
)
TS_TABLE_FORBIDDEN = re.compile(r"_ts_by_state\b|_by_state_vu\b", re.IGNORECASE)


@dataclass
class ProbeResult:
    name: str
    sql: str
    state: str  # SUCCEEDED / FAILED / etc.
    reason: str = ""
    rows: list[list[Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    bytes_scanned: int = 0
    engine_ms: int = 0
    planning_ms: int = 0
    queue_ms: int = 0
    total_ms: int = 0
    execution_id: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "sql": self.sql,
            "state": self.state,
            "reason": self.reason,
            "rows": self.rows,
            "columns": self.columns,
            "bytes_scanned": self.bytes_scanned,
            "engine_ms": self.engine_ms,
            "planning_ms": self.planning_ms,
            "queue_ms": self.queue_ms,
            "total_ms": self.total_ms,
            "execution_id": self.execution_id,
        }


def assert_safe(sql: str, name: str) -> None:
    if TS_TABLE_FORBIDDEN.search(sql):
        raise RuntimeError(f"Probe {name!r} references TS table — refusing to run: {sql}")
    if not METADATA_TABLE_RE.search(sql):
        raise RuntimeError(
            f"Probe {name!r} doesn't match the metadata-only safety regex — refusing to run: {sql}"
        )


def fmt_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:,.2f} {unit}" if unit != "B" else f"{int(n):,} B"
        n /= 1000
    return f"{n:.2f} TB"


def fmt_ms(ms: int) -> str:
    if ms < 1000:
        return f"{ms} ms"
    if ms < 60_000:
        return f"{ms / 1000:.2f} s"
    return f"{ms / 60_000:.2f} min"


def run_probe(
    bsq: BuildStockQuery,
    cache: SqlCache,
    name: str,
    sql: str,
    poll_seconds: float = 1.0,
    timeout_seconds: int = 600,
) -> ProbeResult:
    """Submit `sql` via the BSQ Athena client; poll to completion; fetch full
    statistics + result rows; cache the metadata + rows under hash(sql)."""
    assert_safe(sql, name)

    cached = cache.get_metadata(sql)
    if cached is not None:
        # Reconstruct ProbeResult from cached metadata. The stored dict
        # already includes name/sql; let it round-trip cleanly.
        return ProbeResult(**cached["probe"])

    print(f"  [{name}] submitting...", flush=True)
    response = bsq._aws_athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": bsq.db_name},
        WorkGroup=bsq.workgroup,
    )
    exe_id = response["QueryExecutionId"]
    start = time.time()
    while True:
        info = bsq._aws_athena.get_query_execution(QueryExecutionId=exe_id)
        status = info["QueryExecution"]["Status"]
        state = status["State"].upper()
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Probe {name!r} did not complete in {timeout_seconds}s")
        time.sleep(poll_seconds)

    stats = info["QueryExecution"].get("Statistics", {})
    result = ProbeResult(
        name=name,
        sql=sql,
        state=state,
        reason=status.get("StateChangeReason", ""),
        bytes_scanned=stats.get("DataScannedInBytes", 0),
        engine_ms=stats.get("EngineExecutionTimeInMillis", 0),
        planning_ms=stats.get("QueryPlanningTimeInMillis", 0),
        queue_ms=stats.get("QueryQueueTimeInMillis", 0),
        total_ms=stats.get("TotalExecutionTimeInMillis", 0),
        execution_id=exe_id,
    )

    if state == "SUCCEEDED":
        # Fetch result rows. Probes return at most a few hundred rows, so
        # one get_query_results call (max 1000 rows) suffices.
        rs = bsq._aws_athena.get_query_results(QueryExecutionId=exe_id, MaxResults=1000)
        rows_raw = rs["ResultSet"]["Rows"]
        if rows_raw:
            header = [c.get("VarCharValue", "") for c in rows_raw[0]["Data"]]
            data_rows = [
                [c.get("VarCharValue") for c in r["Data"]] for r in rows_raw[1:]
            ]
            result.columns = header
            result.rows = data_rows

    cache.put_metadata(sql, {"probe": result.to_dict()})
    print(
        f"    state={result.state}  total={fmt_ms(result.total_ms)}  "
        f"plan={fmt_ms(result.planning_ms)}  scan={fmt_bytes(result.bytes_scanned)}",
        flush=True,
    )
    return result


def first_int(rows: list[list[Any]], col: int = 0) -> int | None:
    """Pull an integer out of a probe's result rows, defending against
    Athena's string-only return values."""
    if not rows:
        return None
    v = rows[0][col]
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def probe_table(
    bsq: BuildStockQuery,
    table_full: str,
    schema_label: str,
    state_col: str | None = "state",
    county_col: str | None = "county",
    is_partitioned: bool = True,
) -> dict:
    """Run probes against one fully-qualified Athena table.

    `state_col` and `county_col` name the actual columns (or None if absent).
    Schemas differ: ComStock by-state-and-county uses bare `state`/`county`,
    while ResStock and ComStock national use the dotted `in.state` /
    `in.county` characteristic-prefix convention.
    """
    print(f"\n=== {schema_label} :: {table_full} ===", flush=True)
    cache = SqlCache(PROBE_CACHE_DIR / schema_label)

    probes: dict[str, ProbeResult] = {}

    def col(name: str) -> str:
        # Quote dotted column names; leave bare identifiers unquoted.
        return f'"{name}"' if "." in name else name

    # Pre-flight: does Athena's $path/$file_size pseudocolumn work on this
    # table? If not, the table is Iceberg/Delta and we'd need a different
    # introspection path. For our ComStock/ResStock tables (Hive-on-S3 with
    # parquet files), $path is the standard idiom.
    preflight_sql = f'SELECT "$path" FROM {table_full} LIMIT 1'
    pf = run_probe(bsq, cache, "preflight_path_pseudocol", preflight_sql)
    probes["preflight_path_pseudocol"] = pf
    path_supported = pf.state == "SUCCEEDED"
    if not path_supported:
        print(f"  ! pre-flight failed; $path unsupported. Reason: {pf.reason!r}")

    if path_supported:
        # (1) Total file count + size.
        sql = (
            f'SELECT count(DISTINCT "$path") AS files, '
            f'sum("$file_size") AS total_bytes '
            f"FROM {table_full}"
        )
        probes["total_files"] = run_probe(bsq, cache, "total_files", sql)

        # (2) Files per state, top 5 — only meaningful if the table has state.
        if state_col:
            sc = col(state_col)
            sql = (
                f'SELECT {sc} AS state, count(DISTINCT "$path") AS files '
                f"FROM {table_full} GROUP BY {sc} ORDER BY 2 DESC LIMIT 5"
            )
            probes["files_per_state"] = run_probe(bsq, cache, "files_per_state", sql)

        # (3) Files per (county, upgrade) within CO — needs both columns.
        if state_col and county_col:
            sc, cc = col(state_col), col(county_col)
            sql = (
                f"SELECT {cc} AS county, upgrade, "
                f'count(DISTINCT "$path") AS files, '
                f"count(*) AS rows, "
                f'sum("$file_size") AS total_bytes '
                f"FROM {table_full} "
                f"WHERE {sc} = 'CO' "
                f"GROUP BY {cc}, upgrade ORDER BY files DESC LIMIT 10"
            )
            probes["files_per_county_upgrade_CO"] = run_probe(
                bsq, cache, "files_per_county_upgrade_CO", sql
            )

    # (4) SHOW PARTITIONS — count lines. Skipped on unpartitioned tables.
    if is_partitioned:
        sql = f"SHOW PARTITIONS {table_full}"
        probes["show_partitions"] = run_probe(bsq, cache, "show_partitions", sql)

    # (5) Trivial count(*) benchmark — apples-to-apples by always filtering
    # to CO when state is available. National tables (no state column) get
    # the whole-table count, which we'll annotate explicitly in the summary.
    if state_col:
        sc = col(state_col)
        sql = (
            f"SELECT count(*) AS count "
            f"FROM {table_full} WHERE {sc} = 'CO' AND upgrade = 0"
        )
    else:
        sql = (
            f"SELECT count(*) AS count "
            f"FROM {table_full} WHERE upgrade = 0"
        )
    probes["count_state_CO_upgrade_0"] = run_probe(
        bsq, cache, "count_state_CO_upgrade_0", sql
    )

    return {name: p.to_dict() for name, p in probes.items()}


def collect_failures(probes: dict) -> list[tuple[str, str]]:
    """Return [(probe_name, reason)] for any probe with state != SUCCEEDED.
    Helps surface schema mismatches that would otherwise hide as `n/a`."""
    out = []
    for name, p in probes.items():
        if p.get("state") != "SUCCEEDED":
            out.append((name, p.get("reason", "")[:160]))
    return out


def summarize(table_label: str, table_full: str, probes: dict) -> dict:
    """Compute the human-readable summary fields. Returns a dict the caller
    can both print and dump to JSON."""
    summary = {"table_label": table_label, "table_full": table_full}

    tf = probes.get("total_files", {})
    summary["total_files"] = first_int(tf.get("rows", []), 0)
    summary["total_bytes"] = first_int(tf.get("rows", []), 1)
    if summary["total_files"] and summary["total_bytes"]:
        summary["avg_file_bytes"] = summary["total_bytes"] / summary["total_files"]
    else:
        summary["avg_file_bytes"] = None

    fps = probes.get("files_per_state", {})
    if fps.get("rows"):
        try:
            top_state, top_count = fps["rows"][0][0], int(fps["rows"][0][1])
        except (TypeError, ValueError):
            top_state, top_count = None, None
        summary["max_files_per_state"] = top_count
        summary["max_files_per_state_label"] = top_state
    else:
        summary["max_files_per_state"] = None
        summary["max_files_per_state_label"] = None

    fpc = probes.get("files_per_county_upgrade_CO", {})
    rows_per_file_co: list[float] = []
    if fpc.get("rows"):
        max_files = 0
        max_label = None
        for r in fpc["rows"]:
            try:
                county, upgrade = r[0], r[1]
                files = int(r[2]) if r[2] is not None else 0
                rows = int(r[3]) if r[3] is not None else 0
            except (TypeError, ValueError):
                continue
            if files > max_files:
                max_files, max_label = files, f"{county}/upgrade={upgrade}"
            if files > 0:
                rows_per_file_co.append(rows / files)
        summary["max_files_per_state_county_upgrade"] = max_files
        summary["max_files_per_state_county_upgrade_label"] = max_label
    else:
        summary["max_files_per_state_county_upgrade"] = None
        summary["max_files_per_state_county_upgrade_label"] = None
    summary["median_rows_per_file_CO"] = (
        sorted(rows_per_file_co)[len(rows_per_file_co) // 2]
        if rows_per_file_co
        else None
    )

    sp = probes.get("show_partitions", {})
    summary["athena_partition_count"] = (
        len(sp.get("rows", [])) if sp.get("state") == "SUCCEEDED" else None
    )

    cnt = probes.get("count_state_CO_upgrade_0", {})
    summary["count_co_upgrade0"] = first_int(cnt.get("rows", []), 0)
    summary["count_co_upgrade0_total_ms"] = cnt.get("total_ms")
    summary["count_co_upgrade0_engine_ms"] = cnt.get("engine_ms")
    summary["count_co_upgrade0_planning_ms"] = cnt.get("planning_ms")
    summary["count_co_upgrade0_bytes"] = cnt.get("bytes_scanned")
    summary["count_co_upgrade0_state"] = cnt.get("state")  # SUCCEEDED / FAILED

    summary["failures"] = collect_failures(probes)

    return summary


def print_summary(s: dict) -> None:
    print(f"\n--- summary: {s['table_label']} ---")
    print(f"  Table:                                 {s['table_full']}")

    def _line(label: str, value: Any, suffix: str = "") -> None:
        if value is None:
            v = "n/a"
        elif isinstance(value, float):
            v = f"{value:,.0f}"
        else:
            v = f"{value:,}" if isinstance(value, int) else str(value)
        print(f"  {label:<42} {v}{suffix}")

    _line("Total files:", s["total_files"])
    _line(
        "Total bytes:",
        fmt_bytes(s["total_bytes"]) if s["total_bytes"] else None,
    )
    _line(
        "Avg file size:",
        fmt_bytes(s["avg_file_bytes"]) if s["avg_file_bytes"] else None,
    )
    _line(
        "Max files in any single state:",
        s["max_files_per_state"],
        f"  (state: {s['max_files_per_state_label']})"
        if s["max_files_per_state_label"]
        else "",
    )
    _line(
        "Max files per (state,county,upgrade):",
        s["max_files_per_state_county_upgrade"],
        f"  (CO/{s['max_files_per_state_county_upgrade_label']})"
        if s["max_files_per_state_county_upgrade_label"]
        else "",
    )
    _line("Median rows per file (within CO):", s["median_rows_per_file_CO"])
    _line("Athena partitions reported:", s["athena_partition_count"])
    print()
    _line("count(*) for state=CO, upgrade=0:", s["count_co_upgrade0"])
    if s["count_co_upgrade0_total_ms"] is not None:
        _line("  total wall-clock:", fmt_ms(s["count_co_upgrade0_total_ms"]))
        _line("  engine ms:", fmt_ms(s["count_co_upgrade0_engine_ms"]))
        _line("  planning ms:", fmt_ms(s["count_co_upgrade0_planning_ms"]))
        _line(
            "  bytes scanned:",
            fmt_bytes(s["count_co_upgrade0_bytes"]) if s["count_co_upgrade0_bytes"] else None,
        )

    if s.get("failures"):
        print()
        print("  FAILED probes (schema mismatch likely):")
        for name, reason in s["failures"]:
            print(f"    [{name}] {reason}")


def render_verdict(summaries: list[dict]) -> str:
    """Apply the decision rules from the plan."""
    by_label = {s["table_label"]: s for s in summaries}

    out = ["\n=" * 30, "VERDICT", "=" * 60]

    def get(label: str, key: str):
        s = by_label.get(label)
        return s.get(key) if s else None

    comstock_files = max(
        get("comstock_oedi", "total_files") or 0,
        get("comstock_oedi_agg", "total_files") or 0,
    )
    resstock_files = get("resstock_oedi", "total_files") or 0

    median_rows_co = min(
        v
        for v in (
            get("comstock_oedi", "median_rows_per_file_CO"),
            get("comstock_oedi_agg", "median_rows_per_file_CO"),
        )
        if v is not None
    ) if any(
        get(label, "median_rows_per_file_CO") is not None
        for label in ("comstock_oedi", "comstock_oedi_agg")
    ) else None

    cs_count_planning = max(
        get("comstock_oedi", "count_co_upgrade0_planning_ms") or 0,
        get("comstock_oedi_agg", "count_co_upgrade0_planning_ms") or 0,
    )

    if (
        comstock_files > 30_000
        and median_rows_co is not None
        and median_rows_co < 5_000
    ):
        out.append(
            f"SMALL-FILES PATHOLOGY CONFIRMED. ComStock has {comstock_files:,} "
            f"files with median {median_rows_co:,.0f} rows/file in CO."
        )
        out.append("→ Step 2a: repartition the published parquet (data-pipeline fix).")
    elif (
        comstock_files < 5_000
        and (get("comstock_oedi", "athena_partition_count") or 0) < 1_000
        and cs_count_planning > 5_000
    ):
        out.append(
            "Healthy file/partition count, but planning still >5 s. "
            "Likely Glue catalog overhead."
        )
        out.append("→ Step 2b: investigate Glue/Hive metadata cost.")
    elif resstock_files > 10_000:
        out.append(
            f"ResStock also has {resstock_files:,} files — file-count theory "
            f"alone doesn't explain the asymmetry. Re-investigate."
        )
    else:
        out.append(
            "Inconclusive. Print the per-table summaries above and reason "
            "about what's anomalous."
        )

    # Always show the asymmetry numbers for context. Iterate over every
    # summary so newly-added targets appear without further plumbing.
    labels = [s["table_label"] for s in summaries]
    out.append("")
    out.append("File counts:")
    for label in labels:
        n = get(label, "total_files")
        out.append(f"  {label:<28} {f'{n:,}' if isinstance(n, int) else 'n/a':>12}")
    out.append("")
    out.append("count(*) total wall-clock @ state=CO, upgrade=0:")
    for label in labels:
        ms = get(label, "count_co_upgrade0_total_ms")
        out.append(
            f"  {label:<28} {fmt_ms(ms) if ms is not None else 'n/a':>12}"
        )
    out.append("")
    out.append("count(*) planning ms:")
    for label in labels:
        ms = get(label, "count_co_upgrade0_planning_ms")
        out.append(
            f"  {label:<28} {fmt_ms(ms) if ms is not None else 'n/a':>12}"
        )
    out.append("")
    out.append("count(*) bytes scanned:")
    for label in labels:
        nb = get(label, "count_co_upgrade0_bytes")
        out.append(
            f"  {label:<28} {fmt_bytes(nb) if isinstance(nb, int) else 'n/a':>12}"
        )

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Workload probes: re-issue representative slow queries from the snapshot
# cache against each ComStock table variant. This answers "do the timings
# we see in real test queries also drop when routed to agg_by_state?"
#
# Three shapes (chosen from the cached SQL after sorting by wall-clock):
#   A. Trivial count(*) with NO state filter — the smoking gun.
#   B. National rollup: GROUP BY state, sum(weight).
#   C. State-filtered self-join: applied-buildings pattern (the most common
#      slow shape; ~70 s on the by_state_and_county tables).
#
# Each shape is templated on (table, state_col, county_col, key_cols) so the
# same logical query can be issued against tables with different schemas.
# ---------------------------------------------------------------------------


def workload_probes(
    bsq: BuildStockQuery,
    table_full: str,
    schema_label: str,
    state_col: str | None = "state",
    county_col: str | None = "county",
    is_agg: bool = True,
) -> dict:
    """Run the 3 representative slow-query shapes against `table_full`."""
    print(f"\n--- workload probes: {schema_label} :: {table_full} ---", flush=True)
    cache = SqlCache(PROBE_CACHE_DIR / schema_label)

    def col(name: str) -> str:
        return f'"{name}"' if "." in name else name

    probes: dict[str, ProbeResult] = {}

    # Shape A: trivial count(*) with no state filter.
    sql_a = (
        f"SELECT count(*) AS count FROM {table_full} "
        f"WHERE upgrade = 0 AND applicability = true"
    )
    probes["wA_count_no_state"] = run_probe(bsq, cache, "wA_count_no_state", sql_a)

    # Shape B: GROUP BY state national rollup. Skip if the table has no
    # state column.
    if state_col:
        sc = col(state_col)
        sql_b = (
            f"SELECT {sc} AS state, sum(1) AS sample_count, "
            f"sum(weight) AS weighted_count "
            f"FROM {table_full} "
            f"WHERE upgrade = 0 "
            f"GROUP BY {sc} ORDER BY {sc}"
        )
        probes["wB_groupby_state"] = run_probe(bsq, cache, "wB_groupby_state", sql_b)

    # Shape C: state-filtered self-join applied-buildings pattern. The key
    # tuple varies by schema:
    #   - by_state_and_county: (bldg_id, county, state) — agg variant
    #     uses literal `county`; non-agg uses `in.nhgis_tract_gisjoin`.
    #   - by_state: (bldg_id, state)
    #   - national: (bldg_id) — no county/state to join on.
    if county_col and state_col:
        bc, sc = col(county_col), col(state_col)
        key_tuple = f"(bs.bldg_id, bs.{bc}, bs.{sc})"
        select_cols = f"bs.bldg_id, bs.{bc}, bs.{sc}"
        groupby_cols = f"bs.bldg_id, bs.{bc}, bs.{sc}"
        outer_state_filter = f" AND bs.{sc} = 'CO'"
    elif state_col:
        sc = col(state_col)
        key_tuple = f"(bs.bldg_id, bs.{sc})"
        select_cols = f"bs.bldg_id, bs.{sc}"
        groupby_cols = f"bs.bldg_id, bs.{sc}"
        outer_state_filter = f" AND bs.{sc} = 'CO'"
    else:
        key_tuple = "bs.bldg_id"
        select_cols = "bs.bldg_id"
        groupby_cols = "bs.bldg_id"
        outer_state_filter = ""

    sql_c = (
        f'SELECT bs."in.comstock_building_type" AS comstock_building_type, '
        f"sum(1) AS sample_count, sum(bs.weight) AS units_count "
        f"FROM {table_full} AS bs "
        f"WHERE bs.applicability = true AND bs.upgrade = 0 "
        f"AND {key_tuple} IN ("
        f"SELECT {select_cols} FROM {table_full} AS bs "
        f"WHERE bs.upgrade IN (1) AND bs.applicability = true "
        f"GROUP BY {groupby_cols} HAVING count(distinct(bs.upgrade)) = 1)"
        f"{outer_state_filter} "
        f'GROUP BY 1 ORDER BY 1'
    )
    probes["wC_self_join_applied"] = run_probe(bsq, cache, "wC_self_join_applied", sql_c)

    return {name: p.to_dict() for name, p in probes.items()}


def print_workload_table(workload_results: list[dict]) -> None:
    """Compact table comparing the 3 shapes across all tables."""
    print("\n" + "=" * 100)
    print("WORKLOAD COMPARISON — same logical query across ComStock table variants")
    print("=" * 100)
    shapes = [
        ("wA_count_no_state",    "A. count(*) WHERE upgrade=0 AND applicability=true (NO state filter)"),
        ("wB_groupby_state",     "B. GROUP BY state, sum(weight)"),
        ("wC_self_join_applied", "C. state=CO + self-join applied-buildings, GROUP BY building_type"),
    ]
    for shape_key, label in shapes:
        print(f"\n  {label}")
        print(f"    {'Table':<32}{'wall-clock':>12}{'planning':>12}{'engine':>12}{'scanned':>14}{'state':>10}")
        for entry in workload_results:
            p = entry["probes"].get(shape_key)
            if not p:
                row = (entry["label"], "skipped (column absent)", "", "", "", "")
                print(f"    {row[0]:<32}{row[1]:>12}{row[2]:>12}{row[3]:>12}{row[4]:>14}{row[5]:>10}")
                continue
            wall = fmt_ms(p["total_ms"]) if p["state"] == "SUCCEEDED" else "FAILED"
            plan = fmt_ms(p["planning_ms"]) if p["state"] == "SUCCEEDED" else ""
            eng = fmt_ms(p["engine_ms"]) if p["state"] == "SUCCEEDED" else ""
            scn = fmt_bytes(p["bytes_scanned"]) if p["state"] == "SUCCEEDED" and p["bytes_scanned"] else ""
            print(
                f"    {entry['label']:<32}{wall:>12}{plan:>12}{eng:>12}{scn:>14}"
                f"{p['state']:>10}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete cached probe results before running (forces fresh Athena calls).",
    )
    args = parser.parse_args()

    if args.clear_cache and PROBE_CACHE_DIR.exists():
        import shutil
        shutil.rmtree(PROBE_CACHE_DIR)
        print(f"Cleared {PROBE_CACHE_DIR}.")

    PROBE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Build the three BSQ instances. Mirror tests/conftest.py:113-158.
    print("Constructing BuildStockQuery instances...", flush=True)
    bsq_comstock = BuildStockQuery(
        "rescore", "buildstock_sdr", "comstock_amy2018_r2_2025",
        buildstock_type="comstock",
        db_schema="comstock_oedi_state_and_county",
        skip_reports=True,
        cache_folder=str(PROBE_CACHE_DIR / "_unused_comstock"),
    )
    bsq_comstock_agg = BuildStockQuery(
        "rescore", "buildstock_sdr", "comstock_amy2018_r2_2025",
        buildstock_type="comstock",
        db_schema="comstock_oedi_agg_state_and_county",
        skip_reports=True,
        cache_folder=str(PROBE_CACHE_DIR / "_unused_comstock_agg"),
    )
    bsq_resstock = BuildStockQuery(
        "rescore", "buildstock_sdr", "resstock_2024_amy2018_release_2",
        buildstock_type="resstock",
        db_schema="resstock_oedi_vu",
        skip_reports=True,
        cache_folder=str(PROBE_CACHE_DIR / "_unused_resstock"),
    )

    # Each target names the actual column used for state/county on that
    # table — schemas differ: ComStock by-state-and-county exposes bare
    # `state`/`county` (partition columns), while ResStock and ComStock
    # national use the `in.state` / `in.county` dotted convention.
    targets = [
        # (label, bsq, table, state_col, county_col, is_partitioned)
        ("comstock_oedi", bsq_comstock,
         "comstock_amy2018_r2_2025_md_by_state_and_county_parquet",
         "state", "county", True),
        ("comstock_oedi_agg", bsq_comstock_agg,
         "comstock_amy2018_r2_2025_md_agg_by_state_and_county_parquet",
         "state", "county", True),
        ("comstock_oedi_agg_by_state", bsq_comstock,
         "comstock_amy2018_r2_2025_md_agg_by_state_parquet",
         "state", None, True),
        ("comstock_oedi_agg_national", bsq_comstock,
         "comstock_amy2018_r2_2025_md_agg_national_parquet",
         "in.state", None, False),
        ("resstock_oedi", bsq_resstock,
         "resstock_2024_amy2018_release_2_metadata",
         "in.state", None, False),
    ]

    all_results: dict[str, dict] = {}
    summaries: list[dict] = []
    for label, bsq, table, state_col, county_col, is_partitioned in targets:
        probes = probe_table(
            bsq, table, label,
            state_col=state_col, county_col=county_col,
            is_partitioned=is_partitioned,
        )
        all_results[label] = {"table": table, "probes": probes}
        summary = summarize(label, table, probes)
        summaries.append(summary)
        all_results[label]["summary"] = summary
        print_summary(summary)

    print(render_verdict(summaries))

    # Workload probes — re-issue the 3 representative slow shapes against
    # each table that supports them. Skip ResStock (different `in.*` schema
    # for output columns; the workload_probes shape uses `weight`,
    # `applicability`, `in.comstock_building_type` which only appear on
    # ComStock tables).
    workload_results = []
    for label, bsq, table, state_col, county_col, _ in targets:
        if "comstock" not in label:
            continue
        probes = workload_probes(
            bsq, table, label,
            state_col=state_col, county_col=county_col,
        )
        workload_results.append({
            "label": label, "table": table, "probes": probes,
        })
        all_results[label]["workload_probes"] = probes

    print_workload_table(workload_results)

    # Persist for re-reading without re-running.
    RESULT_JSON.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nFull probe results written to {RESULT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
