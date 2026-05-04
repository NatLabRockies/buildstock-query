"""Two-sided dedup benchmark.

Compares 4 SQL variants of the same logical workload to determine
whether **deduplicating both TS and MD at query time** beats just
routing to a smaller MD table.

Workload: ComStock TS-monthly query, restrict to state IN ('CO','NM'),
no group_by, baseline (upgrade=0), one enduse (electricity total).
Output is one scalar: sum(weight × elec) across all months / states.

Variants:
  V1: primary MD (_md_by_state_and_county_parquet),
      bs_per_bldg GROUP BY (bldg, state),
      join on (bldg_id, state).
      [today's framework SQL]
  V2: alt MD (_md_agg_national_parquet),
      no bs_per_bldg (table is already (bldg, state)-grain),
      join on (bldg_id, state).
      [Piece A+B in the plan: routing-only]
  V3: primary MD,
      TS-side dedup (group by bldg_id, timestamp, arbitrary on values),
      MD-side dedup (group by bldg_id, sum(weight)),
      join on bldg_id.
      [Piece D draft: dedup-only]
  V4: alt MD,
      same dedup shape as V3,
      join on bldg_id.
      [Routing + dedup combined]

Cost guardrail: every probe SQL must touch only the recognized
ComStock tables and must include a state restrict that prunes to two
states. Total estimated spend < $0.05.

Reuses framework primitives from tests/query_snapshots/investigate_partitions.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Reuse infrastructure from the partition investigation script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import investigate_partitions as ip  # noqa: E402

from buildstock_query import BuildStockQuery  # noqa: E402
from buildstock_query.sql_cache import SqlCache  # noqa: E402

CACHE_ROOT = Path(__file__).resolve().parent / "_partition_probe_cache" / "dedup_benchmark"
RESULT_JSON = Path(__file__).resolve().parent / "dedup_benchmark.json"

PRIMARY_MD = "comstock_amy2018_r2_2025_md_by_state_and_county_parquet"
ALT_MD = "comstock_amy2018_r2_2025_md_agg_national_parquet"
TS = "comstock_amy2018_r2_2025_ts_by_state"

STATES = ["CO", "NM"]
STATES_SQL = ", ".join(f"'{s}'" for s in STATES)


# Cost guardrail — these probes are not safe-by-default like the
# partition investigation. Each must (a) touch the TS table with state
# AND upgrade filters, (b) restrict to the small state list above.
_TS_FORBIDDEN_PATTERNS = [
    # No "FROM <ts_table>" without a "state IN (...)" within ~500 chars.
]


_STATE_RESTRICT_RE = re.compile(r"state\s+IN\s*\(\s*'CO'\s*,\s*'NM'\s*\)", re.IGNORECASE)
_UPGRADE_FILTER_RE = re.compile(r"upgrade\s*(=|IN)\s*\(?0\)?", re.IGNORECASE)


def assert_safe(sql: str, name: str) -> None:
    """Strict guardrail: this script DOES touch the TS table, so we
    explicitly check that every probe restricts both TS partitions
    (state) AND TS upgrade column."""
    if not _STATE_RESTRICT_RE.search(sql):
        raise RuntimeError(f"Probe {name!r}: missing state IN ('CO','NM') restrict. SQL head: {sql[:200]}")
    if not _UPGRADE_FILTER_RE.search(sql):
        raise RuntimeError(f"Probe {name!r}: missing upgrade=0 filter. SQL head: {sql[:200]}")
    # Both tables must appear at least once for these joining queries.
    if PRIMARY_MD not in sql and ALT_MD not in sql:
        raise RuntimeError(f"Probe {name!r}: references no recognized MD table.")
    if TS not in sql:
        raise RuntimeError(f"Probe {name!r}: references no TS table.")


def build_v1_primary_with_bs_per_bldg() -> str:
    """V1: today's framework SQL shape against the primary MD table.
    Three-level: ts_flat (per-row scalar projection) -> ts_aggr (per
    bldg-state-month aggregate) -> outer JOIN to bs_per_bldg
    (per bldg-state)."""
    return f"""\
SELECT sum(ts_aggr."bs__out.electricity" * bs_per_bldg.bldg_weight) AS total_kwh,
       sum(ts_aggr._inner_rows) AS total_rows,
       count(DISTINCT (ts_aggr.bldg_id, ts_aggr.state)) AS distinct_bldg_state,
       count(DISTINCT ts_aggr.bldg_id) AS distinct_bldgs
FROM (
  SELECT ts_flat.state AS state, ts_flat.timestamp AS timestamp,
         ts_flat.bldg_id AS bldg_id,
         sum(ts_flat."ts__out.electricity") AS "bs__out.electricity",
         count(*) AS _inner_rows
  FROM (
    SELECT {TS}.state AS state, {TS}.bldg_id AS bldg_id,
           date_trunc('month', date_add('second', -900, {TS}.timestamp)) AS timestamp,
           {TS}."out.electricity.total.energy_consumption" AS "ts__out.electricity"
    FROM {TS}
    WHERE {TS}.upgrade IN (0) AND {TS}.state IN ({STATES_SQL})
  ) AS ts_flat
  GROUP BY ts_flat.state, ts_flat.timestamp, ts_flat.bldg_id
) AS ts_aggr
JOIN (
  SELECT bs.bldg_id AS bldg_id, bs.state AS state,
         sum(bs.weight) AS bldg_weight,
         count(*) AS tract_count
  FROM {PRIMARY_MD} AS bs
  WHERE bs.upgrade = 0 AND bs.state IN ({STATES_SQL})
  GROUP BY bs.bldg_id, bs.state
) AS bs_per_bldg
  ON bs_per_bldg.bldg_id = ts_aggr.bldg_id
 AND bs_per_bldg.state = ts_aggr.state
"""


def build_v2_alt_no_bs_per_bldg() -> str:
    """V2: alt MD table. No bs_per_bldg layer — the alt table already
    has one row per (bldg_id, state) per upgrade with weight pre-summed
    across tract slices. Two-level: ts_flat -> ts_aggr -> JOIN to alt
    MD directly."""
    return f"""\
SELECT sum(ts_aggr."bs__out.electricity" * bs.weight) AS total_kwh,
       sum(ts_aggr._inner_rows) AS total_rows,
       count(DISTINCT (ts_aggr.bldg_id, ts_aggr.state)) AS distinct_bldg_state,
       count(DISTINCT ts_aggr.bldg_id) AS distinct_bldgs
FROM (
  SELECT ts_flat.state AS state, ts_flat.timestamp AS timestamp,
         ts_flat.bldg_id AS bldg_id,
         sum(ts_flat."ts__out.electricity") AS "bs__out.electricity",
         count(*) AS _inner_rows
  FROM (
    SELECT {TS}.state AS state, {TS}.bldg_id AS bldg_id,
           date_trunc('month', date_add('second', -900, {TS}.timestamp)) AS timestamp,
           {TS}."out.electricity.total.energy_consumption" AS "ts__out.electricity"
    FROM {TS}
    WHERE {TS}.upgrade IN (0) AND {TS}.state IN ({STATES_SQL})
  ) AS ts_flat
  GROUP BY ts_flat.state, ts_flat.timestamp, ts_flat.bldg_id
) AS ts_aggr
JOIN {ALT_MD} AS bs
  ON bs.bldg_id = ts_aggr.bldg_id
 AND bs."in.state" = ts_aggr.state
WHERE bs.upgrade = 0 AND bs."in.state" IN ({STATES_SQL})
"""


def build_v3_primary_dedup() -> str:
    """V3: primary MD, but with **two-sided query-time dedup**.
      - TS dedup: collapse (bldg_id, state) duplicates on each timestamp
        via arbitrary() (values are identical across state duplicates).
      - MD dedup: collapse tract+state slices via sum(weight) per bldg.
    Join becomes single-key bldg_id, no fan-out."""
    return f"""\
SELECT sum(ts_dedup."elec_per_bldg_month" * md_dedup.bldg_weight) AS total_kwh,
       sum(ts_dedup._month_rows) AS total_rows,
       count(DISTINCT ts_dedup.bldg_id) AS distinct_bldgs,
       count(DISTINCT ts_dedup.bldg_id) AS distinct_bldg_state  -- same here, since dedup'd
FROM (
  SELECT bldg_id, timestamp,
         sum(elec) AS "elec_per_bldg_month",
         count(*) AS _month_rows
  FROM (
    SELECT bldg_id,
           date_trunc('month', date_add('second', -900, ts.timestamp)) AS timestamp,
           arbitrary(ts."out.electricity.total.energy_consumption") AS elec
    FROM {TS} AS ts
    WHERE ts.upgrade IN (0) AND ts.state IN ({STATES_SQL})
    GROUP BY bldg_id, ts.timestamp
  ) AS ts_dedup_15min
  GROUP BY bldg_id, timestamp
) AS ts_dedup
JOIN (
  SELECT bs.bldg_id AS bldg_id, sum(bs.weight) AS bldg_weight
  FROM {PRIMARY_MD} AS bs
  WHERE bs.upgrade = 0 AND bs.state IN ({STATES_SQL})
  GROUP BY bs.bldg_id
) AS md_dedup
  ON md_dedup.bldg_id = ts_dedup.bldg_id
"""


def build_v4_alt_dedup() -> str:
    """V4: alt MD + two-sided dedup. The alt table is already
    tract-collapsed but still has (bldg_id, state) duplicates, so
    bldg-level dedup still applies. TS dedup is the same as V3."""
    return f"""\
SELECT sum(ts_dedup."elec_per_bldg_month" * md_dedup.bldg_weight) AS total_kwh,
       sum(ts_dedup._month_rows) AS total_rows,
       count(DISTINCT ts_dedup.bldg_id) AS distinct_bldgs,
       count(DISTINCT ts_dedup.bldg_id) AS distinct_bldg_state
FROM (
  SELECT bldg_id, timestamp,
         sum(elec) AS "elec_per_bldg_month",
         count(*) AS _month_rows
  FROM (
    SELECT bldg_id,
           date_trunc('month', date_add('second', -900, ts.timestamp)) AS timestamp,
           arbitrary(ts."out.electricity.total.energy_consumption") AS elec
    FROM {TS} AS ts
    WHERE ts.upgrade IN (0) AND ts.state IN ({STATES_SQL})
    GROUP BY bldg_id, ts.timestamp
  ) AS ts_dedup_15min
  GROUP BY bldg_id, timestamp
) AS ts_dedup
JOIN (
  SELECT bs.bldg_id AS bldg_id, sum(bs.weight) AS bldg_weight
  FROM {ALT_MD} AS bs
  WHERE bs.upgrade = 0 AND bs."in.state" IN ({STATES_SQL})
  GROUP BY bs.bldg_id
) AS md_dedup
  ON md_dedup.bldg_id = ts_dedup.bldg_id
"""


VARIANTS = {
    "V1_primary_bs_per_bldg":       build_v1_primary_with_bs_per_bldg(),
    "V2_alt_no_bs_per_bldg":        build_v2_alt_no_bs_per_bldg(),
    "V3_primary_dedup":             build_v3_primary_dedup(),
    "V4_alt_dedup":                 build_v4_alt_dedup(),
}


def run_one(bsq: BuildStockQuery, cache: SqlCache, name: str, sql: str) -> dict:
    """Inline the probe machinery from investigate_partitions, but with
    OUR safety check (which permits the TS table provided state+upgrade
    filters are present). The IP version's `assert_safe` is metadata-only
    and rejects any TS reference."""
    import time
    assert_safe(sql, name)

    cached = cache.get_metadata(sql)
    if cached is not None:
        return cached["probe"]

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
        if time.time() - start > 900:
            raise TimeoutError(f"Probe {name!r} did not complete in 900s")
        time.sleep(1.5)

    stats = info["QueryExecution"].get("Statistics", {})
    result = {
        "name": name, "sql": sql, "state": state,
        "reason": status.get("StateChangeReason", ""),
        "rows": [], "columns": [],
        "bytes_scanned": stats.get("DataScannedInBytes", 0),
        "engine_ms": stats.get("EngineExecutionTimeInMillis", 0),
        "planning_ms": stats.get("QueryPlanningTimeInMillis", 0),
        "queue_ms": stats.get("QueryQueueTimeInMillis", 0),
        "total_ms": stats.get("TotalExecutionTimeInMillis", 0),
        "execution_id": exe_id,
    }
    if state == "SUCCEEDED":
        rs = bsq._aws_athena.get_query_results(QueryExecutionId=exe_id, MaxResults=100)
        rows_raw = rs["ResultSet"]["Rows"]
        if rows_raw:
            result["columns"] = [c.get("VarCharValue", "") for c in rows_raw[0]["Data"]]
            result["rows"] = [
                [c.get("VarCharValue") for c in r["Data"]] for r in rows_raw[1:]
            ]
    cache.put_metadata(sql, {"probe": result})
    print(
        f"    state={result['state']}  total={ip.fmt_ms(result['total_ms'])}  "
        f"plan={ip.fmt_ms(result['planning_ms'])}  scan={ip.fmt_bytes(result['bytes_scanned'])}",
        flush=True,
    )
    return result


def fmt_us(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, (int, float)):
        return f"{v:,.0f}"
    try:
        f = float(v)
        return f"{f:,.0f}"
    except (ValueError, TypeError):
        return str(v)


def fmt_pct_diff(actual, baseline) -> str:
    if baseline in (0, None) or actual is None:
        return "n/a"
    pct = (actual - baseline) / baseline * 100
    return f"{pct:+.2f}%"


def render(results: dict) -> None:
    print("\n" + "=" * 100)
    print(f"{'Variant':<32}{'Wall':>10}{'Plan':>10}{'Engine':>10}{'Bytes':>14}{'total_kwh':>22}")
    print("-" * 100)
    baseline_kwh = None
    if "V2_alt_no_bs_per_bldg" in results and results["V2_alt_no_bs_per_bldg"]["state"] == "SUCCEEDED":
        baseline_kwh = float(results["V2_alt_no_bs_per_bldg"]["rows"][0][0])
    for name, p in results.items():
        if p["state"] != "SUCCEEDED":
            print(f"{name:<32} FAILED — {p.get('reason','')[:60]}")
            continue
        wall = ip.fmt_ms(p["total_ms"])
        plan = ip.fmt_ms(p["planning_ms"])
        eng = ip.fmt_ms(p["engine_ms"])
        nb = ip.fmt_bytes(p["bytes_scanned"])
        kwh_str = "n/a"
        if p["rows"]:
            try:
                kwh = float(p["rows"][0][0])
                kwh_str = f"{kwh:,.0f}"
                if baseline_kwh is not None and baseline_kwh != 0:
                    diff = (kwh - baseline_kwh) / baseline_kwh
                    kwh_str += f" ({diff*100:+.4f}%)"
            except (ValueError, TypeError):
                pass
        print(f"{name:<32}{wall:>10}{plan:>10}{eng:>10}{nb:>14}{kwh_str:>22}")
    print("-" * 100)

    # Decision-rule output (per the plan).
    print("\nDecision rule (per plan):")
    if all(name in results and results[name]["state"] == "SUCCEEDED" for name in ("V2_alt_no_bs_per_bldg", "V3_primary_dedup", "V4_alt_dedup")):
        v2 = results["V2_alt_no_bs_per_bldg"]["total_ms"]
        v3 = results["V3_primary_dedup"]["total_ms"]
        v4 = results["V4_alt_dedup"]["total_ms"]
        best_dedup = min(v3, v4)
        ratio = best_dedup / v2 if v2 else float("inf")
        print(f"  V2 wall: {ip.fmt_ms(v2)}, best dedup wall: {ip.fmt_ms(best_dedup)} (V{'3' if v3<v4 else '4'})")
        print(f"  ratio = best_dedup / V2 = {ratio:.2f}")
        if ratio < 0.7:
            print("  → SHIP DEDUP. Best dedup variant is >30% faster than routing-only.")
        elif ratio > 1.2:
            print("  → DROP DEDUP. Worse than routing-only; document negative result.")
        else:
            print(f"  → ROUTING ENOUGH. Dedup within ±20% of routing-only ({ratio:+.0%}); not worth complexity.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete cached probe results before running.")
    args = parser.parse_args()

    if args.clear_cache and CACHE_ROOT.exists():
        import shutil
        shutil.rmtree(CACHE_ROOT)
        print(f"Cleared {CACHE_ROOT}.")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    print("Constructing BuildStockQuery (comstock_oedi)...")
    bsq = BuildStockQuery(
        "rescore", "buildstock_sdr", "comstock_amy2018_r2_2025",
        buildstock_type="comstock",
        db_schema="comstock_oedi_state_and_county",
        skip_reports=True,
        cache_folder=str(CACHE_ROOT / "_unused_bsq"),
    )

    cache = SqlCache(CACHE_ROOT)
    results: dict[str, dict] = {}
    for name, sql in VARIANTS.items():
        print(f"\n--- {name} ---")
        results[name] = run_one(bsq, cache, name, sql)

    render(results)

    RESULT_JSON.write_text(json.dumps({"workload": "ts_monthly_two_states_baseline_elec",
                                        "variants": results}, indent=2, default=str))
    print(f"\nFull results written to {RESULT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
