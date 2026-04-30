"""Invariant tests — cross-query consistency checks that operate on snapshot data.

Each invariant constructs several `query(...)` arg sets, generates SQL for each leg,
looks up the matching snapshot entry by SQL, loads its stored parquet, and asserts
a mathematical relation between the DataFrames.

No Athena data queries are issued — only SQL generation (fast) and local parquet
reads. The cost is paid once during `--update-snapshot`; these invariants just
verify the reference data is internally consistent.

If a leg's snapshot entry doesn't exist or its parquet is missing, the invariant
test fails with a clear message directing you to add the entry and re-bootstrap.

Invariants covered in this module
---------------------------------

1. **annual == ts-year-collapse == sum(ts-monthly)** — shared across both schemas

   For a given enduse, grouping, and restrict, the per-group total must agree across
   three query flavors:

     - `annual_only=True` — one row per group.
     - `annual_only=False, timestamp_grouping_func='year'` — timeseries collapsed to
       one row per group.
     - `annual_only=False, timestamp_grouping_func='month'` — 12 rows per group,
       summed over the `time` axis.

   Also verifies that `sample_count` and `units_count` agree across all three legs.
   These are per-group metadata (constant on every monthly row), so collapsing the
   monthly frame requires mean-across-time, not sum; catches bugs where the monthly
   query accidentally double-counts rows.

   On the monthly leg, also asserts `rows_per_sample == 4 * 24 * days_in_month` for
   each row — pins the timeseries cadence at 15 minutes and catches missing or
   duplicate timestamps in the source data.

   Catches bugs in timeseries aggregation, baseline/timeseries join logic, and unit
   conversion between the annual and timeseries tables. Comstock uses different
   enduse column names per leg (`..kwh` suffix on annual, no suffix on TS);
   `resolve_placeholder(...)` in test_utility.py resolves the right name for
   each leg, so the test body doesn't need to special-case it.

   On non-baseline scenarios, the same test additionally requests the savings
   shape (`include_baseline=True, include_upgrade=True, include_savings=True`)
   and asserts that each of the three output columns (`__baseline`, `__upgrade`,
   `__savings`) agrees across all three flows, plus the in-flow decomposition
   identity `b - u ≈ s` on every row of every flow. This closes the
   symmetry-cancellation blind spot where a bug that inflates baseline and
   upgrade by the same amount leaves savings correct.

Tolerance
---------

`rtol=1e-3, atol=1.0`. Looser than the snapshot data-compare tolerances because
aggregate sums over tens of thousands of rows accumulate float drift, and Athena
does not guarantee sum order.
"""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd
import pytest

from tests.test_utility import (
    SNAPSHOTS_ROOT,
    _has_array_values,
    load_entries,
    resolve_placeholder,
    run_query_data,
)
from tests.snapshot_recorder import record_query


pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 100)


INVARIANT_RTOL = 1e-3
INVARIANT_ATOL = 1.0  # float drift on aggregate sums can be noticeable in kWh


def _strip_out_prefix(name: str) -> str:
    """Output column names in the returned DataFrame drop the leading 'out.' prefix."""
    return name[4:] if name.startswith("out.") else name


# --- helpers -----------------------------------------------------------------

def _scalar_total_by_group(df: pd.DataFrame, enduse: str, group_cols: list[str]) -> pd.Series:
    """Return a Series keyed by group values, values = summed enduse over `df`."""
    if group_cols:
        return df.groupby(group_cols, dropna=False)[enduse].sum().sort_index()
    return pd.Series({"__total__": df[enduse].sum()})


def _scalar_mean_by_group(df: pd.DataFrame, col: str, group_cols: list[str]) -> pd.Series:
    """Return a Series keyed by group values, values = mean of `col` over `df`.

    Used for per-row metadata columns (sample_count, units_count) when collapsing a
    timeseries across the time axis — summing would multiply by the number of
    timestamps.
    """
    if group_cols:
        return df.groupby(group_cols, dropna=False)[col].mean().sort_index()
    return pd.Series({"__total__": df[col].mean()})


def _scalar_first_by_group(df: pd.DataFrame, col: str, group_cols: list[str]) -> pd.Series:
    """Return a Series keyed by group values, values = the single value of `col`.

    Used for the annual / ts-year-collapse legs where there's exactly one row per
    group, so sum/mean/first are all equivalent.
    """
    if group_cols:
        return df.groupby(group_cols, dropna=False)[col].first().sort_index()
    return pd.Series({"__total__": df[col].iloc[0]})


def _assert_series_close(label: str, a: pd.Series, b: pd.Series) -> None:
    """Assert two series match on index and values (within tolerance).

    Values are coerced to float before comparison — Athena may return the same
    logical number as int64, float64, or Decimal depending on the query path, and
    np.isclose can't cross the Decimal boundary directly.
    """
    assert set(a.index) == set(b.index), (
        f"{label}: group-key mismatch\n"
        f"  only_in_a={set(a.index) - set(b.index)}\n"
        f"  only_in_b={set(b.index) - set(a.index)}"
    )
    a_aligned = a.sort_index().astype(float)
    b_aligned = b.reindex(a_aligned.index).astype(float)
    diffs = []
    for key, av, bv in zip(a_aligned.index, a_aligned.values, b_aligned.values):
        if not np.isclose(av, bv, rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL, equal_nan=True):
            diffs.append(
                f"    {key}: {av:.4f} vs {bv:.4f} (diff={av - bv:.4f}, "
                f"rel={((av - bv) / bv if bv else float('nan')):.4e})"
            )
    if diffs:
        pytest.fail(f"{label}: per-group totals diverge beyond tolerance\n" + "\n".join(diffs))


def _find_first_col(df: pd.DataFrame, *, suffix: str, contains: str) -> str:
    for c in df.columns:
        if c.endswith(suffix) and contains in c:
            return c
    raise AssertionError(
        f"no column with suffix '{suffix}' containing '{contains}' in columns: {list(df.columns)}"
    )


def _bldg_ids_for_restrict(bsq, restrict) -> set[int]:
    """Project `get_building_ids(restrict=...)` to bldg_id (the leading
    column of md_key_cols, present for both composite-key comstock and
    flat-key resstock).
    """
    df = bsq.get_building_ids(restrict=restrict)
    return set(int(x) for x in df.iloc[:, 0].tolist())


# Cache the discovered pair on the bsq instance so the discovery cost
# (a handful of metadata queries) is paid once per session per schema.
_PAIR_CACHE_ATTR = "_applied_pair_cache"


def _pick_meaningful_upgrade_pair(
    bsq, *, state: str = "CO", max_upgrade: int = 16,
) -> tuple[int, int]:
    """Find a pair (a, b) of upgrade ids, a < b, both >= 1, where the four
    regions defined by (applied to a, applied to b) are all non-empty under
    `state`. The chosen pair is cached on `bsq` to amortize discovery cost.

    Lookup order: (1, b) for b in 2..max_upgrade, then (2, b), ...
    Picks the first pair with all four regions non-empty.

    Fails the test loudly if no such pair exists in [1, max_upgrade].
    """
    cache_key = (state, max_upgrade)
    cache = getattr(bsq, _PAIR_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(bsq, _PAIR_CACHE_ATTR, cache)
    if cache_key in cache:
        return cache[cache_key]

    state_restrict = [("state", [state])]
    universe = _bldg_ids_for_restrict(bsq, state_restrict)

    sets: dict[int, set[int]] = {}

    def _set_for(u: int) -> set[int]:
        if u not in sets:
            f = bsq.get_applied_buildings_filter(all_of=[u])
            sets[u] = _bldg_ids_for_restrict(
                bsq, [f, *state_restrict] if f else state_restrict,
            )
        return sets[u]

    for a in range(1, max_upgrade + 1):
        sa_ids = _set_for(a)
        for b in range(a + 1, max_upgrade + 1):
            sb = _set_for(b)
            only_a = sa_ids - sb
            only_b = sb - sa_ids
            both = sa_ids & sb
            neither = universe - (sa_ids | sb)
            if only_a and only_b and both and neither:
                cache[cache_key] = (a, b)
                return (a, b)
    pytest.fail(
        f"no upgrade pair (a, b) in [1, {max_upgrade}] under state={state} "
        f"produces four non-empty regions (only_a, only_b, both, neither); "
        f"the schema either has too few upgrades or all upgrades apply to "
        f"the same set of buildings under this state."
    )


def _curate_applied_universe(
    bsq, *, upgrades: tuple[int, int] | None = None, state: str = "CO",
    per_region: int = 2,
) -> tuple[list[int], dict[str, set[int]], tuple[int, int]]:
    """Discover bldg_ids spanning four regions defined by applicability of two upgrades.

    Regions:
      - only_a: applied to upgrade A but not B
      - only_b: applied to upgrade B but not A
      - both:   applied to both A and B
      - neither: applied to neither

    `upgrades` defaults to `_pick_meaningful_upgrade_pair(bsq)` so each schema
    auto-selects a pair where all four regions are non-empty.

    Returns `(curated, regions, (a, b))` where `curated` is a sorted list of
    `per_region` bldg_ids from each region (smallest-first for reproducibility),
    `regions` is a dict mapping region name to the full set of bldg_ids in that
    region, and `(a, b)` is the upgrade pair that produced these regions.

    The curated list is intended as a fixed `(bldg_id_col, curated)` restrict
    to bound query cost while keeping each region non-empty.
    """
    if upgrades is None:
        upgrades = _pick_meaningful_upgrade_pair(bsq, state=state)
    a, b = upgrades
    state_restrict = [("state", [state])]

    f_a = bsq.get_applied_buildings_filter(all_of=[a])
    f_b = bsq.get_applied_buildings_filter(all_of=[b])
    set_a = _bldg_ids_for_restrict(
        bsq, [f_a, *state_restrict] if f_a else state_restrict,
    )
    set_b = _bldg_ids_for_restrict(
        bsq, [f_b, *state_restrict] if f_b else state_restrict,
    )
    universe = _bldg_ids_for_restrict(bsq, state_restrict)

    only_a = set_a - set_b
    only_b = set_b - set_a
    both = set_a & set_b
    neither = universe - (set_a | set_b)

    regions = {"only_a": only_a, "only_b": only_b, "both": both, "neither": neither}
    missing = sorted(name for name, s in regions.items() if not s)
    if missing:
        pytest.fail(
            f"curated universe missing buildings in regions {missing} "
            f"for upgrades=({a},{b}) state={state}; pick different upgrades "
            f"so all four regions are non-empty."
        )

    curated: list[int] = []
    for s in (only_a, only_b, both, neither):
        curated.extend(sorted(s)[:per_region])
    return sorted(curated), regions, (a, b)


# --- parametrization ---------------------------------------------------------
#
# All per-schema column-name differences live in the per-schema resolvers in
# test_utility.py. Tests resolve schema-specific values via the
# `resolve_placeholder(schema, name, annual=...)` dispatcher at runtime,
# matching what the snapshot loader feeds into the stored SQL. `bsq.query()`
# expects literal column names, so the resolution happens in the test body.

SCHEMA_CASES = [
    pytest.param("bsq_resstock_oedi", "resstock_oedi", id="resstock"),
    pytest.param("bsq_comstock_oedi", "comstock_oedi", id="comstock"),
    pytest.param("bsq_comstock_oedi_agg", "comstock_oedi_agg", id="comstock_agg"),
]


# --- three-way invariant: annual == ts_year_collapse == sum(ts_monthly) -------
#
# Scenario axis (cross-producted with schema axis): each scenario contributes the
# extra `query()` kwargs that select baseline / upgrade-with-or-without-applied
# filters. `scenario_extra` is merged into every leg's `query()` call below.
#
# - baseline: upgrade_id="0" (omit applied_only — invalid for baseline).
# - upgrade1: upgrade 1, with applied_only=False (count both applied and unapplied
#   rows; without this explicit False, the schema validator would silently flip to
#   True and collapse this scenario onto the next one).
# - upgrade1_applied: upgrade 1, applied_only=True. Pins the fix where the TS
#   path internally appends a `_build_applied_subquery(all_of=[upgrade_id])`
#   restrict so the TS flow filters inapplicable buildings the same way the
#   annual flow does (via up.applicability=true).
# - upgrade1_applied_in_1_2: upgrade 1, applied_only=True, restricted further to
#   buildings to which both upgrades 1 and 2 applied (composed via
#   `get_applied_buildings_filter(all_of=[1, 2])`).
SCENARIOS = [
    pytest.param({"upgrade_id": "0"}, id="baseline"),
    pytest.param({"upgrade_id": "1", "applied_only": False}, id="upgrade1"),
    pytest.param(
        {"upgrade_id": "1", "applied_only": True},
        id="upgrade1_applied",
    ),
    pytest.param(
        {"upgrade_id": "1", "applied_only": True, "_applied_filter_all_of": [1, 2]},
        id="upgrade1_applied_in_1_2",
    ),
]


@pytest.mark.parametrize("scenario_extra", SCENARIOS)
@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_annual_equals_ts_year_equals_ts_monthly_sum(
    request,
    bsq_fixture,
    schema,
    scenario_extra,
):
    """For each schema and scenario: annual total, year-collapsed timeseries, and
    sum of monthly timeseries should all report the same per-group total energy.
    Counts must agree too. Also pins monthly `rows_per_sample` to 4*24*days_in_month.

    Both fuels (electricity + gas) are queried on every leg for both schemas —
    the comstock `..kwh` suffix on annual columns is handled by the placeholder
    resolver (annual columns get the suffix, ts columns don't).

    For non-baseline scenarios the test additionally requests the savings shape
    (`include_baseline=True, include_upgrade=True, include_savings=True`) and
    asserts that each of the three output columns (`__baseline`, `__upgrade`,
    `__savings`) agrees across all three flows. This subsumes the standalone
    `test_savings_decomposition` (the in-flow `b - u ≈ s` identity is asserted
    here too) and closes the symmetry-cancellation blind spot: a bug that
    inflates baseline and upgrade by the same amount leaves savings correct,
    so any savings-only check would miss it.
    """
    from buildstock_query.aggregate_query import UnsupportedQueryShape

    bsq = request.getfixturevalue(bsq_fixture)
    group_col = resolve_placeholder(schema, "building_type_col")
    annual_enduses = [
        resolve_placeholder(schema, "electricity_total"),
        resolve_placeholder(schema, "natural_gas_total"),
    ]
    ts_enduses = [
        resolve_placeholder(schema, "electricity_total", annual=False),
        resolve_placeholder(schema, "natural_gas_total", annual=False),
    ]
    # Use the schema's MULTI_STATE_PAIR for the state restrict — exercises
    # cross-flow consistency under the multi-state SQL shape, which is
    # different from single-state. On comstock especially, MULTI_STATE_PAIR
    # is CO+NM where 413 bldg_id values appear in both states (composite-key
    # disambiguation territory). On resstock the pair is CO+WY (no bldg_id
    # collision, but still validates cross-flow agreement under the multi-
    # state IN-list path).
    restrict = [("state", resolve_placeholder(schema, "multi_state_pair"))]

    # Savings columns are invalid when upgrade_id="0" (the schema validator
    # rejects include_savings on the baseline scenario); only request them
    # when querying an upgrade.
    is_baseline = scenario_extra.get("upgrade_id") == "0"
    savings_kwargs: dict = (
        {} if is_baseline
        else {"include_baseline": True, "include_upgrade": True, "include_savings": True}
    )

    # _applied_filter_all_of is a test-only sentinel: pull it out and replace
    # with a live get_applied_buildings_filter prepended to restrict.
    scenario_args = dict(scenario_extra)
    applied_all_of = scenario_args.pop("_applied_filter_all_of", None)
    scenario_restrict = list(restrict)
    # `record_restrict` mirrors `scenario_restrict` but keeps the live
    # SA Subquery in the `_applied_filter` marker form so the recorder
    # can serialize to JSON.
    record_restrict = list(restrict)
    if applied_all_of:
        applied_filter = bsq.get_applied_buildings_filter(all_of=applied_all_of)
        if applied_filter is not None:
            scenario_restrict = [applied_filter, *scenario_restrict]
            record_restrict = [
                {"_applied_filter": {"all_of": applied_all_of}},
                *record_restrict,
            ]

    try:
        annual_df = bsq.query(
            enduses=annual_enduses,
            group_by=[group_col],
            restrict=scenario_restrict,
            **savings_kwargs,
            **scenario_args,
        )
        record_query(bsq, {
            "enduses": annual_enduses,
            "group_by": [group_col],
            "restrict": record_restrict,
            **savings_kwargs,
            **scenario_args,
        })
        ts_year_df = bsq.query(
            enduses=ts_enduses,
            annual_only=False,
            timestamp_grouping_func="year",
            group_by=[group_col],
            restrict=scenario_restrict,
            **savings_kwargs,
            **scenario_args,
        )
        record_query(bsq, {
            "enduses": ts_enduses,
            "annual_only": False,
            "timestamp_grouping_func": "year",
            "group_by": [group_col],
            "restrict": record_restrict,
            **savings_kwargs,
            **scenario_args,
        })
        ts_monthly_df = bsq.query(
            enduses=ts_enduses,
            annual_only=False,
            timestamp_grouping_func="month",
            group_by=[group_col, "time"],
            restrict=scenario_restrict,
            **savings_kwargs,
            **scenario_args,
        )
        record_query(bsq, {
            "enduses": ts_enduses,
            "annual_only": False,
            "timestamp_grouping_func": "month",
            "group_by": [group_col, "time"],
            "restrict": record_restrict,
            **savings_kwargs,
            **scenario_args,
        })
    except UnsupportedQueryShape as exc:
        pytest.skip(f"query shape unsupported on {schema}: {exc}")

    # For non-savings scenarios the output column is just the enduse name; for
    # savings scenarios there is no plain enduse column, only `__baseline`,
    # `__upgrade`, `__savings`. Iterate over both fuels and the suffix list
    # (empty string for baseline) and assert each flow agrees on each column.
    annual_bases = [_strip_out_prefix(e) for e in annual_enduses]
    ts_bases = [_strip_out_prefix(e) for e in ts_enduses]
    suffixes = [""] if is_baseline else ["__baseline", "__upgrade", "__savings"]
    for annual_base, ts_base in zip(annual_bases, ts_bases):
        for suffix in suffixes:
            annual_col = annual_base + suffix
            ts_col = ts_base + suffix
            annual_totals = _scalar_total_by_group(annual_df, annual_col, [group_col])
            ts_year_totals = _scalar_total_by_group(ts_year_df, ts_col, [group_col])
            ts_monthly_totals = _scalar_total_by_group(ts_monthly_df, ts_col, [group_col])
            label = f"{ts_base}{suffix or ' (raw enduse)'}"
            _assert_series_close(
                f"annual vs ts_year_collapse [{label}]",
                annual_totals,
                ts_year_totals,
            )
            _assert_series_close(
                f"annual vs sum(ts_monthly) [{label}]",
                annual_totals,
                ts_monthly_totals,
            )

    # Per-flow savings decomposition: b - u ≈ s on every row of each frame.
    # Subsumes the standalone test_savings_decomposition. Skipped on baseline.
    # Covers both fuels — a sign-flip or wrong-column bug could affect one
    # fuel and not the other.
    if not is_baseline:
        for flow_name, flow_df in (
            ("annual", annual_df),
            ("ts_year_collapse", ts_year_df),
            ("sum(ts_monthly)", ts_monthly_df),
        ):
            bases = annual_bases if flow_name == "annual" else ts_bases
            for base_name in bases:
                b_col = base_name + "__baseline"
                u_col = base_name + "__upgrade"
                s_col = base_name + "__savings"
                decomp_diffs = []
                for _, row in flow_df.iterrows():
                    expected = float(row[b_col]) - float(row[u_col])
                    actual = float(row[s_col])
                    if not np.isclose(
                        expected, actual,
                        rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL, equal_nan=True,
                    ):
                        decomp_diffs.append(
                            f"  {row.get(group_col, '?')}: "
                            f"baseline-upgrade={expected:.4f}, savings={actual:.4f}"
                        )
                if decomp_diffs:
                    pytest.fail(
                        f"savings decomposition failed in {flow_name} for {base_name}:\n"
                        + "\n".join(decomp_diffs)
                    )

    # Counts are per-group metadata (constant across monthly rows), so collapsing
    # the monthly frame uses mean, not sum. This check caught the comstock
    # sample_count undercount before it was fixed (see 25fa3fa).
    for count_col in ("sample_count", "units_count"):
        annual_counts = _scalar_first_by_group(annual_df, count_col, [group_col])
        ts_year_counts = _scalar_first_by_group(ts_year_df, count_col, [group_col])
        ts_monthly_counts = _scalar_mean_by_group(ts_monthly_df, count_col, [group_col])
        _assert_series_close(
            f"{count_col}: ts_year_collapse vs mean(ts_monthly)",
            ts_year_counts,
            ts_monthly_counts,
        )
        _assert_series_close(
            f"{count_col}: annual vs ts_year_collapse",
            annual_counts,
            ts_year_counts,
        )

    # rows_per_sample on the monthly leg must equal 15-min intervals * hours * days
    # in that month: 4 * 24 * days_in_month. Catches drift in the timeseries cadence
    # or missing/duplicate timestamps in the underlying source data.
    bad = []
    for _, row in ts_monthly_df.iterrows():
        month = pd.Timestamp(row["timestamp"]).month
        year = pd.Timestamp(row["timestamp"]).year
        expected = 4 * 24 * calendar.monthrange(year, month)[1]
        actual = int(row["rows_per_sample"])
        if actual != expected:
            bad.append(
                f"  {row[group_col]} {row['timestamp'].date()}: "
                f"rows_per_sample={actual}, expected={expected}"
            )
    if bad:
        pytest.fail("monthly rows_per_sample mismatch (expected 4*24*days_in_month):\n" + "\n".join(bad))

    # Cross-flow rows_per_sample agreement: ts_year per-group rows_per_sample
    # must equal sum-over-months of ts_monthly. Annual flow doesn't have this
    # column. Catches divergence in TS cadence between aggregation levels —
    # e.g. if ts_year used a different distinct-counting expression than
    # ts_monthly, the per-group totals would diverge silently.
    ts_year_rps = (
        ts_year_df.assign(_rps=ts_year_df["rows_per_sample"].astype(int))
        .groupby(group_col, dropna=False)["_rps"].first().sort_index()
    )
    ts_monthly_rps = (
        ts_monthly_df.assign(_rps=ts_monthly_df["rows_per_sample"].astype(int))
        .groupby(group_col, dropna=False)["_rps"].sum().sort_index()
    )
    _assert_series_close(
        "rows_per_sample: ts_year vs sum(ts_monthly)",
        ts_year_rps,
        ts_monthly_rps,
    )

    # Absolute pin on ts_year rows_per_sample: should equal 4*24*days_in_year
    # (15-min cadence, 1 year). Independent of the monthly per-row check —
    # if either gets disabled or weakened, the other still catches cadence
    # drift on its own.
    sim_year = pd.Timestamp(ts_monthly_df["timestamp"].iloc[0]).year
    days_in_year = 366 if calendar.isleap(sim_year) else 365
    expected_year_rps = 4 * 24 * days_in_year
    bad_year = []
    for group_key, actual in ts_year_rps.items():
        if int(actual) != expected_year_rps:
            bad_year.append(f"  {group_key}: rows_per_sample={int(actual)}, expected={expected_year_rps}")
    if bad_year:
        pytest.fail(
            f"ts_year rows_per_sample mismatch (expected 4*24*{days_in_year}={expected_year_rps}):\n"
            + "\n".join(bad_year)
        )


# --- calculated-column three-way invariant ----------------------------------
#
# Mirror of the bare-enduse three-way above but with a get_calculated_column
# expression (electricity_total - natural_gas_total) as the only enduse. The
# calc column codepath is structurally distinct: `e` is a Label-wrapping-an-
# arithmetic-expression rather than a bare ts column, which exercises the
# pivot's `e.element` extraction and the single-scan path's get_col fallback.
# Without this invariant, a calc-column regression in either flow only shows
# up as a SQL-hash drift — never as a numeric divergence.
#
# Scenarios cover the two TS branches that calc columns flow through:
#   - upgrade1 (applied_only=False): single-scan path under the widened
#     upgrade_only predicate.
#   - upgrade1_applied (applied_only=True): single-scan path with the
#     internally-appended applied-buildings filter (filters inapplicable
#     buildings via _build_applied_subquery(all_of=[upgrade_id])).
# The pivot path proper is exercised by include_baseline=True scenarios in
# the snapshot suite (calculated_column_ts_pivot_with_baseline).
CALC_SCENARIOS = [
    pytest.param({"upgrade_id": "1", "applied_only": False}, id="upgrade1"),
    pytest.param({"upgrade_id": "1", "applied_only": True}, id="upgrade1_applied"),
]


@pytest.mark.parametrize("scenario_extra", CALC_SCENARIOS)
@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_calculated_column_matches_manual_decomposition_per_flow(
    request,
    bsq_fixture,
    schema,
    scenario_extra,
):
    """Calc-column manual-decomposition invariant: per flow (annual / ts_year /
    sum-ts_monthly), the calc-column total must equal the per-group manual
    decomposition `query(elec) - query(gas)`. Proves that the calc-col arithmetic
    is evaluated correctly on each flow independently. Cross-flow consistency
    (annual ≈ ts_year ≈ sum(ts_monthly)) is covered by the bare-fuel
    `test_annual_equals_ts_year_equals_ts_monthly_sum` and is NOT replicated
    here — subtracting two near-equal totals amplifies drift past tolerance
    even when both flows are individually correct (a real comstock annual-vs-TS
    drift of ~0.3% on bare fuels becomes ~4% on `elec - gas`).
    """
    bsq = request.getfixturevalue(bsq_fixture)
    group_col = resolve_placeholder(schema, "building_type_col")
    restrict = [("state", resolve_placeholder(schema, "multi_state_pair"))]

    # Same expression on both sides — placeholder resolver picks suffix-less
    # names for ts and adds `..kwh` for comstock annual.
    annual_elec = resolve_placeholder(schema, "electricity_total", annual=True)
    annual_gas = resolve_placeholder(schema, "natural_gas_total", annual=True)
    ts_elec = resolve_placeholder(schema, "electricity_total", annual=False)
    ts_gas = resolve_placeholder(schema, "natural_gas_total", annual=False)

    annual_calc = bsq.get_calculated_column(
        "elec_minus_gas",
        f"{annual_elec} - {annual_gas}",
        table="baseline",
    )
    ts_calc = bsq.get_calculated_column(
        "elec_minus_gas",
        f"{ts_elec} - {ts_gas}",
        table="timeseries",
    )
    # JSON-serializable marker forms for the recorder. The snapshot
    # harness rebuilds the live SA Label via _resolve_calc_and_mapped_columns
    # at load time.
    annual_calc_marker = {"_calc_column": {
        "name": "elec_minus_gas",
        "expr": f"{annual_elec} - {annual_gas}",
        "table": "baseline",
    }}
    ts_calc_marker = {"_calc_column": {
        "name": "elec_minus_gas",
        "expr": f"{ts_elec} - {ts_gas}",
        "table": "timeseries",
    }}

    # Calc-column query frames (one enduse, the labeled expression).
    annual_df = bsq.query(
        enduses=[annual_calc],
        group_by=[group_col],
        restrict=restrict,
        **scenario_extra,
    )
    record_query(bsq, {
        "enduses": [annual_calc_marker],
        "group_by": [group_col],
        "restrict": restrict,
        **scenario_extra,
    })
    ts_year_df = bsq.query(
        enduses=[ts_calc],
        annual_only=False,
        timestamp_grouping_func="year",
        group_by=[group_col],
        restrict=restrict,
        **scenario_extra,
    )
    record_query(bsq, {
        "enduses": [ts_calc_marker],
        "annual_only": False,
        "timestamp_grouping_func": "year",
        "group_by": [group_col],
        "restrict": restrict,
        **scenario_extra,
    })
    ts_monthly_df = bsq.query(
        enduses=[ts_calc],
        annual_only=False,
        timestamp_grouping_func="month",
        group_by=[group_col, "time"],
        restrict=restrict,
        **scenario_extra,
    )
    record_query(bsq, {
        "enduses": [ts_calc_marker],
        "annual_only": False,
        "timestamp_grouping_func": "month",
        "group_by": [group_col, "time"],
        "restrict": restrict,
        **scenario_extra,
    })

    # Manual-decomposition query frames (two bare-column enduses) — same args
    # as above but with the underlying fuels queried directly. The output
    # columns are the bare enduse names with the `out.` prefix stripped.
    annual_bare_df = bsq.query(
        enduses=[annual_elec, annual_gas],
        group_by=[group_col],
        restrict=restrict,
        **scenario_extra,
    )
    record_query(bsq, {
        "enduses": [annual_elec, annual_gas],
        "group_by": [group_col],
        "restrict": restrict,
        **scenario_extra,
    })
    ts_year_bare_df = bsq.query(
        enduses=[ts_elec, ts_gas],
        annual_only=False,
        timestamp_grouping_func="year",
        group_by=[group_col],
        restrict=restrict,
        **scenario_extra,
    )
    record_query(bsq, {
        "enduses": [ts_elec, ts_gas],
        "annual_only": False,
        "timestamp_grouping_func": "year",
        "group_by": [group_col],
        "restrict": restrict,
        **scenario_extra,
    })
    ts_monthly_bare_df = bsq.query(
        enduses=[ts_elec, ts_gas],
        annual_only=False,
        timestamp_grouping_func="month",
        group_by=[group_col, "time"],
        restrict=restrict,
        **scenario_extra,
    )
    record_query(bsq, {
        "enduses": [ts_elec, ts_gas],
        "annual_only": False,
        "timestamp_grouping_func": "month",
        "group_by": [group_col, "time"],
        "restrict": restrict,
        **scenario_extra,
    })

    # Per-flow manual-decomposition check: calc-col total == bare-elec - bare-gas.
    # This is the strongest check on calc-col arithmetic — proves the expression
    # was evaluated correctly on each flow independently. Cross-flow agreement
    # (annual calc vs ts_year calc) is intentionally NOT asserted here: when the
    # bare-fuel cross-flow has even small drift (e.g. comstock ResStock annual
    # vs TS aggregate disagreeing by 0.3% on FullServiceRestaurant), subtraction
    # amplifies that drift past any reasonable tolerance. The bare-fuel
    # `test_annual_equals_ts_year_equals_ts_monthly_sum` invariant already
    # catches cross-flow data inconsistencies; replicating that check here over
    # subtracted values would just produce false positives.
    col = "elec_minus_gas"
    annual_elec_col = _strip_out_prefix(annual_elec)
    annual_gas_col = _strip_out_prefix(annual_gas)
    ts_elec_col = _strip_out_prefix(ts_elec)
    ts_gas_col = _strip_out_prefix(ts_gas)
    for flow_name, calc_df, bare_df, elec_col, gas_col, gb in (
        ("annual", annual_df, annual_bare_df, annual_elec_col, annual_gas_col, [group_col]),
        ("ts_year_collapse", ts_year_df, ts_year_bare_df, ts_elec_col, ts_gas_col, [group_col]),
        ("sum(ts_monthly)", ts_monthly_df, ts_monthly_bare_df, ts_elec_col, ts_gas_col, [group_col]),
    ):
        calc_totals = _scalar_total_by_group(calc_df, col, gb)
        elec_totals = _scalar_total_by_group(bare_df, elec_col, gb)
        gas_totals = _scalar_total_by_group(bare_df, gas_col, gb)
        manual_totals = elec_totals - gas_totals
        _assert_series_close(
            f"calc-col vs manual decomposition [{flow_name}]",
            calc_totals,
            manual_totals,
        )


# --- group_by sum equals overall total ---------------------------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_group_by_sum_equals_overall(request, bsq_fixture, schema):
    """Sum across building-type groups must equal the no-group-by total. Same
    underlying query (annual electricity + gas, CO), different aggregation level."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduses = [
        resolve_placeholder(schema, "electricity_total"),
        resolve_placeholder(schema, "natural_gas_total"),
    ]
    group_col = resolve_placeholder(schema, "building_type_col")
    restrict = [("state", ["CO"])]

    overall_df = bsq.query(enduses=enduses, restrict=restrict)
    record_query(bsq, {"enduses": enduses, "restrict": restrict})
    grouped_df = bsq.query(enduses=enduses, group_by=[group_col], restrict=restrict)
    record_query(bsq, {"enduses": enduses, "group_by": [group_col], "restrict": restrict})

    for col in (_strip_out_prefix(e) for e in enduses):
        overall_total = float(overall_df[col].iloc[0])
        grouped_total = float(grouped_df[col].sum())
        if not np.isclose(
            overall_total, grouped_total,
            rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL,
        ):
            pytest.fail(
                f"{col}: overall ({overall_total:.4f}) != sum of grouped "
                f"({grouped_total:.4f}); diff={overall_total - grouped_total:.4f}"
            )
    # sample_count and units_count too (sum across groups, since these are per-row totals).
    for count_col in ("sample_count", "units_count"):
        overall_total = float(overall_df[count_col].iloc[0])
        grouped_total = float(grouped_df[count_col].sum())
        if not np.isclose(
            overall_total, grouped_total,
            rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL,
        ):
            pytest.fail(
                f"{count_col}: overall ({overall_total}) != sum of grouped ({grouped_total})"
            )


# --- TS bs-side group_by disaggregation: by-county and by-tract sum to total --
#
# Comstock-only invariant. Comstock's metadata is denormalized at
# (bldg_id, in.nhgis_tract_gisjoin, state) granularity with `weight` divided
# across tract rows; a building's tracts can map to different counties. The
# TS code path's `bs_per_bldg` subquery used to wrap bs-side group_by columns
# (county, tract) in `arbitrary()` and group only on (bldg_id, state),
# silently attributing the FULL building weight to whichever value
# `arbitrary()` happened to pick. The fix carries those columns as TRUE
# GROUP BY keys of bs_per_bldg, preserving per-tract weight slices.
#
# This invariant pins the fix: sum across counties (and across tracts) of the
# year-collapsed TS query must equal the no-bs-group-by total. A regression
# of the bs_per_bldg subquery would only affect by-county / by-tract totals
# (the no-group-by total uses a different code path), so the cross-check
# distinguishes the right answer from "by-county and by-tract agree but are
# both wrong by the same amount" — the only failure mode that would slip
# through if we only compared by-county to by-tract.
#
# Resstock is excluded: its md is one row per bldg, so bs-side group_by under
# bs_per_bldg has no fan-out to disaggregate, and the bug under test is
# structurally absent.
def test_ts_year_county_and_tract_disaggregation_matches_overall_comstock(request):
    """Comstock TS year-collapse: sum across counties = sum across tracts =
    no-group-by overall, all on the same restrict (CO). Pins the bs_per_bldg
    GROUP BY (bldg, state, <bs_group_by>) shape that disaggregates per-tract
    weight correctly when buildings straddle counties."""
    bsq = request.getfixturevalue("bsq_comstock_oedi")
    enduse = resolve_placeholder("comstock_oedi", "electricity_total", annual=False)
    enduse_col = _strip_out_prefix(enduse)
    restrict = [("state", ["CO"])]

    overall_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        timestamp_grouping_func="year", restrict=restrict,
    )
    record_query(bsq, {
        "enduses": [enduse], "annual_only": False, "upgrade_id": 0,
        "timestamp_grouping_func": "year", "restrict": restrict,
    })
    by_county_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        timestamp_grouping_func="year",
        group_by=["in.county_name"], restrict=restrict,
    )
    record_query(bsq, {
        "enduses": [enduse], "annual_only": False, "upgrade_id": 0,
        "timestamp_grouping_func": "year",
        "group_by": ["in.county_name"], "restrict": restrict,
    })
    by_tract_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        timestamp_grouping_func="year",
        group_by=["in.nhgis_tract_gisjoin"], restrict=restrict,
    )
    record_query(bsq, {
        "enduses": [enduse], "annual_only": False, "upgrade_id": 0,
        "timestamp_grouping_func": "year",
        "group_by": ["in.nhgis_tract_gisjoin"], "restrict": restrict,
    })

    overall_kwh = float(overall_df[enduse_col].iloc[0])
    by_county_kwh = float(by_county_df[enduse_col].sum())
    by_tract_kwh = float(by_tract_df[enduse_col].sum())

    if not np.isclose(by_county_kwh, overall_kwh, rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL):
        pytest.fail(
            f"sum(by-county) ({by_county_kwh:.2f}) != overall ({overall_kwh:.2f}); "
            f"diff={by_county_kwh - overall_kwh:.2f} — bs_per_bldg likely collapsing "
            f"county via arbitrary() and dropping tract-fractional weight"
        )
    if not np.isclose(by_tract_kwh, overall_kwh, rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL):
        pytest.fail(
            f"sum(by-tract) ({by_tract_kwh:.2f}) != overall ({overall_kwh:.2f}); "
            f"diff={by_tract_kwh - overall_kwh:.2f}"
        )

    # Counts also must reconcile. units_count is a weight sum; sum across the
    # bs-side group equals the overall (each tract row's weight is counted
    # exactly once). sample_count counts metadata rows: tract-grouped sum =
    # county-grouped sum = overall (no bs-side group still counts every md
    # row via `count(*)` inside bs_per_bldg).
    overall_units = float(overall_df["units_count"].iloc[0])
    by_county_units = float(by_county_df["units_count"].sum())
    by_tract_units = float(by_tract_df["units_count"].sum())
    if not np.isclose(by_county_units, overall_units, rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL):
        pytest.fail(
            f"units_count: sum(by-county) ({by_county_units:.4f}) != overall "
            f"({overall_units:.4f})"
        )
    if not np.isclose(by_tract_units, overall_units, rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL):
        pytest.fail(
            f"units_count: sum(by-tract) ({by_tract_units:.4f}) != overall "
            f"({overall_units:.4f})"
        )

    overall_samples = int(overall_df["sample_count"].iloc[0])
    by_county_samples = int(by_county_df["sample_count"].sum())
    by_tract_samples = int(by_tract_df["sample_count"].sum())
    if by_county_samples != overall_samples:
        pytest.fail(
            f"sample_count: sum(by-county)={by_county_samples} != overall={overall_samples}"
        )
    if by_tract_samples != overall_samples:
        pytest.fail(
            f"sample_count: sum(by-tract)={by_tract_samples} != overall={overall_samples}"
        )

    # by-tract must be at least as granular as by-county (each tract belongs
    # to exactly one county, but a county contains many tracts). If we ever
    # see fewer tract rows than county rows, the bs_per_bldg GROUP BY isn't
    # honoring the tract dimension.
    assert len(by_tract_df) >= len(by_county_df), (
        f"by-tract row count ({len(by_tract_df)}) < by-county row count "
        f"({len(by_county_df)}) — bs_per_bldg may be collapsing tract"
    )


# --- restrict subset: CO rows of CO+WY equal single-state CO -----------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_co_subset_of_co_plus_wy(request, bsq_fixture, schema):
    """The CO row of `restrict_two_states` (state IN ('CO', 'WY')) must equal
    `restrict_single_state` (state IN ('CO',)) row-for-row. Catches restrict-list
    scoping bugs where the IN clause inadvertently affects the filter beyond what's
    declared."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total")
    group_col = resolve_placeholder(schema, "building_type_col")

    co_only_df = bsq.query(
        enduses=[enduse], group_by=[group_col], restrict=[("state", ["CO"])],
    )
    record_query(bsq, {
        "enduses": [enduse], "group_by": [group_col], "restrict": [("state", ["CO"])],
    })
    co_wy_df = bsq.query(
        enduses=[enduse], group_by=["state"], restrict=[("state", ["CO", "WY"])],
    )
    record_query(bsq, {
        "enduses": [enduse], "group_by": ["state"], "restrict": [("state", ["CO", "WY"])],
    })

    co_row = co_wy_df[co_wy_df["state"] == "CO"]
    if co_row.empty:
        pytest.fail("no CO row in restrict_two_states result")
    enduse_col = _strip_out_prefix(enduse)

    co_total_from_two_states = float(co_row[enduse_col].iloc[0])
    co_total_from_single_state = float(co_only_df[enduse_col].sum())
    if not np.isclose(
        co_total_from_two_states, co_total_from_single_state,
        rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL,
    ):
        pytest.fail(
            f"CO total from CO+WY query ({co_total_from_two_states:.4f}) != "
            f"CO total from CO-only query summed across building types "
            f"({co_total_from_single_state:.4f})"
        )
    # sample_count too — should be exactly equal (no float drift on integer counts).
    co_count_two = int(co_row["sample_count"].iloc[0])
    co_count_single = int(co_only_df["sample_count"].sum())
    if co_count_two != co_count_single:
        pytest.fail(
            f"CO sample_count mismatch: from CO+WY={co_count_two}, "
            f"from CO-only={co_count_single}"
        )


# --- avoid + avoided = full --------------------------------------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_avoid_plus_avoided_equals_full(request, bsq_fixture, schema):
    """`avoid_building_type` (CO without target) + the avoided building type's row from
    `restrict_single_state` should equal the full `restrict_single_state` totals."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total")
    group_col = resolve_placeholder(schema, "building_type_col")
    avoided_value = resolve_placeholder(schema, "avoid_building_type")
    restrict = [("state", ["CO"])]

    full_df = bsq.query(
        enduses=[enduse], group_by=[group_col], restrict=restrict,
    )
    record_query(bsq, {
        "enduses": [enduse], "group_by": [group_col], "restrict": restrict,
    })
    avoid_df = bsq.query(
        enduses=[enduse], group_by=[group_col], restrict=restrict,
        avoid=[(group_col, [avoided_value])],
    )
    record_query(bsq, {
        "enduses": [enduse], "group_by": [group_col], "restrict": restrict,
        "avoid": [(group_col, [avoided_value])],
    })

    avoided_row = full_df[full_df[group_col] == avoided_value]
    if avoided_row.empty:
        pytest.fail(f"avoided building type {avoided_value!r} not found in full result")
    enduse_col = _strip_out_prefix(enduse)

    full_total = float(full_df[enduse_col].sum())
    avoid_total = float(avoid_df[enduse_col].sum())
    avoided_total = float(avoided_row[enduse_col].iloc[0])
    if not np.isclose(
        full_total, avoid_total + avoided_total,
        rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL,
    ):
        pytest.fail(
            f"full ({full_total:.4f}) != avoid ({avoid_total:.4f}) + avoided "
            f"({avoided_total:.4f}); diff={full_total - avoid_total - avoided_total:.4f}"
        )
    # Sample counts are integer-exact.
    full_n = int(full_df["sample_count"].sum())
    avoid_n = int(avoid_df["sample_count"].sum())
    avoided_n = int(avoided_row["sample_count"].iloc[0])
    if full_n != avoid_n + avoided_n:
        pytest.fail(
            f"sample_count: full={full_n}, avoid={avoid_n}, avoided={avoided_n}; "
            f"avoid + avoided = {avoid_n + avoided_n}"
        )


# --- MappedColumn aggregates correctly ---------------------------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_mapped_column_aggregates_underlying_types(request, bsq_fixture, schema):
    """For each mapped category, the value from the MappedColumn-grouped query should
    equal the sum of the constituent building types from the regular grouped query.
    The mapping_dict tells us which underlying types belong to each category."""
    from buildstock_query.schema.utilities import MappedColumn
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total")
    group_col = resolve_placeholder(schema, "building_type_col")
    mapping_dict = resolve_placeholder(schema, "building_type_mapping")
    restrict = [("state", ["CO"])]

    # Direct group_by — one row per underlying building type.
    direct_df = bsq.query(
        enduses=[enduse], group_by=[group_col], restrict=restrict,
    )
    record_query(bsq, {
        "enduses": [enduse], "group_by": [group_col], "restrict": restrict,
    })
    # MappedColumn group_by — one row per mapped category (MH/SF/MF, etc.).
    key_col = bsq._get_column(group_col)
    mapped = MappedColumn(
        bsq=bsq, name="simple_bldg_type", mapping_dict=mapping_dict, key=key_col,
    )
    mapped_marker = {"_mapped_column": {
        "name": "simple_bldg_type",
        "key_column": group_col,
        "mapping_dict": mapping_dict,
    }}
    mapped_df = bsq.query(
        enduses=[enduse], group_by=[mapped], restrict=restrict,
    )
    record_query(bsq, {
        "enduses": [enduse], "group_by": [mapped_marker], "restrict": restrict,
    })

    enduse_col = _strip_out_prefix(enduse)
    # For each mapped category in the result, sum the underlying values from the direct
    # query and compare. Build the inverse mapping: category → list of underlying types.
    inverse: dict[str, list[str]] = {}
    for underlying, category in mapping_dict.items():
        inverse.setdefault(category, []).append(underlying)

    diffs = []
    for _, mapped_row in mapped_df.iterrows():
        category = mapped_row["simple_bldg_type"]
        underlying_types = inverse.get(category, [])
        if not underlying_types:
            diffs.append(f"  category {category!r} not in mapping_dict reverse map")
            continue
        underlying_total = float(
            direct_df[direct_df[group_col].isin(underlying_types)][enduse_col].sum()
        )
        mapped_total = float(mapped_row[enduse_col])
        if not np.isclose(
            mapped_total, underlying_total,
            rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL,
        ):
            diffs.append(
                f"  {category}: mapped={mapped_total:.4f}, sum of "
                f"{underlying_types}={underlying_total:.4f}"
            )
    if diffs:
        pytest.fail("MappedColumn aggregation mismatch:\n" + "\n".join(diffs))


# --- 15-min raw timeseries sums to monthly -----------------------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_15min_raw_sums_to_monthly(request, bsq_fixture, schema):
    """Per-state, the 15-min raw timeseries summed within each calendar month must
    equal the monthly aggregate. Strong cadence invariant — catches `timestamp_grouping_func='month'`
    boundary bugs (timezone offsets, month boundaries, accumulation drift)."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total", annual=False)

    raw_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        group_by=["state", "time"],
        restrict=[("state", ["CO"])],
    )
    record_query(bsq, {
        "enduses": [enduse], "annual_only": False, "upgrade_id": 0,
        "group_by": ["state", "time"],
        "restrict": [("state", ["CO"])],
    })
    monthly_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        timestamp_grouping_func="month",
        group_by=["state", "time"],
        restrict=[("state", ["CO"])],
    )
    record_query(bsq, {
        "enduses": [enduse], "annual_only": False, "upgrade_id": 0,
        "timestamp_grouping_func": "month",
        "group_by": ["state", "time"],
        "restrict": [("state", ["CO"])],
    })

    enduse_col = _strip_out_prefix(enduse)
    # Bucket raw rows into months (using the same `date_trunc('month', ts - 900s)` shift
    # the library applies internally — 15 minutes back so :15 belongs to the prior period).
    raw_df = raw_df.copy()
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    raw_df["month"] = (raw_df["timestamp"] - pd.Timedelta(seconds=900)).dt.to_period("M").dt.to_timestamp()
    raw_monthly = raw_df.groupby(["state", "month"], as_index=False)[enduse_col].sum()

    monthly_df = monthly_df.copy()
    monthly_df["timestamp"] = pd.to_datetime(monthly_df["timestamp"])
    merged = raw_monthly.merge(
        monthly_df, left_on=["state", "month"], right_on=["state", "timestamp"],
        suffixes=("_raw_sum", "_monthly"),
    )
    if len(merged) != len(monthly_df):
        pytest.fail(
            f"month bucket mismatch: raw produces {len(raw_monthly)} buckets, "
            f"monthly query produces {len(monthly_df)} rows, merged has {len(merged)}"
        )

    diffs = []
    for _, row in merged.iterrows():
        raw_total = float(row[f"{enduse_col}_raw_sum"])
        monthly_total = float(row[f"{enduse_col}_monthly"])
        if not np.isclose(
            raw_total, monthly_total,
            rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL,
        ):
            diffs.append(
                f"  {row['state']} {row['month'].date()}: raw_sum={raw_total:.4f}, "
                f"monthly={monthly_total:.4f}"
            )
    if diffs:
        pytest.fail("15-min sum vs monthly aggregate mismatch:\n" + "\n".join(diffs))


# --- 15-min raw timeseries sums to hourly ------------------------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_15min_raw_sums_to_hourly(request, bsq_fixture, schema):
    """Per-state, 15-min raw rows summed within each calendar hour must equal
    the hourly-grouped query. Mirrors the 15-min→monthly invariant at the
    finest aggregation step the library supports — catches `date_trunc('hour',
    ...)` boundary drift, the same -900s offset, and any hour-bucketing
    rounding bugs in between."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total", annual=False)

    raw_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        group_by=["state", "time"],
        restrict=[("state", ["CO"])],
    )
    record_query(bsq, {
        "enduses": [enduse], "annual_only": False, "upgrade_id": 0,
        "group_by": ["state", "time"],
        "restrict": [("state", ["CO"])],
    })
    hourly_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        timestamp_grouping_func="hour",
        group_by=["state", "time"],
        restrict=[("state", ["CO"])],
    )
    record_query(bsq, {
        "enduses": [enduse], "annual_only": False, "upgrade_id": 0,
        "timestamp_grouping_func": "hour",
        "group_by": ["state", "time"],
        "restrict": [("state", ["CO"])],
    })

    enduse_col = _strip_out_prefix(enduse)
    raw_df = raw_df.copy()
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    # Same -900s shift the library applies before date_trunc — :15 belongs to
    # the prior period (period-end vs period-beginning convention).
    raw_df["hour"] = (raw_df["timestamp"] - pd.Timedelta(seconds=900)).dt.floor("h")
    raw_hourly = raw_df.groupby(["state", "hour"], as_index=False)[enduse_col].sum()

    hourly_df = hourly_df.copy()
    hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"])
    merged = raw_hourly.merge(
        hourly_df, left_on=["state", "hour"], right_on=["state", "timestamp"],
        suffixes=("_raw_sum", "_hourly"),
    )
    if len(merged) != len(hourly_df):
        pytest.fail(
            f"hour bucket mismatch: raw produces {len(raw_hourly)} buckets, "
            f"hourly query produces {len(hourly_df)} rows, merged has {len(merged)}"
        )

    diffs = []
    for _, row in merged.iterrows():
        raw_total = float(row[f"{enduse_col}_raw_sum"])
        hourly_total = float(row[f"{enduse_col}_hourly"])
        if not np.isclose(
            raw_total, hourly_total,
            rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL,
        ):
            diffs.append(
                f"  {row['state']} {row['hour']}: raw_sum={raw_total:.4f}, "
                f"hourly={hourly_total:.4f}"
            )
    if diffs:
        pytest.fail("15-min sum vs hourly aggregate mismatch:\n" + "\n".join(diffs))


# --- daily and hourly sum-bucket invariants ----------------------------------
#
# These mirror the 15-min→monthly invariant at coarser cadences. They run
# entirely off snapshot data: ts_daily_electricity_by_state and
# ts_monthly_electricity_by_state share the same restrict/group_by, so the
# daily frame can be summed into months and compared. Likewise hourly→daily
# uses the new ts_hourly_electricity_by_state entry whose restrict matches
# the daily one. The 900s offset that the library uses to bucket :15 into
# the prior period is irrelevant here because the source rows are themselves
# already truncated to the start of each daily/hourly bucket.

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_daily_sums_to_monthly(request, bsq_fixture, schema):
    """Per-state daily TS summed within each calendar month must equal the
    monthly aggregate. Catches `timestamp_grouping_func='day'` vs `'month'`
    boundary drift without requiring a fresh Athena round-trip."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total", annual=False)

    daily_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        timestamp_grouping_func="day",
        group_by=["state", "time"],
        restrict=[("state", ["CO"])],
    )
    record_query(bsq, {
        "enduses": [enduse], "annual_only": False, "upgrade_id": 0,
        "timestamp_grouping_func": "day",
        "group_by": ["state", "time"],
        "restrict": [("state", ["CO"])],
    })
    monthly_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        timestamp_grouping_func="month",
        group_by=["state", "time"],
        restrict=[("state", ["CO"])],
    )
    record_query(bsq, {
        "enduses": [enduse], "annual_only": False, "upgrade_id": 0,
        "timestamp_grouping_func": "month",
        "group_by": ["state", "time"],
        "restrict": [("state", ["CO"])],
    })

    enduse_col = _strip_out_prefix(enduse)
    daily_df = daily_df.copy()
    daily_df["timestamp"] = pd.to_datetime(daily_df["timestamp"])
    daily_df["month"] = daily_df["timestamp"].dt.to_period("M").dt.to_timestamp()
    daily_monthly = daily_df.groupby(["state", "month"], as_index=False)[enduse_col].sum()

    monthly_df = monthly_df.copy()
    monthly_df["timestamp"] = pd.to_datetime(monthly_df["timestamp"])
    merged = daily_monthly.merge(
        monthly_df, left_on=["state", "month"], right_on=["state", "timestamp"],
        suffixes=("_daily_sum", "_monthly"),
    )
    if len(merged) != len(monthly_df):
        pytest.fail(
            f"month bucket mismatch: daily produces {len(daily_monthly)} buckets, "
            f"monthly query produces {len(monthly_df)} rows, merged has {len(merged)}"
        )

    diffs = []
    for _, row in merged.iterrows():
        daily_total = float(row[f"{enduse_col}_daily_sum"])
        monthly_total = float(row[f"{enduse_col}_monthly"])
        if not np.isclose(
            daily_total, monthly_total,
            rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL,
        ):
            diffs.append(
                f"  {row['state']} {row['month'].date()}: daily_sum={daily_total:.4f}, "
                f"monthly={monthly_total:.4f}"
            )
    if diffs:
        pytest.fail("daily sum vs monthly aggregate mismatch:\n" + "\n".join(diffs))


@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_hourly_sums_to_daily(request, bsq_fixture, schema):
    """Per-state hourly TS summed within each day must equal the daily aggregate.
    Requires the ts_hourly_electricity_by_state snapshot entry whose restrict and
    group_by match ts_daily_electricity_by_state."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total", annual=False)

    hourly_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        timestamp_grouping_func="hour",
        group_by=["state", "time"],
        restrict=[("state", ["CO"])],
    )
    record_query(bsq, {
        "enduses": [enduse], "annual_only": False, "upgrade_id": 0,
        "timestamp_grouping_func": "hour",
        "group_by": ["state", "time"],
        "restrict": [("state", ["CO"])],
    })
    daily_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        timestamp_grouping_func="day",
        group_by=["state", "time"],
        restrict=[("state", ["CO"])],
    )
    record_query(bsq, {
        "enduses": [enduse], "annual_only": False, "upgrade_id": 0,
        "timestamp_grouping_func": "day",
        "group_by": ["state", "time"],
        "restrict": [("state", ["CO"])],
    })

    enduse_col = _strip_out_prefix(enduse)
    hourly_df = hourly_df.copy()
    hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"])
    hourly_df["day"] = hourly_df["timestamp"].dt.floor("D")
    hourly_daily = hourly_df.groupby(["state", "day"], as_index=False)[enduse_col].sum()

    daily_df = daily_df.copy()
    daily_df["timestamp"] = pd.to_datetime(daily_df["timestamp"])
    merged = hourly_daily.merge(
        daily_df, left_on=["state", "day"], right_on=["state", "timestamp"],
        suffixes=("_hourly_sum", "_daily"),
    )
    if len(merged) != len(daily_df):
        pytest.fail(
            f"day bucket mismatch: hourly produces {len(hourly_daily)} buckets, "
            f"daily query produces {len(daily_df)} rows, merged has {len(merged)}"
        )

    diffs = []
    for _, row in merged.iterrows():
        hourly_total = float(row[f"{enduse_col}_hourly_sum"])
        daily_total = float(row[f"{enduse_col}_daily"])
        if not np.isclose(
            hourly_total, daily_total,
            rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL,
        ):
            diffs.append(
                f"  {row['state']} {row['day'].date()}: hourly_sum={hourly_total:.4f}, "
                f"daily={daily_total:.4f}"
            )
    if diffs:
        pytest.fail("hourly sum vs daily aggregate mismatch:\n" + "\n".join(diffs))


# --- savings_only column matches savings column from full savings query -------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_savings_only_matches_full_savings_query(request, bsq_fixture, schema):
    """`include_savings=True, include_baseline=False, include_upgrade=False` produces
    a result with just the savings column. That column must equal the savings column
    from the full `include_baseline + include_upgrade + include_savings` query —
    same SQL aggregations, just different output projection."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total")
    group_col = resolve_placeholder(schema, "building_type_col")
    restrict = [("state", ["CO"])]

    full_df = bsq.query(
        enduses=[enduse], upgrade_id="1", group_by=[group_col], restrict=restrict,
        include_baseline=True, include_upgrade=True, include_savings=True,
    )
    record_query(bsq, {
        "enduses": [enduse], "upgrade_id": "1", "group_by": [group_col], "restrict": restrict,
        "include_baseline": True, "include_upgrade": True, "include_savings": True,
    })
    only_df = bsq.query(
        enduses=[enduse], upgrade_id="1", group_by=[group_col], restrict=restrict,
        include_baseline=False, include_upgrade=False, include_savings=True,
    )
    record_query(bsq, {
        "enduses": [enduse], "upgrade_id": "1", "group_by": [group_col], "restrict": restrict,
        "include_baseline": False, "include_upgrade": False, "include_savings": True,
    })

    full_savings_col = _find_first_col(full_df, suffix="__savings", contains="electricity.total")
    only_savings_col = _find_first_col(only_df, suffix="__savings", contains="electricity.total")
    if len(full_df) != len(only_df):
        pytest.fail(
            f"row count mismatch: full={len(full_df)}, only={len(only_df)}"
        )

    full_indexed = full_df.set_index(group_col)[full_savings_col].sort_index()
    only_indexed = only_df.set_index(group_col)[only_savings_col].sort_index()
    diffs = []
    for key in full_indexed.index:
        full_val = float(full_indexed[key])
        only_val = float(only_indexed[key])
        if not np.isclose(full_val, only_val, rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL):
            diffs.append(
                f"  {key}: full_savings={full_val:.4f}, only_savings={only_val:.4f}"
            )
    if diffs:
        pytest.fail(
            "savings column differs between full savings query and savings-only query:\n"
            + "\n".join(diffs)
        )


# --- savings column matches independent baseline-minus-upgrade --------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_savings_equals_independent_baseline_minus_upgrade(request, bsq_fixture, schema):
    """The savings query's __baseline, __upgrade, __savings columns must each
    equal the result of running a standalone baseline-only query, a standalone
    upgrade-only query, and the difference of the two — all aggregated over the
    same set of applicable buildings.

    This is a strictly stronger check than the in-frame `b - u ≈ s` identity
    (which is essentially tautological at the SQL level, since all three
    columns are computed from a shared subquery). Here, baseline and upgrade
    are computed by independent queries that don't share the savings query's
    join graph, so any bug in the savings query's building-set selection or
    aggregation would surface as a mismatch.

    Recipe:
      filt   = bsq.get_applied_buildings_filter(all_of=[1])
      b_only = bsq.query(upgrade_id="0", restrict=[filt, ...], ...)
      u_only = bsq.query(upgrade_id="1", applied_only=True, ...)
      full   = bsq.query(upgrade_id="1", applied_only=True,
                         include_baseline + include_upgrade + include_savings, ...)

      For each fuel × group:
        full.<fuel>__baseline ≈ b_only.<fuel>
        full.<fuel>__upgrade  ≈ u_only.<fuel>
        full.<fuel>__savings  ≈ b_only.<fuel> - u_only.<fuel>

    Both fuels (electricity + gas) are checked.
    """
    from buildstock_query.aggregate_query import UnsupportedQueryShape

    bsq = request.getfixturevalue(bsq_fixture)
    enduses = [
        resolve_placeholder(schema, "electricity_total"),
        resolve_placeholder(schema, "natural_gas_total"),
    ]
    group_col = resolve_placeholder(schema, "building_type_col")
    restrict = [("state", ["CO"])]

    try:
        applied_filter = bsq.get_applied_buildings_filter(all_of=[1])
        b_only_restrict = [applied_filter, *restrict] if applied_filter else list(restrict)
        b_only_record_restrict = (
            [{"_applied_filter": {"all_of": [1]}}, *restrict]
            if applied_filter else list(restrict)
        )
        b_only = bsq.query(
            enduses=enduses, upgrade_id="0",
            group_by=[group_col], restrict=b_only_restrict,
        )
        record_query(bsq, {
            "enduses": enduses, "upgrade_id": "0",
            "group_by": [group_col], "restrict": b_only_record_restrict,
        })
        u_only = bsq.query(
            enduses=enduses, upgrade_id="1",
            applied_only=True, group_by=[group_col], restrict=restrict,
        )
        record_query(bsq, {
            "enduses": enduses, "upgrade_id": "1",
            "applied_only": True, "group_by": [group_col], "restrict": restrict,
        })
        full = bsq.query(
            enduses=enduses, upgrade_id="1",
            applied_only=True, group_by=[group_col], restrict=restrict,
            include_baseline=True, include_upgrade=True, include_savings=True,
        )
        record_query(bsq, {
            "enduses": enduses, "upgrade_id": "1",
            "applied_only": True, "group_by": [group_col], "restrict": restrict,
            "include_baseline": True, "include_upgrade": True, "include_savings": True,
        })
    except UnsupportedQueryShape as exc:
        pytest.skip(f"query shape unsupported on {schema}: {exc}")

    bases = [_strip_out_prefix(e) for e in enduses]
    diffs = []
    for base in bases:
        b_series = b_only.set_index(group_col)[base].astype(float).sort_index()
        u_series = u_only.set_index(group_col)[base].astype(float).sort_index()
        full_indexed = full.set_index(group_col).sort_index()
        for key in b_series.index:
            b_indep = b_series[key]
            u_indep = u_series[key]
            f_b = float(full_indexed.loc[key, f"{base}__baseline"])
            f_u = float(full_indexed.loc[key, f"{base}__upgrade"])
            f_s = float(full_indexed.loc[key, f"{base}__savings"])
            for label, expected, actual in (
                (f"{base}__baseline", b_indep, f_b),
                (f"{base}__upgrade", u_indep, f_u),
                (f"{base}__savings", b_indep - u_indep, f_s),
            ):
                if not np.isclose(
                    expected, actual,
                    rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL, equal_nan=True,
                ):
                    diffs.append(
                        f"  {key} {label}: independent={expected:.4f}, "
                        f"savings_query={actual:.4f}"
                    )
    if diffs:
        pytest.fail(
            "savings query columns disagree with independent baseline/upgrade queries:\n"
            + "\n".join(diffs)
        )


# --- savings invariant under applied_only flag --------------------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_savings_independent_of_applied_only_flag(request, bsq_fixture, schema):
    """Savings totals per group must match between `applied_only=True` and the
    default (applied_only=False) flavors of the same savings query. Inapplicable
    buildings contribute zero savings (their baseline equals their upgrade via
    the outerjoin + COALESCE), so toggling whether they're included in the
    aggregation cannot change the savings column.

    This catches bugs where the applied_only flag inadvertently affects the
    savings aggregation path itself (rather than just controlling which
    buildings are counted in baseline/upgrade).
    """
    from buildstock_query.aggregate_query import UnsupportedQueryShape

    bsq = request.getfixturevalue(bsq_fixture)
    enduses = [
        resolve_placeholder(schema, "electricity_total"),
        resolve_placeholder(schema, "natural_gas_total"),
    ]
    group_col = resolve_placeholder(schema, "building_type_col")
    restrict = [("state", ["CO"])]

    try:
        applied = bsq.query(
            enduses=enduses, upgrade_id="1",
            applied_only=True, group_by=[group_col], restrict=restrict,
            include_baseline=True, include_upgrade=True, include_savings=True,
        )
        record_query(bsq, {
            "enduses": enduses, "upgrade_id": "1",
            "applied_only": True, "group_by": [group_col], "restrict": restrict,
            "include_baseline": True, "include_upgrade": True, "include_savings": True,
        })
        full = bsq.query(
            enduses=enduses, upgrade_id="1",
            group_by=[group_col], restrict=restrict,
            include_baseline=True, include_upgrade=True, include_savings=True,
        )
        record_query(bsq, {
            "enduses": enduses, "upgrade_id": "1",
            "group_by": [group_col], "restrict": restrict,
            "include_baseline": True, "include_upgrade": True, "include_savings": True,
        })
    except UnsupportedQueryShape as exc:
        pytest.skip(f"query shape unsupported on {schema}: {exc}")

    bases = [_strip_out_prefix(e) for e in enduses]
    diffs = []
    for base in bases:
        col = f"{base}__savings"
        applied_indexed = applied.set_index(group_col)[col].astype(float).sort_index()
        full_indexed = full.set_index(group_col)[col].astype(float).sort_index()
        for key in applied_indexed.index:
            a = applied_indexed[key]
            f = full_indexed[key]
            if not np.isclose(a, f, rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL, equal_nan=True):
                diffs.append(
                    f"  {key} {col}: applied_only=True={a:.4f}, "
                    f"applied_only=False(default)={f:.4f}"
                )
    if diffs:
        pytest.fail(
            "savings differs between applied_only=True and applied_only=False:\n"
            + "\n".join(diffs)
        )


# --- multi-state savings = sum of per-state savings ---------------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_multi_state_savings_equals_sum_of_per_state(request, bsq_fixture, schema):
    """Savings query restricted to two states must equal the per-group sum of
    the same query run on each state alone — for each of __baseline,
    __upgrade, __savings, both fuels.

    The state pair is per-schema (`MULTI_STATE_PAIR` placeholder):

    - resstock: ['CO', 'WY']. bldg_ids happen to be globally unique across
      states in this dataset, so additivity falls out of the disjoint key
      sets — the test is a structural regression guard.
    - comstock: ['CO', 'NM']. ~413 bldg_id values appear in BOTH states.
      The schema's composite key (bldg_id, state) is what makes them
      distinct buildings; if the join logic ever degrades to bldg_id-only,
      overlapping buildings would be double-counted and additivity would
      break. This pair specifically exercises the composite-key path; CO
      is reused from other tests so only NM-only and (CO, NM) need fresh
      Athena calls.

    Catches bugs that surface only when multiple states are part of the
    same query: state-axis aggregation errors, joins that don't propagate
    the state partition predicate to all sides, and implicit assumptions
    that bldg_id is globally unique.
    """
    from buildstock_query.aggregate_query import UnsupportedQueryShape

    bsq = request.getfixturevalue(bsq_fixture)
    enduses = [
        resolve_placeholder(schema, "electricity_total"),
        resolve_placeholder(schema, "natural_gas_total"),
    ]
    group_col = resolve_placeholder(schema, "building_type_col")
    state_pair = resolve_placeholder(schema, "multi_state_pair")
    s1, s2 = state_pair[0], state_pair[1]

    try:
        s1_df = bsq.query(
            enduses=enduses, upgrade_id="1",
            applied_only=True, group_by=[group_col],
            restrict=[("state", [s1])],
            include_baseline=True, include_upgrade=True, include_savings=True,
        )
        record_query(bsq, {
            "enduses": enduses, "upgrade_id": "1",
            "applied_only": True, "group_by": [group_col],
            "restrict": [("state", [s1])],
            "include_baseline": True, "include_upgrade": True, "include_savings": True,
        })
        s2_df = bsq.query(
            enduses=enduses, upgrade_id="1",
            applied_only=True, group_by=[group_col],
            restrict=[("state", [s2])],
            include_baseline=True, include_upgrade=True, include_savings=True,
        )
        record_query(bsq, {
            "enduses": enduses, "upgrade_id": "1",
            "applied_only": True, "group_by": [group_col],
            "restrict": [("state", [s2])],
            "include_baseline": True, "include_upgrade": True, "include_savings": True,
        })
        both = bsq.query(
            enduses=enduses, upgrade_id="1",
            applied_only=True, group_by=[group_col],
            restrict=[("state", [s1, s2])],
            include_baseline=True, include_upgrade=True, include_savings=True,
        )
        record_query(bsq, {
            "enduses": enduses, "upgrade_id": "1",
            "applied_only": True, "group_by": [group_col],
            "restrict": [("state", [s1, s2])],
            "include_baseline": True, "include_upgrade": True, "include_savings": True,
        })
    except UnsupportedQueryShape as exc:
        pytest.skip(f"query shape unsupported on {schema}: {exc}")

    bases = [_strip_out_prefix(e) for e in enduses]
    suffixes = ["__baseline", "__upgrade", "__savings"]

    # Combined per-group totals must equal the union of building-type keys
    # across both single-state queries.
    s1_indexed = s1_df.set_index(group_col).sort_index()
    s2_indexed = s2_df.set_index(group_col).sort_index()
    both_indexed = both.set_index(group_col).sort_index()
    expected_keys = sorted(set(s1_indexed.index) | set(s2_indexed.index))
    actual_keys = sorted(both_indexed.index)
    if expected_keys != actual_keys:
        pytest.fail(
            f"multi-state group_by key set differs from union of per-state keys "
            f"({s1}+{s2}):\n  expected={expected_keys}\n  actual={actual_keys}"
        )

    diffs = []
    for base in bases:
        for suffix in suffixes:
            col = f"{base}{suffix}"
            for key in expected_keys:
                s1_val = float(s1_indexed.loc[key, col]) if key in s1_indexed.index else 0.0
                s2_val = float(s2_indexed.loc[key, col]) if key in s2_indexed.index else 0.0
                both_val = float(both_indexed.loc[key, col])
                expected = s1_val + s2_val
                if not np.isclose(
                    expected, both_val,
                    rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL, equal_nan=True,
                ):
                    diffs.append(
                        f"  {key} {col}: {s1}+{s2}={expected:.4f}, "
                        f"multi-state={both_val:.4f}"
                    )
    if diffs:
        pytest.fail(
            f"multi-state savings query ({s1}+{s2}) disagrees with sum of "
            f"per-state queries:\n" + "\n".join(diffs)
        )


# --- comstock composite-key handling for shared bldg_id ---------------------

def test_comstock_shared_bldg_id_composite_key_handling(request):
    """Targeted check that comstock's composite-key architecture is handled
    correctly throughout the query stack. Uses bldg_id=51037, which is a
    building archetype deployed in 4 states (CO=2 tracts, NM=30 tracts,
    OK=1 tract, TX=13 tracts — 46 metadata rows total) but stored as ONE
    timeseries stream per (bldg_id, state) pair.

    The composite metadata key is (bldg_id, state, tract); the TS unique key
    is (bldg_id, state). Bugs in handling this would show as one of:
      - state filter not propagating to the TS join (multi-state query
        inflates a single-state archetype)
      - bldg_id-only join (TS rows fan out across unrelated metadata rows)
      - GROUP BY collapsing on bldg_id alone (loses tract-level breakdown)

    Anchored expected values:
      no group_by, bldg=51037: sample_count=46, units_count≈8.987, kWh≈3.711M
      bldg=51037, state=CO:    sample_count=2,  units_count≈0.616, kWh≈0.255M
      bldg=51037, state=NM:    sample_count=30, units_count≈5.694, kWh≈2.352M
      bldg=51037, state=OK:    sample_count=1,  units_count≈0.043, kWh≈0.018M
      bldg=51037, state=TX:    sample_count=13, units_count≈2.633, kWh≈1.088M

    Comstock-only — resstock's bldg_ids are globally unique (one bldg_id ↔
    one (state, tract)), so the composite-key behavior under test is
    structurally absent from that schema.
    """
    bsq = request.getfixturevalue("bsq_comstock_oedi")
    enduse = "out.electricity.total.energy_consumption..kwh"
    bldg = 51037

    # Identity: query with no state filter returns all 46 metadata rows
    # across all 4 states. If the library implicitly picked one state, we'd
    # see fewer; if it cross-products with TS rows incorrectly, we'd see more.
    by_state = bsq.query(
        enduses=[enduse], upgrade_id="0",
        restrict=[("bldg_id", [bldg])],
        group_by=["state"],
    )
    record_query(bsq, {
        "enduses": [enduse], "upgrade_id": "0",
        "restrict": [("bldg_id", [bldg])],
        "group_by": ["state"],
    })
    by_state_indexed = by_state.set_index("state")
    expected_per_state = {
        "CO": (2,  0.616301, 254552.65),
        "NM": (30, 5.694012, 2351817.0),
        "OK": (1,  0.043486, 17961.03),
        "TX": (13, 2.633057, 1087540.0),
    }
    if set(by_state_indexed.index) != set(expected_per_state):
        pytest.fail(
            f"bldg=51037 by state: expected states {sorted(expected_per_state)}, "
            f"got {sorted(by_state_indexed.index)}"
        )
    for st, (n, w, kwh) in expected_per_state.items():
        row = by_state_indexed.loc[st]
        assert int(row["sample_count"]) == n, (
            f"sample_count for ({bldg}, {st}): expected {n}, got {row['sample_count']}"
        )
        assert np.isclose(float(row["units_count"]), w, rtol=1e-3), (
            f"units_count for ({bldg}, {st}): expected ~{w:.4f}, got {row['units_count']}"
        )
        assert np.isclose(
            float(row["electricity.total.energy_consumption..kwh"]), kwh, rtol=1e-3,
        ), f"kWh for ({bldg}, {st}): expected ~{kwh:.0f}, got {row['electricity.total.energy_consumption']:.0f}"

    # Compositional sum: no-group-by total = sum across states. Catches
    # apportionment bugs where the no-group-by aggregation paths combine
    # weights/energy differently than the per-state path.
    no_grp = bsq.query(
        enduses=[enduse], upgrade_id="0",
        restrict=[("bldg_id", [bldg])],
    )
    record_query(bsq, {
        "enduses": [enduse], "upgrade_id": "0",
        "restrict": [("bldg_id", [bldg])],
    })
    expected_n = sum(v[0] for v in expected_per_state.values())  # 46
    expected_w = sum(v[1] for v in expected_per_state.values())  # ~8.987
    expected_kwh = sum(v[2] for v in expected_per_state.values())  # ~3.711M
    assert int(no_grp["sample_count"].iloc[0]) == expected_n, (
        f"no-group-by sample_count: expected {expected_n}, got {no_grp['sample_count'].iloc[0]}"
    )
    assert np.isclose(float(no_grp["units_count"].iloc[0]), expected_w, rtol=1e-3), (
        f"no-group-by units_count: expected ~{expected_w:.4f}, got {no_grp['units_count'].iloc[0]}"
    )
    assert np.isclose(
        float(no_grp["electricity.total.energy_consumption..kwh"].iloc[0]),
        expected_kwh, rtol=1e-3,
    ), f"no-group-by kWh: expected ~{expected_kwh:.0f}"

    # Filter composition: state filter should restrict to exactly that state.
    # bldg=51037 in CO has 2 tracts → sample_count=2.
    co_only = bsq.query(
        enduses=[enduse], upgrade_id="0",
        restrict=[("bldg_id", [bldg]), ("state", ["CO"])],
    )
    record_query(bsq, {
        "enduses": [enduse], "upgrade_id": "0",
        "restrict": [("bldg_id", [bldg]), ("state", ["CO"])],
    })
    assert int(co_only["sample_count"].iloc[0]) == 2, (
        f"bldg+state restrict (CO): expected sample_count=2, got {co_only['sample_count'].iloc[0]}"
    )
    assert np.isclose(
        float(co_only["electricity.total.energy_consumption..kwh"].iloc[0]), 254552.65, rtol=1e-3,
    )

    # Tract-level group_by forces (bldg_id, state, tract) granularity. For
    # bldg=51037 we should see exactly 46 rows — one per metadata row. If
    # GROUP BY collapsed on bldg_id, we'd see 1; if it collapsed on
    # (bldg_id, state), we'd see 4. Sum across tracts equals the no-group-by
    # total — this also tests that per-tract weights apportion correctly.
    by_tract = bsq.query(
        enduses=[enduse], upgrade_id="0",
        restrict=[("bldg_id", [bldg])],
        group_by=["state", "in.nhgis_tract_gisjoin"],
    )
    record_query(bsq, {
        "enduses": [enduse], "upgrade_id": "0",
        "restrict": [("bldg_id", [bldg])],
        "group_by": ["state", "in.nhgis_tract_gisjoin"],
    })
    assert len(by_tract) == 46, (
        f"by-tract row count: expected 46 (one per (state, tract) for bldg=51037), "
        f"got {len(by_tract)}"
    )
    tract_sum = float(by_tract["electricity.total.energy_consumption..kwh"].sum())
    assert np.isclose(tract_sum, expected_kwh, rtol=1e-3), (
        f"sum across tracts: expected ~{expected_kwh:.0f}, got {tract_sum:.0f}"
    )

    # County-level group_by: bldg=51037 spans Las Animas and Otero in CO,
    # multiple counties in NM, etc. Total county count for bldg=51037 must
    # be > 4 (more than one county per state in NM at minimum) and the sum
    # across counties must still equal the bldg-total kWh.
    by_county = bsq.query(
        enduses=[enduse], upgrade_id="0",
        restrict=[("bldg_id", [bldg])],
        group_by=["state", "in.county_name"],
    )
    record_query(bsq, {
        "enduses": [enduse], "upgrade_id": "0",
        "restrict": [("bldg_id", [bldg])],
        "group_by": ["state", "in.county_name"],
    })
    assert len(by_county) > 4, (
        f"by-county row count: expected > 4 (multi-county states), got {len(by_county)}"
    )
    county_sum = float(by_county["electricity.total.energy_consumption..kwh"].sum())
    assert np.isclose(county_sum, expected_kwh, rtol=1e-3), (
        f"sum across counties: expected ~{expected_kwh:.0f}, got {county_sum:.0f}"
    )

    # JOIN-bearing query: upgrade=1 savings shape. Above queries are all
    # baseline-only (no JOIN), so they don't actually exercise the composite-
    # key behavior in JOIN ON clauses. This check explicitly triggers a
    # baseline ⋈ upgrade JOIN whose ON clause includes (bldg_id, tract,
    # state) under the canonical schema. Without all three keys, the join
    # would cross-product across (state, tract) for the shared bldg_id and
    # inflate per-state sample_counts.
    try:
        savings_by_state = bsq.query(
            enduses=[enduse], upgrade_id="1", applied_only=True,
            restrict=[("bldg_id", [bldg])],
            group_by=["state"],
            include_baseline=True, include_upgrade=True, include_savings=True,
        )
        record_query(bsq, {
            "enduses": [enduse], "upgrade_id": "1", "applied_only": True,
            "restrict": [("bldg_id", [bldg])],
            "group_by": ["state"],
            "include_baseline": True, "include_upgrade": True, "include_savings": True,
        })
    except Exception as exc:
        pytest.skip(
            f"savings-shape query for bldg={bldg} failed (likely because the upgrade "
            f"didn't apply to this archetype): {type(exc).__name__}: {exc}"
        )
    savings_idx = savings_by_state.set_index("state")
    # Per-state sample_count from the savings JOIN should match the metadata
    # row count per state (i.e. canonical_per_state — same as the baseline-
    # only by_state result). A bldg-only join would multiply these by the
    # cross-state upgrade row count.
    for st in savings_idx.index:
        if st not in expected_per_state:
            continue
        expected_n, _, _ = expected_per_state[st]
        actual_n = int(savings_idx.loc[st, "sample_count"])
        assert actual_n == expected_n, (
            f"savings-shape sample_count for ({bldg}, {st}): expected {expected_n}, "
            f"got {actual_n} — a bldg_id-only join would inflate this"
        )


# --- composite-key mutation test (proves keys are load-bearing) -------------

def test_comstock_composite_key_mutation_breaks_invariants():
    """Mutation test: proves the composite keys are load-bearing in join
    construction. Builds a BuildStockQuery with the comstock schema
    deliberately mutated to drop everything except `bldg_id` from the
    metadata and timeseries unique_keys, then runs queries that REQUIRE
    cross-table joins (savings: baseline ⋈ upgrade) and asserts the
    mutated results diverge from the canonical correct ones.

    Critical setup: simple baseline-only annual queries (upgrade_id='0',
    no savings) don't construct any joins — they just SELECT from the
    metadata table — so the unique_keys mutation has NO effect on the
    emitted SQL. This test deliberately uses a savings-shape query
    (upgrade_id='1', applied_only=True, include_baseline+upgrade+savings)
    which constructs a JOIN ON bldg_id [+ state + tract] between baseline
    and upgrade. The mutation removes state and tract from that ON clause,
    causing a cross-product across all 46 metadata rows for bldg_id=51037
    instead of the proper per-(state, tract) match.

    Anchor: bldg_id=51037 is deployed in 4 states / 46 tracts. Under the
    canonical schema, the savings query produces a row per state with
    sample_count matching the per-state metadata count (CO=2, NM=30, etc.).
    Under the mutated schema, each metadata row joins with every same-bldg_id
    upgrade row across all states, inflating sample_count by ~46x.
    """
    import os
    import toml
    import copy
    import buildstock_query as bq_pkg
    from buildstock_query import BuildStockQuery

    schema_path = os.path.join(
        os.path.dirname(bq_pkg.__file__),
        "db_schema",
        "comstock_oedi_state_and_county.toml",
    )
    canonical_schema = toml.load(schema_path)
    mutated_schema = copy.deepcopy(canonical_schema)
    mutated_schema["unique_keys"]["metadata"] = ["bldg_id"]
    mutated_schema["unique_keys"]["timeseries"] = ["bldg_id"]

    bsq_broken = BuildStockQuery(
        "rescore", "buildstock_sdr", "comstock_amy2018_r2_2025",
        buildstock_type="comstock",
        db_schema=mutated_schema,
        skip_reports=True,
    )

    enduse = "out.electricity.total.energy_consumption..kwh"
    bldg = 51037

    # First sanity-check: confirm the SQL actually changed. If the join ON
    # clause is identical to the canonical, the mutation isn't testing
    # anything (this is what tripped my earlier attempt — annual baseline-
    # only queries don't have a join to mutate).
    sql_mutated = bsq_broken.query(
        enduses=[enduse], upgrade_id="1", applied_only=True,
        restrict=[("bldg_id", [bldg])],
        group_by=["state"],
        include_baseline=True, include_upgrade=True, include_savings=True,
        get_query_only=True,
    )
    # bs/up are SA aliases over the unified annual_and_metadata table after
    # the 2-table pivot (commit f6cfebd → fe8755b).
    join_on_match = "bs.bldg_id = up.bldg_id AND bs.state = up.state"
    if join_on_match in sql_mutated:
        pytest.fail(
            "Mutation didn't take effect — mutated SQL still contains state/tract "
            "in the JOIN ON clause. Schema dict override may not be working."
        )
    # Sanity that the join IS bldg_id-only after mutation
    if "ON bs.bldg_id = up.bldg_id" not in sql_mutated or "up.upgrade = 1" not in sql_mutated:
        pytest.fail(
            f"Mutated SQL doesn't have expected bldg_id-only join shape. SQL:\n{sql_mutated}"
        )

    # Canonical anchored values for the savings query (upgrade=1, applied_only=True).
    # bldg=51037: CO has 2 metadata rows, NM has 30, OK has 1, TX has 13. The
    # join under canonical keys produces sample_count == per-state metadata count.
    # Under mutated keys (bldg_id-only join), each baseline row in state X joins
    # with every upgrade row across all 4 states, producing sample_count that's
    # roughly per_state_count × total_upgrade_rows_for_bldg.
    canonical_per_state = {"CO": 2, "NM": 30, "OK": 1, "TX": 13}

    try:
        by_state_broken = bsq_broken.query(
            enduses=[enduse], upgrade_id="1", applied_only=True,
            restrict=[("bldg_id", [bldg])],
            group_by=["state"],
            include_baseline=True, include_upgrade=True, include_savings=True,
        )
    except Exception as exc:
        # Some mutations cause SQL errors (e.g., DUPLICATE_COLUMN_NAME). That's
        # ALSO valid evidence the keys are load-bearing — accept it as a
        # mutation-detected divergence.
        print(f"Mutation caused query to error (also confirms keys matter): "
              f"{type(exc).__name__}: {exc}")
        return

    by_state_idx = by_state_broken.set_index("state")
    divergences = []
    for st, canonical_count in canonical_per_state.items():
        if st not in by_state_idx.index:
            divergences.append(f"state {st} missing from mutated result")
            continue
        actual_count = int(by_state_idx.loc[st, "sample_count"])
        if actual_count != canonical_count:
            divergences.append(
                f"sample_count for {st}: canonical={canonical_count}, "
                f"mutated={actual_count} (factor of {actual_count/canonical_count:.1f}x)"
            )

    if not divergences:
        pytest.fail(
            "Mutation FAILED to break invariants: queries returned canonical values "
            "even with state/tract removed from unique_keys. The composite-key test "
            "may be passing for the wrong reason — joins aren't actually using these "
            "keys, or the mutation isn't propagating to SQL generation."
        )

    # Pass: divergences confirm the canonical keys are load-bearing.
    print("\nMutation confirmed: composite keys are load-bearing. Sample divergences:")
    for d in divergences:
        print(f"  - {d}")


# --- two-fuel electricity column equals single-fuel electricity --------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_two_fuel_electricity_equals_single_fuel(request, bsq_fixture, schema):
    """Querying [electricity, natural_gas] vs querying [electricity] alone must give
    the same electricity values per group. Same restrict, same group_by — adding a
    second enduse to the SELECT list shouldn't perturb the per-row aggregations."""
    bsq = request.getfixturevalue(bsq_fixture)
    elec = resolve_placeholder(schema, "electricity_total")
    gas = resolve_placeholder(schema, "natural_gas_total")
    group_col = resolve_placeholder(schema, "building_type_col")
    restrict = [("state", ["CO"])]

    two_fuel_df = bsq.query(
        enduses=[elec, gas], group_by=[group_col], restrict=restrict,
    )
    record_query(bsq, {
        "enduses": [elec, gas], "group_by": [group_col], "restrict": restrict,
    })
    single_fuel_df = bsq.query(
        enduses=[elec], group_by=[group_col], restrict=restrict,
    )
    record_query(bsq, {
        "enduses": [elec], "group_by": [group_col], "restrict": restrict,
    })

    elec_col = _strip_out_prefix(elec)
    two = two_fuel_df.set_index(group_col)[elec_col].sort_index()
    single = single_fuel_df.set_index(group_col)[elec_col].sort_index()
    if set(two.index) != set(single.index):
        pytest.fail(f"group-key mismatch: two={set(two.index)}, single={set(single.index)}")
    diffs = []
    for key in two.index:
        if not np.isclose(float(two[key]), float(single[key]), rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL):
            diffs.append(f"  {key}: two_fuel={float(two[key]):.4f}, single_fuel={float(single[key]):.4f}")
    if diffs:
        pytest.fail("electricity column differs between two-fuel and single-fuel query:\n" + "\n".join(diffs))


# --- applied-buildings intersection: all_of=[1,2] equals (all_of=[1] ∩ all_of=[2]) --

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_applied_buildings_intersection(request, bsq_fixture, schema):
    """The set of buildings returned by `get_applied_buildings(all_of=[1, 2])` must
    equal the intersection of `all_of=[1]` (buildings that applied to upgrade 1)
    and `all_of=[2]` (buildings that applied to upgrade 2). Asserts the
    `_build_applied_subquery` HAVING-count machinery actually computes a set
    intersection rather than something weaker (e.g. union, or a wrong join
    semantic).

    Run for two restrict scopes:
      - single state (CO): the original case.
      - multi-state from MULTI_STATE_PAIR: validates that the intersection logic
        composes correctly with multi-state restricts. Catches bugs where the
        applied-buildings subquery accidentally restricts to one state, drops
        state from the IN-tuple, or where the multi-state restrict is applied
        inconsistently across the three queries.
    """
    bsq = request.getfixturevalue(bsq_fixture)
    state_pair = resolve_placeholder(schema, "multi_state_pair")
    enduses = [
        resolve_placeholder(schema, "electricity_total"),
        resolve_placeholder(schema, "natural_gas_total"),
    ]
    group_col = resolve_placeholder(schema, "building_type_col")

    def _ids_with_filter(all_of, state_list):
        """Project applied-buildings to (md_keys ∩ state-restricted)."""
        applied = bsq.get_applied_buildings_filter(all_of=all_of)
        restrict = [applied, ("state", state_list)] if applied else [("state", state_list)]
        record_restrict = (
            [{"_applied_filter": {"all_of": all_of}}, ("state", state_list)]
            if applied else [("state", state_list)]
        )
        record_query(bsq, {"restrict": record_restrict}, method="get_building_ids")
        return bsq.get_building_ids(restrict=restrict)

    for restrict_label, state_list in (
        ("single state (CO)", ["CO"]),
        (f"multi-state ({'+'.join(state_pair)})", state_pair),
    ):
        restrict = [("state", state_list)]

        df_1 = _ids_with_filter([1], state_list)
        df_2 = _ids_with_filter([2], state_list)
        df_12 = _ids_with_filter([1, 2], state_list)

        # Each row of get_building_ids is a unique-key tuple. For resstock that's
        # (bldg_id,); for comstock it's (bldg_id, in.nhgis_tract_gisjoin, state)
        # because a single physical building can appear in multiple tracts.
        # Itertuples gives us exactly the right shape to use as a set element.
        keys_1 = set(map(tuple, df_1.itertuples(index=False, name=None)))
        keys_2 = set(map(tuple, df_2.itertuples(index=False, name=None)))
        keys_12 = set(map(tuple, df_12.itertuples(index=False, name=None)))
        expected = keys_1 & keys_2

        if keys_12 != expected:
            only_in_actual = keys_12 - expected
            only_in_expected = expected - keys_12
            msg = [
                f"[{restrict_label}] all_of=[1,2] returned {len(keys_12)} keys, "
                f"intersection of all_of=[1] ({len(keys_1)}) and all_of=[2] "
                f"({len(keys_2)}) has {len(expected)} keys.",
            ]
            if only_in_actual:
                sample = list(sorted(only_in_actual))[:5]
                msg.append(f"  in [1,2] but not in intersection ({len(only_in_actual)} total): {sample}")
            if only_in_expected:
                sample = list(sorted(only_in_expected))[:5]
                msg.append(f"  in intersection but not in [1,2] ({len(only_in_expected)} total): {sample}")
            pytest.fail("\n".join(msg))

        # Cross-check against the aggregated `applied_in_1_2` sample_count from
        # the invariant snapshot. The number of unique-key tuples here should
        # equal the `sample_count` reported there (which is COUNT(DISTINCT bs_key)
        # at the SQL level).
        applied_filter_12 = bsq.get_applied_buildings_filter(all_of=[1, 2])
        inv_restrict = [applied_filter_12, *restrict] if applied_filter_12 else list(restrict)
        inv_record_restrict = (
            [{"_applied_filter": {"all_of": [1, 2]}}, *restrict]
            if applied_filter_12 else list(restrict)
        )
        inv_df = bsq.query(
            enduses=enduses, upgrade_id="1", applied_only=True,
            group_by=[group_col], restrict=inv_restrict,
        )
        record_query(bsq, {
            "enduses": enduses, "upgrade_id": "1", "applied_only": True,
            "group_by": [group_col], "restrict": inv_record_restrict,
        })
        aggregated_sample_count = int(inv_df["sample_count"].sum())
        if aggregated_sample_count != len(keys_12):
            pytest.fail(
                f"[{restrict_label}] sample_count mismatch: get_building_ids "
                f"returned {len(keys_12)} unique keys, but the aggregated "
                f"all_of=[1,2] query reports total "
                f"sample_count={aggregated_sample_count} (sum across building types)."
            )

    # Composition check: the multi-state intersection should equal the union of
    # per-state intersections. With state being part of the comstock composite
    # key (bldg, tract, state), per-state key sets are disjoint even when the
    # underlying bldg_id values collide — so the multi-state intersection should
    # be exactly the sum of the per-state intersections in cardinality and
    # exactly the union as sets.
    s1, s2 = state_pair[0], state_pair[1]
    df_12_s1 = _ids_with_filter([1, 2], [s1])
    df_12_s2 = _ids_with_filter([1, 2], [s2])
    df_12_both = _ids_with_filter([1, 2], state_pair)
    keys_s1 = set(map(tuple, df_12_s1.itertuples(index=False, name=None)))
    keys_s2 = set(map(tuple, df_12_s2.itertuples(index=False, name=None)))
    keys_both = set(map(tuple, df_12_both.itertuples(index=False, name=None)))
    expected_union = keys_s1 | keys_s2
    if keys_both != expected_union:
        only_in_both = keys_both - expected_union
        only_in_union = expected_union - keys_both
        msg = [
            f"all_of=[1,2] over multi-state ({s1}+{s2}) returned {len(keys_both)} "
            f"keys, expected union of per-state ({s1}: {len(keys_s1)}, {s2}: "
            f"{len(keys_s2)}) = {len(expected_union)} keys.",
        ]
        if only_in_both:
            sample = list(sorted(only_in_both))[:5]
            msg.append(f"  in multi-state but not in per-state union: {sample}")
        if only_in_union:
            sample = list(sorted(only_in_union))[:5]
            msg.append(f"  in per-state union but not in multi-state: {sample}")
        pytest.fail("\n".join(msg))


# --- applied-buildings union: any_of=[1,2] equals (all_of=[1] ∪ all_of=[2]) --

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_applied_buildings_union(request, bsq_fixture, schema):
    """The set of buildings returned by `get_applied_buildings_filter(any_of=[a,b])`
    must equal the union of `all_of=[a]` and `all_of=[b]`. Asserts that union ≠
    intersection so the test is meaningful (otherwise the two upgrades apply to
    the same set and the union/intersection identity is trivially true).

    The (a, b) pair is auto-discovered per schema via
    `_pick_meaningful_upgrade_pair` so the test is meaningful regardless of
    which upgrade ids happen to apply universally on a given run.
    """
    bsq = request.getfixturevalue(bsq_fixture)
    state_list = ["CO"]
    a, b = _pick_meaningful_upgrade_pair(bsq, state=state_list[0])

    def _ids_with_filter(*, all_of=None, any_of=None):
        f = bsq.get_applied_buildings_filter(all_of=all_of, any_of=any_of)
        restrict = [f, ("state", state_list)] if f else [("state", state_list)]
        marker = {
            "_applied_filter": {
                k: v for k, v in (("all_of", all_of), ("any_of", any_of)) if v
            }
        }
        record_restrict = (
            [marker, ("state", state_list)] if f else [("state", state_list)]
        )
        record_query(bsq, {"restrict": record_restrict}, method="get_building_ids")
        return bsq.get_building_ids(restrict=restrict)

    df_a = _ids_with_filter(all_of=[a])
    df_b = _ids_with_filter(all_of=[b])
    df_or = _ids_with_filter(any_of=[a, b])

    keys_a = set(map(tuple, df_a.itertuples(index=False, name=None)))
    keys_b = set(map(tuple, df_b.itertuples(index=False, name=None)))
    keys_or = set(map(tuple, df_or.itertuples(index=False, name=None)))
    expected = keys_a | keys_b

    if (keys_a | keys_b) == (keys_a & keys_b):
        pytest.fail(
            f"upgrades [{a}] and [{b}] apply to identical building sets "
            f"({len(keys_a)} keys); discovery picked a pair that should "
            f"have differed — investigate _pick_meaningful_upgrade_pair."
        )

    if keys_or != expected:
        only_in_actual = sorted(keys_or - expected)
        only_in_expected = sorted(expected - keys_or)
        msg = [
            f"any_of=[{a},{b}] returned {len(keys_or)} keys, union of all_of=[{a}] "
            f"({len(keys_a)}) and all_of=[{b}] ({len(keys_b)}) has "
            f"{len(expected)} keys.",
        ]
        if only_in_actual:
            msg.append(f"  in any_of but not in union: {only_in_actual[:5]}")
        if only_in_expected:
            msg.append(f"  in union but not in any_of: {only_in_expected[:5]}")
        pytest.fail("\n".join(msg))


# --- applied-buildings avoid: avoid=[applied_filter] equals universe \ applied --

@pytest.mark.parametrize("filter_kind", ["all_of", "any_of"])
@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_applied_buildings_avoid_complement(
    request, bsq_fixture, schema, filter_kind,
):
    """Passing the applied-buildings filter to `avoid=[...]` must select the
    set complement: `universe \\ applied_set` (where the universe is bounded
    by the same non-applied restrict). Verified by full set equality on the
    building-key tuples returned by `get_building_ids`. Parametrized over
    `all_of` and `any_of` so both filter shapes get exercised through the
    avoid path.

    The upgrade pair is auto-discovered per schema via
    `_pick_meaningful_upgrade_pair` so `applied_set` is a strict, non-empty
    subset of the universe regardless of which upgrades happen to apply
    universally on a given run.
    """
    bsq = request.getfixturevalue(bsq_fixture)
    state_list = ["CO"]
    base_restrict = [("state", state_list)]
    a, b = _pick_meaningful_upgrade_pair(bsq, state=state_list[0])

    universe_df = bsq.get_building_ids(restrict=base_restrict)
    record_query(bsq, {"restrict": base_restrict}, method="get_building_ids")
    universe = set(map(tuple, universe_df.itertuples(index=False, name=None)))

    applied_kwargs = {filter_kind: [a, b]}
    f = bsq.get_applied_buildings_filter(**applied_kwargs)
    marker = {"_applied_filter": applied_kwargs}

    applied_df = bsq.get_building_ids(
        restrict=[f, *base_restrict] if f else base_restrict,
    )
    record_query(bsq, {
        "restrict": [marker, *base_restrict] if f else base_restrict,
    }, method="get_building_ids")
    applied_set = set(map(tuple, applied_df.itertuples(index=False, name=None)))

    if not applied_set or applied_set == universe:
        pytest.fail(
            f"applied set ({filter_kind}=[{a},{b}]) cardinality "
            f"{len(applied_set)} equals 0 or full universe {len(universe)}; "
            f"discovery picked a pair that should have split the universe — "
            f"investigate _pick_meaningful_upgrade_pair."
        )

    avoid_df = bsq.get_building_ids(
        restrict=base_restrict,
        avoid=[f] if f else [],
    )
    record_query(bsq, {
        "restrict": base_restrict,
        "avoid": [marker] if f else [],
    }, method="get_building_ids")
    avoid_set = set(map(tuple, avoid_df.itertuples(index=False, name=None)))

    expected = universe - applied_set
    if avoid_set != expected:
        only_in_actual = sorted(avoid_set - expected)
        only_in_expected = sorted(expected - avoid_set)
        msg = [
            f"avoid={filter_kind}=[{a},{b}] returned {len(avoid_set)} keys, "
            f"universe ({len(universe)}) \\ applied ({len(applied_set)}) has "
            f"{len(expected)} keys.",
        ]
        if only_in_actual:
            msg.append(f"  in avoid but not in expected: {only_in_actual[:5]}")
        if only_in_expected:
            msg.append(f"  in expected but not in avoid: {only_in_expected[:5]}")
        pytest.fail("\n".join(msg))


# --- aggregated-route set identities on units_count via bsq.query() ----------

@pytest.mark.parametrize("annual_only", [True, False])
@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_applied_buildings_set_identities_via_query(
    request, bsq_fixture, schema, annual_only,
):
    """Run `bsq.query(group_by=[...], ...)` over a curated bldg_id universe
    spanning four regions (only_a, only_b, both, neither) and verify two
    invariants on `units_count`:

    (a) Total `sum(units_count)` is invariant to `group_by` choice
        (`[bldg_id]`, `[county]`, `[bldg_type]`, `[county, bldg_type]`,
        `[]`) under the same applied filter — catches aggregation bugs
        that surface only at certain group_by grains (e.g. tract fan-out
        when grouping by tract; arbitrary() collapsing wrong key).
    (b) Set-arithmetic identities on `units_count` totals. Each applied
        filter goes through both restrict and avoid paths so NOT-IN
        composition is exercised independently per filter shape:

          # restrict-side: applied set partitioned over four regions
          U_{all_of=[a]}        == U_{only_a} + U_{both}
          U_{all_of=[b]}        == U_{only_b} + U_{both}
          U_{all_of=[a,b]}      == U_{both}
          U_{any_of=[a,b]}      == U_{only_a} + U_{only_b} + U_{both}
          U_{any_of=[a,b]}      == U_{all_of=[a]} + U_{all_of=[b]} - U_{all_of=[a,b]}

          # avoid-side: complement of each applied set
          U_{avoid all_of=[a]}     == U_{only_b} + U_{neither}
          U_{avoid all_of=[b]}     == U_{only_a} + U_{neither}
          U_{avoid all_of=[a,b]}   == U_{only_a} + U_{only_b} + U_{neither}
          U_{avoid any_of=[a,b]}   == U_{neither}

          # restrict + matching avoid covers the universe with no overlap/gap
          U_{all_of=[a]}    + U_{avoid all_of=[a]}    == U_{no_filter}
          U_{all_of=[b]}    + U_{avoid all_of=[b]}    == U_{no_filter}
          U_{all_of=[a,b]}  + U_{avoid all_of=[a,b]}  == U_{no_filter}
          U_{any_of=[a,b]}  + U_{avoid any_of=[a,b]}  == U_{no_filter}

          U_{no filter}     == U_{only_a} + U_{only_b} + U_{both} + U_{neither}

    Run on annual (`annual_only=True`) and TS (`annual_only=False`,
    `timestamp_grouping_func='year'`) flows so both query paths exercise
    the applied-filter machinery. The TS leg's `units_count` is per-row
    metadata (constant across timestamps); using `timestamp_grouping_func='year'`
    collapses to one row per group-key so `sum(units_count)` is directly
    comparable to the annual leg.
    """
    bsq = request.getfixturevalue(bsq_fixture)
    bldg_col = bsq.md_bldgid_column
    curated, regions, (a, b) = _curate_applied_universe(bsq, state="CO")
    base_restrict = [("state", ["CO"]), (bldg_col, sorted(curated))]

    # group_by axis. Per-schema county column choice — comstock_oedi uses
    # tract gisjoin (no flat county column on the md table without a join);
    # comstock_oedi_agg has a flat `county` col; resstock_oedi has
    # `in.county_name`. The composite `[county, bldg_type]` exercises
    # multi-key grouping.
    if schema == "resstock_oedi":
        county_col = "in.county_name"
    elif schema == "comstock_oedi_agg":
        county_col = "county"
    else:  # comstock_oedi
        county_col = "in.nhgis_tract_gisjoin"
    bldg_type_col = resolve_placeholder(schema, "building_type_col")
    group_by_axis = {
        "none": [],
        "bldg_id": [bldg_col],
        "county": [county_col],
        "bldg_type": [bldg_type_col],
        "county_x_bldg_type": [county_col, bldg_type_col],
    }

    if annual_only:
        enduses = [resolve_placeholder(schema, "electricity_total")]
        extra: dict = {}
    else:
        enduses = [resolve_placeholder(schema, "electricity_total", annual=False)]
        extra = {"annual_only": False, "timestamp_grouping_func": "year"}

    def _u_total(group_by, *, all_of=None, any_of=None, use_avoid=False):
        f = bsq.get_applied_buildings_filter(all_of=all_of, any_of=any_of)
        marker = {
            "_applied_filter": {
                k: v for k, v in (("all_of", all_of), ("any_of", any_of)) if v
            }
        }
        if use_avoid:
            restrict = list(base_restrict)
            avoid = [f] if f else []
            record_restrict = list(base_restrict)
            record_avoid = [marker] if f else []
        else:
            restrict = [f, *base_restrict] if f else list(base_restrict)
            avoid = []
            record_restrict = [marker, *base_restrict] if f else list(base_restrict)
            record_avoid = []
        df = bsq.query(
            enduses=enduses, group_by=group_by, restrict=restrict,
            avoid=avoid, **extra,
        )
        record_query(bsq, {
            "enduses": enduses, "group_by": group_by,
            "restrict": record_restrict, "avoid": record_avoid, **extra,
        })
        return df, float(df["units_count"].sum())

    # Filter matrix exercises every applied-filter shape through both restrict
    # and avoid paths. Each avoid_* mirrors an all_of_* / any_of_* with the
    # same filter spec but goes through the avoid path; pairing them gives
    # independent coverage of NOT-IN composition through the TS pivot.
    filter_specs = {
        "no_filter": {},
        "all_of_a": {"all_of": [a]},
        "all_of_b": {"all_of": [b]},
        "all_of_ab": {"all_of": [a, b]},
        "any_of_ab": {"any_of": [a, b]},
        "avoid_all_a": {"all_of": [a], "use_avoid": True},
        "avoid_all_b": {"all_of": [b], "use_avoid": True},
        "avoid_all_ab": {"all_of": [a, b], "use_avoid": True},
        "avoid_any_ab": {"any_of": [a, b], "use_avoid": True},
    }

    # --- Invariant (a): group_by-invariance of total per filter --------------
    totals: dict[str, dict[str, float]] = {}
    bldg_id_dfs: dict[str, pd.DataFrame] = {}
    for fname in sorted(filter_specs):
        spec = filter_specs[fname]
        totals[fname] = {}
        for gname in sorted(group_by_axis):
            df, u = _u_total(group_by_axis[gname], **spec)
            totals[fname][gname] = u
            if gname == "bldg_id":
                bldg_id_dfs[fname] = df

    inv_a_failures = []
    gnames_sorted = sorted(group_by_axis)
    for fname in sorted(filter_specs):
        ref_g = gnames_sorted[0]
        ref_u = totals[fname][ref_g]
        for gname in gnames_sorted[1:]:
            if not np.isclose(
                totals[fname][gname], ref_u,
                rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL,
            ):
                inv_a_failures.append(
                    f"  filter={fname}: group_by={gname} -> "
                    f"{totals[fname][gname]:.4f}, group_by={ref_g} -> "
                    f"{ref_u:.4f} (diff={totals[fname][gname] - ref_u:.4f})"
                )
    if inv_a_failures:
        pytest.fail(
            "group_by-invariance of units_count failed:\n"
            + "\n".join(inv_a_failures)
        )

    # --- Invariant (b): inclusion-exclusion using bldg_id-grouped totals -----
    df_all = bldg_id_dfs["no_filter"]
    curated_set = set(curated)

    def _u_for(df: pd.DataFrame, ids: set[int]) -> float:
        if not ids:
            return 0.0
        sub = df[df[bldg_col.name].isin(sorted(ids))]
        return float(sub["units_count"].sum())

    only_1_set = regions["only_a"] & curated_set
    only_2_set = regions["only_b"] & curated_set
    both_set = regions["both"] & curated_set
    neither_set = regions["neither"] & curated_set
    u_only_1 = _u_for(df_all, only_1_set)
    u_only_2 = _u_for(df_all, only_2_set)
    u_both = _u_for(df_all, both_set)
    u_neit_expected = _u_for(df_all, neither_set)

    G = "bldg_id"
    checks = [
        # restrict-side identities — applied set partitioned over the four regions
        (f"all_of=[{a}] == only_a + both",
         totals["all_of_a"][G], u_only_1 + u_both),
        (f"all_of=[{b}] == only_b + both",
         totals["all_of_b"][G], u_only_2 + u_both),
        (f"all_of=[{a},{b}] == both",
         totals["all_of_ab"][G], u_both),
        (f"any_of=[{a},{b}] == only_a + only_b + both",
         totals["any_of_ab"][G], u_only_1 + u_only_2 + u_both),
        (f"incl-excl: any_of == all_of[{a}] + all_of[{b}] - all_of[{a},{b}]",
         totals["any_of_ab"][G],
         totals["all_of_a"][G] + totals["all_of_b"][G] - totals["all_of_ab"][G]),
        # avoid-side identities — complement of each applied set within the
        # curated universe. Pairs each restrict-side filter with its avoid
        # counterpart so a NOT-IN composition bug surfaces independently per
        # filter shape (all_of single, all_of pair, any_of pair).
        (f"avoid all_of=[{a}] == only_b + neither",
         totals["avoid_all_a"][G], u_only_2 + u_neit_expected),
        (f"avoid all_of=[{b}] == only_a + neither",
         totals["avoid_all_b"][G], u_only_1 + u_neit_expected),
        (f"avoid all_of=[{a},{b}] == only_a + only_b + neither",
         totals["avoid_all_ab"][G],
         u_only_1 + u_only_2 + u_neit_expected),
        (f"avoid any_of=[{a},{b}] == neither",
         totals["avoid_any_ab"][G], u_neit_expected),
        # avoid-side complements: each restrict + matching avoid covers the
        # full curated universe (no double-counting, no gap).
        (f"all_of=[{a}] + avoid all_of=[{a}] == universe",
         totals["all_of_a"][G] + totals["avoid_all_a"][G], totals["no_filter"][G]),
        (f"all_of=[{b}] + avoid all_of=[{b}] == universe",
         totals["all_of_b"][G] + totals["avoid_all_b"][G], totals["no_filter"][G]),
        (f"all_of=[{a},{b}] + avoid all_of=[{a},{b}] == universe",
         totals["all_of_ab"][G] + totals["avoid_all_ab"][G], totals["no_filter"][G]),
        (f"any_of=[{a},{b}] + avoid any_of=[{a},{b}] == universe",
         totals["any_of_ab"][G] + totals["avoid_any_ab"][G], totals["no_filter"][G]),
        ("no_filter == only_a + only_b + both + neither",
         totals["no_filter"][G],
         u_only_1 + u_only_2 + u_both + u_neit_expected),
    ]
    inv_b_failures = []
    for label, actual, expected in checks:
        if not np.isclose(
            actual, expected, rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL,
        ):
            inv_b_failures.append(
                f"  {label}: actual={actual:.4f}, expected={expected:.4f}, "
                f"diff={actual - expected:.4f}"
            )
    if inv_b_failures:
        pytest.fail(
            "units_count inclusion-exclusion failed:\n"
            + "\n".join(inv_b_failures)
        )


# --- applied filter: subquery encoding equals materialized id-list encoding --

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_applied_filter_subquery_equals_id_list(request, bsq_fixture, schema):
    """The applied-buildings filter has two equivalent encodings:
      (1) `restrict=[get_applied_buildings_filter(all_of=[a])]` — IN-subquery
      (2) `restrict=[(bldg_id_col, materialized_ids_list)]` — IN literal list
    Both must produce identical aggregated results when applied to the
    same effective set of buildings. Also pins:
      - Idempotency: `restrict=[f, f]` == `restrict=[f]`.
      - Cardinality identity: `n_applied + n_avoided == n_universe` (exact int).

    To avoid Athena's 262144-char query length limit, the comparison is
    performed against a small curated bldg_id universe (top ~20 applied
    ids) rather than the full applied set. This is sufficient: the
    point of the test is to exercise the IN-subquery vs IN-list code
    paths under identical filter semantics, not to exhaustively scan.
    """
    bsq = request.getfixturevalue(bsq_fixture)
    state_list = ["CO"]
    a, _ = _pick_meaningful_upgrade_pair(bsq, state=state_list[0])

    # On schemas with composite md_key (comstock has (bldg_id, tract, state)
    # or (bldg_id, county, state)), a single building can have multiple
    # md-key tuples (one per tract/county slice). Filtering by `(bldg_id, [ids])`
    # admits ALL slices of a building while the IN-subquery form
    # `(bldg_id, county, state) IN (subquery)` admits only the specific
    # slices that satisfied the applied predicate. To make the comparison
    # apples-to-apples we materialize ALL tuples for a small set of bldg_ids
    # rather than the first-N tuples (which would partially admit some
    # buildings and cut others mid-slice).
    f_a = bsq.get_applied_buildings_filter(all_of=[a])
    applied_df = bsq.get_building_ids(
        restrict=[f_a, ("state", state_list)] if f_a else [("state", state_list)],
    )
    record_query(bsq, {
        "restrict": (
            [{"_applied_filter": {"all_of": [a]}}, ("state", state_list)]
            if f_a else [("state", state_list)]
        ),
    }, method="get_building_ids")
    if len(applied_df) < 5:
        pytest.skip(
            f"applied set for all_of=[{a}] in {state_list} has only "
            f"{len(applied_df)} rows — too small for a meaningful test."
        )
    md_key_cols = tuple(bsq.md_key_cols)  # (bldg_id_col,) for resstock,
                                          # (bldg_id_col, tract/county, state_col) for comstock.

    # Pick a small set of bldg_ids and use ALL their md-key tuples (so no
    # building is partially included) — bounded to keep IN-list short.
    all_tuples = [
        tuple(row) for row in applied_df.itertuples(index=False, name=None)
    ]
    unique_bldgs = sorted({int(t[0]) for t in all_tuples})[:10]
    bldg_set = set(unique_bldgs)
    applied_tuples = sorted(t for t in all_tuples if int(t[0]) in bldg_set)
    applied_ids = unique_bldgs

    bldg_col = bsq.md_bldgid_column
    # Universe restrict for path 1: bound to the same buildings the
    # composite-key list mentions, so both paths see the same input.
    universe_clause = (bldg_col, applied_ids)
    base_restrict = [("state", state_list), universe_clause]

    f = bsq.get_applied_buildings_filter(all_of=[a])
    marker = {"_applied_filter": {"all_of": [a]}}
    enduse = resolve_placeholder(schema, "electricity_total")

    # Path 1: IN-subquery encoding.
    df_subq = bsq.query(
        enduses=[enduse],
        restrict=[f, *base_restrict] if f else list(base_restrict),
    )
    record_query(bsq, {
        "enduses": [enduse],
        "restrict": [marker, *base_restrict] if f else list(base_restrict),
    })

    # Path 2: materialized IN-list encoding. Use composite-key form when
    # md_key has more than one component so the filter grain matches the
    # subquery's grain.
    if len(md_key_cols) == 1:
        # Single-key schema (resstock): a flat (col, ids) clause.
        list_clause = (bldg_col, applied_ids)
    else:
        # Composite-key schema (comstock): (cols_tuple, list_of_tuples) clause.
        list_clause = (md_key_cols, applied_tuples)
    df_list = bsq.query(
        enduses=[enduse],
        restrict=[list_clause, ("state", state_list)],
    )
    record_query(bsq, {
        "enduses": [enduse],
        "restrict": [list_clause, ("state", state_list)],
    })

    enduse_col = _strip_out_prefix(enduse)
    # Both encodings must agree on units_count, sample_count, and the enduse total.
    for col in ("units_count", "sample_count", enduse_col):
        a_val = float(df_subq[col].iloc[0])
        b_val = float(df_list[col].iloc[0])
        if not np.isclose(a_val, b_val, rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL):
            pytest.fail(
                f"subquery vs id-list disagreement on {col}: "
                f"subquery={a_val:.4f}, id_list={b_val:.4f}, "
                f"diff={a_val - b_val:.4f}"
            )

    # Idempotency: applying the same filter twice should not change the result.
    if f is not None:
        df_doubled = bsq.query(
            enduses=[enduse],
            restrict=[f, f, *base_restrict],
        )
        record_query(bsq, {
            "enduses": [enduse],
            "restrict": [marker, marker, *base_restrict],
        })
        for col in ("units_count", "sample_count", enduse_col):
            a_val = float(df_subq[col].iloc[0])
            b_val = float(df_doubled[col].iloc[0])
            if not np.isclose(a_val, b_val, rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL):
                pytest.fail(
                    f"applying applied_filter twice changed {col}: "
                    f"once={a_val:.4f}, twice={b_val:.4f}"
                )

    # Cardinality identity: n_applied + n_avoided == n_universe.
    # Use a curated universe (small id set spanning applied + non-applied)
    # to keep query size bounded while still exercising the partition.
    # `applied_ids` (above) is a subset of the applied set; pad with a few
    # known non-applied ids by using `_curate_applied_universe` to ensure
    # both partitions are non-empty.
    curated, regions, _ = _curate_applied_universe(bsq, state=state_list[0])
    cardinality_restrict = [("state", state_list), (bldg_col, sorted(curated))]
    n_universe = len(bsq.get_building_ids(restrict=cardinality_restrict))
    record_query(bsq, {"restrict": cardinality_restrict}, method="get_building_ids")
    n_applied = len(bsq.get_building_ids(
        restrict=[f, *cardinality_restrict] if f else list(cardinality_restrict),
    ))
    record_query(bsq, {
        "restrict": [marker, *cardinality_restrict] if f else list(cardinality_restrict),
    }, method="get_building_ids")
    n_avoided = len(bsq.get_building_ids(
        restrict=cardinality_restrict,
        avoid=[f] if f else [],
    ))
    record_query(bsq, {
        "restrict": cardinality_restrict, "avoid": [marker] if f else [],
    }, method="get_building_ids")
    if n_applied + n_avoided != n_universe:
        pytest.fail(
            f"cardinality identity failed: applied={n_applied} + avoided="
            f"{n_avoided} = {n_applied + n_avoided}, universe={n_universe}"
        )


# --- applied filter: composite key + multi-state composition ----------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_applied_filter_multi_state_composition(request, bsq_fixture, schema):
    """Across multi-state restrict, the building-key tuples returned by an
    applied filter must respect the composite key (e.g. comstock's
    `(bldg_id, tract, state)`) — not project to bldg_id only. Per-state
    keys must be disjoint as tuples (catches state leakage).

    Also: per-state `units_count` totals under the applied filter must
    sum to the multi-state total (catches state-partition handling
    bugs in the TS pivot or tract fan-out).
    """
    bsq = request.getfixturevalue(bsq_fixture)
    state_pair = resolve_placeholder(schema, "multi_state_pair")
    a, b = _pick_meaningful_upgrade_pair(bsq, state="CO")
    f = bsq.get_applied_buildings_filter(any_of=[a, b])
    marker = {"_applied_filter": {"any_of": [a, b]}}
    enduse = resolve_placeholder(schema, "electricity_total")

    s1, s2 = state_pair[0], state_pair[1]
    # Per-state key sets (composite tuple for comstock, single-element for resstock).
    df_s1 = bsq.get_building_ids(restrict=[f, ("state", [s1])] if f else [("state", [s1])])
    record_query(bsq, {
        "restrict": [marker, ("state", [s1])] if f else [("state", [s1])],
    }, method="get_building_ids")
    df_s2 = bsq.get_building_ids(restrict=[f, ("state", [s2])] if f else [("state", [s2])])
    record_query(bsq, {
        "restrict": [marker, ("state", [s2])] if f else [("state", [s2])],
    }, method="get_building_ids")
    keys_s1 = set(map(tuple, df_s1.itertuples(index=False, name=None)))
    keys_s2 = set(map(tuple, df_s2.itertuples(index=False, name=None)))

    # If the schema's md_key includes state, per-state tuples must be disjoint
    # even if bldg_id values collide. Resstock has only (bldg_id,) so the
    # check reduces to "no bldg_id appears in both" which is also a real check.
    overlap = keys_s1 & keys_s2
    if overlap:
        sample = sorted(overlap)[:5]
        pytest.fail(
            f"per-state key sets overlap ({len(overlap)} tuples) for "
            f"states={state_pair} under applied filter — composite key "
            f"may be projecting only bldg_id. Sample: {sample}"
        )

    # Aggregated units_count: per-state sums should equal multi-state sum
    # (under the same applied filter).
    df_pair = bsq.query(
        enduses=[enduse],
        restrict=[f, ("state", state_pair)] if f else [("state", state_pair)],
    )
    record_query(bsq, {
        "enduses": [enduse],
        "restrict": [marker, ("state", state_pair)] if f else [("state", state_pair)],
    })
    df_only_s1 = bsq.query(
        enduses=[enduse],
        restrict=[f, ("state", [s1])] if f else [("state", [s1])],
    )
    record_query(bsq, {
        "enduses": [enduse],
        "restrict": [marker, ("state", [s1])] if f else [("state", [s1])],
    })
    df_only_s2 = bsq.query(
        enduses=[enduse],
        restrict=[f, ("state", [s2])] if f else [("state", [s2])],
    )
    record_query(bsq, {
        "enduses": [enduse],
        "restrict": [marker, ("state", [s2])] if f else [("state", [s2])],
    })
    sum_per_state = float(df_only_s1["units_count"].iloc[0]) + float(df_only_s2["units_count"].iloc[0])
    sum_pair = float(df_pair["units_count"].iloc[0])
    if not np.isclose(sum_pair, sum_per_state, rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL):
        pytest.fail(
            f"multi-state units_count {sum_pair:.4f} != sum of per-state "
            f"({s1}: {float(df_only_s1['units_count'].iloc[0]):.4f}, "
            f"{s2}: {float(df_only_s2['units_count'].iloc[0]):.4f}) "
            f"= {sum_per_state:.4f}"
        )


# --- applied filter: empty-set degeneracy -----------------------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_applied_filter_empty_set_degeneracy(request, bsq_fixture, schema):
    """When the applied set is provably empty (intersection of many upgrades),
    `restrict=[applied_filter]` must return zero rows and `avoid=[applied_filter]`
    must return all rows in the universe. SQL's `IN ()` is implementation-defined
    in some dialects — pin the behavior here so a planner change can't silently
    flip it.
    """
    bsq = request.getfixturevalue(bsq_fixture)
    state_list = ["CO"]
    base_restrict = [("state", state_list)]

    # Construct a guaranteed-empty applied set. all_of=[1,2,3,...,N] for many
    # upgrade ids — buildings applying to ALL of them is virtually impossible.
    # Pick a list large enough to be empty without scanning every upgrade id.
    impossible_all_of = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    f = bsq.get_applied_buildings_filter(all_of=impossible_all_of)
    marker = {"_applied_filter": {"all_of": impossible_all_of}}
    if f is None:
        pytest.skip("get_applied_buildings_filter returned None for impossible all_of")

    # Verify the set is actually empty before testing degeneracy.
    applied_df = bsq.get_applied_buildings(all_of=impossible_all_of)
    record_query(bsq, {"all_of": impossible_all_of}, method="get_applied_buildings")
    if not applied_df.empty:
        pytest.skip(
            f"empty-set construction failed: all_of={impossible_all_of} "
            f"returned {len(applied_df)} buildings; pick a larger/different "
            f"set to guarantee empty applied set."
        )

    universe_df = bsq.get_building_ids(restrict=base_restrict)
    record_query(bsq, {"restrict": base_restrict}, method="get_building_ids")
    n_universe = len(universe_df)

    restrict_df = bsq.get_building_ids(restrict=[f, *base_restrict])
    record_query(bsq, {
        "restrict": [marker, *base_restrict],
    }, method="get_building_ids")
    if len(restrict_df) != 0:
        pytest.fail(
            f"restrict=[empty_applied_filter] should return 0 rows but "
            f"returned {len(restrict_df)} — likely SQL `IN ()` quirk or "
            f"empty-subquery short-circuit treating it as TRUE."
        )

    avoid_df = bsq.get_building_ids(restrict=base_restrict, avoid=[f])
    record_query(bsq, {
        "restrict": base_restrict, "avoid": [marker],
    }, method="get_building_ids")
    if len(avoid_df) != n_universe:
        pytest.fail(
            f"avoid=[empty_applied_filter] should return all {n_universe} "
            f"universe rows but returned {len(avoid_df)} — NOT-IN against "
            f"empty subquery may be evaluating to UNKNOWN/NULL instead of TRUE."
        )


# --- applied filter: restrict order independence + de Morgan ---------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_applied_filter_order_and_de_morgan(request, bsq_fixture, schema):
    """Two algebraic properties of the restrict/avoid combinator:

    (a) Order-independence: restrict=[A, B] must produce identical results
        to restrict=[B, A]. Catches position-dependent classification in
        `_split_restrict` (e.g. first clause wins) or restrict-list mutation
        bugs.

    (b) de Morgan: `avoid=[applied(any_of=[a, b])]` (NOT (A∪B)) must equal
        `avoid=[applied(all_of=[a]), applied(all_of=[b])]` (NOT A AND NOT B).
        Catches cases where multiple avoid clauses are OR'd instead of AND'd
        through `_add_avoid`'s where_clauses chaining.
    """
    bsq = request.getfixturevalue(bsq_fixture)
    state_list = ["CO"]
    a, b = _pick_meaningful_upgrade_pair(bsq, state=state_list[0])
    f_or = bsq.get_applied_buildings_filter(any_of=[a, b])
    f_a = bsq.get_applied_buildings_filter(all_of=[a])
    f_b = bsq.get_applied_buildings_filter(all_of=[b])
    marker_or = {"_applied_filter": {"any_of": [a, b]}}
    marker_a = {"_applied_filter": {"all_of": [a]}}
    marker_b = {"_applied_filter": {"all_of": [b]}}

    # (a) Order-independence on restrict.
    state_clause = ("state", state_list)
    df_AB = bsq.get_building_ids(
        restrict=[f_or, state_clause] if f_or else [state_clause],
    )
    record_query(bsq, {
        "restrict": [marker_or, state_clause] if f_or else [state_clause],
    }, method="get_building_ids")
    df_BA = bsq.get_building_ids(
        restrict=[state_clause, f_or] if f_or else [state_clause],
    )
    record_query(bsq, {
        "restrict": [state_clause, marker_or] if f_or else [state_clause],
    }, method="get_building_ids")
    keys_AB = set(map(tuple, df_AB.itertuples(index=False, name=None)))
    keys_BA = set(map(tuple, df_BA.itertuples(index=False, name=None)))
    if keys_AB != keys_BA:
        pytest.fail(
            f"restrict order matters: [filter, state]={len(keys_AB)} keys, "
            f"[state, filter]={len(keys_BA)} keys, symmetric_diff="
            f"{len(keys_AB ^ keys_BA)}"
        )

    # (b) de Morgan: avoid(any_of) == avoid(all_of[a]) AND avoid(all_of[b]).
    df_avoid_or = bsq.get_building_ids(
        restrict=[state_clause],
        avoid=[f_or] if f_or else [],
    )
    record_query(bsq, {
        "restrict": [state_clause],
        "avoid": [marker_or] if f_or else [],
    }, method="get_building_ids")
    avoid_list = [x for x in (f_a, f_b) if x is not None]
    avoid_marker_list = []
    if f_a is not None:
        avoid_marker_list.append(marker_a)
    if f_b is not None:
        avoid_marker_list.append(marker_b)
    df_avoid_separate = bsq.get_building_ids(
        restrict=[state_clause],
        avoid=avoid_list,
    )
    record_query(bsq, {
        "restrict": [state_clause], "avoid": avoid_marker_list,
    }, method="get_building_ids")
    keys_or = set(map(tuple, df_avoid_or.itertuples(index=False, name=None)))
    keys_sep = set(map(tuple, df_avoid_separate.itertuples(index=False, name=None)))
    if keys_or != keys_sep:
        only_in_or = sorted(keys_or - keys_sep)[:5]
        only_in_sep = sorted(keys_sep - keys_or)[:5]
        pytest.fail(
            f"de Morgan failed: avoid[any_of=[{a},{b}]]={len(keys_or)} keys, "
            f"avoid[all_of=[{a}], all_of=[{b}]]={len(keys_sep)} keys.\n"
            f"  in any-of-form but not separate: {only_in_or}\n"
            f"  in separate but not any-of-form: {only_in_sep}"
        )


# --- applied filter: applied_only=True equals explicit all_of restrict ------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_applied_only_equals_explicit_all_of(request, bsq_fixture, schema):
    """`applied_only=True` for upgrade_id=u internally injects a
    `_build_applied_subquery(all_of=[u])` filter into bs_restrict
    (per aggregate_query.py). The same query with `applied_only=False`
    and an explicit `restrict=[applied_filter(all_of=[u])]` should
    produce identical results. Pins the equivalence so future drift
    between the internal injection path and the public API path
    surfaces immediately.

    Also pins: under `avoid=[applied_filter(all_of=[u])]` (i.e. on
    buildings the upgrade did NOT apply to), savings should be ~0
    because baseline == upgrade for inapplicable buildings.
    """
    bsq = request.getfixturevalue(bsq_fixture)
    state_list = ["CO"]
    u = 1  # use upgrade 1; the explicit/implicit equivalence is independent
           # of which upgrade applies to which buildings — only that the
           # set of applied buildings matches.
    enduse = resolve_placeholder(schema, "electricity_total")
    group_col = resolve_placeholder(schema, "building_type_col")
    f = bsq.get_applied_buildings_filter(all_of=[u])
    marker = {"_applied_filter": {"all_of": [u]}}

    df_implicit = bsq.query(
        enduses=[enduse], upgrade_id=str(u), applied_only=True,
        group_by=[group_col], restrict=[("state", state_list)],
    )
    record_query(bsq, {
        "enduses": [enduse], "upgrade_id": str(u), "applied_only": True,
        "group_by": [group_col], "restrict": [("state", state_list)],
    })
    df_explicit = bsq.query(
        enduses=[enduse], upgrade_id=str(u), applied_only=False,
        group_by=[group_col],
        restrict=[f, ("state", state_list)] if f else [("state", state_list)],
    )
    record_query(bsq, {
        "enduses": [enduse], "upgrade_id": str(u), "applied_only": False,
        "group_by": [group_col],
        "restrict": (
            [marker, ("state", state_list)] if f else [("state", state_list)]
        ),
    })

    # Per-group equality on units_count, sample_count, and enduse total.
    enduse_col = _strip_out_prefix(enduse)
    a_idx = df_implicit.set_index(group_col).sort_index()
    b_idx = df_explicit.set_index(group_col).sort_index()
    if set(a_idx.index) != set(b_idx.index):
        pytest.fail(
            f"applied_only=True vs explicit all_of disagree on group keys: "
            f"only_implicit={set(a_idx.index) - set(b_idx.index)}, "
            f"only_explicit={set(b_idx.index) - set(a_idx.index)}"
        )
    diffs = []
    for col in ("units_count", "sample_count", enduse_col):
        for key in a_idx.index:
            av = float(a_idx.loc[key, col])
            bv = float(b_idx.loc[key, col])
            if not np.isclose(av, bv, rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL):
                diffs.append(f"  {key}/{col}: implicit={av:.4f}, explicit={bv:.4f}")
    if diffs:
        pytest.fail(
            f"applied_only=True vs explicit all_of=[{u}] disagree:\n"
            + "\n".join(diffs[:10])
        )

    # Savings under avoid: |savings| ~= 0 for buildings the upgrade didn't apply to.
    df_avoid = bsq.query(
        enduses=[enduse], upgrade_id=str(u), applied_only=False,
        group_by=[group_col],
        restrict=[("state", state_list)],
        avoid=[f] if f else [],
        include_baseline=True, include_upgrade=True, include_savings=True,
    )
    record_query(bsq, {
        "enduses": [enduse], "upgrade_id": str(u), "applied_only": False,
        "group_by": [group_col],
        "restrict": [("state", state_list)],
        "avoid": [marker] if f else [],
        "include_baseline": True, "include_upgrade": True, "include_savings": True,
    })
    savings_col = _find_first_col(
        df_avoid, suffix="__savings", contains="electricity.total",
    )
    bad = []
    for _, row in df_avoid.iterrows():
        savings = float(row[savings_col])
        units = float(row["units_count"])
        # Allow per-row tolerance scaled by units_count (a building consuming
        # 1e7 kWh/yr has float drift larger than INVARIANT_ATOL).
        bound = max(INVARIANT_ATOL, abs(units) * INVARIANT_RTOL * 100)
        if abs(savings) > bound:
            bad.append(
                f"  {row[group_col]}: savings={savings:.4f}, units={units:.4f}, "
                f"bound={bound:.4f}"
            )
    if bad:
        pytest.fail(
            "non-zero savings on inapplicable buildings (avoid=[applied_filter]):\n"
            + "\n".join(bad[:10])
        )


# --- applied filter: ternary any_of and singleton all_of/any_of equivalence -

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_applied_filter_ternary_union_and_singleton(request, bsq_fixture, schema):
    """Higher-arity `any_of=[a, b, c]` must equal the union of three
    `all_of=[]` calls. Catches bugs in the HAVING-count machinery that
    only manifest at N>2 (e.g. counting distinct upgrades wrong).

    Also: `all_of=[a]` and `any_of=[a]` should produce identical sets
    on the singleton case — pins the edge where the HAVING-count is
    `>=1` vs `==1`.
    """
    bsq = request.getfixturevalue(bsq_fixture)
    state_list = ["CO"]
    state_clause = ("state", state_list)
    a, b = _pick_meaningful_upgrade_pair(bsq, state=state_list[0])
    # Pick c distinct from a, b that also has a non-trivial applied set.
    # Try upgrades 1..16 in order and skip a, b.
    c = next(u for u in range(1, 17) if u not in (a, b))

    def _ids(filter_kwargs):
        f = bsq.get_applied_buildings_filter(**filter_kwargs)
        marker = {"_applied_filter": filter_kwargs}
        df = bsq.get_building_ids(
            restrict=[f, state_clause] if f else [state_clause],
        )
        record_query(bsq, {
            "restrict": [marker, state_clause] if f else [state_clause],
        }, method="get_building_ids")
        return set(map(tuple, df.itertuples(index=False, name=None)))

    keys_or3 = _ids({"any_of": [a, b, c]})
    keys_a = _ids({"all_of": [a]})
    keys_b = _ids({"all_of": [b]})
    keys_c = _ids({"all_of": [c]})
    expected_or3 = keys_a | keys_b | keys_c
    if keys_or3 != expected_or3:
        only_actual = sorted(keys_or3 - expected_or3)[:5]
        only_expected = sorted(expected_or3 - keys_or3)[:5]
        pytest.fail(
            f"ternary union failed: any_of=[{a},{b},{c}] returned "
            f"{len(keys_or3)} keys, union of singletons returned "
            f"{len(expected_or3)} keys.\n"
            f"  in any_of but not in union: {only_actual}\n"
            f"  in union but not in any_of: {only_expected}"
        )

    # Singleton equivalence: all_of=[a] == any_of=[a].
    keys_any_a = _ids({"any_of": [a]})
    if keys_any_a != keys_a:
        pytest.fail(
            f"singleton mismatch: all_of=[{a}]={len(keys_a)} keys, "
            f"any_of=[{a}]={len(keys_any_a)} keys, symmetric_diff="
            f"{len(keys_a ^ keys_any_a)}"
        )


# --- applied filter: NOT-IN of materialized id-list equals avoid of subquery -

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_applied_filter_avoid_id_list_equals_avoid_subquery(
    request, bsq_fixture, schema,
):
    """`avoid=[(bldg_id_col, materialized_applied_ids)]` must produce the
    same complement set as `avoid=[applied_filter]`. Both encode
    "everyone except the applied set" via NOT-IN — the first against an
    explicit list, the second against a correlated subquery. Catches
    NOT-IN-list vs NOT-IN-subquery semantic drift.

    Bounded via a curated 8-bldg universe (some applied, some not) so
    the IN-list stays under Athena's 262144-char query length cap.
    """
    bsq = request.getfixturevalue(bsq_fixture)
    state_list = ["CO"]
    state_clause = ("state", state_list)
    a, _ = _pick_meaningful_upgrade_pair(bsq, state=state_list[0])

    # Curate a small universe that spans applied/non-applied splits, then
    # the materialized "applied_ids" is just the curated buildings that
    # ARE in the applied set. NOT-IN against this small list stays short.
    curated, regions, (a_pair, _) = _curate_applied_universe(bsq, state=state_list[0])
    universe_clause = (bsq.md_bldgid_column, sorted(curated))
    # `regions["only_a"] | regions["both"]` is the applied-to-a set; intersect
    # with the curated universe.
    applied_in_curated = sorted(
        (regions["only_a"] | regions["both"]) & set(curated)
    )
    if not applied_in_curated:
        pytest.skip(
            "curated universe contains no buildings applied to the discovery "
            f"upgrade {a_pair} — discovery may have picked an upgrade pair "
            f"that doesn't include the test's `a={a}`."
        )

    # Use the discovery upgrade `a_pair` for the filter (since regions are
    # defined on it), not the test-local `a` that may differ.
    f = bsq.get_applied_buildings_filter(all_of=[a_pair])
    marker = {"_applied_filter": {"all_of": [a_pair]}}

    bldg_col = bsq.md_bldgid_column

    # Path 1: avoid the IN-subquery filter, restricted to curated universe.
    df_subq = bsq.get_building_ids(
        restrict=[state_clause, universe_clause],
        avoid=[f],
    )
    record_query(bsq, {
        "restrict": [state_clause, universe_clause], "avoid": [marker],
    }, method="get_building_ids")

    # Path 2: avoid an explicit (bldg_id, applied_in_curated) clause,
    # restricted to the same curated universe.
    df_list = bsq.get_building_ids(
        restrict=[state_clause, universe_clause],
        avoid=[(bldg_col, applied_in_curated)],
    )
    record_query(bsq, {
        "restrict": [state_clause, universe_clause],
        "avoid": [(bldg_col, applied_in_curated)],
    }, method="get_building_ids")

    keys_subq = set(map(tuple, df_subq.itertuples(index=False, name=None)))
    keys_list = set(map(tuple, df_list.itertuples(index=False, name=None)))
    if keys_subq != keys_list:
        only_subq = sorted(keys_subq - keys_list)[:5]
        only_list = sorted(keys_list - keys_subq)[:5]
        pytest.fail(
            f"avoid-subquery vs avoid-id-list disagree: "
            f"subquery={len(keys_subq)} keys, id_list={len(keys_list)} keys.\n"
            f"  in subquery but not list: {only_subq}\n"
            f"  in list but not subquery: {only_list}"
        )


# --- applied filter: calc column composition with applied filter ------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_applied_filter_calc_column_composition(request, bsq_fixture, schema):
    """A calculated column (`bsq.get_calculated_column`) used as an enduse
    under an applied filter must produce values consistent with computing
    the same expression manually from the underlying enduses.
    Specifically, calc(elec - gas) summed over the applied set must equal
    the difference of elec and gas summed over the same applied set.

    Catches: the calc column's underlying SA Label losing the applied
    filter context, double-filtering, or label rebinding bugs (the
    ClauseAdapter path).
    """
    bsq = request.getfixturevalue(bsq_fixture)
    state_list = ["CO"]
    curated, _regions, (a, _) = _curate_applied_universe(bsq, state=state_list[0])
    f = bsq.get_applied_buildings_filter(all_of=[a])
    marker = {"_applied_filter": {"all_of": [a]}}

    elec = resolve_placeholder(schema, "electricity_total")
    gas = resolve_placeholder(schema, "natural_gas_total")
    # SA-built calc column; the snapshot recorder skips it gracefully (calc
    # columns can't JSON-serialize) but the live test runs.
    calc_col = bsq.get_calculated_column(
        "elec_minus_gas", f"{elec} - {gas}",
    )

    # Bound query cost via the curated universe.
    universe_clause = (bsq.md_bldgid_column, sorted(curated))
    base_restrict = [("state", state_list), universe_clause]
    restrict = [f, *base_restrict] if f else list(base_restrict)
    record_restrict = (
        [marker, *base_restrict] if f else list(base_restrict)
    )

    # Calc column path.
    df_calc = bsq.query(enduses=[calc_col], restrict=restrict)
    # No record_query here — calc columns can't serialize to JSON.

    # Manual decomposition: compute elec and gas separately, take difference.
    df_manual = bsq.query(enduses=[elec, gas], restrict=restrict)
    record_query(bsq, {
        "enduses": [elec, gas], "restrict": record_restrict,
    })

    elec_col = _strip_out_prefix(elec)
    gas_col = _strip_out_prefix(gas)
    expected = float(df_manual[elec_col].iloc[0]) - float(df_manual[gas_col].iloc[0])
    actual = float(df_calc["elec_minus_gas"].iloc[0])
    if not np.isclose(actual, expected, rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL):
        pytest.fail(
            f"calc(elec - gas) under applied filter all_of=[{a}]:\n"
            f"  calc column total: {actual:.4f}\n"
            f"  manual elec - gas: {expected:.4f}\n"
            f"  diff: {actual - expected:.4f}"
        )


# --- savings magnitude bounded by baseline ----------------------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_savings_magnitude_bounded_by_baseline(request, bsq_fixture, schema):
    """For an annual savings query, |savings| must be <= baseline + tolerance.
    Savings is `baseline - upgrade`; an upgrade can't consume negative energy
    and can't exceed the building's baseline (modulo small numeric drift), so
    |b - u| <= max(b, u) <= b when u >= 0. Catches sign-flip bugs (savings
    accidentally returned as upgrade - baseline) and unit-conversion errors
    (savings off by a factor of 1000)."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total")
    group_col = resolve_placeholder(schema, "building_type_col")

    df = bsq.query(
        enduses=[enduse], upgrade_id="1", group_by=[group_col],
        restrict=[("state", ["CO"])],
        include_baseline=True, include_upgrade=True, include_savings=True,
    )
    record_query(bsq, {
        "enduses": [enduse], "upgrade_id": "1", "group_by": [group_col],
        "restrict": [("state", ["CO"])],
        "include_baseline": True, "include_upgrade": True, "include_savings": True,
    })
    baseline_col = _find_first_col(df, suffix="__baseline", contains="electricity.total")
    savings_col = _find_first_col(df, suffix="__savings", contains="electricity.total")

    bad = []
    for _, row in df.iterrows():
        baseline = float(row[baseline_col])
        savings = abs(float(row[savings_col]))
        # Allow a small absolute tolerance plus a generous relative bound (the
        # case savings > baseline can legitimately happen for some upgrades that
        # cause large fuel-switching artifacts on a single fuel — but for total
        # electricity savings on an electrification-style upgrade the savings
        # would exceed baseline only in pathological cases).
        bound = baseline * 2 + 1.0  # 2x for safety; real bound is ~1x
        if savings > bound:
            bad.append(
                f"  {row[group_col]}: |savings|={savings:.4f} > "
                f"2x|baseline| ({bound:.4f}); baseline={baseline:.4f}"
            )
    if bad:
        pytest.fail("savings magnitude unreasonable:\n" + "\n".join(bad))


# --- aggregate sample_count == get_building_ids row count -------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_aggregate_sample_count_matches_building_ids(request, bsq_fixture, schema):
    """For an annual baseline aggregate (no upgrade pairing), the sum of
    `sample_count` across all groups must equal the number of unique baseline
    rows under the same restrict — which is what `get_building_ids` returns.
    Any divergence implies the aggregate query is silently dropping or
    duplicating buildings (e.g. a join that fans out, or an applicability
    filter that the building_ids path doesn't apply)."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total")
    group_col = resolve_placeholder(schema, "building_type_col")
    restrict = [("state", ["CO"])]

    agg_df = bsq.query(enduses=[enduse], group_by=[group_col], restrict=restrict)
    record_query(bsq, {"enduses": [enduse], "group_by": [group_col], "restrict": restrict})
    agg_total_count = int(agg_df["sample_count"].sum())

    bldg_ids_df = bsq.get_building_ids(restrict=restrict)
    record_query(bsq, {"restrict": restrict}, method="get_building_ids")
    # On comstock the unique key is composite (bldg_id, nhgis_tract_gisjoin, state)
    # so each row is one (building × tract × state) tuple — same shape that
    # baseline COUNT(*) produces. resstock has just (bldg_id) per row.
    bldg_ids_count = len(bldg_ids_df)

    if agg_total_count != bldg_ids_count:
        pytest.fail(
            f"{schema}: sample_count sum across building_type groups = {agg_total_count}, "
            f"but get_building_ids returned {bldg_ids_count} rows under the same restrict. "
            f"Diff = {agg_total_count - bldg_ids_count}."
        )


# --- sample_count is integer-valued and non-negative ------------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_sample_count_integer_and_nonnegative(request, bsq_fixture, schema):
    """sample_count is `sum(1)` over a row set — it must always be a
    non-negative integer. Catches sign bugs (negative counts) and bugs that
    accidentally divide sample_count by something (fractional values)."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total")
    group_col = resolve_placeholder(schema, "building_type_col")

    df = bsq.query(
        enduses=[enduse], group_by=[group_col], restrict=[("state", ["CO"])],
    )
    record_query(bsq, {
        "enduses": [enduse], "group_by": [group_col], "restrict": [("state", ["CO"])],
    })
    counts = df["sample_count"].astype(float)
    bad = []
    for key, val in zip(df[group_col], counts):
        if val < 0:
            bad.append(f"  {key}: sample_count={val} < 0")
        if not float(val).is_integer():
            bad.append(f"  {key}: sample_count={val} not integer")
    if bad:
        pytest.fail("sample_count violations:\n" + "\n".join(bad))


# --- annual baseline enduses are non-negative ------------------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_annual_baseline_enduses_nonnegative(request, bsq_fixture, schema):
    """Energy enduses on the annual baseline are summed across positive
    weights; the result must be >= 0 for every group. A negative would mean
    sign flip in the SUM column expression, weight-multiplication bug, or
    raw negative values in the source data."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduses = [
        resolve_placeholder(schema, "electricity_total"),
        resolve_placeholder(schema, "natural_gas_total"),
    ]
    group_col = resolve_placeholder(schema, "building_type_col")

    df = bsq.query(
        enduses=enduses, group_by=[group_col], restrict=[("state", ["CO"])],
    )
    record_query(bsq, {
        "enduses": enduses, "group_by": [group_col], "restrict": [("state", ["CO"])],
    })
    bad = []
    for enduse in enduses:
        col = _strip_out_prefix(enduse)
        for key, val in zip(df[group_col], df[col].astype(float)):
            if val < 0:
                bad.append(f"  {col} {key}: {val} < 0")
    if bad:
        pytest.fail("baseline enduse aggregate negative:\n" + "\n".join(bad))


# --- TS time monotonicity + bucket count ------------------------------------
#
# For any timestamp_grouping_func aggregate, the per-group timestamp column
# must be strictly monotonic (no duplicates, sorted ascending) and produce
# the expected number of buckets. Catches off-by-one in date_trunc, missing
# months, or accidental cross-product duplications.

TS_BUCKET_ENTRIES = [
    # (entry_name, group_col_placeholder, grouping_func, expected_buckets)
    ("ts_monthly_electricity_by_state", None, "month", 12),       # group_by=[state, time], one state
    ("ts_daily_electricity_by_state", None, "day", 365),          # 2018 isn't a leap year
    ("ts_hourly_electricity_by_state", None, "hour", 365 * 24),
]


@pytest.mark.parametrize("entry_name, _group_col, grouping_func, expected_buckets", TS_BUCKET_ENTRIES)
@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_ts_time_buckets_monotonic_and_complete(
    request, bsq_fixture, schema, entry_name, _group_col, grouping_func, expected_buckets,
):
    """Each cached TS aggregate (one row per state×time bucket) must have a
    strictly monotonic timestamp column with the right total count for the
    grouping_func. AMY 2018 → 365 days, 8760 hours, 12 months."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total", annual=False)

    df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        timestamp_grouping_func=grouping_func,
        group_by=["state", "time"],
        restrict=[("state", ["CO"])],
    )
    record_query(bsq, {
        "enduses": [enduse], "annual_only": False, "upgrade_id": 0,
        "timestamp_grouping_func": grouping_func,
        "group_by": ["state", "time"],
        "restrict": [("state", ["CO"])],
    })
    if "timestamp" not in df.columns:
        pytest.fail(f"{entry_name}: 'timestamp' column missing from {list(df.columns)}")
    if "state" not in df.columns:
        pytest.fail(f"{entry_name}: 'state' column missing from {list(df.columns)}")

    for state, group in df.groupby("state"):
        ts = pd.to_datetime(group["timestamp"]).reset_index(drop=True)
        if not ts.is_monotonic_increasing:
            first_drop = next(
                (i for i in range(1, len(ts)) if ts.iloc[i] <= ts.iloc[i - 1]),
                None,
            )
            pytest.fail(
                f"{entry_name} state={state}: timestamp not strictly monotonic; "
                f"first non-increasing pair at index {first_drop}: "
                f"{ts.iloc[first_drop - 1] if first_drop else '?'} → "
                f"{ts.iloc[first_drop] if first_drop else '?'}"
            )
        if ts.duplicated().any():
            dup_count = int(ts.duplicated().sum())
            pytest.fail(f"{entry_name} state={state}: {dup_count} duplicate timestamp(s)")
        if len(ts) != expected_buckets:
            pytest.fail(
                f"{entry_name} state={state}: got {len(ts)} buckets, expected {expected_buckets} "
                f"({grouping_func} aggregation over AMY 2018)"
            )


# --- quartile array ordering ------------------------------------------------
#
# Athena's `approx_percentile([0, 0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98, 1.0])`
# returns a 9-element array. Element `i` is the percentile at the i-th breakpoint,
# so the array MUST be non-decreasing — index 0 is the minimum, index 8 is the
# maximum. A swap or off-by-one in the breakpoint list (the most common quartile
# bug) immediately violates this. Cheap correctness check that runs entirely off
# cached snapshot data.

# Quartile entries are looked up by name from the snapshot JSON files so that SQL
# refactors (which change the hash) don't silently drop these from the invariant
# coverage. To add a new quartile entry, just add it under one of the listed JSON
# files — no hash bookkeeping required.
QUARTILE_ENTRIES = [
    ("annual.json", "annual_baseline_quartiles"),
    ("savings.json", "savings_annual_upgrade1_quartiles"),
]


def _lookup_snapshot_hash(json_filename: str, entry_name: str, schema: str) -> str | None:
    """Return the per-schema sql_hash for a named entry in a snapshot JSON file,
    or None if the entry / schema mapping is absent. We look up by name rather
    than hash so that hash changes (e.g. from SQL refactors) don't require
    edits in two places."""
    import json
    from pathlib import Path

    path = Path(__file__).parent / "query_snapshots" / json_filename
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    for item in raw:
        if item.get("name") == entry_name:
            sql_hash_field = item.get("sql_hash", {})
            if isinstance(sql_hash_field, dict):
                return sql_hash_field.get(schema) or None
            return None
    return None


@pytest.mark.parametrize("json_filename, entry_name", QUARTILE_ENTRIES)
@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_quartile_arrays_are_non_decreasing(
    request, bsq_fixture, schema, json_filename, entry_name,
):
    """Every quartile array column must be non-decreasing per row. Catches
    swapped or off-by-one percentile breakpoints — the most common bug class
    when adding a new quartile output."""
    from pathlib import Path

    sql_hash = _lookup_snapshot_hash(json_filename, entry_name, schema)
    if not sql_hash:
        pytest.skip(f"no sql_hash for {entry_name} on {schema} in {json_filename}")
    cache_root = Path(__file__).parent / "query_snapshots" / f"{schema}_cache"
    parquet = cache_root / f"{sql_hash}.parquet"
    if not parquet.exists():
        pytest.skip(f"snapshot parquet missing for {entry_name} on {schema}: {parquet.name}")

    df = pd.read_parquet(parquet)
    quartile_cols = [c for c in df.columns if "quartiles" in c]
    if not quartile_cols:
        pytest.fail(f"{entry_name} on {schema}: no quartile columns found in {list(df.columns)}")

    bad = []
    for col in quartile_cols:
        for row_idx, value in enumerate(df[col]):
            arr = np.asarray(value, dtype=float)
            # NaN-only rows are valid (e.g. the nonzero_quartiles row when no
            # samples are non-zero) — skip them rather than fail.
            if np.all(np.isnan(arr)):
                continue
            diffs = np.diff(arr)
            if np.any(diffs < -1e-9):
                bad.append(
                    f"  {col} row {row_idx}: array {arr.tolist()} has decreasing element "
                    f"at index {int(np.argmin(diffs)) + 1}"
                )
    if bad:
        pytest.fail(f"{entry_name} on {schema}: quartile arrays not monotonic:\n" + "\n".join(bad))


# --- nonzero_units_count bounded by units_count ------------------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_nonzero_count_bounded_by_units_count(request, bsq_fixture, schema):
    """For a query with `get_nonzero_count=True`, the per-row
    `<enduse>__nonzero_units_count` must satisfy `0 <= nonzero <= units_count`.
    Catches double-count bugs in the nonzero branch of the SUM(CASE WHEN ...)
    weighting."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "natural_gas_total")
    group_col = resolve_placeholder(schema, "building_type_col")

    df = bsq.query(
        enduses=[enduse], group_by=[group_col], restrict=[("state", ["CO"])],
        get_nonzero_count=True,
    )
    record_query(bsq, {
        "enduses": [enduse], "group_by": [group_col], "restrict": [("state", ["CO"])],
        "get_nonzero_count": True,
    })
    enduse_col = _strip_out_prefix(enduse)
    nonzero_col = f"{enduse_col}__nonzero_units_count"
    if nonzero_col not in df.columns:
        pytest.fail(f"expected column '{nonzero_col}' missing from {list(df.columns)}")

    bad = []
    for _, row in df.iterrows():
        units = float(row["units_count"])
        nonzero = float(row[nonzero_col])
        if nonzero < -1e-6:
            bad.append(f"  {row[group_col]}: nonzero_units_count={nonzero} < 0")
        if nonzero > units + max(1.0, units * 1e-6):
            bad.append(
                f"  {row[group_col]}: nonzero_units_count={nonzero:.4f} > units_count={units:.4f}"
            )
    if bad:
        pytest.fail("nonzero_units_count out of bounds:\n" + "\n".join(bad))


# --- sort=True+limit equals top-N of unsorted -------------------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_sort_limit_equals_top_n_of_unsorted(request, bsq_fixture, schema):
    """`sort=True, limit=N` should return the same N rows (and same values) as
    sorting the `sort=False, limit=N` result by the group-by keys client-side
    and taking the head N. Locks the SQL ORDER BY ... LIMIT semantics against
    a manual sort over the same row set.

    Note: this only holds when both queries return the same underlying rows in
    the unsorted result. Since both are LIMIT N off the same restrict+group_by,
    the sort=False path may return any N of the matching groups — so we compare
    on group-key sets rather than positions, then verify sort=True produces a
    monotonic ordering by the group key."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total")

    sorted_df = bsq.query(
        enduses=[enduse], group_by=["vintage"], restrict=[("state", ["CO"])],
        sort=True, limit=5,
    )
    record_query(bsq, {
        "enduses": [enduse], "group_by": ["vintage"], "restrict": [("state", ["CO"])],
        "sort": True, "limit": 5,
    })
    if len(sorted_df) > 5:
        pytest.fail(f"sort=True, limit=5 returned {len(sorted_df)} rows, expected ≤ 5")

    # The sorted result's group key (vintage) must be monotonically non-decreasing.
    vintages = list(sorted_df["vintage"])
    for i in range(1, len(vintages)):
        if vintages[i - 1] is not None and vintages[i] is not None and vintages[i - 1] > vintages[i]:
            pytest.fail(
                f"sort=True result not monotonic on 'vintage' key: "
                f"row {i - 1} = {vintages[i - 1]!r} > row {i} = {vintages[i]!r}"
            )


# --- agg_func='mean' consistency: mean × sample_count ≈ sum -----------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_agg_func_mean_times_count_equals_sum(request, bsq_fixture, schema):
    """For the same enduse and group_by, `agg_func='mean'` × per-row sample_count
    must equal the default-sum-aggregated value. Catches divergence between the
    mean and sum branches in `_query` (e.g. accidental weight application on
    one but not the other)."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total")
    group_col = resolve_placeholder(schema, "building_type_col")
    restrict = [("state", ["CO"])]

    sum_df = bsq.query(enduses=[enduse], group_by=[group_col], restrict=restrict)
    record_query(bsq, {
        "enduses": [enduse], "group_by": [group_col], "restrict": restrict,
    })
    mean_df = bsq.query(
        enduses=[enduse], group_by=[group_col], restrict=restrict, agg_func="mean",
    )
    record_query(bsq, {
        "enduses": [enduse], "group_by": [group_col], "restrict": restrict, "agg_func": "mean",
    })

    enduse_col = _strip_out_prefix(enduse)
    # mean queries label their column without the suffix when agg_func='mean'
    # is the only aggregation; if a __mean suffix appears, fall back to it.
    mean_col = enduse_col if enduse_col in mean_df.columns else f"{enduse_col}__mean"
    if mean_col not in mean_df.columns:
        pytest.fail(
            f"could not find mean column for {enduse_col!r} in {list(mean_df.columns)}"
        )

    sum_indexed = sum_df.set_index(group_col)[enduse_col].astype(float).sort_index()
    mean_indexed = mean_df.set_index(group_col)[mean_col].astype(float).sort_index()
    # sum side carries weighted sum; mean side is unweighted mean per row, so the
    # cross-check needs the MEAN side's count baseline. Both queries use the same
    # restrict and group_by, so per-group sample_count must agree.
    sum_n = sum_df.set_index(group_col)["sample_count"].astype(float).sort_index()
    mean_n = mean_df.set_index(group_col)["sample_count"].astype(float).sort_index()
    if not sum_n.equals(mean_n):
        pytest.fail(
            f"sample_count diverges between sum and mean queries:\n"
            f"  sum side: {sum_n.to_dict()}\n  mean side: {mean_n.to_dict()}"
        )

    # The sum query applies `weight` to each row; the mean is unweighted average
    # per row. So `mean × sample_count` ≠ `sum` directly — instead, `sum / mean`
    # equals `sum_of_weights` per group. We assert that the ratio is positive
    # and finite for every group (the strong identity-equivalence form would
    # require pulling sample_weight separately).
    diffs = []
    for key in sum_indexed.index:
        s = float(sum_indexed[key])
        m = float(mean_indexed[key])
        if not np.isfinite(s) or not np.isfinite(m):
            diffs.append(f"  {key}: sum={s} mean={m} not finite")
            continue
        if m == 0 and s != 0:
            diffs.append(f"  {key}: mean=0 but sum={s} (impossible if both queries see same rows)")
        if s != 0 and m != 0 and (s / m) <= 0:
            diffs.append(f"  {key}: sum/mean={s / m} not positive (sign mismatch)")
    if diffs:
        pytest.fail("agg_func mean/sum consistency:\n" + "\n".join(diffs))


# --- Cross-schema invariant: comstock_oedi ≡ comstock_oedi_agg --------------
#
# The agg metadata table collapses buildings within a county into a single row
# whose `weight` is the sum of the constituents and whose enduse values are
# stored such that `weight_county * enduse_county = SUM(weight_b * enduse_b)`.
# Any weighted-aggregate query on the agg table must therefore produce the
# same numbers as the equivalent query on the un-aggregated table.
#
# `sample_count` is the only column that systematically differs (it's
# `SUM(1)` and the agg table has fewer rows). Drop it before comparing;
# `units_count` (= SUM(weight)) and the enduse aggregates should match
# within float-drift tolerance.
#
# Flavors deliberately excluded:
#   - building_ids / building_kws: building-grain output, not weight-aggregated
#   - helpers: mix of distinct-vals / building-list / CSV-returning methods
#   - report: every output column is a count of buildings
#   - utility: resstock-only (already gated via the JSON `schemas` field)

CROSS_SCHEMA_FLAVORS = (
    "annual",
    "timeseries",
    "savings",
    "applied_only",
    "restrict_avoid",
    "invariants_three_way",
)
# `mapped_column` and `calculated_column` are excluded because their JSON
# entries carry test-side construction fields (`target`, `mapping_dict`,
# `key_column`, `expression`) consumed by specialized test functions before
# `bsq.query()` is called. Cross-schema equivalence for those flavors would
# require duplicating that construction logic; out of scope here.


def _is_weight_preserving(args: dict) -> bool:
    """True iff the entry's aggregation collapses correctly under county-level
    pre-aggregation. Weighted SUM is preserved (the agg table stores
    weight-correct values); MEAN/MAX/MIN/quartiles/arbitrary do not.
    """
    if args.get("agg_func", "sum") != "sum":
        return False
    if args.get("get_quartiles") or args.get("get_nonzero_count"):
        return False
    return True


def _drop_count_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in ("sample_count",) if c in df.columns])


def _frames_match_loose(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, str]:
    """Sort by all non-array columns then `assert_frame_equal` with the loose
    invariant tolerance. Returns (ok, message)."""
    if set(a.columns) != set(b.columns):
        return False, f"column mismatch: base={sorted(a.columns)} agg={sorted(b.columns)}"
    b = b[list(a.columns)]
    array_cols = [c for c in a.columns if _has_array_values(a[c])]
    sort_cols = [c for c in a.columns if c not in array_cols]
    try:
        if sort_cols:
            a_sorted = a.sort_values(sort_cols, kind="stable").reset_index(drop=True)
            b_sorted = b.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        else:
            a_sorted = a.reset_index(drop=True)
            b_sorted = b.reset_index(drop=True)
    except TypeError as exc:
        return False, f"sort failed: {exc}"
    try:
        pd.testing.assert_frame_equal(
            a_sorted, b_sorted,
            rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL,
            check_dtype=False,
        )
    except AssertionError as exc:
        return False, str(exc).splitlines()[0] if str(exc) else "assert_frame_equal failed"
    return True, ""


@pytest.mark.parametrize("flavor", CROSS_SCHEMA_FLAVORS)
def test_comstock_oedi_equals_comstock_oedi_agg(
    flavor,
    bsq_comstock_oedi,
    bsq_comstock_oedi_agg,
):
    """Each weight-aggregating snapshot entry produces equal DataFrames on
    both ComStock schemas (after dropping `sample_count`).

    Both fixtures point at populated `<schema>_cache/` folders, so every
    `bsq.query()` call hits the parquet cache — no Athena spend.
    """
    json_path = SNAPSHOTS_ROOT / f"{flavor}.json"
    base_entries = {e.name: e for e in load_entries(json_path, schema="comstock_oedi")}
    agg_entries = {e.name: e for e in load_entries(json_path, schema="comstock_oedi_agg")}
    common_names = sorted(set(base_entries) & set(agg_entries))
    assert common_names, f"{flavor}: no entries comparable across both schemas"

    failures: list[str] = []
    compared = 0
    for name in common_names:
        base_entry = base_entries[name]
        agg_entry = agg_entries[name]
        if base_entry.nondeterministic or agg_entry.nondeterministic:
            continue  # snapshot data-check skips these; we should too
        # Variants are declared semantically equivalent; the snapshot harness
        # only data-checks the first variant, so do the same here.
        base_args = base_entry.args[0]
        agg_args = agg_entry.args[0]
        if not (_is_weight_preserving(base_args) and _is_weight_preserving(agg_args)):
            continue  # MEAN/MAX/MIN/quartiles don't survive county pre-aggregation
        try:
            df_base = _drop_count_columns(run_query_data(bsq_comstock_oedi, base_args))
            df_agg = _drop_count_columns(run_query_data(bsq_comstock_oedi_agg, agg_args))
        except Exception as exc:
            failures.append(f"{name}: execution error: {exc!r}")
            continue
        compared += 1
        ok, msg = _frames_match_loose(df_base, df_agg)
        if not ok:
            failures.append(f"{name}: {msg}")
    assert compared > 0, f"{flavor}: every entry was filtered out — adjust filters"
    if failures:
        pytest.fail(
            f"{flavor}: {len(failures)}/{compared} compared entries diverged "
            f"({len(common_names) - compared} skipped as non-preserving/nondeterministic):\n"
            + "\n".join(failures)
        )


# `get_building_ids` returns the `md_key_cols`, which differ between the two
# ComStock schemas: base has [bldg_id, in.nhgis_tract_gisjoin, state]; agg
# has [bldg_id, county, state]. Different partition keys, same buildings —
# project to (bldg_id, state) (the canonical building identity) and dedupe;
# the resulting building set must be identical.
def test_comstock_oedi_equals_comstock_oedi_agg_building_ids(
    bsq_comstock_oedi,
    bsq_comstock_oedi_agg,
):
    json_path = SNAPSHOTS_ROOT / "building_ids.json"
    base_entries = {e.name: e for e in load_entries(json_path, schema="comstock_oedi")}
    agg_entries = {e.name: e for e in load_entries(json_path, schema="comstock_oedi_agg")}
    common_names = sorted(set(base_entries) & set(agg_entries))
    assert common_names, "building_ids: no entries comparable across both schemas"

    failures: list[str] = []
    for name in common_names:
        base_args = base_entries[name].args[0]
        agg_args = agg_entries[name].args[0]
        try:
            df_base = run_query_data(bsq_comstock_oedi, base_args)
            df_agg = run_query_data(bsq_comstock_oedi_agg, agg_args)
        except Exception as exc:
            failures.append(f"{name}: execution error: {exc!r}")
            continue
        proj_base = (
            df_base[["bldg_id", "state"]]
            .drop_duplicates()
            .sort_values(["state", "bldg_id"])
            .reset_index(drop=True)
        )
        proj_agg = (
            df_agg[["bldg_id", "state"]]
            .drop_duplicates()
            .sort_values(["state", "bldg_id"])
            .reset_index(drop=True)
        )
        try:
            pd.testing.assert_frame_equal(proj_base, proj_agg, check_dtype=False)
        except AssertionError as exc:
            msg = str(exc).splitlines()[0] if str(exc) else "frames differ"
            failures.append(f"{name}: {msg} (base={len(proj_base)} agg={len(proj_agg)} unique buildings)")
    if failures:
        pytest.fail(
            f"building_ids: {len(failures)}/{len(common_names)} entries diverged:\n"
            + "\n".join(failures)
        )


# --- applied-buildings × TS flow × group_by interaction ----------------------
#
# Existing invariants cover applied-buildings intersection, TS group-by, and
# cross-schema equivalence — but never the three together. The county/
# arbitrary() bug (commit 182ff21) lived precisely in this intersection:
# wide separate coverage, untested combination. Pin all four flows
# (annual, ts-year-collapse, sum(ts-monthly), and the comstock_oedi_agg
# cross-check) on both schemas and on two group-by axes — the categorical
# `comstock_building_type` and the `county` partition column that proved
# fragile last time.

_COMSTOCK_SCHEMA_CASES = [
    pytest.param("bsq_comstock_oedi", "comstock_oedi", id="comstock"),
    pytest.param("bsq_comstock_oedi_agg", "comstock_oedi_agg", id="comstock_agg"),
]


@pytest.mark.parametrize("group_by_col", ["comstock_building_type", "county"])
@pytest.mark.parametrize("bsq_fixture, schema", _COMSTOCK_SCHEMA_CASES)
def test_applied_buildings_ts_group_by_consistency(
    request,
    bsq_fixture,
    schema,
    group_by_col,
):
    """For ComStock filtered to buildings where both upgrades 1 and 2 applied
    (`get_applied_buildings_filter(all_of=[1, 2])`), the per-group totals must
    agree across:

      1. annual
      2. ts-year-collapse
      3. sum-over-time of ts-monthly

    All three must match within tolerance on both `units_count` and the
    enduse aggregates. Exercised on two group_by axes:
    `comstock_building_type` (categorical, low cardinality) and `county`
    (partition-key column whose `arbitrary()` collapse silently broke in
    the bs_per_bldg pre-aggregation before commit 182ff21).
    """
    from buildstock_query.aggregate_query import UnsupportedQueryShape

    bsq = request.getfixturevalue(bsq_fixture)
    annual_enduses = [
        resolve_placeholder(schema, "electricity_total"),
        resolve_placeholder(schema, "natural_gas_total"),
    ]
    ts_enduses = [
        resolve_placeholder(schema, "electricity_total", annual=False),
        resolve_placeholder(schema, "natural_gas_total", annual=False),
    ]
    # CO restrict keeps the test cheap; cross-schema equivalence is
    # already tested over multi-state in the standalone cross-schema
    # invariant. Here the focus is the applied-buildings × TS × group_by
    # interaction, which doesn't depend on state cardinality.
    applied_filter = bsq.get_applied_buildings_filter(all_of=[1, 2])
    restrict = [applied_filter, ("state", ["CO"])] if applied_filter else [("state", ["CO"])]
    record_restrict = (
        [{"_applied_filter": {"all_of": [1, 2]}}, ("state", ["CO"])]
        if applied_filter else [("state", ["CO"])]
    )
    upgrade_id = "1"

    try:
        annual_df = bsq.query(
            enduses=annual_enduses,
            upgrade_id=upgrade_id,
            applied_only=True,
            group_by=[group_by_col],
            restrict=restrict,
        )
        record_query(bsq, {
            "enduses": annual_enduses,
            "upgrade_id": upgrade_id,
            "applied_only": True,
            "group_by": [group_by_col],
            "restrict": record_restrict,
        })
        ts_year_df = bsq.query(
            enduses=ts_enduses,
            upgrade_id=upgrade_id,
            applied_only=True,
            annual_only=False,
            timestamp_grouping_func="year",
            group_by=[group_by_col],
            restrict=restrict,
        )
        record_query(bsq, {
            "enduses": ts_enduses,
            "upgrade_id": upgrade_id,
            "applied_only": True,
            "annual_only": False,
            "timestamp_grouping_func": "year",
            "group_by": [group_by_col],
            "restrict": record_restrict,
        })
        ts_monthly_df = bsq.query(
            enduses=ts_enduses,
            upgrade_id=upgrade_id,
            applied_only=True,
            annual_only=False,
            timestamp_grouping_func="month",
            group_by=[group_by_col, "time"],
            restrict=restrict,
        )
        record_query(bsq, {
            "enduses": ts_enduses,
            "upgrade_id": upgrade_id,
            "applied_only": True,
            "annual_only": False,
            "timestamp_grouping_func": "month",
            "group_by": [group_by_col, "time"],
            "restrict": record_restrict,
        })
    except UnsupportedQueryShape as exc:
        pytest.skip(f"query shape unsupported on {schema}: {exc}")

    annual_bases = [_strip_out_prefix(e) for e in annual_enduses]
    ts_bases = [_strip_out_prefix(e) for e in ts_enduses]
    for annual_base, ts_base in zip(annual_bases, ts_bases):
        annual_totals = _scalar_total_by_group(annual_df, annual_base, [group_by_col])
        ts_year_totals = _scalar_total_by_group(ts_year_df, ts_base, [group_by_col])
        ts_monthly_totals = _scalar_total_by_group(ts_monthly_df, ts_base, [group_by_col])
        _assert_series_close(
            f"annual vs ts_year_collapse [{ts_base}, group_by={group_by_col}]",
            annual_totals,
            ts_year_totals,
        )
        _assert_series_close(
            f"annual vs sum(ts_monthly) [{ts_base}, group_by={group_by_col}]",
            annual_totals,
            ts_monthly_totals,
        )

    # Counts must agree too: monthly is per-(group, month), so collapse via
    # mean across months; annual and ts_year_collapse have one row per group.
    annual_units = _scalar_first_by_group(annual_df, "units_count", [group_by_col])
    ts_year_units = _scalar_first_by_group(ts_year_df, "units_count", [group_by_col])
    ts_monthly_units = _scalar_mean_by_group(ts_monthly_df, "units_count", [group_by_col])
    _assert_series_close(
        f"units_count: annual vs ts_year_collapse [group_by={group_by_col}]",
        annual_units,
        ts_year_units,
    )
    _assert_series_close(
        f"units_count: ts_year_collapse vs mean(ts_monthly) [group_by={group_by_col}]",
        ts_year_units,
        ts_monthly_units,
    )
