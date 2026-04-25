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

2. **savings decomposition: baseline - upgrade ≈ savings** — shared across both
   schemas. For a `query(..., include_baseline=True, include_upgrade=True,
   include_savings=True)` call, the returned DataFrame must satisfy
   `baseline_col - upgrade_col ≈ savings_col` for every row.

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

from tests.test_utility import resolve_placeholder


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
# - upgrade1_applied: upgrade 1, applied_only=True.
# - upgrade1_applied_in_1_2: upgrade 1, applied_only=True, restricted further to
#   buildings to which both upgrades 1 and 2 applied.
# `upgrade1_applied` is xfail because the timeseries upgrade-pair flow doesn't apply
# the same `up.applicability='true'` filter that the annual flow does — the TS table
# includes inapplicable buildings when `inapplicables_have_ts=true`. Annual reports
# fewer building types / lower totals than TS for any `applied_only=True` query.
# Fix would touch user-facing "applied" semantics; deferred until that decision lands.
APPLICABILITY_DIVERGENCE = pytest.mark.xfail(
    reason="Known: TS upgrade-pair flow doesn't filter inapplicable buildings the way annual does.",
    strict=True,
)

SCENARIOS = [
    pytest.param({"upgrade_id": "0"}, id="baseline"),
    pytest.param({"upgrade_id": "1", "applied_only": False}, id="upgrade1"),
    pytest.param(
        {"upgrade_id": "1", "applied_only": True},
        id="upgrade1_applied",
        marks=APPLICABILITY_DIVERGENCE,
    ),
    pytest.param(
        {"upgrade_id": "1", "applied_only": True, "applied_in": [1, 2]},
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
    restrict = [("state", ["CO"])]

    try:
        annual_df = bsq.query(
            enduses=annual_enduses,
            group_by=[group_col],
            restrict=restrict,
            **scenario_extra,
        )
        ts_year_df = bsq.query(
            enduses=ts_enduses,
            annual_only=False,
            timestamp_grouping_func="year",
            group_by=[group_col],
            restrict=restrict,
            **scenario_extra,
        )
        ts_monthly_df = bsq.query(
            enduses=ts_enduses,
            annual_only=False,
            timestamp_grouping_func="month",
            group_by=[group_col, "time"],
            restrict=restrict,
            **scenario_extra,
        )
    except UnsupportedQueryShape as exc:
        pytest.skip(f"query shape unsupported on {schema}: {exc}")

    # Pick the electricity enduse from each leg (always element 0) for the totals check.
    annual_col = _strip_out_prefix(annual_enduses[0])
    ts_col = _strip_out_prefix(ts_enduses[0])

    annual_totals = _scalar_total_by_group(annual_df, annual_col, [group_col])
    ts_year_totals = _scalar_total_by_group(ts_year_df, ts_col, [group_col])
    ts_monthly_totals = _scalar_total_by_group(ts_monthly_df, ts_col, [group_col])

    _assert_series_close("annual vs ts_year_collapse", annual_totals, ts_year_totals)
    _assert_series_close("annual vs sum(ts_monthly)", annual_totals, ts_monthly_totals)

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


# --- savings decomposition: baseline - upgrade ≈ savings ---------------------

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_savings_decomposition(request, bsq_fixture, schema):
    """For a savings query with include_baseline + include_upgrade + include_savings,
    the stored DataFrame should satisfy baseline - upgrade ≈ savings per group."""
    bsq = request.getfixturevalue(bsq_fixture)
    enduse = resolve_placeholder(schema, "electricity_total")
    group_col = resolve_placeholder(schema, "building_type_col")

    df = bsq.query(
        enduses=[enduse],
        upgrade_id="1",
        group_by=[group_col],
        restrict=[("state", ["CO"])],
        include_baseline=True,
        include_upgrade=True,
        include_savings=True,
    )

    baseline_col = _find_first_col(df, suffix="__baseline", contains="electricity.total")
    upgrade_col = _find_first_col(df, suffix="__upgrade", contains="electricity.total")
    savings_col = _find_first_col(df, suffix="__savings", contains="electricity.total")

    diffs = []
    for _, row in df.iterrows():
        expected = row[baseline_col] - row[upgrade_col]
        actual = row[savings_col]
        if not np.isclose(expected, actual, rtol=INVARIANT_RTOL, atol=INVARIANT_ATOL, equal_nan=True):
            diffs.append(
                f"  {row.get(group_col, '?')}: baseline-upgrade={expected:.4f}, savings={actual:.4f}"
            )
    if diffs:
        pytest.fail("savings decomposition failed:\n" + "\n".join(diffs))


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
    grouped_df = bsq.query(enduses=enduses, group_by=[group_col], restrict=restrict)

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
    co_wy_df = bsq.query(
        enduses=[enduse], group_by=["state"], restrict=[("state", ["CO", "WY"])],
    )

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
    avoid_df = bsq.query(
        enduses=[enduse], group_by=[group_col], restrict=restrict,
        avoid=[(group_col, [avoided_value])],
    )

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
    # MappedColumn group_by — one row per mapped category (MH/SF/MF, etc.).
    key_col = bsq._get_column(group_col)
    mapped = MappedColumn(
        bsq=bsq, name="simple_bldg_type", mapping_dict=mapping_dict, key=key_col,
    )
    mapped_df = bsq.query(
        enduses=[enduse], group_by=[mapped], restrict=restrict,
    )

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
    monthly_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        timestamp_grouping_func="month",
        group_by=["state", "time"],
        restrict=[("state", ["CO"])],
    )

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
    only_df = bsq.query(
        enduses=[enduse], upgrade_id="1", group_by=[group_col], restrict=restrict,
        include_baseline=False, include_upgrade=False, include_savings=True,
    )

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
    single_fuel_df = bsq.query(
        enduses=[elec], group_by=[group_col], restrict=restrict,
    )

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


# --- applied_in intersection: applied_in=[1,2] equals (applied_in=[1] ∩ applied_in=[2]) --

@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_applied_in_intersection(request, bsq_fixture, schema):
    """The set of buildings returned by `applied_in=[1, 2]` must equal the intersection
    of `applied_in=[1]` (buildings that applied to upgrade 1) and `applied_in=[2]`
    (buildings that applied to upgrade 2). Asserts the `_get_applied_in_subquery`
    HAVING-count machinery actually computes a set intersection rather than something
    weaker (e.g. union, or a wrong join semantic)."""
    bsq = request.getfixturevalue(bsq_fixture)

    df_1 = bsq.get_building_ids(applied_in=[1], restrict=[("state", ["CO"])])
    df_2 = bsq.get_building_ids(applied_in=[2], restrict=[("state", ["CO"])])
    df_12 = bsq.get_building_ids(applied_in=[1, 2], restrict=[("state", ["CO"])])

    # Each row of get_building_ids is a unique-key tuple. For resstock that's
    # (bldg_id,); for comstock it's (bldg_id, in.nhgis_tract_gisjoin, state) because
    # a single physical building can appear in multiple tracts. Itertuples gives us
    # exactly the right shape to use as a set element.
    keys_1 = set(map(tuple, df_1.itertuples(index=False, name=None)))
    keys_2 = set(map(tuple, df_2.itertuples(index=False, name=None)))
    keys_12 = set(map(tuple, df_12.itertuples(index=False, name=None)))
    expected = keys_1 & keys_2

    if keys_12 != expected:
        only_in_actual = keys_12 - expected
        only_in_expected = expected - keys_12
        msg = [
            f"applied_in=[1,2] returned {len(keys_12)} keys, "
            f"intersection of applied_in=[1] ({len(keys_1)}) and applied_in=[2] "
            f"({len(keys_2)}) has {len(expected)} keys.",
        ]
        if only_in_actual:
            sample = list(sorted(only_in_actual))[:5]
            msg.append(f"  in [1,2] but not in intersection ({len(only_in_actual)} total): {sample}")
        if only_in_expected:
            sample = list(sorted(only_in_expected))[:5]
            msg.append(f"  in intersection but not in [1,2] ({len(only_in_expected)} total): {sample}")
        pytest.fail("\n".join(msg))

    # Cross-check against the aggregated `applied_in_1_2` sample_count from the
    # invariant snapshot. The number of unique-key tuples here should equal the
    # `sample_count` reported there (which is COUNT(DISTINCT bs_key) at the SQL level).
    enduses = [
        resolve_placeholder(schema, "electricity_total"),
        resolve_placeholder(schema, "natural_gas_total"),
    ]
    group_col = resolve_placeholder(schema, "building_type_col")
    inv_df = bsq.query(
        enduses=enduses, upgrade_id="1", applied_only=True, applied_in=[1, 2],
        group_by=[group_col], restrict=[("state", ["CO"])],
    )
    aggregated_sample_count = int(inv_df["sample_count"].sum())
    if aggregated_sample_count != len(keys_12):
        pytest.fail(
            f"sample_count mismatch: get_building_ids returned {len(keys_12)} unique "
            f"keys, but the aggregated applied_in=[1,2] query reports total "
            f"sample_count={aggregated_sample_count} (sum across building types)."
        )
