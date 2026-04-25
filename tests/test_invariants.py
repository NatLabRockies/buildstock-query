"""Invariant tests — cross-query consistency checks that operate on snapshot data.

Each invariant constructs several `query(...)` arg sets, generates SQL for each leg,
looks up the matching snapshot entry by SQL, loads its stored parquet, and asserts
a mathematical relation between the DataFrames.

No Athena data queries are issued — only SQL generation (fast) and local parquet
reads. The cost is paid once during `--update-sql-and-data`; these invariants just
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

   Catches bugs in timeseries aggregation, baseline/timeseries join logic, and unit
   conversion between the annual and timeseries tables. Comstock uses different
   enduse column names for each (`..kwh` suffix on annual, no suffix on TS), so the
   parametrized case table carries both names.

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

import numpy as np
import pandas as pd
import pytest

from tests.test_utility import SNAPSHOTS_ROOT, find_data_for_sql


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
# Each case declares the fixture name (resolved at test time via
# `request.getfixturevalue`), the snapshot subdirectory, and the per-schema
# specifics (enduse names differ on comstock because baseline uses `..kwh`
# suffix while TS does not). Column names like `state`, `vintage`,
# `geometry_building_type_recs`, and `comstock_building_type` are bare — the
# library's _get_column fallback auto-prepends the `in.` prefix.

# Each case pins the electricity enduse name for the annual leg and the TS leg
# (they differ on comstock: baseline has the `..kwh` suffix, TS does not).
# `annual_enduses` and `ts_monthly_enduses` are the full lists used for the
# stored snapshots of those legs (resstock stores a two-fuel monthly; comstock
# stores a single-fuel monthly).

THREE_WAY_CASES = [
    pytest.param(
        "bsq_resstock_oedi",
        "resstock_oedi",
        [
            "out.electricity.total.energy_consumption",
            "out.natural_gas.total.energy_consumption",
        ],
        "out.electricity.total.energy_consumption",
        [
            "out.electricity.total.energy_consumption",
            "out.natural_gas.total.energy_consumption",
        ],
        "geometry_building_type_recs",
        id="resstock_electricity_by_building_type",
    ),
    pytest.param(
        "bsq_comstock_oedi",
        "comstock_oedi",
        [
            "out.electricity.total.energy_consumption..kwh",
            "out.natural_gas.total.energy_consumption..kwh",
        ],
        "out.electricity.total.energy_consumption",
        ["out.electricity.total.energy_consumption"],
        "comstock_building_type",
        id="comstock_electricity_by_building_type",
    ),
]


SAVINGS_CASES = [
    pytest.param(
        "bsq_resstock_oedi",
        "resstock_oedi",
        "out.electricity.total.energy_consumption",
        "geometry_building_type_recs",
        id="resstock_electricity_by_building_type",
    ),
    pytest.param(
        "bsq_comstock_oedi",
        "comstock_oedi",
        "out.electricity.total.energy_consumption..kwh",
        "comstock_building_type",
        id="comstock_electricity_by_building_type",
    ),
]


# --- three-way invariant: annual == ts_year_collapse == sum(ts_monthly) -------

@pytest.mark.parametrize(
    "bsq_fixture, schema, annual_enduses, ts_enduse, ts_monthly_enduses, group_col",
    THREE_WAY_CASES,
)
def test_annual_equals_ts_year_equals_ts_monthly_sum(
    request, bsq_fixture, schema, annual_enduses, ts_enduse, ts_monthly_enduses, group_col
):
    """For each schema: annual total, year-collapsed timeseries, and sum of monthly
    timeseries should all report the same per-group total energy. Counts must agree
    too."""
    bsq = request.getfixturevalue(bsq_fixture)
    schema_dir = SNAPSHOTS_ROOT / schema
    restrict = [("state", ["CO"])]

    annual_sql = bsq.query(
        enduses=annual_enduses,
        group_by=[group_col],
        restrict=restrict,
        get_query_only=True,
    )
    annual_df = find_data_for_sql(schema_dir, annual_sql)

    ts_year_sql = bsq.query(
        enduses=[ts_enduse],
        annual_only=False,
        timestamp_grouping_func="year",
        group_by=[group_col],
        restrict=restrict,
        get_query_only=True,
    )
    ts_year_df = find_data_for_sql(schema_dir, ts_year_sql)

    ts_monthly_sql = bsq.query(
        enduses=ts_monthly_enduses,
        annual_only=False,
        timestamp_grouping_func="month",
        group_by=[group_col, "time"],
        restrict=restrict,
        get_query_only=True,
    )
    ts_monthly_df = find_data_for_sql(schema_dir, ts_monthly_sql)

    # Pick the electricity enduse from each leg (always element 0) for the totals check.
    annual_col = _strip_out_prefix(annual_enduses[0])
    ts_col = _strip_out_prefix(ts_enduse)

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


# --- savings decomposition: baseline - upgrade ≈ savings ---------------------

@pytest.mark.parametrize(
    "bsq_fixture, schema, enduse, group_col",
    SAVINGS_CASES,
)
def test_savings_decomposition(request, bsq_fixture, schema, enduse, group_col):
    """For a savings query with include_baseline + include_upgrade + include_savings,
    the stored DataFrame should satisfy baseline - upgrade ≈ savings per group."""
    bsq = request.getfixturevalue(bsq_fixture)
    schema_dir = SNAPSHOTS_ROOT / schema

    sql = bsq.query(
        enduses=[enduse],
        upgrade_id="1",
        group_by=[group_col],
        restrict=[("state", ["CO"])],
        include_baseline=True,
        include_upgrade=True,
        include_savings=True,
        get_query_only=True,
    )
    df = find_data_for_sql(schema_dir, sql)

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
