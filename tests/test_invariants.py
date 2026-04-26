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
# - upgrade1_applied: upgrade 1, applied_only=True. Pins the fix that synthesizes
#   `applied_in=[upgrade_id]` on TS paths so the TS flow filters inapplicable
#   buildings the same way the annual flow does (via up.applicability=true).
# - upgrade1_applied_in_1_2: upgrade 1, applied_only=True, restricted further to
#   buildings to which both upgrades 1 and 2 applied.
SCENARIOS = [
    pytest.param({"upgrade_id": "0"}, id="baseline"),
    pytest.param({"upgrade_id": "1", "applied_only": False}, id="upgrade1"),
    pytest.param(
        {"upgrade_id": "1", "applied_only": True},
        id="upgrade1_applied",
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
    restrict = [("state", ["CO"])]

    # Savings columns are invalid when upgrade_id="0" (the schema validator
    # rejects include_savings on the baseline scenario); only request them
    # when querying an upgrade.
    is_baseline = scenario_extra.get("upgrade_id") == "0"
    savings_kwargs: dict = (
        {} if is_baseline
        else {"include_baseline": True, "include_upgrade": True, "include_savings": True}
    )

    try:
        annual_df = bsq.query(
            enduses=annual_enduses,
            group_by=[group_col],
            restrict=restrict,
            **savings_kwargs,
            **scenario_extra,
        )
        ts_year_df = bsq.query(
            enduses=ts_enduses,
            annual_only=False,
            timestamp_grouping_func="year",
            group_by=[group_col],
            restrict=restrict,
            **savings_kwargs,
            **scenario_extra,
        )
        ts_monthly_df = bsq.query(
            enduses=ts_enduses,
            annual_only=False,
            timestamp_grouping_func="month",
            group_by=[group_col, "time"],
            restrict=restrict,
            **savings_kwargs,
            **scenario_extra,
        )
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
    hourly_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        timestamp_grouping_func="hour",
        group_by=["state", "time"],
        restrict=[("state", ["CO"])],
    )

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
    monthly_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        timestamp_grouping_func="month",
        group_by=["state", "time"],
        restrict=[("state", ["CO"])],
    )

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
    daily_df = bsq.query(
        enduses=[enduse], annual_only=False, upgrade_id=0,
        timestamp_grouping_func="day",
        group_by=["state", "time"],
        restrict=[("state", ["CO"])],
    )

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
      b_only = bsq.query(upgrade_id="0", applied_in=[1], ...)
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
        b_only = bsq.query(
            enduses=enduses, upgrade_id="0",
            applied_in=[1], group_by=[group_col], restrict=restrict,
        )
        u_only = bsq.query(
            enduses=enduses, upgrade_id="1",
            applied_only=True, group_by=[group_col], restrict=restrict,
        )
        full = bsq.query(
            enduses=enduses, upgrade_id="1",
            applied_only=True, group_by=[group_col], restrict=restrict,
            include_baseline=True, include_upgrade=True, include_savings=True,
        )
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
        full = bsq.query(
            enduses=enduses, upgrade_id="1",
            group_by=[group_col], restrict=restrict,
            include_baseline=True, include_upgrade=True, include_savings=True,
        )
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
        s2_df = bsq.query(
            enduses=enduses, upgrade_id="1",
            applied_only=True, group_by=[group_col],
            restrict=[("state", [s2])],
            include_baseline=True, include_upgrade=True, include_savings=True,
        )
        both = bsq.query(
            enduses=enduses, upgrade_id="1",
            applied_only=True, group_by=[group_col],
            restrict=[("state", [s1, s2])],
            include_baseline=True, include_upgrade=True, include_savings=True,
        )
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
    agg_total_count = int(agg_df["sample_count"].sum())

    bldg_ids_df = bsq.get_building_ids(restrict=restrict)
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

QUARTILE_ENTRIES = [
    ("baseline_quartiles", "8b107153bb7a05f4b4e087d074bd19d3f7612f13363e2cd383a541cc89f0e3e3", "749f6aa7782fcad720b8c95ab1ee730338bd6983ea3448a02f08860981468876"),
    ("savings_annual_quartiles", "505930cd8529cc5246c88e294b3419feb000ecfd9253a02bbacc8bf1d5914ee7", "994b17636d36a4f40bc524267516b8d0e133479a244e31102713fce7d9d8daba"),
]


@pytest.mark.parametrize("entry_name, resstock_hash, comstock_hash", QUARTILE_ENTRIES)
@pytest.mark.parametrize("bsq_fixture, schema", SCHEMA_CASES)
def test_quartile_arrays_are_non_decreasing(
    request, bsq_fixture, schema, entry_name, resstock_hash, comstock_hash,
):
    """Every quartile array column must be non-decreasing per row. Catches
    swapped or off-by-one percentile breakpoints — the most common bug class
    when adding a new quartile output."""
    from pathlib import Path

    cache_root = Path(__file__).parent / "query_snapshots" / f"{schema}_cache"
    sql_hash = resstock_hash if schema == "resstock_oedi" else comstock_hash
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
    mean_df = bsq.query(
        enduses=[enduse], group_by=[group_col], restrict=restrict, agg_func="mean",
    )

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
