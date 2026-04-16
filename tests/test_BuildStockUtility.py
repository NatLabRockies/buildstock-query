from __future__ import annotations

from typing import Generator

import numpy as np
import pandas as pd
import pytest

from buildstock_query.main import BuildStockQuery


@pytest.fixture(scope="module")
def bsq() -> Generator[BuildStockQuery, None, None]:
    """Shared BuildStockQuery instance backed by the sdr_magic17 run."""
    obj = BuildStockQuery(
        db_name="resstock_core",
        table_name="sdr_magic17",
        workgroup="rescore",
        buildstock_type="resstock",
        skip_reports=True,
    )
    obj.save_cache()
    yield obj
    obj.save_cache()


@pytest.fixture(scope="module")
def sample_eiaids(bsq: BuildStockQuery) -> list[str]:
    ids = bsq.utility.get_eiaids()
    if len(ids) < 3:
        pytest.skip("Not enough EIA IDs available for utility integration tests.")
    return ids[:5]


@pytest.fixture(scope="module")
def sample_counties(bsq: BuildStockQuery, sample_eiaids: list[str]) -> list[str]:
    counties = bsq.utility.get_locations_by_eiaids(sample_eiaids[:1])
    if not counties:
        pytest.skip("No counties returned for sample EIA IDs.")
    return counties[:3]


@pytest.fixture(scope="module")
def seasonal_rate_map() -> dict[tuple[int, int, int], float]:
    return {
        (month, weekend, hour): (0.1 if 1 <= month <= 7 else 0.0)
        for month in range(1, 13)
        for weekend in (0, 1)
        for hour in range(24)
    }


def test_aggregate_ts_by_eiaid_monthly(bsq: BuildStockQuery, sample_eiaids: list[str]) -> None:
    target_ids = sample_eiaids[:-1]
    df = bsq.utility.aggregate_ts_by_eiaid(
        eiaid_list=target_ids,
        enduses=["fuel_use__electricity__total__kwh"],
        group_by=["time"],
        timestamp_grouping_func="month",
    )

    assert not df.empty
    expected_cols = {
        "eiaid",
        "time",
        "sample_count",
        "units_count",
        "rows_per_sample",
        "fuel_use__electricity__total__kwh",
    }
    assert expected_cols <= set(df.columns)
    assert set(df["eiaid"].unique()) <= set(target_ids)
    assert df["sample_count"].gt(0).all()
    assert df["units_count"].ge(0).all()
    assert df["fuel_use__electricity__total__kwh"].ge(0).all()
    assert pd.api.types.is_datetime64_any_dtype(df["time"]) or isinstance(df["time"].iloc[0], pd.Timestamp)
    assert df.groupby("eiaid")["time"].nunique().gt(0).all()

    totals = (
        df.groupby("time")[["sample_count", "units_count", "fuel_use__electricity__total__kwh"]]
        .sum()
        .sort_index()
    )
    portfolio = (
        bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
        )
        .set_index("time")[["sample_count", "units_count", "fuel_use__electricity__total__kwh"]]
        .sort_index()
    )
    # There are only a tiny subset set of counties in the test run for the eiaids
    # So, the energy 
    coverage = totals.div(portfolio.replace(0, np.nan)).fillna(1.0)
    assert coverage.ge(0.02).all().all()
    assert coverage.le(1).all().all()


def test_aggregate_ts_by_eiaid_collapse(bsq: BuildStockQuery, sample_eiaids: list[str]) -> None:
    target_ids = sample_eiaids[:-1]
    monthly = bsq.utility.aggregate_ts_by_eiaid(
        eiaid_list=target_ids,
        enduses=["fuel_use__electricity__total__kwh"],
        group_by=["time"],
        timestamp_grouping_func="month",
    )
    collapsed = bsq.utility.aggregate_ts_by_eiaid(
        eiaid_list=target_ids,
        enduses=["fuel_use__electricity__total__kwh"],
        collapse_ts=True,
    )

    assert not collapsed.empty
    assert not monthly.empty
    assert "time" not in collapsed.columns
    assert set(collapsed["eiaid"].unique()) <= set(target_ids)
    energy_col = "fuel_use__electricity__total__kwh"

    summed_energy = monthly.groupby("eiaid")[energy_col].sum().sort_index()
    collapsed_energy = collapsed.set_index("eiaid")[energy_col].sort_index()
    pd.testing.assert_series_equal(collapsed_energy, summed_energy, check_dtype=False)

    collapsed_counts = collapsed.set_index("eiaid")[["units_count", "sample_count"]].sort_index()
    expected_counts = (
        monthly.groupby("eiaid")[["units_count", "sample_count"]]
        .first()
        .sort_index()
    )
    pd.testing.assert_frame_equal(collapsed_counts, expected_counts, check_dtype=False, check_like=True)


def test_aggregate_unit_counts_by_eiaid(bsq: BuildStockQuery, sample_eiaids: list[str]) -> None:
    target_ids = sample_eiaids[:3]
    df = bsq.utility.aggregate_unit_counts_by_eiaid(
        eiaid_list=target_ids,
        group_by=["build_existing_model.geometry_building_type_recs"],
    )

    assert not df.empty
    assert {"eiaid", "sample_count", "units_count"} <= set(df.columns)
    geometry_cols = [col for col in df.columns if "geometry_building_type_recs" in col]
    assert geometry_cols, "Expected geometry_building_type_recs column in results."
    geom_col = geometry_cols[0]
    assert set(df["eiaid"].unique()) <= set(target_ids)
    assert df["units_count"].ge(0).all()
    assert df["sample_count"].gt(0).all()
    assert df[geom_col].notna().all()

    portfolio = bsq.agg.aggregate_annual(
        enduses=[],
        group_by=["build_existing_model.geometry_building_type_recs"],
        sort=True,
    )
    portfolio = portfolio.set_index(geom_col)[["units_count", "sample_count"]].sort_index()
    summed = df.groupby(geom_col)[["units_count", "sample_count"]].sum().sort_index()
    coverage = summed.div(portfolio.replace(0, np.nan)).fillna(1.0)
    # Reduced EIAIDs set only covert tiny part of the portfolio, but should capture at least 2%.
    assert coverage.ge(0.02).all().all()
    assert coverage.le(1).all().all()


def test_aggregate_annual_by_eiaid_matches_overall(bsq: BuildStockQuery) -> None:
    enduse = "fuel_use_electricity_total_m_btu"
    df = bsq.utility.aggregate_annual_by_eiaid(enduses=[enduse], get_nonzero_count=True)
    overall = bsq.agg.aggregate_annual(enduses=[enduse])

    assert not df.empty
    expected_cols = {"eiaid", "sample_count", "units_count", enduse}
    assert expected_cols <= set(df.columns)
    nonzero_col = f"{enduse}__nonzero_units_count"
    assert nonzero_col in df.columns
    assert df[nonzero_col].ge(0).all()
    assert df[enduse].ge(0).all()
    coverage = df[enduse].sum() / max(overall[enduse].iloc[0], np.finfo(float).eps)
    # EIAIDs do not span the full portfolio; expect at least 10% coverage.
    assert 0.1 <= coverage <= 1.01
    assert df.groupby("eiaid")["units_count"].first().gt(0).all()
    df["eiaid"] = df["eiaid"].astype(str)
    unit_counts = bsq.utility.aggregate_unit_counts_by_eiaid(
        eiaid_list=df["eiaid"].unique().tolist())
    merged = (
        df[["eiaid", "units_count"]]
        .merge(unit_counts[["eiaid", "units_count"]], on="eiaid", suffixes=("_energy", "_units"))
    )
    pd.testing.assert_series_equal(
        merged.sort_values("eiaid")["units_count_energy"].reset_index(drop=True),
        merged.sort_values("eiaid")["units_count_units"].reset_index(drop=True),
        check_dtype=False,
        check_names=False,
    )


def test_get_buildings_by_eiaids(bsq: BuildStockQuery, sample_eiaids: list[str]) -> None:
    target_ids = sample_eiaids[:-1]
    df = bsq.utility.get_buildings_by_eiaids(target_ids)

    assert not df.empty
    building_col = bsq.bs_bldgid_column.name
    assert building_col in df.columns
    assert df[building_col].notna().all()
    assert df[building_col].is_unique
    filtered = bsq.utility.get_filtered_results_csv_by_eiaid(target_ids)
    assert set(df[building_col]) <= set(filtered[building_col].unique())


def test_get_filtered_results_csv_by_eiaid(bsq: BuildStockQuery, sample_eiaids: list[str]) -> None:
    target_ids = sample_eiaids[:-1]
    df = bsq.utility.get_filtered_results_csv_by_eiaid(target_ids)

    assert not df.empty
    assert {"eiaid", "weight"} <= set(df.columns)
    assert set(df["eiaid"].unique()) <= set(target_ids)
    assert (df["weight"] > 0).all()
    building_col = bsq.bs_bldgid_column.name
    assert building_col in df.columns
    grouped = df.groupby("eiaid")[building_col].nunique()
    assert grouped.gt(0).all()


def test_get_locations_by_eiaids(bsq: BuildStockQuery, sample_eiaids: list[str]) -> None:
    target_ids = sample_eiaids[:-1]
    locations = bsq.utility.get_locations_by_eiaids(target_ids)

    assert isinstance(locations, list)
    assert locations
    assert all(isinstance(loc, str) and loc for loc in locations)


def test_get_buildings_by_locations(
    bsq: BuildStockQuery, sample_counties: list[str]
) -> None:
    df = bsq.get_buildings_by_locations(
        location_col="build_existing_model.county",
        locations=sample_counties,
    )

    assert not df.empty
    building_col = bsq.bs_bldgid_column.name
    assert building_col in df.columns
    assert df[building_col].notna().all()


def test_calculate_tou_bill_monthly(
    bsq: BuildStockQuery,
    seasonal_rate_map: dict[tuple[int, int, int], float],
) -> None:
    df = bsq.utility.calculate_tou_bill(
        rate_map=seasonal_rate_map,
        meter_col="fuel_use__electricity__total__kwh",
        timestamp_grouping_func="month",
    )

    assert not df.empty
    expected_col = "fuel_use__electricity__total__kwh__tou__dollars"
    assert expected_col in df.columns
    assert "time" in df.columns
    assert df["sample_count"].gt(0).all()
    assert df[expected_col].ge(0).all()
    assert pd.api.types.is_datetime64_any_dtype(df["time"]) or isinstance(df["time"].iloc[0], pd.Timestamp)

    df = df.sort_values("time").reset_index(drop=True)
    df["month"] = df["time"].dt.month

    baseline = bsq.agg.aggregate_timeseries(
        enduses=["fuel_use__electricity__total__kwh"],
        timestamp_grouping_func="month",
    ).sort_values("time")
    merged = df.merge(
        baseline[["time", "fuel_use__electricity__total__kwh"]],
        on="time",
        suffixes=("", "__expected"),
    )
    merged["rate"] = np.where(merged["month"].between(1, 7), 0.1, 0.0)
    expected_values = merged["fuel_use__electricity__total__kwh"] * merged["rate"] / 100.0
    obtained_values = merged[expected_col].astype(float).to_numpy()
    np.testing.assert_allclose(
        obtained_values,
        expected_values.astype(float).to_numpy(),
        rtol=5e-4,
        atol=1e-6,
    )
    zero_months = merged.loc[~merged["month"].between(1, 7), expected_col]
    assert np.allclose(zero_months.to_numpy(), 0.0, atol=1e-8)
    assert merged.loc[merged["month"].between(1, 7), expected_col].gt(0).all()


def test_calculate_tou_bill_collapse_multiple_meters(
    bsq: BuildStockQuery,
    seasonal_rate_map: dict[tuple[int, int, int], float],
) -> None:
    monthly = bsq.utility.calculate_tou_bill(
        rate_map=seasonal_rate_map,
        meter_col=(
            "fuel_use__electricity__total__kwh",
            "end_use__electricity__cooling__kwh",
        ),
        timestamp_grouping_func="month",
    )
    df = bsq.utility.calculate_tou_bill(
        rate_map=seasonal_rate_map,
        meter_col=(
            "fuel_use__electricity__total__kwh",
            "end_use__electricity__cooling__kwh",
        ),
        collapse_ts=True,
    )

    assert not df.empty
    expected_cols = [
        "fuel_use__electricity__total__kwh__tou__dollars",
        "end_use__electricity__cooling__kwh__tou__dollars",
    ]
    assert set(expected_cols) <= set(df.columns)
    assert "time" not in df.columns
    assert df["sample_count"].gt(0).all()
    assert len(df) == 1
    for col in expected_cols:
        assert df[col].ge(0).all()

    totals = monthly[expected_cols].sum()
    pd.testing.assert_series_equal(
        df.iloc[0][expected_cols],
        totals,
        check_dtype=False,
        check_names=False,
    )
