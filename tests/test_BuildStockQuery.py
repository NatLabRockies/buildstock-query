from __future__ import annotations

import pandas as pd
import pytest
import ast
from typing import Generator
from buildstock_query.main import BuildStockQuery
from buildstock_query.schema.query_params import Query, BaseQuery


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
    # Warm cache once so subsequent calls can reuse local artifacts where possible.
    obj.save_cache()
    yield obj
    obj.save_cache()


class TestBuildStockQuery:
    def test_clean_group_by(self, bsq: BuildStockQuery) -> None:
        group_by = [
            "time",
            '"sdr_magic17_baseline"."build_existing_model.state"',
            '"build_existing_model.county"',
        ]
        clean_group_by = bsq._clean_group_by(group_by)  # pylint: disable=protected-access
        assert clean_group_by == ["time", "build_existing_model.state", "build_existing_model.county"]

        group_by_with_alias = ["time", ("month(time)", "moy"), '"build_existing_model.county"']
        clean_group_by = bsq._clean_group_by(group_by_with_alias)  # pylint: disable=protected-access
        assert clean_group_by == ["time", "moy", "build_existing_model.county"]

    def test_execute_returns_dataframe(self, bsq: BuildStockQuery) -> None:
        df = bsq.execute(f"select count(*) as total_rows from {bsq.bs_table.name}")
        assert list(df.columns) == ["total_rows"]
        assert len(df) == 1
        assert int(df.loc[0, "total_rows"]) > 0

    def test_aggregate_annual_basic(self, bsq: BuildStockQuery) -> None:
        enduses = ["fuel_use_electricity_total_m_btu", "fuel_use_natural_gas_total_m_btu"]
        df = bsq.agg.aggregate_annual(enduses=enduses)

        assert len(df) == 1
        for column in ["sample_count", "units_count", *enduses]:
            assert column in df.columns
        assert df["sample_count"].iloc[0] > 0
        assert df["units_count"].iloc[0] > 0
        assert df[enduses].apply(pd.api.types.is_numeric_dtype).all()
        assert (df[enduses] >= 0).to_numpy().all()

    def test_aggregate_annual_group_by(self, bsq: BuildStockQuery) -> None:
        df = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
        )

        assert not df.empty
        assert "geometry_building_type_recs" in df.columns
        assert df["geometry_building_type_recs"].notna().all()
        assert df["sample_count"].gt(0).all()

    def test_aggregate_annual_restrict_state(self, bsq: BuildStockQuery) -> None:
        df = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs", "build_existing_model.state"],
            restrict=[("build_existing_model.state", ["CA"])],
        )

        assert not df.empty
        assert "state" in df.columns
        assert (df["state"] == "CA").all()

    def test_aggregate_annual_upgrade(self, bsq: BuildStockQuery) -> None:
        df = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            upgrade_id="1",
        )

        assert not df.empty
        assert df["geometry_building_type_recs"].notna().all()
        assert df["fuel_use_electricity_total_m_btu"].ge(0).all()

    def test_aggregate_annual_nonzero_count(self, bsq: BuildStockQuery) -> None:
        df = bsq.agg.aggregate_annual(
            enduses=["fuel_use_natural_gas_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            get_nonzero_count=True,
        )

        col = "fuel_use_natural_gas_total_m_btu__nonzero_units_count"
        assert col in df.columns
        assert df[col].ge(0).all()

    def test_aggregate_annual_quartiles(self, bsq: BuildStockQuery) -> None:
        df = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            get_quartiles=True,
        )

        quartile_column = "fuel_use_electricity_total_m_btu__quartiles"
        assert quartile_column in df.columns
        quartiles = ast.literal_eval(df[quartile_column].iloc[0])
        assert hasattr(quartiles, "__len__")
        assert len(quartiles) == 9

    def test_aggregate_annual_matches_query(self, bsq: BuildStockQuery) -> None:
        param_dict = {
            "enduses": ["fuel_use_electricity_total_m_btu"],
            "group_by": ["geometry_building_type_recs"],
            "upgrade_id": "1",
        }

        agg_df = bsq.agg.aggregate_annual(params=BaseQuery.model_validate(param_dict))
        query_df = bsq.query(params=Query.model_validate(param_dict))
        assert isinstance(agg_df, pd.DataFrame)
        assert isinstance(query_df, pd.DataFrame)
        pd.testing.assert_frame_equal(
            agg_df.sort_values(list(agg_df.columns)).reset_index(drop=True),
            query_df.sort_values(list(query_df.columns)).reset_index(drop=True),
        )

    def test_timeseries_basic(self, bsq: BuildStockQuery) -> None:
        df = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
        )

        for column in ["time", "sample_count", "units_count", "rows_per_sample", "fuel_use__electricity__total__kwh"]:
            assert column in df.columns
        assert len(df) == 12
        assert pd.api.types.is_datetime64_any_dtype(df["time"]) or isinstance(df["time"].iloc[0], pd.Timestamp)
        assert df["fuel_use__electricity__total__kwh"].ge(0).all()

    def test_timeseries_group_by_restrict(self, bsq: BuildStockQuery) -> None:
        df = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs", "build_existing_model.state"],
            restrict=[("build_existing_model.state", ["TX"])],
        )

        assert not df.empty
        assert {"geometry_building_type_recs", "state", "time"} <= set(df.columns)
        assert (df["state"] == "TX").all()
        assert df["fuel_use__electricity__total__kwh"].ge(0).all()

    def test_timeseries_matches_query(self, bsq: BuildStockQuery) -> None:
        agg_df = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs", "build_existing_model.state"],
            restrict=[("build_existing_model.state", ["TX"])],
        )
        query_df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs", "build_existing_model.state", "time"],
            restrict=[("build_existing_model.state", ["TX"])],
        )

        sort_cols = ["geometry_building_type_recs", "state", "time"]
        pd.testing.assert_frame_equal(
            agg_df.sort_values(sort_cols).reset_index(drop=True),
            query_df.sort_values(sort_cols).reset_index(drop=True),
        )

    def test_timeseries_collapse(self, bsq: BuildStockQuery) -> None:
        df = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            collapse_ts=True,
        )
        assert "time" not in df.columns
        assert len(df) == 1
        assert df["sample_count"].iloc[0] > 0
        assert df["units_count"].iloc[0] > 0

    def test_query_limit_and_sort(self, bsq: BuildStockQuery) -> None:
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            limit=5,
            sort=True,
        )

        assert 1 <= len(df) <= 5
        assert df["geometry_building_type_recs"].is_monotonic_increasing

    def test_get_building_average_kws_at(self, bsq: BuildStockQuery) -> None:
        df = bsq.agg.get_building_average_kws_at(
            at_hour=14.0,
            at_days=[1, 100, 200],
            enduses=["fuel_use__electricity__total__kwh"],
        )

        required_columns = {"building_id", "sample_count", "units_count", "fuel_use__electricity__total__kwh"}
        assert required_columns <= set(df.columns)
        assert df["building_id"].notna().all()
        assert df["fuel_use__electricity__total__kwh"].ge(0).all()
        assert df["sample_count"].gt(0).all()
        assert df["units_count"].gt(0).all()
