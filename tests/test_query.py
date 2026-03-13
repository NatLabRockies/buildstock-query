from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typing import Generator
from buildstock_query import BuildStockQuery
from buildstock_query.schema.utilities import MappedColumn

@pytest.fixture(scope="module")
def bsq() -> Generator[BuildStockQuery, None, None]:  # pylint: disable=invalid-name
    """Shared BuildStockQuery instance for all tests."""
    """Shared BuildStockQuery instance for all tests."""
    obj = BuildStockQuery(
        db_name="resstock_core",
        table_name="sdr_magic17",
        workgroup="rescore",
        buildstock_type="resstock",
    )
    # Warm-up – ensures that subsequent queries can leverage the local cache when available
    obj.save_cache()
    yield obj
    # Save cache after all tests using this fixture are complete
    obj.save_cache()


class TestBuildStockQuery:
    # ------------------------------------------------------------------
    # Basic annual aggregations
    # ------------------------------------------------------------------

    def test_annual_electricity_agg_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
        )
        df2 = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_annual_electricity_agg_vs_query_grpby(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
        )
        df2 = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
        )
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_annual_electricity_agg_vs_query_grpby_restric(self, bsq: BuildStockQuery):
        df1_query = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs", "build_existing_model.state"],
            restrict=[("build_existing_model.state", ["CA"])],
            get_query_only=True,
        )
        df2_query = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs", "build_existing_model.state"],
            restrict=[("build_existing_model.state", ["CA"])],
            get_query_only=True,
        )
        print(f"Query1:\n{df1_query}\n")
        print(f"Query2:\n{df2_query}\n")
        df1 = bsq.execute(df1_query)
        df2 = bsq.execute(df2_query)
        print(f"DF1 states: {df1['state'].unique()}")
        print(f"DF2 states: {df2['state'].unique()}")
        pd.testing.assert_frame_equal(df1, df2)

    def test_annual_electricity_agg_vs_query_grpby_restric2(self, bsq: BuildStockQuery):
        df1_query = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs", "state"],
            restrict=[("state", ["CA"])],
            get_query_only=True,
        )
        df2_query = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs", "state"],
            restrict=[("state", ["CA"])],
            get_query_only=True,
        )
        df1 = bsq.execute(df1_query)
        df2 = bsq.execute(df2_query)
        pd.testing.assert_frame_equal(df1, df2)

    def test_annual_electricity_agg_vs_query_grpby_avoid(self, bsq: BuildStockQuery):
        df1_query = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs", "build_existing_model.state"],
            avoid=[("build_existing_model.state", ["CA"])],
            get_query_only=True,
        )
        df2_query = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs", "build_existing_model.state"],
            avoid=[("build_existing_model.state", ["CA"])],
            get_query_only=True,
        )
        df1 = bsq.execute(df1_query)
        df2 = bsq.execute(df2_query)
        pd.testing.assert_frame_equal(df1, df2)

    def test_annual_electricity_agg_vs_query_upg1(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            upgrade_id="1",
        )
        df2 = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            upgrade_id="1",
        )
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_annual_electricity_agg_vs_query_quartiles(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            get_quartiles=True,
        )
        df2 = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            get_quartiles=True,
        ).rename({'fuel_use_electricity_total_m_btu__upgrade__quartiles':
                  'fuel_use_electricity_total_m_btu__quartiles'}, axis=1)
        pd.testing.assert_frame_equal(df1, df2)

    def test_annual_electricity_agg_max_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            agg_func="max",
            get_nonzero_count=True,
        )
        df2 = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            agg_func="max",
            get_nonzero_count=True,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_annual_natural_gas_agg_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_annual(
            enduses=["fuel_use_natural_gas_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            get_nonzero_count=True,
        )
        df2 = bsq.query(
            enduses=["fuel_use_natural_gas_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            get_nonzero_count=True,
        )
        pd.testing.assert_frame_equal(df1, df2)

    # ------------------------------------------------------------------
    # Time-series aggregations
    # ------------------------------------------------------------------

    def test_timeseries_tx_agg_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[("build_existing_model.state", ["TX"])],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs", "build_existing_model.state"],
            get_query_only=False,
        )
        df2 = bsq.query(
            annual_only=False,
            timestamp_grouping_func="month",
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[("build_existing_model.state", ["TX"])],
            group_by=[
                "geometry_building_type_recs",
                "build_existing_model.state",
                "time",
            ],
            get_query_only=False,
        )
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_timeseries_tx_agg_vs_query_upg1(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[("build_existing_model.state", ["TX"])],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs", "build_existing_model.state"],
            get_query_only=False,
            upgrade_id="1",
        )
        df2 = bsq.query(
            annual_only=False,
            timestamp_grouping_func="month",
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[("build_existing_model.state", ["TX"])],
            group_by=[
                "geometry_building_type_recs",
                "build_existing_model.state",
                "time",
            ],
            get_query_only=False,
            upgrade_id="1"
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_peak_electricity_per_building_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            collapse_ts=True,
            agg_func="max",
            group_by=[bsq.bs_bldgid_column],
            get_query_only=False,
        )
        df2 = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="year",
            agg_func="max",
            group_by=[bsq.bs_bldgid_column],
            get_query_only=False,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_timeseries_electricity_agg_vs_query_basic(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
        )
        df2 = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_timeseries_electricity_agg_vs_query_grpby(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs"],
        )
        df2 = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs", "time"],
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_timeseries_electricity_agg_vs_query_avoid(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs", "build_existing_model.state"],
            avoid=[("build_existing_model.state", ["CA"])],
        )
        df2 = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs", "build_existing_model.state", "time"],
            avoid=[("build_existing_model.state", ["CA"])],
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_timeseries_electricity_agg_max_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs"],
            agg_func="max",
        )
        df2 = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs", "time"],
            agg_func="max",
        )
        pd.testing.assert_frame_equal(df1, df2)


    # ------------------------------------------------------------------
    # Savings-shape & upgrade tests
    # ------------------------------------------------------------------

    def test_savings_shape_tx_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.savings.savings_shape(
            upgrade_id=2,
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[("build_existing_model.state", ["TX"])],
            annual_only=False,
        )
        df2 = bsq.query(
            upgrade_id=2,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[("build_existing_model.state", ["TX"])],
            annual_only=False,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_savings_shape_geometry_vs_query(self, bsq: BuildStockQuery):
        df1 = bsq.savings.savings_shape(
            upgrade_id=2,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
            annual_only=False,
        )
        df2 = bsq.query(
            annual_only=False,
            upgrade_id=2,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_quartile_savings_shape(self, bsq: BuildStockQuery):
        df1 = bsq.savings.savings_shape(
            upgrade_id=2,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
            get_quartiles=True,
            annual_only=False,
        )
        df2 = bsq.query(
            upgrade_id=2,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
            get_quartiles=True,
            annual_only=False,
        ).drop(columns=["fuel_use__electricity__total__kwh__baseline__quartiles"])
        pd.testing.assert_frame_equal(df1, df2)

    def test_savings_shape_annual_vs_query(self, bsq: BuildStockQuery):
        """Test annual savings query to cover __get_annual_bs_up_table."""
        df1 = bsq.savings.savings_shape(
            upgrade_id=2,
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            annual_only=True,
        )
        df2 = bsq.query(
            upgrade_id=2,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            annual_only=True,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_savings_shape_with_timestamp_grouping(self, bsq: BuildStockQuery):
        """Test savings with timestamp_grouping_func to cover elif branch."""
        df1 = bsq.savings.savings_shape(
            upgrade_id=2,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
            annual_only=False,
            timestamp_grouping_func="month",
        )
        df2 = bsq.query(
            upgrade_id=2,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
            annual_only=False,
            timestamp_grouping_func="month",
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_savings_shape_annual_applied_only_true(self, bsq: BuildStockQuery):
        """Test annual savings with applied_only=True to cover join branch in __get_annual_bs_up_table."""
        df1 = bsq.savings.savings_shape(
            upgrade_id=2,
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            annual_only=True,
            applied_only=True,
        )
        df2 = bsq.query(
            upgrade_id=2,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            annual_only=True,
            applied_only=True,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_savings_shape_annual_applied_only_false(self, bsq: BuildStockQuery):
        """Test annual savings with applied_only=False to cover outerjoin branch in __get_annual_bs_up_table."""
        df1 = bsq.savings.savings_shape(
            upgrade_id=2,
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            annual_only=True,
            applied_only=False,
        )
        df2 = bsq.query(
            upgrade_id=2,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            annual_only=True,
            applied_only=False,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_savings_shape_timeseries_applied_only_true(self, bsq: BuildStockQuery):
        """Test timeseries savings with applied_only=True to cover join branch in __get_timeseries_bs_up_table."""
        df1 = bsq.savings.savings_shape(
            upgrade_id=2,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
            annual_only=False,
            applied_only=True,
            restrict=[("build_existing_model.state", ["TX"])],
        )
        df2 = bsq.query(
            upgrade_id=2,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
            annual_only=False,
            applied_only=True,
            restrict=[("build_existing_model.state", ["TX"])],
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_savings_shape_timeseries_applied_only_false(self, bsq: BuildStockQuery):
        """Test timeseries savings with applied_only=False to cover outerjoin branch in __get_timeseries_bs_up_table."""
        df1 = bsq.savings.savings_shape(
            upgrade_id=2,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
            annual_only=False,
            applied_only=False,
            restrict=[("build_existing_model.state", ["TX"])],
        )
        df2 = bsq.query(
            upgrade_id=2,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
            annual_only=False,
            applied_only=False,
            restrict=[("build_existing_model.state", ["TX"])],
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_savings_shape_collapse_ts(self, bsq: BuildStockQuery):
        """Test savings with collapse_ts using timestamp_grouping_func='year' to cover collapse_ts branch."""
        df1 = bsq.savings.savings_shape(
            upgrade_id=2,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
            annual_only=False,
            collapse_ts=True,
        )
        df2 = bsq.query(
            upgrade_id=2,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["fuel_use__electricity__total__kwh"],
            group_by=["geometry_building_type_recs"],
            annual_only=False,
            timestamp_grouping_func="year",
        )
        pd.testing.assert_frame_equal(df1, df2)

    # ------------------------------------------------------------------
    # Upgrade-10 – applied vs all buildings
    # ------------------------------------------------------------------

    def _upgrade10_applied_bldgs(self, bsq: BuildStockQuery):  # helper
        return bsq.execute(
            f"select distinct(building_id) from {bsq.up_table.name} "
            "where upgrade = '10' and completed_status = 'Success'"
        )["building_id"].tolist()

    def _upgrade10_all_bldgs(self, bsq: BuildStockQuery):  # helper
        return bsq.execute(
            f"select distinct(building_id) from {bsq.up_table.name} where upgrade = '10'"
        )["building_id"].tolist()

    def test_upgrade10_applied_building_list(self, bsq: BuildStockQuery):
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            upgrade_id="10",
            group_by=["building_id"],
            applied_only=True,
        )
        expected = sorted(self._upgrade10_applied_bldgs(bsq))
        assert sorted(df["building_id"].tolist()) == expected

    def test_upgrade10_all_building_list(self, bsq: BuildStockQuery):
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            upgrade_id="10",
            group_by=["building_id"],
            applied_only=False,
        )
        expected = sorted(self._upgrade10_all_bldgs(bsq))
        assert sorted(df["building_id"].tolist()) == expected

    # ------------------------------------------------------------------
    # Time-series checks for upgrade 10
    # ------------------------------------------------------------------

    def test_timeseries_upgrade10_applied_vs_aggregate(self, bsq: BuildStockQuery):
        ts_df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            include_savings=False,
            include_baseline=False,
            include_upgrade=True,
            timestamp_grouping_func="month",
            upgrade_id="10",
            group_by=["building_id"],
            applied_only=True,
            get_query_only=False,
        )
        ts_df2 = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="10",
            group_by=["time", "building_id"],
            get_query_only=False,
        )
        pd.testing.assert_frame_equal(ts_df, ts_df2)

    def test_timeseries_upgrade10_applied_bldgs_match(self, bsq: BuildStockQuery):
        applied = self._upgrade10_applied_bldgs(bsq)
        ts_df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="10",
            group_by=["building_id", "time"],
            applied_only=True,
        )
        assert sorted(set(ts_df["building_id"].tolist())) == sorted(applied)

    def test_timeseries_upgrade10_all_bldgs_match(self, bsq: BuildStockQuery):
        all_bldgs = self._upgrade10_all_bldgs(bsq)
        ts_df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="10",
            group_by=["building_id", "time"],
            applied_only=False,
        )
        assert sorted(set(ts_df["building_id"].tolist())) == sorted(all_bldgs)

    # ------------------------------------------------------------------
    # Baseline verification for a single unapplied building
    # ------------------------------------------------------------------

    def test_baseline_for_unapplied_building(self, bsq: BuildStockQuery):
        all_bldgs = set(self._upgrade10_all_bldgs(bsq))
        applied = set(self._upgrade10_applied_bldgs(bsq))
        unapplied_bldgs = list(all_bldgs - applied)
        assert unapplied_bldgs, "Expected at least one unapplied building"
        ubldg = int(unapplied_bldgs[0])

        ts_df_baseline = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="0",
            group_by=["building_id", "time"],
            restrict=[("building_id", ubldg)],
        )
        ts_df_baseline_2 = bsq.agg.aggregate_timeseries(
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="0",
            group_by=["building_id", "time"],
            restrict=[("building_id", ubldg)],
        )
        pd.testing.assert_frame_equal(ts_df_baseline, ts_df_baseline_2)

        ts_df_all = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="10",
            group_by=["building_id", "time"],
            applied_only=False,
        )
        pd.testing.assert_frame_equal(
            ts_df_baseline,
            ts_df_all[ts_df_all["building_id"] == ubldg].reset_index(drop=True),
        )

    def test_baseline_column_match_applied(self, bsq: BuildStockQuery):
        all_bldgs = set(self._upgrade10_all_bldgs(bsq))
        applied = set(self._upgrade10_applied_bldgs(bsq))
        unapplied_bldgs = list(all_bldgs - applied)
        other_df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="0",
            group_by=["building_id", "time"],
            avoid=[("building_id", [int(u) for u in unapplied_bldgs])],
        ).rename(
            columns={"fuel_use__electricity__total__kwh": "fuel_use__electricity__total__kwh__baseline"}
        )
        applied_with_baseline = bsq.query(
            annual_only=False,
            include_baseline=True,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="10",
            group_by=["building_id", "time"],
            applied_only=True,
        )
        pd.testing.assert_frame_equal(
            other_df[["fuel_use__electricity__total__kwh__baseline"]],
            applied_with_baseline[["fuel_use__electricity__total__kwh__baseline"]],
        )

    # ------------------------------------------------------------------
    # Simple length checks
    # ------------------------------------------------------------------

    def test_query_len_10_avoid(self, bsq: BuildStockQuery):
        all_bldgs = set(self._upgrade10_all_bldgs(bsq))
        applied = set(self._upgrade10_applied_bldgs(bsq))
        unapplied_bldgs = list(all_bldgs - applied)
        df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="year",
            upgrade_id="0",
            group_by=["building_id", "time"],
            avoid=[("building_id", [int(u) for u in unapplied_bldgs])],
            limit=10,
            get_nonzero_count=False,
            sort=True,
        )
        assert len(df) == 10

    def test_query_len_10_upgrade1_annual(self, bsq: BuildStockQuery):
        df = bsq.query(
            annual_only=True,
            enduses=["fuel_use_natural_gas_total_m_btu"],
            upgrade_id="1",
            group_by=[bsq.up_bldgid_column, "time"],
            limit=10,
            get_nonzero_count=False,
            sort=True,
        )
        assert len(df) == 10

    # ------------------------------------------------------------------
    # Data sanity checks for annual queries
    # ------------------------------------------------------------------

    def test_annual_sanity_basic_columns_and_rows(self, bsq: BuildStockQuery):
        """Verify basic annual query returns sensible data structure."""
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
        )
        # Check columns exist
        assert "fuel_use_electricity_total_m_btu" in df.columns

        # Basic query should return 1 row (aggregate across all buildings)
        assert len(df) == 1, f"Expected 1 row for basic aggregate query, got {len(df)}"

        # Check no NaN in the enduse column
        assert not df["fuel_use_electricity_total_m_btu"].isna().any(), "Found NaN values in enduse column"

        # Check data type is numeric
        assert pd.api.types.is_numeric_dtype(df["fuel_use_electricity_total_m_btu"]), "Enduse column should be numeric"

    def test_annual_sanity_with_groupby(self, bsq: BuildStockQuery):
        """Verify annual query with group_by returns sensible data."""
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
        )

        # Check both columns exist
        assert "fuel_use_electricity_total_m_btu" in df.columns
        assert "geometry_building_type_recs" in df.columns

        # Should have multiple groups (RECS has at least a few building types)
        assert len(df) >= 2, f"Expected at least 2 groups, got {len(df)}"
        assert len(df) <= 1000, f"Expected at most 1000 rows, got {len(df)}"

        # Check no NaN in groupby column
        assert not df["geometry_building_type_recs"].isna().any(), "Found NaN values in group_by column"

        # Check no NaN in enduse column
        assert not df["fuel_use_electricity_total_m_btu"].isna().any(), "Found NaN values in enduse column"

    def test_annual_sanity_with_restrict(self, bsq: BuildStockQuery):
        """Verify annual query with restrict returns sensible filtered data."""
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs", "build_existing_model.state"],
            restrict=[("build_existing_model.state", ["CA"])],
        )

        # Check required columns exist
        assert "fuel_use_electricity_total_m_btu" in df.columns
        assert "state" in df.columns
        assert "geometry_building_type_recs" in df.columns

        # Should have at least 1 row (CA exists in dataset)
        assert len(df) >= 1, "Expected at least 1 row for CA"
        assert len(df) <= 1000, f"Expected at most 1000 rows, got {len(df)}"

        # Verify restrict actually worked - all rows should be CA
        assert (df["state"] == "CA").all(), "Found rows not matching restrict filter"

    def test_annual_sanity_with_upgrade(self, bsq: BuildStockQuery):
        """Verify annual query with upgrade returns sensible data."""
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            upgrade_id="1",
        )

        # Check columns exist
        assert "fuel_use_electricity_total_m_btu" in df.columns
        assert "geometry_building_type_recs" in df.columns

        # Should have reasonable row count
        assert 2 <= len(df) <= 1000, f"Expected 2-1000 rows, got {len(df)}"

        # Check no NaN in critical columns
        assert not df["fuel_use_electricity_total_m_btu"].isna().any()
        assert not df["geometry_building_type_recs"].isna().any()

    def test_annual_sanity_with_nonzero_count(self, bsq: BuildStockQuery):
        """Verify get_nonzero_count returns valid count column."""
        df = bsq.query(
            enduses=["fuel_use_natural_gas_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            get_nonzero_count=True,
        )

        # Check count column exists
        assert "fuel_use_natural_gas_total_m_btu__nonzero_units_count" in df.columns

        # Count should be numeric type
        assert pd.api.types.is_numeric_dtype(df["fuel_use_natural_gas_total_m_btu__nonzero_units_count"])

        # Count should be >= 0
        assert (df["fuel_use_natural_gas_total_m_btu__nonzero_units_count"] >= 0).all()

    def test_annual_sanity_with_quartiles(self, bsq: BuildStockQuery):
        """Verify get_quartiles returns valid quartile columns."""
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            get_quartiles=True,
        )

        # Check quartile column exists
        assert "fuel_use_electricity_total_m_btu__upgrade__quartiles" in df.columns

        # Should have reasonable row count
        assert 2 <= len(df) <= 1000, f"Expected 2-1000 rows, got {len(df)}"

    # ------------------------------------------------------------------
    # Data sanity checks for timeseries queries
    # ------------------------------------------------------------------

    def test_timeseries_sanity_basic_columns_and_rows(self, bsq: BuildStockQuery):
        """Verify basic timeseries query returns sensible data structure."""
        df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
        )

        # Check columns exist
        assert "fuel_use__electricity__total__kwh" in df.columns
        assert "time" in df.columns

        # Should have many rows (12 months minimum, likely more with building groups)
        assert len(df) >= 12, f"Expected at least 12 rows (monthly data), got {len(df)}"
        assert len(df) <= 100000, f"Expected reasonable number of rows, got {len(df)}"

        # Check no NaN in enduse column
        assert not df["fuel_use__electricity__total__kwh"].isna().any(), "Found NaN values in enduse column"

        # Check data type is numeric
        assert pd.api.types.is_numeric_dtype(df["fuel_use__electricity__total__kwh"])

    def test_timeseries_sanity_with_groupby(self, bsq: BuildStockQuery):
        """Verify timeseries query with group_by returns sensible data."""
        df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs", "time"],
        )

        # Check columns exist
        assert "fuel_use__electricity__total__kwh" in df.columns
        assert "geometry_building_type_recs" in df.columns
        assert "time" in df.columns

        # Should have at least 12 rows per group
        assert len(df) >= 12, f"Expected at least 12 rows, got {len(df)}"

        # Check no NaN in key columns
        assert not df["geometry_building_type_recs"].isna().any()
        assert not df["fuel_use__electricity__total__kwh"].isna().any()

    def test_timeseries_sanity_with_restrict(self, bsq: BuildStockQuery):
        """Verify timeseries query with restrict returns sensible filtered data."""
        df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            restrict=[("build_existing_model.state", ["TX"])],
            group_by=["geometry_building_type_recs", "build_existing_model.state", "time"],
        )

        # Check required columns exist
        assert "fuel_use__electricity__total__kwh" in df.columns
        assert "state" in df.columns
        assert "time" in df.columns

        # Should have at least 12 rows (12 months minimum)
        assert len(df) >= 12, f"Expected at least 12 rows, got {len(df)}"

        # Verify restrict actually worked
        assert (df["state"] == "TX").all(), "Found rows not matching restrict filter"

    def test_timeseries_sanity_with_upgrade(self, bsq: BuildStockQuery):
        """Verify timeseries query with upgrade returns sensible data."""
        df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="1",
            group_by=["geometry_building_type_recs", "time"],
        )

        # Check columns exist
        assert "fuel_use__electricity__total__kwh" in df.columns
        assert "geometry_building_type_recs" in df.columns
        assert "time" in df.columns

        # Should have at least 12 rows
        assert len(df) >= 12, f"Expected at least 12 rows, got {len(df)}"

        # Check no NaN in critical columns
        assert not df["fuel_use__electricity__total__kwh"].isna().any()

    def test_timeseries_sanity_timestamp_values(self, bsq: BuildStockQuery):
        """Verify timeseries timestamps are sensible."""
        df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            group_by=["time"],
        )

        # Check time column exists
        assert "time" in df.columns

        # Should have exactly 12 months (Jan-Dec)
        assert len(df) == 12, f"Expected 12 months, got {len(df)}"

        # Time column should be datetime or timestamp type
        assert pd.api.types.is_datetime64_any_dtype(df["time"]) or isinstance(df["time"].iloc[0], pd.Timestamp), \
            "Time column should be datetime/timestamp type"

    def test_timeseries_sanity_collapse_ts(self, bsq: BuildStockQuery):
        """Verify collapsed timeseries (peak analysis) returns sensible data."""
        df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="year",
            agg_func="max",
            group_by=[bsq.bs_bldgid_column],
        )

        # Check columns exist (with __max suffix for agg_func)
        assert "fuel_use__electricity__total__kwh__max" in df.columns
        assert "building_id" in df.columns

        # Should have reasonable number of buildings
        assert 2 <= len(df) <= 1000, f"Expected 2-1000 buildings, got {len(df)}"

        # Peak values should be positive (max electricity usage)
        assert (df["fuel_use__electricity__total__kwh__max"] >= 0).all(), "Peak values should be non-negative"

    # ------------------------------------------------------------------
    # Data sanity checks for get_building_average_kws_at
    # ------------------------------------------------------------------

    def test_get_building_average_kws_at_single_hour(self, bsq: BuildStockQuery):
        """Verify get_building_average_kws_at with single hour returns sensible data."""
        df = bsq.agg.get_building_average_kws_at(
            at_hour=14.0,  # 2 PM
            at_days=[1, 100, 200],  # Three days
            enduses=["fuel_use__electricity__total__kwh"],
        )

        # Check required columns exist
        assert "building_id" in df.columns
        assert "sample_count" in df.columns
        assert "units_count" in df.columns
        assert "fuel_use__electricity__total__kwh" in df.columns

        # Should have reasonable number of buildings
        assert 2 <= len(df) <= 1000, f"Expected 2-1000 buildings, got {len(df)}"

        # Check building_id is numeric
        assert pd.api.types.is_numeric_dtype(df["building_id"])

        # Check sample_count and units_count are numeric
        assert pd.api.types.is_numeric_dtype(df["sample_count"])
        assert pd.api.types.is_numeric_dtype(df["units_count"])

        # Check enduse column is numeric
        assert pd.api.types.is_numeric_dtype(df["fuel_use__electricity__total__kwh"])

        # Values should be non-negative (kW can't be negative)
        assert (df["fuel_use__electricity__total__kwh"] >= 0).all(), "kW values should be non-negative"

        # Check no NaN values in critical columns
        assert not df["building_id"].isna().any()
        assert not df["fuel_use__electricity__total__kwh"].isna().any()

        # sample_count and units_count should be positive
        assert (df["sample_count"] > 0).all(), "sample_count should be positive"
        assert (df["units_count"] > 0).all(), "units_count should be positive"

    def test_get_building_average_kws_at_multiple_hours(self, bsq: BuildStockQuery):
        """Verify get_building_average_kws_at with multiple hours (list) returns sensible data."""
        df = bsq.agg.get_building_average_kws_at(
            at_hour=[10.0, 14.0, 18.0],  # Different hours for different days
            at_days=[1, 100, 200],
            enduses=["fuel_use__electricity__total__kwh"],
        )

        # Check required columns exist
        assert "building_id" in df.columns
        assert "sample_count" in df.columns
        assert "units_count" in df.columns
        assert "fuel_use__electricity__total__kwh" in df.columns

        # Should have reasonable number of buildings
        assert 2 <= len(df) <= 1000, f"Expected 2-1000 buildings, got {len(df)}"

        # Check data types
        assert pd.api.types.is_numeric_dtype(df["building_id"])
        assert pd.api.types.is_numeric_dtype(df["fuel_use__electricity__total__kwh"])

        # Values should be non-negative
        assert (df["fuel_use__electricity__total__kwh"] >= 0).all()

        # Check no NaN values
        assert not df["building_id"].isna().any()
        assert not df["fuel_use__electricity__total__kwh"].isna().any()

    def test_get_building_average_kws_at_interpolation(self, bsq: BuildStockQuery):
        """Verify get_building_average_kws_at with non-exact hour (tests interpolation)."""
        df = bsq.agg.get_building_average_kws_at(
            at_hour=14.5,  # 2:30 PM - likely falls between timestamps
            at_days=[1, 100],
            enduses=["fuel_use__electricity__total__kwh"],
        )

        # Check required columns exist
        assert "building_id" in df.columns
        assert "fuel_use__electricity__total__kwh" in df.columns

        # Should have reasonable number of buildings
        assert 2 <= len(df) <= 1000, f"Expected 2-1000 buildings, got {len(df)}"

        # Values should be non-negative and finite
        assert (df["fuel_use__electricity__total__kwh"] >= 0).all()
        assert df["fuel_use__electricity__total__kwh"].notna().all()
        assert np.isfinite(df["fuel_use__electricity__total__kwh"]).all()


    def test_get_building_average_kws_at_edge_days(self, bsq: BuildStockQuery):
        """Verify get_building_average_kws_at works at edge days (start/end of year)."""
        df = bsq.agg.get_building_average_kws_at(
            at_hour=12.0,  # Noon
            at_days=[1, 365],  # First and last day of year
            enduses=["fuel_use__electricity__total__kwh"],
        )

        # Check required columns exist
        assert "building_id" in df.columns
        assert "fuel_use__electricity__total__kwh" in df.columns

        # Should have reasonable number of buildings
        assert 2 <= len(df) <= 1000, f"Expected 2-1000 buildings, got {len(df)}"

        # Values should be valid
        assert (df["fuel_use__electricity__total__kwh"] >= 0).all()
        assert df["fuel_use__electricity__total__kwh"].notna().all()

    def test_savings_shape_with_timestamp_grouping(self, bsq: BuildStockQuery):
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        group_by = ["geometry_building_type_recs"]
        df1 = bsq.savings.savings_shape(
            upgrade_id="1",
            enduses=ts_enduses,
            group_by=group_by,
            timestamp_grouping_func="hour",
            get_query_only=False,
            annual_only=False,
            sort=True,
        )
        df2 = bsq.query(
            upgrade_id="1",
            enduses=ts_enduses,
            group_by=group_by,
            annual_only=False,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            applied_only=False,
            timestamp_grouping_func="hour",
            get_query_only=False,
            sort=True,
        )
        pd.testing.assert_frame_equal(df1, df2)

    def test_mapped_column_with_query(self, bsq: BuildStockQuery):
        """Test that MappedColumn works with both aggregate_annual and query methods."""
        # Create a MappedColumn that maps building types to simplified categories
        building_type_map = {
            "Mobile Home": "MH",
            "Single-Family Detached": "SF",
            "Single-Family Attached": "SF",
            "Multi-Family with 2 - 4 Units": "MF",
            "Multi-Family with 5+ Units": "MF",
        }
        bldg_col = bsq._get_column("build_existing_model.geometry_building_type_recs")
        simple_bldg_col = MappedColumn(
            bsq=bsq,
            name="simple_bldg_type",
            mapping_dict=building_type_map,
            key=bldg_col,
        )

        # Test that aggregate_annual works with MappedColumn
        df1 = bsq.agg.aggregate_annual(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=[simple_bldg_col],
            get_query_only=False,
        )
        assert not df1.empty
        assert "simple_bldg_type" in df1.columns

        # Test that query() works with MappedColumn (the previously broken case)
        df2 = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=[simple_bldg_col],
            get_query_only=False,
        )
        assert not df2.empty
        assert "simple_bldg_type" in df2.columns

        # Results should be identical
        pd.testing.assert_frame_equal(df1, df2)
        assert set(df2['simple_bldg_type'].unique()) == {'SF', 'MF', 'MH'}

    def test_mapped_column_with_query_and_groupby(self, bsq: BuildStockQuery):
        """Test that MappedColumn works with query() when using group_by."""
        # Create a MappedColumn for dryer CO2 impact
        usage_weight = {"High": 1.2, 
                    "Medium": 1,
                    "Low": 0.8 ,
                    "None": 0}
        bldg_col = bsq._get_column('build_existing_model.usage_level')
        impact_col = MappedColumn(bsq=bsq, name='usage_weight', mapping_dict=usage_weight,
                                    key=bldg_col)
        df1 = bsq.query(enduses=[impact_col],get_query_only=False)
        assert not df1.empty
        assert "usage_weight" in df1.columns

        df2 = bsq.query(
            enduses=[impact_col],
            get_query_only=False,
        )
        assert not df2.empty
        assert "usage_weight" in df2.columns
        pd.testing.assert_frame_equal(df1, df2)
        assert df2['usage_weight'].sum() > 0