from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typing import Generator
from pyathena.error import OperationalError
from buildstock_query import BuildStockQuery
from buildstock_query.schema.utilities import MappedColumn


@pytest.fixture(scope="module")
def bsq() -> Generator[BuildStockQuery, None, None]:  # pylint: disable=invalid-name
    """Shared BuildStockQuery instance for all tests."""
    try:
        obj = BuildStockQuery(
            db_name="resstock_core",
            table_name="sdr_magic17",
            workgroup="rescore",
            buildstock_type="resstock",
        )
    except OperationalError as exc:
        pytest.skip(f"Athena integration tests unavailable: {exc}")
    obj.save_cache()
    yield obj
    obj.save_cache()


class TestBuildStockQuery:
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
        assert "fuel_use_electricity_total_m_btu" in df.columns
        assert len(df) == 1, f"Expected 1 row for basic aggregate query, got {len(df)}"
        assert not df["fuel_use_electricity_total_m_btu"].isna().any()
        assert pd.api.types.is_numeric_dtype(df["fuel_use_electricity_total_m_btu"])

    def test_annual_sanity_with_groupby(self, bsq: BuildStockQuery):
        """Verify annual query with group_by returns sensible data."""
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
        )
        assert "fuel_use_electricity_total_m_btu" in df.columns
        assert "geometry_building_type_recs" in df.columns
        assert 2 <= len(df) <= 1000, f"Expected 2-1000 groups, got {len(df)}"
        assert not df["geometry_building_type_recs"].isna().any()
        assert not df["fuel_use_electricity_total_m_btu"].isna().any()

    def test_annual_sanity_with_restrict(self, bsq: BuildStockQuery):
        """Verify annual query with restrict returns sensible filtered data."""
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs", "build_existing_model.state"],
            restrict=[("build_existing_model.state", ["CA"])],
        )
        assert "fuel_use_electricity_total_m_btu" in df.columns
        assert "state" in df.columns
        assert "geometry_building_type_recs" in df.columns
        assert 1 <= len(df) <= 1000
        assert (df["state"] == "CA").all(), "Found rows not matching restrict filter"

    def test_annual_sanity_with_upgrade(self, bsq: BuildStockQuery):
        """Verify annual query with upgrade returns sensible data."""
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            upgrade_id="1",
        )
        assert "fuel_use_electricity_total_m_btu" in df.columns
        assert "geometry_building_type_recs" in df.columns
        assert 2 <= len(df) <= 1000
        assert not df["fuel_use_electricity_total_m_btu"].isna().any()
        assert not df["geometry_building_type_recs"].isna().any()

    def test_annual_sanity_with_nonzero_count(self, bsq: BuildStockQuery):
        """Verify get_nonzero_count returns valid count column."""
        df = bsq.query(
            enduses=["fuel_use_natural_gas_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            get_nonzero_count=True,
        )
        assert "fuel_use_natural_gas_total_m_btu__nonzero_units_count" in df.columns
        assert pd.api.types.is_numeric_dtype(df["fuel_use_natural_gas_total_m_btu__nonzero_units_count"])
        assert (df["fuel_use_natural_gas_total_m_btu__nonzero_units_count"] >= 0).all()

    def test_annual_sanity_with_quartiles(self, bsq: BuildStockQuery):
        """Verify get_quartiles returns valid quartile columns."""
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=["geometry_building_type_recs"],
            get_quartiles=True,
        )
        assert "fuel_use_electricity_total_m_btu__upgrade__quartiles" in df.columns
        assert 2 <= len(df) <= 1000

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
        assert "fuel_use__electricity__total__kwh" in df.columns
        assert "time" in df.columns
        assert 12 <= len(df) <= 100000
        assert not df["fuel_use__electricity__total__kwh"].isna().any()
        assert pd.api.types.is_numeric_dtype(df["fuel_use__electricity__total__kwh"])

    def test_timeseries_sanity_with_groupby(self, bsq: BuildStockQuery):
        """Verify timeseries query with group_by returns sensible data."""
        df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            group_by=["geometry_building_type_recs", "time"],
        )
        assert "fuel_use__electricity__total__kwh" in df.columns
        assert "geometry_building_type_recs" in df.columns
        assert "time" in df.columns
        assert len(df) >= 12
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
        assert "fuel_use__electricity__total__kwh" in df.columns
        assert "state" in df.columns
        assert "time" in df.columns
        assert len(df) >= 12
        assert (df["state"] == "TX").all()

    def test_timeseries_sanity_with_upgrade(self, bsq: BuildStockQuery):
        """Verify timeseries query with upgrade returns sensible data."""
        df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            upgrade_id="1",
            group_by=["geometry_building_type_recs", "time"],
        )
        assert "fuel_use__electricity__total__kwh" in df.columns
        assert "geometry_building_type_recs" in df.columns
        assert "time" in df.columns
        assert len(df) >= 12
        assert not df["fuel_use__electricity__total__kwh"].isna().any()

    def test_timeseries_sanity_timestamp_values(self, bsq: BuildStockQuery):
        """Verify timeseries timestamps are sensible."""
        df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="month",
            group_by=["time"],
        )
        assert "time" in df.columns
        assert len(df) == 12
        assert pd.api.types.is_datetime64_any_dtype(df["time"]) or isinstance(df["time"].iloc[0], pd.Timestamp)

    def test_timeseries_sanity_collapse_ts(self, bsq: BuildStockQuery):
        """Verify collapsed timeseries (peak analysis via timestamp_grouping_func='year')."""
        df = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            timestamp_grouping_func="year",
            agg_func="max",
            group_by=[bsq.bs_bldgid_column],
        )
        assert "fuel_use__electricity__total__kwh__max" in df.columns
        assert "building_id" in df.columns
        assert 2 <= len(df) <= 1000
        assert (df["fuel_use__electricity__total__kwh__max"] >= 0).all()

    # ------------------------------------------------------------------
    # Data sanity checks for get_building_average_kws_at
    # ------------------------------------------------------------------

    def test_get_building_average_kws_at_single_hour(self, bsq: BuildStockQuery):
        """Verify get_building_average_kws_at with single hour returns sensible data."""
        df = bsq.agg.get_building_average_kws_at(
            at_hour=14.0,
            at_days=[1, 100, 200],
            enduses=["fuel_use__electricity__total__kwh"],
        )
        assert "building_id" in df.columns
        assert "sample_count" in df.columns
        assert "units_count" in df.columns
        assert "fuel_use__electricity__total__kwh" in df.columns
        assert 2 <= len(df) <= 1000
        assert pd.api.types.is_numeric_dtype(df["building_id"])
        assert pd.api.types.is_numeric_dtype(df["sample_count"])
        assert pd.api.types.is_numeric_dtype(df["units_count"])
        assert pd.api.types.is_numeric_dtype(df["fuel_use__electricity__total__kwh"])
        assert (df["fuel_use__electricity__total__kwh"] >= 0).all()
        assert not df["building_id"].isna().any()
        assert not df["fuel_use__electricity__total__kwh"].isna().any()
        assert (df["sample_count"] > 0).all()
        assert (df["units_count"] > 0).all()

    def test_get_building_average_kws_at_multiple_hours(self, bsq: BuildStockQuery):
        """Verify get_building_average_kws_at with multiple hours (list) returns sensible data."""
        df = bsq.agg.get_building_average_kws_at(
            at_hour=[10.0, 14.0, 18.0],
            at_days=[1, 100, 200],
            enduses=["fuel_use__electricity__total__kwh"],
        )
        assert "building_id" in df.columns
        assert "fuel_use__electricity__total__kwh" in df.columns
        assert 2 <= len(df) <= 1000
        assert (df["fuel_use__electricity__total__kwh"] >= 0).all()
        assert not df["building_id"].isna().any()
        assert not df["fuel_use__electricity__total__kwh"].isna().any()

    def test_get_building_average_kws_at_interpolation(self, bsq: BuildStockQuery):
        """Verify get_building_average_kws_at with non-exact hour (tests interpolation)."""
        df = bsq.agg.get_building_average_kws_at(
            at_hour=14.5,
            at_days=[1, 100],
            enduses=["fuel_use__electricity__total__kwh"],
        )
        assert "building_id" in df.columns
        assert "fuel_use__electricity__total__kwh" in df.columns
        assert 2 <= len(df) <= 1000
        assert (df["fuel_use__electricity__total__kwh"] >= 0).all()
        assert df["fuel_use__electricity__total__kwh"].notna().all()
        assert np.isfinite(df["fuel_use__electricity__total__kwh"]).all()

    def test_get_building_average_kws_at_edge_days(self, bsq: BuildStockQuery):
        """Verify get_building_average_kws_at works at edge days (start/end of year)."""
        df = bsq.agg.get_building_average_kws_at(
            at_hour=12.0,
            at_days=[1, 365],
            enduses=["fuel_use__electricity__total__kwh"],
        )
        assert "building_id" in df.columns
        assert "fuel_use__electricity__total__kwh" in df.columns
        assert 2 <= len(df) <= 1000
        assert (df["fuel_use__electricity__total__kwh"] >= 0).all()
        assert df["fuel_use__electricity__total__kwh"].notna().all()

    # ------------------------------------------------------------------
    # MappedColumn integration
    # ------------------------------------------------------------------

    def test_mapped_column_with_query(self, bsq: BuildStockQuery):
        """Test that MappedColumn works with query()."""
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
        df = bsq.query(
            enduses=["fuel_use_electricity_total_m_btu"],
            group_by=[simple_bldg_col],
            get_query_only=False,
        )
        assert not df.empty
        assert "simple_bldg_type" in df.columns
        assert set(df['simple_bldg_type'].unique()) == {'SF', 'MF', 'MH'}

    def test_mapped_column_with_query_and_groupby(self, bsq: BuildStockQuery):
        """Test that MappedColumn works with query() when used as an enduse."""
        usage_weight = {"High": 1.2, "Medium": 1, "Low": 0.8, "None": 0}
        bldg_col = bsq._get_column('build_existing_model.usage_level')
        impact_col = MappedColumn(
            bsq=bsq, name='usage_weight', mapping_dict=usage_weight, key=bldg_col,
        )
        df = bsq.query(enduses=[impact_col], get_query_only=False)
        assert not df.empty
        assert "usage_weight" in df.columns
        assert df['usage_weight'].sum() > 0
