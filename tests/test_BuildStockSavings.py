import numpy as np
from unittest.mock import MagicMock
from buildstock_query.main import BuildStockQuery, SimInfo
import pytest
from tests.utils import assert_query_equal, load_tbl_from_pkl, load_cache_from_pkl
from buildstock_query.helpers import KWH2MBTU
import re
import tempfile
import pandas as pd
from typing import Union
from typing_extensions import assert_type
from typing import Generator

@pytest.fixture
def temp_history_file():
    history_file = tempfile.NamedTemporaryFile()
    name = history_file.name
    history_file.close()
    return name

@pytest.fixture(scope="module")
def my_athena() -> Generator[BuildStockQuery, None, None]:  # pylint: disable=invalid-name
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

class TestResStockSavings:
    def test_get_available_upgrades(self, my_athena):
        available_upgrade = [int(upg) for upg in my_athena.get_available_upgrades()]
        assert len(available_upgrade) > 1
        assert sorted(available_upgrade) == list(range(0, len(available_upgrade)))

    def test_savings_shape(self, my_athena: BuildStockQuery):
        enduses = ["fuel_use_electricity_total_m_btu"]
        success_report = my_athena.report.get_success_report()
        annual_savings_full = my_athena.savings.savings_shape(upgrade_id="1", enduses=enduses)
        annual_savings_applied = my_athena.savings.savings_shape(upgrade_id="1", enduses=enduses, applied_only=True)
        annual_bs_consumption = my_athena.agg.aggregate_annual(enduses=enduses)
        annual_up_consumption = my_athena.agg.aggregate_annual(upgrade_id="1", enduses=enduses)
        assert len(annual_savings_full) == len(annual_savings_applied) == 1
        assert annual_savings_full["sample_count"].iloc[0] == success_report.loc["0"].success
        assert annual_savings_applied["sample_count"].iloc[0] == success_report.loc["1"].success
        assert annual_up_consumption["sample_count"].iloc[0] == success_report.loc["1"].success
        assert np.isclose(
            annual_savings_full[f"{enduses[0]}__baseline"], annual_bs_consumption[f"{enduses[0]}"], rtol=1e-3
        )
        # Absolute savings between applied and full should be same
        assert np.isclose(
            annual_savings_applied[f"{enduses[0]}__savings"], annual_savings_full[f"{enduses[0]}__savings"], rtol=1e-3
        )
        # The savings for applied buildings must equal diff of upgrade vs corresponding baseline
        diff = annual_savings_applied[f"{enduses[0]}__baseline"] - annual_up_consumption[f"{enduses[0]}"]
        assert np.isclose(diff, annual_savings_applied[f"{enduses[0]}__savings"], rtol=1e-3)
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        ts_savings_full = my_athena.savings.savings_shape(upgrade_id="1", enduses=ts_enduses, annual_only=False)
        ts_savings_applied = my_athena.savings.savings_shape(
            upgrade_id="1", enduses=ts_enduses, applied_only=True, annual_only=False
        )
        assert len(ts_savings_full) == len(ts_savings_applied) == 35040
        assert ts_savings_full["sample_count"].iloc[0] == success_report.loc["0"].success
        assert ts_savings_applied["sample_count"].iloc[0] == success_report.loc["1"].success
        # Match with annual result
        ts_full_bsline = ts_savings_full[f"{ts_enduses[0]}__baseline"].sum() * KWH2MBTU
        assert np.isclose(ts_full_bsline, annual_savings_full[f"{enduses[0]}__baseline"], rtol=1e-3)
        ts_applied_bsline = ts_savings_applied[f"{ts_enduses[0]}__baseline"].sum() * KWH2MBTU
        assert np.isclose(ts_applied_bsline, annual_savings_applied[f"{enduses[0]}__baseline"], rtol=1e-3)
        # Match savings with annual savings
        ts_full_savings = ts_savings_full[f"{ts_enduses[0]}__savings"].sum() * KWH2MBTU
        assert np.isclose(ts_full_savings, annual_savings_full[f"{enduses[0]}__savings"], rtol=1e-3)
        ts_applied_savings = ts_savings_applied[f"{ts_enduses[0]}__savings"].sum() * KWH2MBTU
        assert np.isclose(ts_applied_savings, annual_savings_applied[f"{enduses[0]}__savings"], rtol=1e-3)
        # Verify that the absolute value is not zero
        assert ts_full_savings != 0

    def test_savings_shape_with_query(self, my_athena: BuildStockQuery):
        enduses = ["fuel_use_electricity_total_m_btu"]
        success_report = my_athena.report.get_success_report()
        annual_savings_full = my_athena.query(
            upgrade_id="1",
            enduses=enduses,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            applied_only=False,
        )
        annual_savings_applied = my_athena.query(
            upgrade_id="1",
            enduses=enduses,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            applied_only=True,
        )
        annual_bs_consumption = my_athena.query(enduses=enduses)
        annual_up_consumption = my_athena.query(upgrade_id="1", enduses=enduses)

        assert len(annual_savings_full) == len(annual_savings_applied) == 1
        assert annual_savings_full["sample_count"].iloc[0] == success_report.loc["0"].success
        assert annual_savings_applied["sample_count"].iloc[0] == success_report.loc["1"].success
        assert annual_up_consumption["sample_count"].iloc[0] == success_report.loc["1"].success
        assert np.isclose(
            annual_savings_full[f"{enduses[0]}__baseline"], annual_bs_consumption[f"{enduses[0]}"], rtol=1e-3
        )
        # Absolute savings between applied and full should be same
        assert np.isclose(
            annual_savings_applied[f"{enduses[0]}__savings"], annual_savings_full[f"{enduses[0]}__savings"], rtol=1e-3
        )
        # The savings for applied buildings must equal diff of upgrade vs corresponding baseline
        diff = annual_savings_applied[f"{enduses[0]}__baseline"] - annual_up_consumption[f"{enduses[0]}"]
        assert np.isclose(diff, annual_savings_applied[f"{enduses[0]}__savings"], rtol=1e-3)
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        ts_savings_full = my_athena.query(
            upgrade_id="1",
            enduses=ts_enduses,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            annual_only=False,
            applied_only=False,
        )
        ts_savings_applied = my_athena.query(
            upgrade_id="1",
            enduses=ts_enduses,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            annual_only=False,
            applied_only=True,
        )
        assert len(ts_savings_full) == len(ts_savings_applied) == 35040
        assert ts_savings_full["sample_count"].iloc[0] == success_report.loc["0"].success
        assert ts_savings_applied["sample_count"].iloc[0] == success_report.loc["1"].success
        # Match with annual result
        ts_full_bsline = ts_savings_full[f"{ts_enduses[0]}__baseline"].sum() * KWH2MBTU
        assert np.isclose(ts_full_bsline, annual_savings_full[f"{enduses[0]}__baseline"], rtol=1e-3)
        ts_applied_bsline = ts_savings_applied[f"{ts_enduses[0]}__baseline"].sum() * KWH2MBTU
        assert np.isclose(ts_applied_bsline, annual_savings_applied[f"{enduses[0]}__baseline"], rtol=1e-3)
        # Match savings with annual savings
        ts_full_savings = ts_savings_full[f"{ts_enduses[0]}__savings"].sum() * KWH2MBTU
        assert np.isclose(ts_full_savings, annual_savings_full[f"{enduses[0]}__savings"], rtol=1e-3)
        ts_applied_savings = ts_savings_applied[f"{ts_enduses[0]}__savings"].sum() * KWH2MBTU
        assert np.isclose(ts_applied_savings, annual_savings_applied[f"{enduses[0]}__savings"], rtol=1e-3)
        # Verify that the absolute value is correct
        assert ts_full_savings != 0

    def test_savings_shape_with_grouping(self, my_athena: BuildStockQuery):
        enduses = ["fuel_use_electricity_total_m_btu"]
        group_by = ["geometry_building_type_recs"]
        rtol = 2e-3  # 0.2% relative tolerance for matching annual values to timeseries values
        mbtu_atol = 0.01  # absolute mbtu tolerance per unit for closeness comparison
        # group_by = [my_athena.bs_bldgid_column]
        n_groups = my_athena.execute(f"""select count(distinct("build_existing_model.geometry_building_type_recs"))
                                     from {my_athena.bs_table.name}""").values[0][0]
        success_report = my_athena.report.get_success_report()
        annual_savings_full = my_athena.savings.savings_shape(
            upgrade_id="1", enduses=enduses, group_by=group_by, sort=True
        )
        annual_savings_applied = my_athena.savings.savings_shape(
            upgrade_id="1", enduses=enduses, group_by=group_by, applied_only=True, sort=True
        )
        annual_bs_consumption = my_athena.agg.aggregate_annual(enduses=enduses, group_by=group_by, sort=True)
        annual_up_consumption = my_athena.agg.aggregate_annual(
            upgrade_id="1", enduses=enduses, group_by=group_by, sort=True
        )
        assert len(annual_savings_full) == len(annual_savings_applied) == n_groups
        assert annual_savings_full["sample_count"].sum() == success_report.loc["0"].success
        assert annual_savings_applied["sample_count"].sum() == success_report.loc["1"].success
        assert annual_up_consumption["sample_count"].sum() == success_report.loc["1"].success
        applied_units = annual_savings_applied["units_count"].iloc[0]
        assert np.isclose(
            annual_savings_full[f"{enduses[0]}__baseline"],
            annual_bs_consumption[f"{enduses[0]}"],
            rtol=rtol,
            atol=mbtu_atol * applied_units,
        ).all()
        # Absolute savings between applied and full should be same
        assert np.isclose(
            annual_savings_applied[f"{enduses[0]}__savings"],
            annual_savings_full[f"{enduses[0]}__savings"],
            rtol=rtol,
            atol=mbtu_atol * applied_units,
        ).all()
        # The savings for applied buildings must equal diff of upgrade vs corresponding baseline
        diff = annual_savings_applied[f"{enduses[0]}__baseline"] - annual_up_consumption[f"{enduses[0]}"]
        assert np.isclose(
            diff, annual_savings_applied[f"{enduses[0]}__savings"], rtol=rtol, atol=mbtu_atol * applied_units
        ).all()
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        ts_savings_full = my_athena.savings.savings_shape(
            upgrade_id="1", enduses=ts_enduses, annual_only=False, group_by=group_by
        )
        ts_savings_applied = my_athena.savings.savings_shape(
            upgrade_id="1", enduses=ts_enduses, applied_only=True, annual_only=False, group_by=group_by
        )
        assert len(ts_savings_full) == len(ts_savings_applied) == 35040 * n_groups
        assert ts_savings_full.groupby(group_by)["sample_count"].mean().sum() == success_report.loc["0"].success
        assert ts_savings_applied.groupby(group_by)["sample_count"].mean().sum() == success_report.loc["1"].success
        # Match with annual result
        ts_full_bsline = ts_savings_full.groupby(group_by)[f"{ts_enduses[0]}__baseline"].sum() * KWH2MBTU
        assert np.isclose(
            ts_full_bsline, annual_savings_full[f"{enduses[0]}__baseline"], rtol=rtol, atol=mbtu_atol * applied_units
        ).all()
        ts_applied_bsline = ts_savings_applied.groupby(group_by)[f"{ts_enduses[0]}__baseline"].sum() * KWH2MBTU
        assert np.isclose(
            ts_applied_bsline,
            annual_savings_applied[f"{enduses[0]}__baseline"],
            rtol=rtol,
            atol=mbtu_atol * applied_units,
        ).all()
        # Match savings with annual savings
        ts_full_savings = ts_savings_full.groupby(group_by)[f"{ts_enduses[0]}__savings"].sum() * KWH2MBTU
        assert np.isclose(
            ts_full_savings, annual_savings_full[f"{enduses[0]}__savings"], rtol=rtol, atol=mbtu_atol * applied_units
        ).all()
        ts_applied_savings = ts_savings_applied.groupby(group_by)[f"{ts_enduses[0]}__savings"].sum() * KWH2MBTU
        assert np.isclose(
            ts_applied_savings,
            annual_savings_applied[f"{enduses[0]}__savings"],
            rtol=rtol,
            atol=mbtu_atol * applied_units,
        ).all()

    def test_savings_shape_with_grouping_using_query(self, my_athena: BuildStockQuery):
        enduses = ["fuel_use_electricity_total_m_btu"]
        group_by = ["geometry_building_type_recs"]
        rtol = 2e-3  # 0.2% relative tolerance for matching annual values to timeseries values
        mbtu_atol = 0.01  # absolute mbtu tolerance per unit for closeness comparison
        # group_by = [my_athena.bs_bldgid_column]
        n_groups = my_athena.execute(f"""select count(distinct("build_existing_model.geometry_building_type_recs"))
                                     from {my_athena.bs_table.name}""").values[0][0]
        success_report = my_athena.report.get_success_report()
        annual_savings_full = my_athena.query(
            upgrade_id="1",
            enduses=enduses,
            group_by=group_by,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            annual_only=True,
            applied_only=False,
            sort=True,
        )
        annual_savings_applied = my_athena.query(
            upgrade_id="1",
            enduses=enduses,
            group_by=group_by,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            annual_only=True,
            applied_only=True,
            sort=True,
        )
        annual_bs_consumption = my_athena.query(enduses=enduses, group_by=group_by, sort=True)
        annual_up_consumption = my_athena.query(upgrade_id="1", enduses=enduses, group_by=group_by, sort=True)
        assert len(annual_savings_full) == len(annual_savings_applied) == n_groups
        assert annual_savings_full["sample_count"].sum() == success_report.loc["0"].success
        assert annual_savings_applied["sample_count"].sum() == success_report.loc["1"].success
        assert annual_up_consumption["sample_count"].sum() == success_report.loc["1"].success
        applied_units = annual_savings_applied["units_count"].iloc[0]
        assert np.isclose(
            annual_savings_full[f"{enduses[0]}__baseline"],
            annual_bs_consumption[f"{enduses[0]}"],
            rtol=rtol,
            atol=mbtu_atol * applied_units,
        ).all()
        # Absolute savings between applied and full should be same
        assert np.isclose(
            annual_savings_applied[f"{enduses[0]}__savings"],
            annual_savings_full[f"{enduses[0]}__savings"],
            rtol=rtol,
            atol=mbtu_atol * applied_units,
        ).all()
        # The savings for applied buildings must equal diff of upgrade vs corresponding baseline
        diff = annual_savings_applied[f"{enduses[0]}__baseline"] - annual_up_consumption[f"{enduses[0]}"]
        assert np.isclose(
            diff, annual_savings_applied[f"{enduses[0]}__savings"], rtol=rtol, atol=mbtu_atol * applied_units
        ).all()
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        ts_savings_full = my_athena.query(
            upgrade_id="1",
            enduses=ts_enduses,
            annual_only=False,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            applied_only=False,
            group_by=group_by,
        )
        ts_savings_applied = my_athena.query(
            upgrade_id="1",
            enduses=ts_enduses,
            applied_only=True,
            annual_only=False,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            group_by=group_by,
        )
        assert len(ts_savings_full) == len(ts_savings_applied) == 35040 * n_groups
        assert ts_savings_full.groupby(group_by)["sample_count"].mean().sum() == success_report.loc["0"].success
        assert ts_savings_applied.groupby(group_by)["sample_count"].mean().sum() == success_report.loc["1"].success
        # Match with annual result
        ts_full_bsline = ts_savings_full.groupby(group_by)[f"{ts_enduses[0]}__baseline"].sum() * KWH2MBTU
        assert np.isclose(
            ts_full_bsline, annual_savings_full[f"{enduses[0]}__baseline"], rtol=rtol, atol=mbtu_atol * applied_units
        ).all()
        ts_applied_bsline = ts_savings_applied.groupby(group_by)[f"{ts_enduses[0]}__baseline"].sum() * KWH2MBTU
        assert np.isclose(
            ts_applied_bsline,
            annual_savings_applied[f"{enduses[0]}__baseline"],
            rtol=rtol,
            atol=mbtu_atol * applied_units,
        ).all()
        # Match savings with annual savings
        ts_full_savings = ts_savings_full.groupby(group_by)[f"{ts_enduses[0]}__savings"].sum() * KWH2MBTU
        assert np.isclose(
            ts_full_savings, annual_savings_full[f"{enduses[0]}__savings"], rtol=rtol, atol=mbtu_atol * applied_units
        ).all()
        ts_applied_savings = ts_savings_applied.groupby(group_by)[f"{ts_enduses[0]}__savings"].sum() * KWH2MBTU
        assert np.isclose(
            ts_applied_savings,
            annual_savings_applied[f"{enduses[0]}__savings"],
            rtol=rtol,
            atol=mbtu_atol * applied_units,
        ).all()

    def test_collapse_ts(self, my_athena: BuildStockQuery):
        group_by = ["geometry_building_type_recs"]
        n_groups = my_athena.execute(f"""select count(distinct("build_existing_model.geometry_building_type_recs"))
                                     from {my_athena.bs_table.name}""").values[0][0]
        rtol = 2e-3  # 0.2% relative tolerance for matching annual values to timeseries values
        mbtu_atol = 0.01  # absolute mbtu tolerance per unit for closeness comparison
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        ts_savings_full = my_athena.savings.savings_shape(
            upgrade_id="1", enduses=ts_enduses, annual_only=False, sort=True, group_by=group_by
        )
        ts_savings_applied = my_athena.savings.savings_shape(
            upgrade_id="1", enduses=ts_enduses, applied_only=True, annual_only=False, group_by=group_by, sort=True
        )
        ts_savings_full_collapsed = my_athena.savings.savings_shape(
            upgrade_id="1", enduses=ts_enduses, annual_only=False, group_by=group_by, sort=True, collapse_ts=True
        )
        ts_savings_applied_collapsed = my_athena.savings.savings_shape(
            upgrade_id="1",
            enduses=ts_enduses,
            applied_only=True,
            annual_only=False,
            group_by=group_by,
            sort=True,
            collapse_ts=True,
        )

        assert len(ts_savings_applied_collapsed) == n_groups
        assert len(ts_savings_full_collapsed) == n_groups
        assert (
            ts_savings_full_collapsed["sample_count"].values
            == ts_savings_full.groupby(group_by)["sample_count"].mean().values
        ).all()  # type: ignore
        assert (
            ts_savings_applied_collapsed["sample_count"].values
            == ts_savings_applied.groupby(group_by)["sample_count"].mean().values
        ).all()  # type: ignore
        applied_units = ts_savings_applied_collapsed["units_count"].iloc[0]
        value_cols = [f"{ts_enduses[0]}__baseline", f"{ts_enduses[0]}__savings"]
        for value_col in value_cols:
            assert np.isclose(
                ts_savings_full_collapsed[value_col],
                ts_savings_full.groupby(group_by)[value_col].sum(),
                rtol=rtol,
                atol=mbtu_atol * applied_units,
            ).all()
            assert np.isclose(
                ts_savings_applied_collapsed[value_col],
                ts_savings_applied.groupby(group_by)[value_col].sum(),
                rtol=rtol,
                atol=mbtu_atol * applied_units,
            ).all()

    def test_collapse_ts_with_query(self, my_athena: BuildStockQuery):
        group_by = ["geometry_building_type_recs"]
        n_groups = my_athena.execute(f"""select count(distinct("build_existing_model.geometry_building_type_recs"))
                                     from {my_athena.bs_table.name}""").values[0][0]
        rtol = 2e-3  # 0.2% relative tolerance for matching annual values to timeseries values
        mbtu_atol = 0.01  # absolute mbtu tolerance per unit for closeness comparison
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        ts_savings_full = my_athena.query(
            upgrade_id="1",
            enduses=ts_enduses,
            annual_only=False,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            sort=True,
            group_by=group_by,
        )
        ts_savings_applied = my_athena.query(
            upgrade_id="1",
            enduses=ts_enduses,
            applied_only=True,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            annual_only=False,
            group_by=group_by,
            sort=True,
        )
        ts_savings_full_collapsed = my_athena.query(
            upgrade_id="1",
            enduses=ts_enduses,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            annual_only=False,
            group_by=group_by,
            sort=True,
            timestamp_grouping_func="year",
        )
        ts_savings_applied_collapsed = my_athena.query(
            upgrade_id="1",
            enduses=ts_enduses,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            applied_only=True,
            annual_only=False,
            group_by=group_by,
            sort=True,
            timestamp_grouping_func="year",
        )

        assert len(ts_savings_applied_collapsed) == n_groups
        assert len(ts_savings_full_collapsed) == n_groups
        assert (
            ts_savings_full_collapsed["sample_count"].values
            == ts_savings_full.groupby(group_by)["sample_count"].mean().values
        ).all()  # type: ignore
        assert (
            ts_savings_applied_collapsed["sample_count"].values
            == ts_savings_applied.groupby(group_by)["sample_count"].mean().values
        ).all()  # type: ignore
        applied_units = ts_savings_applied_collapsed["units_count"].iloc[0]
        value_cols = [f"{ts_enduses[0]}__baseline", f"{ts_enduses[0]}__savings"]
        for value_col in value_cols:
            assert np.isclose(
                ts_savings_full_collapsed[value_col],
                ts_savings_full.groupby(group_by)[value_col].sum(),
                rtol=rtol,
                atol=mbtu_atol * applied_units,
            ).all()
            assert np.isclose(
                ts_savings_applied_collapsed[value_col],
                ts_savings_applied.groupby(group_by)[value_col].sum(),
                rtol=rtol,
                atol=mbtu_atol * applied_units,
            ).all()

    def test_restrict(self, my_athena: BuildStockQuery):
        group_by = ["state", "geometry_building_type_recs"]
        n_groups = my_athena.execute(f"""select count(distinct("build_existing_model.geometry_building_type_recs"))
                                     from {my_athena.bs_table.name} where "build_existing_model.state" = 'CA'""").values[0][0]
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        restrict = [("state", ["CA"])]
        ts_savings = my_athena.savings.savings_shape(
            upgrade_id="1",
            enduses=ts_enduses,
            applied_only=True,
            annual_only=False,
            group_by=group_by,
            sort=True,
            restrict=restrict,
        )
        assert len(ts_savings) == n_groups * 35040
        assert (ts_savings["state"] == "CA").all()

    def test_restrict_with_query(self, my_athena: BuildStockQuery):
        group_by = ["state", "geometry_building_type_recs"]
        n_groups = my_athena.execute(f"""select count(distinct("build_existing_model.geometry_building_type_recs"))
                                     from {my_athena.bs_table.name} where "build_existing_model.state" = 'CA'""").values[0][0]
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        restrict = [("state", ["CA"])]
        ts_savings = my_athena.query(
            upgrade_id="1",
            enduses=ts_enduses,
            applied_only=True,
            annual_only=False,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            group_by=group_by,
            sort=True,
            restrict=restrict,
        )
        assert len(ts_savings) == n_groups * 35040
        assert (ts_savings["state"] == "CA").all()

    def test_unload(self, my_athena: BuildStockQuery):
        group_by = ["state", "geometry_building_type_recs"]
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        restrict = [("state", ["CO"])]
        part_by = ["geometry_building_type_recs"]
        unload_loc = "buildstock-testing/unload_test/test1/state=CO"
        ts_savings_unload_query = my_athena.savings.savings_shape(
            upgrade_id="1",
            enduses=ts_enduses,
            applied_only=True,
            annual_only=False,
            group_by=group_by,
            sort=True,
            unload_to=unload_loc,
            partition_by=part_by,
            restrict=restrict,
            get_query_only=True,
        )
        pattern = r"UNLOAD\s+\((.*)\)\s+TO 's3://" + unload_loc + r".* partitioned_by = ARRAY\['" + part_by[0] + r"'\]"
        match = re.match(pattern, ts_savings_unload_query, re.DOTALL)
        assert match
        ts_savings_query = my_athena.savings.savings_shape(
            upgrade_id="1",
            enduses=ts_enduses,
            applied_only=True,
            annual_only=False,
            group_by=group_by,
            sort=True,
            restrict=restrict,
            get_query_only=True,
        )
        assert_query_equal(ts_savings_query, match.groups()[0])

    def test_unload_with_query(self, my_athena: BuildStockQuery):
        group_by = ["state", "geometry_building_type_recs"]
        ts_enduses = ["fuel_use__electricity__total__kwh"]
        restrict = [("state", ["CO"])]
        part_by = ["geometry_building_type_recs"]
        unload_loc = "buildstock-testing/unload_test/test1/state=CO"
        ts_savings_unload_query = my_athena.query(
            upgrade_id="1",
            enduses=ts_enduses,
            applied_only=True,
            annual_only=False,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            group_by=group_by,
            sort=True,
            unload_to=unload_loc,
            partition_by=part_by,
            restrict=restrict,
            get_query_only=True,
        )
        pattern = r"UNLOAD\s+\((.*)\)\s+TO 's3://" + unload_loc + r".* partitioned_by = ARRAY\['" + part_by[0] + r"'\]"
        match = re.match(pattern, ts_savings_unload_query, re.DOTALL)
        assert match
        ts_savings_query = my_athena.query(
            upgrade_id="1",
            enduses=ts_enduses,
            applied_only=True,
            annual_only=False,
            include_baseline=True,
            include_savings=True,
            include_upgrade=False,
            group_by=group_by,
            sort=True,
            restrict=restrict,
            get_query_only=True,
        )
        assert_query_equal(ts_savings_query, match.groups()[0])

    def static_test_savings_shape_return_type(self, my_athena: BuildStockQuery):
        """
        This test doesn't run. If there are any type related error, it should show up in the IDE through
        type checking interface. Make sure to either run mypy or have type checking enabled in your IDE.
        """
        enduses = ["fuel_use_electricity_total_m_btu"]

        my_bool = 3 < 5
        assert_type(my_athena.savings.savings_shape(upgrade_id="1", enduses=enduses), pd.DataFrame)
        assert_type(my_athena.savings.savings_shape(upgrade_id="1", enduses=enduses, get_query_only=True), str)
        assert_type(
            my_athena.savings.savings_shape(upgrade_id="1", enduses=enduses, get_query_only=False), pd.DataFrame
        )
        assert_type(
            my_athena.savings.savings_shape(upgrade_id="1", enduses=enduses, get_query_only=my_bool),
            Union[str, pd.DataFrame],
        )

    def static_test_savings_shape_return_type_with_query(self, my_athena: BuildStockQuery):
        """
        This test doesn't run. If there are any type related error, it should show up in the IDE through
        type checking interface. Make sure to either run mypy or have type checking enabled in your IDE.
        """
        enduses = ["fuel_use_electricity_total_m_btu"]

        my_bool = 3 < 5
        assert_type(my_athena.query(upgrade_id="1", enduses=enduses), pd.DataFrame)
        assert_type(my_athena.query(upgrade_id="1", enduses=enduses, get_query_only=True), str)
        assert_type(my_athena.query(upgrade_id="1", enduses=enduses, get_query_only=False), pd.DataFrame)
        assert_type(my_athena.query(upgrade_id="1", enduses=enduses, get_query_only=my_bool), Union[str, pd.DataFrame])