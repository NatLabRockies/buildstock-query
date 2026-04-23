from __future__ import annotations

import pathlib
from typing import Generator

import pandas as pd
import pytest
import sqlalchemy as sa
import toml
from pyathena.error import OperationalError

from buildstock_query.main import BuildStockQuery
from buildstock_query.db_schema.db_schema_model import DBSchema
from buildstock_query.schema.query_params import Query, BaseQuery, SavingsQuery


@pytest.fixture(scope="module")
def bsq() -> Generator[BuildStockQuery, None, None]:
    """Shared BuildStockQuery instance backed by the sdr_magic17 run."""
    try:
        obj = BuildStockQuery(
            db_name="resstock_core",
            table_name="sdr_magic17",
            workgroup="rescore",
            buildstock_type="resstock",
            skip_reports=True,
        )
    except OperationalError as exc:
        pytest.skip(f"Athena integration tests unavailable: {exc}")
    # Warm cache once so subsequent calls can reuse local artifacts where possible.
    obj.save_cache()
    yield obj
    obj.save_cache()


class TestBuildStockQuery:
    @staticmethod
    def _custom_join_key_bsq(monkeypatch: pytest.MonkeyPatch) -> BuildStockQuery:
        schema_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "buildstock_query"
            / "db_schema"
            / "comstock_oedi_state_and_county.toml"
        )
        db_schema_dict = toml.load(schema_path)
        metadata = sa.MetaData()
        baseline = sa.Table(
            "custom_run_metadata",
            metadata,
            sa.Column("bldg_id", sa.Integer),
            sa.Column("county", sa.String),
            sa.Column("state", sa.String),
            sa.Column("applicability", sa.String),
            sa.Column("out.electricity.total.energy_consumption", sa.Float),
        )
        timeseries = sa.Table(
            "custom_run_by_state",
            metadata,
            sa.Column("bldg_id", sa.Integer),
            sa.Column("state", sa.String),
            sa.Column("timestamp", sa.DateTime),
            sa.Column("upgrade", sa.String),
            sa.Column("out.electricity.total.energy_consumption", sa.Float),
        )
        upgrades = sa.Table(
            "custom_run_upgrades",
            metadata,
            sa.Column("bldg_id", sa.Integer),
            sa.Column("county", sa.String),
            sa.Column("state", sa.String),
            sa.Column("upgrade", sa.String),
            sa.Column("applicability", sa.String),
            sa.Column("out.electricity.total.energy_consumption", sa.Float),
        )

        def _get_local_tables(self, requested_table_name):
            assert requested_table_name == "custom_run"
            return baseline, timeseries, upgrades

        monkeypatch.setattr(BuildStockQuery, "_get_tables", _get_local_tables)
        bsq = BuildStockQuery(
            db_name="comstock_oedi",
            table_name="custom_run",
            workgroup="comstock",
            buildstock_type="comstock",
            db_schema=db_schema_dict,
            skip_reports=True,
            sample_weight_override=1,
        )
        monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0", "1"])
        return bsq

    def test_schema_accepts_custom_unique_keys(self) -> None:
        schema_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "buildstock_query"
            / "db_schema"
            / "comstock_oedi_state_and_county.toml"
        )
        db_schema_dict = toml.load(schema_path)
        db_schema_dict["unique_keys"] = {
            "metadata": ["bldg_id", "county", "state"],
            "timeseries": ["bldg_id", "state"],
        }

        db_schema = DBSchema.model_validate(db_schema_dict)

        assert db_schema.unique_keys.metadata == ["bldg_id", "county", "state"]
        assert db_schema.unique_keys.timeseries == ["bldg_id", "state"]

    def test_schema_rejects_timeseries_keys_not_in_metadata(self) -> None:
        schema_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "buildstock_query"
            / "db_schema"
            / "comstock_oedi_state_and_county.toml"
        )
        db_schema_dict = toml.load(schema_path)
        db_schema_dict["unique_keys"] = {
            "metadata": ["bldg_id", "county"],
            "timeseries": ["bldg_id", "state"],
        }

        with pytest.raises(ValueError, match="subset of unique_keys.metadata"):
            DBSchema.model_validate(db_schema_dict)

    def test_annual_query_uses_metadata_unique_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bsq = self._custom_join_key_bsq(monkeypatch)

        query_sql = bsq.query(
            upgrade_id="1",
            annual_only=True,
            enduses=["out.electricity.total.energy_consumption"],
            get_query_only=True,
        )
        aggregate_sql = bsq.agg.aggregate_annual(
            upgrade_id="1",
            enduses=["out.electricity.total.energy_consumption"],
            get_query_only=True,
        )

        for query in [query_sql, aggregate_sql]:
            assert "custom_run_metadata.bldg_id = custom_run_upgrades.bldg_id" in query
            assert "custom_run_metadata.county = custom_run_upgrades.county" in query
            assert "custom_run_metadata.state = custom_run_upgrades.state" in query

    def test_timeseries_query_uses_timeseries_unique_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bsq = self._custom_join_key_bsq(monkeypatch)

        query = bsq.agg.aggregate_timeseries(
            enduses=["out.electricity.total.energy_consumption"],
            get_query_only=True,
        )

        assert "custom_run_metadata.bldg_id = custom_run_by_state.bldg_id" in query
        assert "custom_run_metadata.state = custom_run_by_state.state" in query

    def test_timeseries_savings_uses_unique_keys_in_subqueries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bsq = self._custom_join_key_bsq(monkeypatch)

        query = bsq.query(
            upgrade_id="1",
            annual_only=False,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["out.electricity.total.energy_consumption"],
            get_query_only=True,
        )

        assert "ts_b.bldg_id = ts_u.bldg_id" in query
        assert "ts_b.state = ts_u.state" in query
        assert "ts_b.timestamp = ts_u.timestamp" in query
        assert "custom_run_metadata.state = ts_b.state" in query

    def test_timeseries_pair_join_defaults_to_building_id_when_unconfigured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bsq = self._custom_join_key_bsq(monkeypatch)
        bsq.db_schema.unique_keys.metadata = None
        bsq.db_schema.unique_keys.timeseries = None

        query = bsq.query(
            upgrade_id="1",
            annual_only=False,
            include_savings=True,
            include_baseline=True,
            include_upgrade=False,
            enduses=["out.electricity.total.energy_consumption"],
            get_query_only=True,
        )

        assert "ts_b.bldg_id = ts_u.bldg_id" in query
        assert "ts_b.state = ts_u.state" not in query
        assert "ts_b.timestamp = ts_u.timestamp" in query

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

    def test_get_available_upgrades_does_not_use_success_report(self) -> None:
        metadata = sa.MetaData()
        upgrades = sa.Table(
            "test_upgrades",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("upgrade", sa.Integer),
        )
        bsq = BuildStockQuery.__new__(BuildStockQuery)
        bsq.up_table = upgrades

        class Report:
            def get_success_report(self):
                pytest.fail("get_available_upgrades should not call get_success_report")

        executed_queries = []

        def execute(query):
            executed_queries.append(query)
            return pd.DataFrame({"upgrade": [1, 2, 3]})

        bsq.report = Report()
        bsq.execute = execute

        assert bsq.get_available_upgrades() == ["0", "1", "2", "3"]
        assert executed_queries

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
        quartiles = df[quartile_column].iloc[0]
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

    def test_timeseries_query_keeps_ts_restrict_without_upgrades(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        table_name = "small_run_baseline_20230810_100"

        metadata = sa.MetaData()
        baseline = sa.Table(
            f"{table_name}_baseline",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("completed_status", sa.String),
        )
        timeseries = sa.Table(
            f"{table_name}_timeseries",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("time", sa.DateTime),
            sa.Column("upgrade", sa.String),
            sa.Column("fuel_use__electricity__total__kwh", sa.Float),
        )

        def _get_local_tables(self, requested_table_name):
            assert requested_table_name == table_name
            return baseline, timeseries, None

        monkeypatch.setattr(BuildStockQuery, "_get_tables", _get_local_tables)
        bsq = BuildStockQuery(
            db_name="resstock_core",
            table_name=table_name,
            workgroup="rescore",
            buildstock_type="resstock",
            skip_reports=True,
            sample_weight_override=1,
        )
        monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0"])

        query = bsq.query(
            annual_only=False,
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[(bsq.building_id_column_name, [1])],
            get_query_only=True,
        )

        assert f"{bsq.ts_table.name}.{bsq.building_id_column_name} = 1" in query

    @pytest.mark.parametrize("table_name", ["shared_run", ("shared_run_baseline", None, "shared_run_baseline")])
    def test_get_tables_keeps_exported_columns_for_shared_baseline_table(
        self, table_name: str | tuple[str, None, str]
    ) -> None:
        schema_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "buildstock_query"
            / "db_schema"
            / "resstock_default.toml"
        )
        db_schema_dict = toml.load(schema_path)
        db_schema_dict["table_suffix"]["upgrades"] = db_schema_dict["table_suffix"]["baseline"]

        bsq = BuildStockQuery.__new__(BuildStockQuery)
        bsq.region_name = "us-west-2"
        bsq.db_name = "resstock_core"
        bsq.workgroup = "rescore"
        bsq.db_schema = DBSchema.model_validate(db_schema_dict)

        metadata = sa.MetaData()
        source_table = sa.Table(
            "shared_run_baseline",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("upgrade", sa.String),
            sa.Column("completed_status", sa.String),
        )
        tables = {source_table.name: source_table}

        bsq._create_athena_engine = lambda **kwargs: object()

        def _get_local_table(requested_table_name, missing_ok=False):
            if requested_table_name in tables:
                return tables[requested_table_name]
            if missing_ok:
                return None
            raise sa.exc.NoSuchTableError(requested_table_name)

        bsq._get_table = _get_local_table

        baseline_table, ts_table, upgrade_table = bsq._get_tables(table_name)

        assert ts_table is None
        assert baseline_table.c["building_id"].name == "building_id"
        assert upgrade_table.c["building_id"].name == "building_id"

        compiled_baseline = " ".join(bsq._compile(sa.select(baseline_table.c["building_id"])).split())
        compiled_upgrade = " ".join(bsq._compile(sa.select(upgrade_table.c["building_id"])).split())
        assert f"SELECT * FROM {source_table.name}" in compiled_baseline
        assert f"SELECT * FROM {source_table.name}" in compiled_upgrade
        assert "AS upgrade" in compiled_upgrade

        compiled_available_upgrades = []

        def execute(query):
            compiled_available_upgrades.append(" ".join(bsq._compile(query).split()))
            return pd.DataFrame({"upgrade": [1, 2, 3]})

        bsq.up_table = upgrade_table
        bsq.execute = execute

        assert bsq.get_available_upgrades() == ["0", "1", "2", "3"]
        assert "SELECT DISTINCT upgrade.upgrade" in compiled_available_upgrades[0]
        assert "ORDER BY upgrade.upgrade" not in compiled_available_upgrades[0]
        assert "ORDER BY 1" in compiled_available_upgrades[0]

    def test_timeseries_query_supports_subquery_restrict_with_ts_column(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        table_name = "small_run_baseline_20230810_100"

        metadata = sa.MetaData()
        baseline = sa.Table(
            f"{table_name}_baseline",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("completed_status", sa.String),
        )
        timeseries = sa.Table(
            f"{table_name}_timeseries",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("time", sa.DateTime),
            sa.Column("upgrade", sa.String),
            sa.Column("fuel_use__electricity__total__kwh", sa.Float),
        )
        upgrades = sa.Table(
            f"{table_name}_upgrades",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("upgrade", sa.String),
            sa.Column("state", sa.String),
            sa.Column("applicability", sa.Boolean),
        )

        def _get_local_tables(self, requested_table_name):
            assert requested_table_name == table_name
            return baseline, timeseries, upgrades

        monkeypatch.setattr(BuildStockQuery, "_get_tables", _get_local_tables)
        bsq = BuildStockQuery(
            db_name="resstock_core",
            table_name=table_name,
            workgroup="rescore",
            buildstock_type="resstock",
            skip_reports=True,
            sample_weight_override=1,
        )
        monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0", "1", "2", "3", "4"])

        eligible_buildings = (
            sa.select(upgrades.c.building_id)
            .where(
                upgrades.c.state == "CO",
                upgrades.c.upgrade.in_(["1", "2", "3", "4"]),
                upgrades.c.applicability.is_(True),
            )
            .group_by(upgrades.c.building_id)
            .having(sa.func.count() == 4)
        )

        query = bsq.query(
            annual_only=False,
            upgrade_id="1",
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[(bsq.ts_bldgid_column, eligible_buildings)],
            get_query_only=True,
        )

        assert "IN (SELECT" in query
        assert "HAVING count(*) = 4" in query
        assert "applicability IS true" in query

    def test_query_model_rejects_applied_in_without_applied_only(self) -> None:
        with pytest.raises(ValueError, match="applied_in cannot be set when applied_only is False"):
            Query.model_validate(
                {
                    "upgrade_id": "1",
                    "enduses": ["fuel_use__electricity__total__kwh"],
                    "applied_only": False,
                    "applied_in": ["1", "2"],
                }
            )

        with pytest.raises(ValueError, match="applied_in cannot be set when applied_only is False"):
            SavingsQuery.model_validate(
                {
                    "upgrade_id": "1",
                    "enduses": ["fuel_use__electricity__total__kwh"],
                    "applied_only": False,
                    "applied_in": ["1", "2"],
                }
            )

    @pytest.mark.parametrize(
        "query_runner",
        [
            lambda bsq, restrict: bsq.query(
                annual_only=False,
                upgrade_id="1",
                enduses=["fuel_use__electricity__total__kwh"],
                restrict=restrict,
                get_query_only=True,
            ),
            lambda bsq, restrict: bsq.savings.savings_shape(
                annual_only=False,
                upgrade_id="1",
                enduses=["fuel_use__electricity__total__kwh"],
                restrict=restrict,
                get_query_only=True,
            ),
        ],
    )
    def test_timeseries_upgrade_restrict_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, query_runner
    ) -> None:
        table_name = "small_run_baseline_20230810_100"

        metadata = sa.MetaData()
        baseline = sa.Table(
            f"{table_name}_baseline",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("completed_status", sa.String),
        )
        timeseries = sa.Table(
            f"{table_name}_timeseries",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("time", sa.DateTime),
            sa.Column("upgrade", sa.Integer),
            sa.Column("fuel_use__electricity__total__kwh", sa.Float),
        )
        upgrades = sa.Table(
            f"{table_name}_upgrades",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("upgrade", sa.String),
            sa.Column("completed_status", sa.String),
        )

        def _get_local_tables(self, requested_table_name):
            assert requested_table_name == table_name
            return baseline, timeseries, upgrades

        monkeypatch.setattr(BuildStockQuery, "_get_tables", _get_local_tables)
        bsq = BuildStockQuery(
            db_name="resstock_core",
            table_name=table_name,
            workgroup="rescore",
            buildstock_type="resstock",
            skip_reports=True,
            sample_weight_override=1,
        )
        monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0", "1"])

        with pytest.raises(
            ValueError,
            match="Use `upgrade_id` instead of a `restrict` on the timeseries `upgrade` column",
        ):
            query_runner(bsq, [("upgrade", [1])])

    def test_timeseries_query_applied_in_adds_subquery_restrict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        table_name = "small_run_baseline_20230810_100"

        metadata = sa.MetaData()
        baseline = sa.Table(
            f"{table_name}_baseline",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("completed_status", sa.String),
        )
        timeseries = sa.Table(
            f"{table_name}_timeseries",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("time", sa.DateTime),
            sa.Column("upgrade", sa.String),
            sa.Column("fuel_use__electricity__total__kwh", sa.Float),
        )
        upgrades = sa.Table(
            f"{table_name}_upgrades",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("upgrade", sa.String),
            sa.Column("completed_status", sa.String),
        )

        def _get_local_tables(self, requested_table_name):
            assert requested_table_name == table_name
            return baseline, timeseries, upgrades

        monkeypatch.setattr(BuildStockQuery, "_get_tables", _get_local_tables)
        bsq = BuildStockQuery(
            db_name="resstock_core",
            table_name=table_name,
            workgroup="rescore",
            buildstock_type="resstock",
            skip_reports=True,
            sample_weight_override=1,
        )
        monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0", "1", "2", "3", "4"])

        query = bsq.query(
            annual_only=False,
            upgrade_id="1",
            enduses=["fuel_use__electricity__total__kwh"],
            applied_in=["1", "2", "3", "4"],
            get_query_only=True,
        )

        assert "IN (SELECT" in query
        assert "HAVING count(distinct" in query
        assert "IN ('1', '2', '3', '4')" in query
        assert "completed_status = 'Success'" in query

    def test_savings_shape_applied_in_adds_subquery_restrict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        table_name = "small_run_baseline_20230810_100"

        metadata = sa.MetaData()
        baseline = sa.Table(
            f"{table_name}_baseline",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("completed_status", sa.String),
        )
        timeseries = sa.Table(
            f"{table_name}_timeseries",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("time", sa.DateTime),
            sa.Column("upgrade", sa.String),
            sa.Column("fuel_use__electricity__total__kwh", sa.Float),
        )
        upgrades = sa.Table(
            f"{table_name}_upgrades",
            metadata,
            sa.Column("building_id", sa.Integer),
            sa.Column("upgrade", sa.String),
            sa.Column("completed_status", sa.String),
        )

        def _get_local_tables(self, requested_table_name):
            assert requested_table_name == table_name
            return baseline, timeseries, upgrades

        monkeypatch.setattr(BuildStockQuery, "_get_tables", _get_local_tables)
        bsq = BuildStockQuery(
            db_name="resstock_core",
            table_name=table_name,
            workgroup="rescore",
            buildstock_type="resstock",
            skip_reports=True,
            sample_weight_override=1,
        )
        monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0", "1", "2", "3", "4"])

        query = bsq.savings.savings_shape(
            annual_only=False,
            upgrade_id="1",
            enduses=["fuel_use__electricity__total__kwh"],
            applied_only=True,
            applied_in=["1", "2", "3", "4"],
            get_query_only=True,
        )

        assert "IN (SELECT" in query
        assert "HAVING count(distinct" in query
        assert "IN ('1', '2', '3', '4')" in query
        assert "completed_status = 'Success'" in query

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

    def test_get_calculated_column_simple(self, bsq: BuildStockQuery) -> None:
        col = bsq.get_calculated_column("total", "fuel_use_electricity_total_m_btu + fuel_use_natural_gas_total_m_btu")
        sql = str(col.compile(compile_kwargs={"literal_binds": True}))
        # Both columns should appear with a + between them
        assert "+" in sql
        assert "total" in col.key

    def test_get_calculated_column_left_associativity(self, bsq: BuildStockQuery) -> None:
        """A - B - C must produce ((A - B) - C), not (A - (B - C))."""
        a, b, c = (
            "fuel_use_electricity_total_m_btu",
            "fuel_use_natural_gas_total_m_btu",
            "fuel_use_propane_total_m_btu",
        )
        col = bsq.get_calculated_column("result", f"{a} - {b} - {c}")
        sql = str(col.compile(compile_kwargs={"literal_binds": True}))
        # Left-associative: should be (A - B) - C, i.e. two subtractions at the same level
        # Right-associative would nest: A - (B - C) which SQLAlchemy renders with parens around B - C
        assert sql.count("(") <= 1 or f"({b.split('_m_btu')[0]}" not in sql  # no nested parens around B-C
        # More direct: compile A - (B - C) and verify our result differs
        cols = bsq._get_enduse_cols([a, b, c])
        right_assoc_sql = str((cols[0] - (cols[1] - cols[2])).compile(compile_kwargs={"literal_binds": True}))
        left_assoc_sql = str(((cols[0] - cols[1]) - cols[2]).compile(compile_kwargs={"literal_binds": True}))
        assert sql == left_assoc_sql
        assert sql != right_assoc_sql

    def test_get_calculated_column_parentheses(self, bsq: BuildStockQuery) -> None:
        a = "fuel_use_electricity_total_m_btu"
        b = "fuel_use_natural_gas_total_m_btu"
        c = "fuel_use_propane_total_m_btu"
        col = bsq.get_calculated_column("result", f"{a} * ({b} + {c})")
        sql = str(col.compile(compile_kwargs={"literal_binds": True}))
        # Should have multiplication and a grouped addition
        assert "*" in sql
        assert "+" in sql

    def test_get_calculated_column_numeric_literal(self, bsq: BuildStockQuery) -> None:
        col = bsq.get_calculated_column("scaled", "fuel_use_electricity_total_m_btu * 1000")
        sql = str(col.compile(compile_kwargs={"literal_binds": True}))
        assert "1000" in sql
        assert "*" in sql

    def test_get_calculated_column_invalid_chars(self, bsq: BuildStockQuery) -> None:
        with pytest.raises(ValueError, match="Invalid characters"):
            bsq.get_calculated_column("bad", "col1; DROP TABLE")

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
