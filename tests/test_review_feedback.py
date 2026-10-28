from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import sqlalchemy as sa
import toml

from buildstock_query.aggregate_query import BuildStockAggregate
from buildstock_query.db_schema.db_schema_model import DBSchema
from buildstock_query.main import BuildStockQuery
from buildstock_query.query_core import QueryCore
from buildstock_query.report_query import BuildStockReport


def _make_bsq(baseline: sa.Table, upgrades: sa.Table | None = None) -> BuildStockQuery:
    schema_path = Path(__file__).resolve().parents[1] / "buildstock_query" / "db_schema" / "resstock_default.toml"
    bsq = BuildStockQuery.__new__(BuildStockQuery)
    bsq.db_schema = DBSchema.model_validate(toml.load(schema_path))
    bsq.bs_table = baseline
    bsq.up_table = upgrades
    bsq.ts_table = None
    bsq.building_id_column_name = bsq.db_schema.column_names.building_id
    bsq.run_params = SimpleNamespace(keep_column_prefix=False)
    bsq.sample_wt = sa.literal(1)
    return bsq


def test_annual_count_only_query_counts_baseline_rows_without_status() -> None:
    metadata = sa.MetaData()
    baseline = sa.Table("baseline", metadata, sa.Column("building_id", sa.Integer))
    bsq = _make_bsq(baseline)
    bsq.agg = BuildStockAggregate(bsq)
    bsq._compile = lambda query: query
    engine = sa.create_engine("sqlite://")
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(baseline.insert(), [{"building_id": 1}, {"building_id": 2}])

    query = bsq.agg.aggregate_annual(enduses=[], get_query_only=True)

    with engine.connect() as connection:
        result = connection.execute(query).mappings().one()

    assert result["sample_count"] == 2
    assert result["units_count"] == 2


def test_baseline_success_report_selects_from_baseline_table() -> None:
    metadata = sa.MetaData()
    baseline = sa.Table("baseline", metadata, sa.Column("building_id", sa.Integer))
    bsq = _make_bsq(baseline)

    query = BuildStockReport(bsq)._get_bs_success_report(get_query_only=True)

    assert "FROM baseline" in query


def test_missing_upgrade_status_preserves_null_for_unmatched_rows() -> None:
    metadata = sa.MetaData()
    baseline = sa.Table("baseline", metadata, sa.Column("building_id", sa.Integer))
    upgrades = sa.Table(
        "upgrades",
        metadata,
        sa.Column("building_id", sa.Integer),
        sa.Column("upgrade", sa.String),
    )
    bsq = _make_bsq(baseline, upgrades)
    status = bsq._up_completed_status_col
    engine = sa.create_engine("sqlite://")
    metadata.create_all(engine)

    with engine.begin() as connection:
        connection.execute(baseline.insert(), [{"building_id": 1}, {"building_id": 2}])
        connection.execute(upgrades.insert(), [{"building_id": 1, "upgrade": "1"}])
        rows = connection.execute(
            sa.select(status)
            .select_from(baseline.outerjoin(upgrades, baseline.c.building_id == upgrades.c.building_id))
            .order_by(baseline.c.building_id)
        ).scalars().all()

    assert rows == [bsq.db_schema.completion_values.success, None]


def test_missing_table_does_not_swallow_reflection_query_failures() -> None:
    query_core = QueryCore.__new__(QueryCore)
    query_core._tables = {}
    query_core._meta = sa.MetaData()
    query_core._engine = sa.create_engine("sqlite://")
    query_core.db_name = "test"

    with pytest.raises(sa.exc.OperationalError):
        query_core._get_table("missing", missing_ok=True)


def test_information_schema_reflection_reuses_existing_table() -> None:
    query_core = QueryCore.__new__(QueryCore)
    query_core._meta = sa.MetaData()
    query_core.db_name = "test"
    existing_table = sa.Table("present", query_core._meta, sa.Column("building_id", sa.Integer))
    query_core._engine = MagicMock()
    connection = query_core._engine.connect.return_value.__enter__.return_value
    connection.execute.return_value.fetchall.return_value = [("added", "varchar")]

    table = query_core._reflect_via_information_schema("present")

    assert table is existing_table
    assert table.c.keys() == ["building_id", "added"]


def test_empty_reflection_uses_information_schema_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    query_core = QueryCore.__new__(QueryCore)
    query_core._tables = {}
    query_core._meta = sa.MetaData()
    query_core._engine = sa.create_engine("sqlite://")
    query_core.db_name = "test"
    empty_table = sa.Table("empty", query_core._meta)

    def reflect_table(table_name: str) -> sa.Table:
        empty_table.append_column(sa.Column("building_id", sa.Integer))
        return empty_table

    fallback = Mock(side_effect=reflect_table)
    monkeypatch.setattr(query_core, "_reflect_via_information_schema", fallback)

    table = query_core._get_table("empty")

    assert table is empty_table
    assert table.c.keys() == ["building_id"]
    fallback.assert_called_once_with("empty")
