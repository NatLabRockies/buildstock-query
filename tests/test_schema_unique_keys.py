"""Unit tests for schema-config / unique_keys plumbing.

These tests don't need Athena — they construct synthetic SQLAlchemy tables
in-memory and assert that BuildStockQuery emits the right SQL shapes for
various unique_keys configurations. Pinning the JOIN ON clauses, IN-tuple
vs scalar branches, and count(distinct(...)) shapes catches regressions
in the composite-key handling that the snapshot tests don't directly
exercise (snapshots compare full SQL hashes, so they CAN catch these but
without isolating the specific contract).

Salvaged from `tests/legacy/test_BuildStockQuery.py` (which targeted the
old sdr_magic17 classic table and is otherwise subsumed by the OEDI
snapshot suite). The Athena-bound tests in that file were dropped; the
schema/unique_keys/validator unit tests survive here.
"""
from __future__ import annotations

import pathlib

import pandas as pd
import pytest
import sqlalchemy as sa
import toml

from buildstock_query.main import BuildStockQuery
from buildstock_query.db_schema.db_schema_model import DBSchema
from buildstock_query.schema.query_params import Query


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_COMSTOCK_OEDI_SCHEMA_PATH = (
    _PROJECT_ROOT / "buildstock_query" / "db_schema" / "comstock_oedi_state_and_county.toml"
)
_RESSTOCK_DEFAULT_SCHEMA_PATH = (
    _PROJECT_ROOT / "buildstock_query" / "db_schema" / "resstock_default.toml"
)


# ---------------------------------------------------------------------------
# Helpers


def _custom_join_key_bsq(
    monkeypatch: pytest.MonkeyPatch,
    *,
    buildstock_type: str = "comstock",
) -> BuildStockQuery:
    """Construct a BSQ with a 3-column metadata key (bldg_id, county, state) and
    2-column timeseries key (bldg_id, state), backed by in-memory SA tables.
    Mirrors the comstock_oedi composite-key shape but with synthetic table names
    so the SQL diffs are easy to read in assertion failures.

    `buildstock_type="resstock"` switches off the comstock-only guardrail that
    blocks TS upgrade-pair savings queries — needed for tests that exercise
    that SQL shape against the synthetic schema."""
    db_schema_dict = toml.load(_COMSTOCK_OEDI_SCHEMA_PATH)
    db_schema_dict["unique_keys"] = {
        "metadata": ["bldg_id", "county", "state"],
        "timeseries": ["bldg_id", "state"],
    }
    metadata = sa.MetaData()
    baseline = sa.Table(
        "custom_run_metadata", metadata,
        sa.Column("bldg_id", sa.Integer),
        sa.Column("county", sa.String),
        sa.Column("state", sa.String),
        sa.Column("applicability", sa.String),
        sa.Column("out.electricity.total.energy_consumption", sa.Float),
    )
    timeseries = sa.Table(
        "custom_run_by_state", metadata,
        sa.Column("bldg_id", sa.Integer),
        sa.Column("state", sa.String),
        sa.Column("timestamp", sa.DateTime),
        sa.Column("upgrade", sa.String),
        sa.Column("out.electricity.total.energy_consumption", sa.Float),
    )
    upgrades = sa.Table(
        "custom_run_upgrades", metadata,
        sa.Column("bldg_id", sa.Integer),
        sa.Column("county", sa.String),
        sa.Column("state", sa.String),
        sa.Column("upgrade", sa.String),
        sa.Column("applicability", sa.String),
        sa.Column("out.electricity.total.energy_consumption", sa.Float),
    )

    def _get_local_tables(self, requested_table_name):
        assert requested_table_name == "custom_run"
        # md_table is None because this fixture builds a classic 3-table shape
        # (baseline and upgrades are physically distinct).
        return baseline, timeseries, upgrades, None

    monkeypatch.setattr(BuildStockQuery, "_get_tables", _get_local_tables)
    bsq = BuildStockQuery(
        db_name="comstock_oedi",
        table_name="custom_run",
        workgroup="comstock",
        buildstock_type=buildstock_type,
        db_schema=db_schema_dict,
        skip_reports=True,
        sample_weight_override=1,
    )
    monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0", "1"])
    return bsq


def _stub_sim_info(bsq: BuildStockQuery, monkeypatch: pytest.MonkeyPatch) -> None:
    from buildstock_query.main import SimInfo
    monkeypatch.setattr(bsq, "_get_simulation_info", lambda: SimInfo(2018, 3600, 0, "second"))


def _classic_resstock_bsq(
    monkeypatch: pytest.MonkeyPatch,
    *,
    table_name: str = "small_run_baseline_20230810_100",
    include_upgrades: bool = True,
    include_state_on_upgrades: bool = False,
) -> BuildStockQuery:
    """BSQ with classic-ResStock-style synthetic tables. Used to test query-model
    behavior on a single-key (bldg_id only) schema."""
    metadata = sa.MetaData()
    baseline = sa.Table(
        f"{table_name}_baseline", metadata,
        sa.Column("building_id", sa.Integer),
        sa.Column("completed_status", sa.String),
    )
    timeseries = sa.Table(
        f"{table_name}_timeseries", metadata,
        sa.Column("building_id", sa.Integer),
        sa.Column("time", sa.DateTime),
        sa.Column("upgrade", sa.String if include_upgrades else sa.Integer),
        sa.Column("fuel_use__electricity__total__kwh", sa.Float),
    )
    upgrades = None
    if include_upgrades:
        cols = [
            sa.Column("building_id", sa.Integer),
            sa.Column("upgrade", sa.String),
            sa.Column("completed_status", sa.String),
        ]
        if include_state_on_upgrades:
            cols.append(sa.Column("state", sa.String))
            cols.append(sa.Column("applicability", sa.Boolean))
        upgrades = sa.Table(f"{table_name}_upgrades", metadata, *cols)

    def _get_local_tables(self, requested_table_name):
        assert requested_table_name == table_name
        # Classic resstock shape: distinct baseline/upgrades parquets, md_table=None.
        return baseline, timeseries, upgrades, None

    monkeypatch.setattr(BuildStockQuery, "_get_tables", _get_local_tables)
    bsq = BuildStockQuery(
        db_name="resstock_core",
        table_name=table_name,
        workgroup="rescore",
        buildstock_type="resstock",
        skip_reports=True,
        sample_weight_override=1,
    )
    return bsq


# ---------------------------------------------------------------------------
# Schema model validation


def test_schema_accepts_custom_unique_keys() -> None:
    db_schema_dict = toml.load(_COMSTOCK_OEDI_SCHEMA_PATH)
    db_schema_dict["unique_keys"] = {
        "metadata": ["bldg_id", "county", "state"],
        "timeseries": ["bldg_id", "state"],
    }
    db_schema = DBSchema.model_validate(db_schema_dict)
    assert db_schema.unique_keys.metadata == ["bldg_id", "county", "state"]
    assert db_schema.unique_keys.timeseries == ["bldg_id", "state"]


def test_schema_rejects_timeseries_keys_not_in_metadata() -> None:
    db_schema_dict = toml.load(_COMSTOCK_OEDI_SCHEMA_PATH)
    db_schema_dict["unique_keys"] = {
        "metadata": ["bldg_id", "county"],
        "timeseries": ["bldg_id", "state"],
    }
    with pytest.raises(ValueError, match="subset of unique_keys.metadata"):
        DBSchema.model_validate(db_schema_dict)


# ---------------------------------------------------------------------------
# Composite-key SQL shape (annual / TS / savings) — emitted joins use the
# configured unique_keys.


def test_annual_query_uses_metadata_unique_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    bsq = _custom_join_key_bsq(monkeypatch)
    query = bsq.query(
        upgrade_id="1", annual_only=True,
        enduses=["out.electricity.total.energy_consumption"],
        get_query_only=True,
    )
    assert "custom_run_metadata.bldg_id = custom_run_upgrades.bldg_id" in query
    assert "custom_run_metadata.county = custom_run_upgrades.county" in query
    assert "custom_run_metadata.state = custom_run_upgrades.state" in query


def test_timeseries_query_uses_timeseries_unique_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    bsq = _custom_join_key_bsq(monkeypatch)
    query = bsq.query(
        annual_only=False,
        enduses=["out.electricity.total.energy_consumption"],
        get_query_only=True,
    )
    assert "custom_run_metadata.bldg_id = custom_run_by_state.bldg_id" in query
    assert "custom_run_metadata.state = custom_run_by_state.state" in query


def test_timeseries_savings_uses_unique_keys_in_subqueries(monkeypatch: pytest.MonkeyPatch) -> None:
    # buildstock_type="resstock" — comstock blocks TS upgrade-pair savings via
    # UnsupportedQueryShape; the unique_keys plumbing under test is identical
    # across schemas, so we test it on the resstock variant.
    bsq = _custom_join_key_bsq(monkeypatch, buildstock_type="resstock")
    query = bsq.query(
        upgrade_id="1", annual_only=False,
        include_savings=True, include_baseline=True, include_upgrade=False,
        enduses=["out.electricity.total.energy_consumption"],
        get_query_only=True,
    )
    assert "ts_b.bldg_id = ts_u.bldg_id" in query
    assert "ts_b.state = ts_u.state" in query
    assert "ts_b.timestamp = ts_u.timestamp" in query
    assert "custom_run_metadata.state = ts_b.state" in query


def test_timeseries_pair_join_defaults_to_building_id_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bsq = _custom_join_key_bsq(monkeypatch, buildstock_type="resstock")
    bsq.db_schema.unique_keys.metadata = None
    bsq.db_schema.unique_keys.timeseries = None
    # Re-initialize so cached bs_key/ts_key tuples reflect the wiped config.
    bsq._initialize_tables()
    query = bsq.query(
        upgrade_id="1", annual_only=False,
        include_savings=True, include_baseline=True, include_upgrade=False,
        enduses=["out.electricity.total.energy_consumption"],
        get_query_only=True,
    )
    assert "ts_b.bldg_id = ts_u.bldg_id" in query
    assert "ts_b.state = ts_u.state" not in query
    assert "ts_b.timestamp = ts_u.timestamp" in query


# ---------------------------------------------------------------------------
# Key-attribute and helper-projection contracts


def test_key_attributes_reflect_configured_unique_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    bsq = _custom_join_key_bsq(monkeypatch)
    assert bsq.bs_key == ("bldg_id", "county", "state")
    assert bsq.up_key == ("bldg_id", "county", "state")
    assert bsq.ts_key == ("bldg_id", "state")
    assert [c.name for c in bsq.bs_key_cols] == list(bsq.bs_key)
    assert [c.name for c in bsq.ts_key_cols] == list(bsq.ts_key)


def test_key_attributes_default_to_building_id(monkeypatch: pytest.MonkeyPatch) -> None:
    bsq = _custom_join_key_bsq(monkeypatch)
    bsq.db_schema.unique_keys.metadata = None
    bsq.db_schema.unique_keys.timeseries = None
    bsq._initialize_tables()
    assert bsq.bs_key == ("bldg_id",)
    assert bsq.up_key == ("bldg_id",)
    assert bsq.ts_key == ("bldg_id",)


def test_get_building_ids_returns_all_metadata_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    bsq = _custom_join_key_bsq(monkeypatch)
    query = bsq.get_building_ids(get_query_only=True)
    assert (
        "SELECT custom_run_metadata.bldg_id, custom_run_metadata.county, custom_run_metadata.state"
        in query
    )


# ---------------------------------------------------------------------------
# applied_in subquery shape — tuple-IN vs scalar-IN branches


def test_applied_in_uses_tuple_filter_for_multi_metadata_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    bsq = _custom_join_key_bsq(monkeypatch)
    query = bsq.query(
        upgrade_id="1", annual_only=True,
        enduses=["out.electricity.total.energy_consumption"],
        applied_in=["1"],
        get_query_only=True,
    )
    assert (
        "(custom_run_metadata.bldg_id, custom_run_metadata.county, custom_run_metadata.state) IN"
        in query
    )
    assert (
        "SELECT custom_run_upgrades.bldg_id, custom_run_upgrades.county, custom_run_upgrades.state"
        in query
    )


def test_applied_in_single_key_preserves_scalar_in_clause(monkeypatch: pytest.MonkeyPatch) -> None:
    bsq = _custom_join_key_bsq(monkeypatch)
    bsq.db_schema.unique_keys.metadata = None
    bsq.db_schema.unique_keys.timeseries = None
    bsq._initialize_tables()
    query = bsq.query(
        upgrade_id="1", annual_only=True,
        enduses=["out.electricity.total.energy_consumption"],
        applied_in=["1"],
        get_query_only=True,
    )
    assert "custom_run_metadata.bldg_id IN (SELECT custom_run_upgrades.bldg_id" in query
    assert "(custom_run_metadata.bldg_id," not in query


# ---------------------------------------------------------------------------
# count(distinct(...)) shape — multi-column tuple vs scalar


def test_aggregate_uses_multi_column_count_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    """sample_count for TS-aggregated queries uses count(DISTINCT(metadata_keys))
    so each physical building is counted once even when it has many TS rows.
    Counts from the metadata table (not TS) so the count is exact for the
    composite key — TS may not carry every key column."""
    bsq = _custom_join_key_bsq(monkeypatch)
    _stub_sim_info(bsq, monkeypatch)
    query = bsq.query(
        annual_only=False,
        enduses=["out.electricity.total.energy_consumption"],
        timestamp_grouping_func="month",
        get_query_only=True,
    )
    assert (
        "count(DISTINCT (custom_run_metadata.bldg_id, custom_run_metadata.county, "
        "custom_run_metadata.state))"
    ) in query


def test_aggregate_single_key_preserves_scalar_count_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-key schema falls back to the scalar count(distinct(bldg_id)) form
    rather than count(DISTINCT (bldg_id,)) which would be a degenerate tuple."""
    bsq = _custom_join_key_bsq(monkeypatch)
    bsq.db_schema.unique_keys.metadata = None
    bsq.db_schema.unique_keys.timeseries = None
    bsq._initialize_tables()
    _stub_sim_info(bsq, monkeypatch)
    query = bsq.query(
        annual_only=False,
        enduses=["out.electricity.total.energy_consumption"],
        timestamp_grouping_func="month",
        get_query_only=True,
    )
    assert "count(distinct(custom_run_metadata.bldg_id))" in query
    # Negative: must NOT be a tuple form
    assert "count(distinct(custom_run_metadata.bldg_id," not in query
    assert "count(DISTINCT (custom_run_metadata.bldg_id" not in query


# ---------------------------------------------------------------------------
# Other monkeypatch-based query-shape tests salvaged from legacy


def test_get_available_upgrades_does_not_use_success_report() -> None:
    """Regression guard: get_available_upgrades must derive from the upgrade
    table directly, not by calling report.get_success_report (which is heavier
    and has its own dependencies)."""
    metadata = sa.MetaData()
    upgrades = sa.Table(
        "test_upgrades", metadata,
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


def test_timeseries_query_keeps_ts_restrict_without_upgrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the upgrade table is missing, a restrict on building_id should still
    propagate to the TS table (not silently dropped)."""
    bsq = _classic_resstock_bsq(monkeypatch, include_upgrades=False)
    monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0"])
    query = bsq.query(
        annual_only=False,
        enduses=["fuel_use__electricity__total__kwh"],
        restrict=[(bsq.building_id_column_name, [1])],
        get_query_only=True,
    )
    assert f"{bsq.ts_table.name}.{bsq.building_id_column_name} = 1" in query


@pytest.mark.parametrize(
    "table_name",
    ["shared_run", ("shared_run_md", None, None)],
)
def test_get_tables_aliases_unified_metadata_table(
    table_name: str | tuple[str, None, None],
) -> None:
    """The 2-table schema shape (`annual_and_metadata` + `timeseries`) wraps the
    single physical table in two SQLAlchemy aliases — one labelled "baseline",
    one "upgrade". This produces clean `FROM md AS baseline JOIN md AS upgrade`
    SQL with no synthesized `(SELECT * FROM md WHERE upgrade=0)` subquery
    wrapper. The `md_table` attribute exposes the underlying table; `bs_table`
    and `up_table` are the aliases."""
    # Build a minimal 2-table schema dict (no Athena needed).
    db_schema_dict = toml.load(_RESSTOCK_DEFAULT_SCHEMA_PATH)
    db_schema_dict["table_suffix"] = {
        "annual_and_metadata": "_md",
        "timeseries": "_ts",
    }

    bsq = BuildStockQuery.__new__(BuildStockQuery)
    bsq.region_name = "us-west-2"
    bsq.db_name = "resstock_core"
    bsq.workgroup = "rescore"
    bsq.db_schema = DBSchema.model_validate(db_schema_dict)

    metadata = sa.MetaData()
    source_table = sa.Table(
        "shared_run_md", metadata,
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

    baseline_table, ts_table, upgrade_table, md_table = bsq._get_tables(table_name)
    assert ts_table is None
    assert md_table is source_table
    # Aliases preserve the underlying column list.
    assert baseline_table.c["building_id"].name == "building_id"
    assert upgrade_table.c["building_id"].name == "building_id"
    # Aliases must be distinct SA handles so identity-based dispatch keeps
    # discriminating bs vs up usage.
    assert baseline_table is not upgrade_table

    # SELECTing from each alias should produce `FROM <real_table> AS <alias>`,
    # not a `(SELECT * FROM ...)` subquery wrapper.
    compiled_baseline = " ".join(bsq._compile(sa.select(baseline_table.c["building_id"])).split())
    compiled_upgrade = " ".join(bsq._compile(sa.select(upgrade_table.c["building_id"])).split())
    assert "SELECT *" not in compiled_baseline
    assert "SELECT *" not in compiled_upgrade
    assert f"FROM {source_table.name} AS baseline" in compiled_baseline
    assert f"FROM {source_table.name} AS upgrade" in compiled_upgrade

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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SA subquery passed as a restrict criterion should be inlined as
    `IN (SELECT ...)` on the timeseries column. Catches regressions where
    only literal lists were supported."""
    bsq = _classic_resstock_bsq(monkeypatch, include_state_on_upgrades=True)
    monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0", "1", "2", "3", "4"])

    upgrades = bsq.up_table
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
        annual_only=False, upgrade_id="1",
        enduses=["fuel_use__electricity__total__kwh"],
        restrict=[(bsq.ts_bldgid_column, eligible_buildings)],
        get_query_only=True,
    )
    assert "IN (SELECT" in query
    assert "HAVING count(*) = 4" in query
    assert "applicability IS true" in query


def test_query_model_rejects_applied_in_without_applied_only() -> None:
    """The Query Pydantic model rejects applied_in on non-baseline upgrades
    when applied_only is False — applied_in semantics require the applicable-
    buildings filter to be active. Note: applied_in IS allowed on baseline
    queries (upgrade_id='0') as of 2026-04-26 (commit e85e741)."""
    with pytest.raises(ValueError, match="applied_in cannot be set when applied_only is False"):
        Query.model_validate(
            {
                "upgrade_id": "1",
                "enduses": ["fuel_use__electricity__total__kwh"],
                "applied_only": False,
                "applied_in": ["1", "2"],
            }
        )


def test_timeseries_upgrade_restrict_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passing a restrict on the TS-side `upgrade` column when annual_only=False
    and upgrade_id is explicitly set should error — the upgrade_id kwarg is
    the canonical way to filter upgrades. Catches users mixing the two."""
    bsq = _classic_resstock_bsq(monkeypatch)
    monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0", "1"])
    with pytest.raises(
        ValueError,
        match="Use `upgrade_id` instead of a `restrict` on the timeseries `upgrade` column",
    ):
        bsq.query(
            annual_only=False, upgrade_id="1",
            enduses=["fuel_use__electricity__total__kwh"],
            restrict=[("upgrade", [1])],
            get_query_only=True,
        )


def test_timeseries_query_applied_in_adds_subquery_restrict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """applied_in=[1,2,3,4] on a TS query should produce a subquery restrict
    with a HAVING count(distinct(upgrade)) clause that ensures the building
    appears in ALL listed upgrades (set intersection semantics)."""
    bsq = _classic_resstock_bsq(monkeypatch)
    monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0", "1", "2", "3", "4"])
    query = bsq.query(
        annual_only=False, upgrade_id="1",
        enduses=["fuel_use__electricity__total__kwh"],
        applied_in=["1", "2", "3", "4"],
        get_query_only=True,
    )
    assert "IN (SELECT" in query
    assert "HAVING count(distinct" in query
    assert "IN ('1', '2', '3', '4')" in query
    assert "completed_status = 'Success'" in query
