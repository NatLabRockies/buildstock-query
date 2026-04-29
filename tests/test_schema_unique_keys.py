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
_RESSTOCK_OEDI_SCHEMA_PATH = (
    _PROJECT_ROOT / "buildstock_query" / "db_schema" / "resstock_oedi.toml"
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
    md = sa.Table(
        "custom_run_metadata", metadata,
        sa.Column("bldg_id", sa.Integer),
        sa.Column("county", sa.String),
        sa.Column("state", sa.String),
        sa.Column("upgrade", sa.String),
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

    def _get_local_tables(self, requested_table_name):
        assert requested_table_name == "custom_run"
        return md, timeseries

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
    assert "bs.bldg_id = up.bldg_id" in query
    assert "bs.county = up.county" in query
    assert "bs.state = up.state" in query


def test_timeseries_query_uses_timeseries_unique_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """The TS aggregation joins ts_aggr (per-(bldg,bucket,state)) to a
    `bs_per_bldg` subquery (one row per (bldg,state)) on the ts unique
    keys. ts_unique_keys for the custom_join_key fixture is (bldg_id, state)."""
    bsq = _custom_join_key_bsq(monkeypatch)
    query = bsq.query(
        annual_only=False,
        enduses=["out.electricity.total.energy_consumption"],
        get_query_only=True,
    )
    # Outer JOIN ts_aggr ⋈ bs_per_bldg uses the ts unique keys.
    assert "bs_per_bldg.bldg_id = ts_aggr.bldg_id" in query
    assert "bs_per_bldg.state = ts_aggr.state" in query
    # bs_per_bldg's inner SELECT references the underlying bs columns.
    assert "FROM custom_run_metadata AS bs" in query
    assert "GROUP BY bs.bldg_id, bs.state" in query


def test_timeseries_savings_uses_unique_keys_in_subqueries(monkeypatch: pytest.MonkeyPatch) -> None:
    """TS upgrade-pair savings now uses a two-level pivot:
    - Innermost `ts_flat` subquery projects per-row enduse scalars (arithmetic
      pushed into scan).
    - Outer `ts_pivot` subquery GROUPs BY (ts_keys + timestamp) with FILTER
      aggregates per side: `SUM(_v__col) FILTER (WHERE upgrade=N)`.
    - Outer SELECT joins to metadata once on the ts unique keys.
    """
    bsq = _custom_join_key_bsq(monkeypatch, buildstock_type="resstock")
    query = bsq.query(
        upgrade_id="1", annual_only=False,
        include_savings=True, include_baseline=True, include_upgrade=False,
        enduses=["out.electricity.total.energy_consumption"],
        get_query_only=True,
    )
    # Per-side FILTER aggregates over the precomputed `_v__<enduse>` columns
    assert "FILTER (WHERE ts_flat.upgrade = '0')" in query
    assert "FILTER (WHERE ts_flat.upgrade = '1')" in query
    # Outer pivot GROUP BY uses the ts unique keys + timestamp (state-first
    # for partition-aligned hashing). Custom join schema has unique_keys
    # = (bldg_id, state) on ts.
    assert "GROUP BY ts_flat.state, ts_flat.timestamp, ts_flat.bldg_id" in query
    # Outer JOIN now goes to bs_per_bldg (per-bldg pre-aggregation),
    # not raw bs. Joins on ts unique keys.
    assert "bs_per_bldg.bldg_id = ts_aggr.bldg_id AND bs_per_bldg.state = ts_aggr.state" in query
    # No self-join on the TS table
    assert "ts_b.bldg_id = ts_u.bldg_id" not in query


def test_timeseries_pair_join_defaults_to_building_id_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When unique_keys are unconfigured, the TS pivot's GROUP BY falls back
    to (bldg_id, timestamp) — no extra key columns."""
    bsq = _custom_join_key_bsq(monkeypatch, buildstock_type="resstock")
    bsq.db_schema.unique_keys.metadata = None
    bsq.db_schema.unique_keys.timeseries = None
    bsq._initialize_tables()
    query = bsq.query(
        upgrade_id="1", annual_only=False,
        include_savings=True, include_baseline=True, include_upgrade=False,
        enduses=["out.electricity.total.energy_consumption"],
        get_query_only=True,
    )
    # Outer pivot GROUP BY on the default ts unique key (bldg_id) + timestamp.
    # No partition keys present (state was unconfigured), so only timestamp
    # leads bldg_id.
    assert "GROUP BY ts_flat.timestamp, ts_flat.bldg_id" in query
    # No state in the inner GROUP BY (since unique_keys was wiped) — but the
    # bare `state` column may still appear in WHERE/restrict clauses if the
    # query uses it; check specifically for state in the GROUP BY context.
    assert "ts_flat.state" not in query


# ---------------------------------------------------------------------------
# Key-attribute and helper-projection contracts


def test_key_attributes_reflect_configured_unique_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    bsq = _custom_join_key_bsq(monkeypatch)
    assert bsq.md_key == ("bldg_id", "county", "state")
    assert bsq.ts_key == ("bldg_id", "state")
    assert [c.name for c in bsq.md_key_cols] == list(bsq.md_key)
    assert [c.name for c in bsq.ts_key_cols] == list(bsq.ts_key)


def test_key_attributes_default_to_building_id(monkeypatch: pytest.MonkeyPatch) -> None:
    bsq = _custom_join_key_bsq(monkeypatch)
    bsq.db_schema.unique_keys.metadata = None
    bsq.db_schema.unique_keys.timeseries = None
    bsq._initialize_tables()
    assert bsq.md_key == ("bldg_id",)
    assert bsq.ts_key == ("bldg_id",)


def test_get_building_ids_returns_all_metadata_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    bsq = _custom_join_key_bsq(monkeypatch)
    query = bsq.get_building_ids(get_query_only=True)
    # Selects through the canonical bs alias of md_table.
    assert "SELECT bs.bldg_id, bs.county, bs.state" in query
    assert "FROM custom_run_metadata AS bs" in query


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
    # Outer LHS uses the bs alias (canonical metadata-side handle).
    assert "(bs.bldg_id, bs.county, bs.state) IN" in query
    # Inner subquery scans the same metadata table aliased "bs" in its own scope.
    assert "SELECT bs.bldg_id, bs.county, bs.state" in query


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
    # Single-key form: no tuple-IN; scalar IN.
    assert "bs.bldg_id IN (SELECT bs.bldg_id" in query
    assert "(bs.bldg_id," not in query


# ---------------------------------------------------------------------------
# count(distinct(...)) shape — multi-column tuple vs scalar


def test_aggregate_uses_multi_column_count_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    """sample_count for TS-aggregated queries collapses tract fan-out via
    `bs_per_bldg`: GROUP BY ts_unique_keys, count(*) AS tract_count. Outer
    sample_count = sum(bs_per_bldg.tract_count). For the custom_join_key
    fixture, ts_unique_keys = (bldg_id, state), and tract_count = number of
    metadata rows (i.e., distinct counties) per (bldg, state). Outer
    sum(tract_count) per (state, hour) bucket equals the prior
    count(DISTINCT (bldg_id, county, state)) value."""
    bsq = _custom_join_key_bsq(monkeypatch)
    _stub_sim_info(bsq, monkeypatch)
    query = bsq.query(
        annual_only=False,
        enduses=["out.electricity.total.energy_consumption"],
        timestamp_grouping_func="month",
        get_query_only=True,
    )
    # bs_per_bldg subquery groups by ts unique keys (bldg, state) with count(*).
    assert "GROUP BY bs.bldg_id, bs.state" in query
    assert "count(*) AS tract_count" in query
    # Outer sample_count sums the per-bldg tract counts.
    assert "sum(bs_per_bldg.tract_count) AS sample_count" in query


def test_aggregate_single_key_preserves_scalar_count_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-key schema: bs_per_bldg GROUPs BY bldg_id alone (no tuple).
    Outer sample_count = sum(tract_count) follows the same pattern as the
    multi-column case."""
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
    # Single-key bs_per_bldg.
    assert "GROUP BY bs.bldg_id" in query
    assert "GROUP BY bs.bldg_id, bs.state" not in query  # not the multi-key form
    assert "count(*) AS tract_count" in query
    assert "sum(bs_per_bldg.tract_count) AS sample_count" in query


# ---------------------------------------------------------------------------
# Other monkeypatch-based query-shape tests salvaged from legacy


def test_get_available_upgrades_does_not_use_success_report() -> None:
    """Regression guard: get_available_upgrades must derive from the metadata
    table directly, not by calling report.get_success_report (which is heavier
    and has its own dependencies)."""
    metadata = sa.MetaData()
    md = sa.Table(
        "test_metadata", metadata,
        sa.Column("building_id", sa.Integer),
        sa.Column("upgrade", sa.Integer),
    )
    bsq = BuildStockQuery.__new__(BuildStockQuery)
    bsq.md_table = md

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
    """For a baseline-only timeseries query (no upgrade-pair join), a restrict
    on building_id should still propagate to the TS table — not silently dropped
    just because there's no second metadata-side join to anchor it on."""
    bsq = _custom_join_key_bsq(monkeypatch, buildstock_type="resstock")
    monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0"])
    query = bsq.query(
        annual_only=False, upgrade_id="0",
        enduses=["out.electricity.total.energy_consumption"],
        restrict=[(bsq.building_id_column_name, [1])],
        get_query_only=True,
    )
    # Restrict materializes on both sides: TS (the FROM table) and bs (joined md).
    assert f"{bsq.ts_table.name}.{bsq.building_id_column_name} = 1" in query
    assert "bs.bldg_id = 1" in query


@pytest.mark.parametrize(
    "table_name",
    ["shared_run", ("shared_run_md", None)],
)
def test_get_tables_returns_md_and_ts(
    table_name: str | tuple[str, None],
) -> None:
    """`_get_tables` returns (md_table, ts_table) — the unified annual_and_metadata
    parquet plus the timeseries table. Self-join sites that need a baseline-vs-
    upgrade split construct `md.alias("bs")` / `.alias("up")` locally; there are
    no module-level baseline/upgrade attributes."""
    db_schema_dict = toml.load(_RESSTOCK_OEDI_SCHEMA_PATH)
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

    md_table, ts_table = bsq._get_tables(table_name)
    assert ts_table is None
    assert md_table is source_table

    # Local self-join aliases produce clean `FROM <real_table> AS bs / AS up`
    # — no synthesized `(SELECT * FROM ...)` subquery wrapper.
    bs = md_table.alias("bs")
    up = md_table.alias("up")
    compiled_bs = " ".join(bsq._compile(sa.select(bs.c["building_id"])).split())
    compiled_up = " ".join(bsq._compile(sa.select(up.c["building_id"])).split())
    assert "SELECT *" not in compiled_bs
    assert "SELECT *" not in compiled_up
    assert f"FROM {source_table.name} AS bs" in compiled_bs
    assert f"FROM {source_table.name} AS up" in compiled_up

    compiled_available_upgrades = []

    def execute(query):
        compiled_available_upgrades.append(" ".join(bsq._compile(query).split()))
        return pd.DataFrame({"upgrade": [1, 2, 3]})

    bsq.md_table = md_table
    bsq.execute = execute
    assert bsq.get_available_upgrades() == ["0", "1", "2", "3"]
    assert f"SELECT DISTINCT {source_table.name}.upgrade" in compiled_available_upgrades[0]
    assert "ORDER BY 1" in compiled_available_upgrades[0]


def test_timeseries_query_supports_subquery_restrict_with_ts_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A SA subquery passed as a restrict criterion should be inlined as
    `IN (SELECT ...)` on the timeseries column. Catches regressions where
    only literal lists were supported."""
    bsq = _custom_join_key_bsq(monkeypatch, buildstock_type="resstock")
    monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0", "1", "2", "3", "4"])

    md = bsq.md_table
    eligible_buildings = (
        sa.select(md.c.bldg_id)
        .where(
            md.c.state == "CO",
            md.c.upgrade.in_(["1", "2", "3", "4"]),
            md.c.applicability == "true",
        )
        .group_by(md.c.bldg_id)
        .having(sa.func.count() == 4)
    )

    query = bsq.query(
        annual_only=False, upgrade_id="1",
        enduses=["out.electricity.total.energy_consumption"],
        restrict=[(bsq.ts_bldgid_column, eligible_buildings)],
        get_query_only=True,
    )
    assert "IN (SELECT" in query
    assert "HAVING count(*) = 4" in query
    assert "applicability = 'true'" in query
    # Subquery body should select from the actual md_table (not an alias of it).
    assert f"FROM {bsq.md_table.name} WHERE" in query


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
    bsq = _custom_join_key_bsq(monkeypatch, buildstock_type="resstock")
    monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0", "1"])
    with pytest.raises(
        ValueError,
        match="Use `upgrade_id` instead of a `restrict` on the timeseries `upgrade` column",
    ):
        bsq.query(
            annual_only=False, upgrade_id="1",
            enduses=["out.electricity.total.energy_consumption"],
            restrict=[("upgrade", [1])],
            get_query_only=True,
        )


def test_timeseries_query_applied_in_adds_subquery_restrict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """applied_in=[1,2,3,4] on a TS query should produce a subquery restrict
    with a HAVING count(distinct(upgrade)) clause that ensures the building
    appears in ALL listed upgrades (set intersection semantics).

    For composite-key schemas the LHS of the IN is the tuple of TS unique keys
    `(bldg_id, state)`, and the subquery selects the same tuple from the bs
    alias of the metadata table — mirrors the applied_in shape used by
    snapshot tests against the OEDI schemas."""
    bsq = _custom_join_key_bsq(monkeypatch, buildstock_type="resstock")
    monkeypatch.setattr(bsq, "get_available_upgrades", lambda: ["0", "1", "2", "3", "4"])
    query = bsq.query(
        annual_only=False, upgrade_id="1",
        enduses=["out.electricity.total.energy_consumption"],
        applied_in=["1", "2", "3", "4"],
        get_query_only=True,
    )
    assert "IN (SELECT" in query
    assert "HAVING count(distinct(bs.upgrade)) = 4" in query
    assert "bs.upgrade IN ('1', '2', '3', '4')" in query
    # Composite-key form: tuple-IN on the TS unique keys.
    assert "(custom_run_by_state.bldg_id, custom_run_by_state.state) IN" in query
    assert "SELECT bs.bldg_id, bs.state" in query
    # The applied-in subquery filters by applicability, not completed_status,
    # because the unified metadata table uses applicability for upgrade-applied
    # filtering (was completed_status='Success' on the legacy 3-table shape).
    assert "applicability = 'true'" in query
