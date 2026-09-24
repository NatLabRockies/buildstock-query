"""Unit tests for the timeseries/baseline join condition.

Offline: the tables are built in memory, so nothing here needs Athena.
"""
from __future__ import annotations

import pytest
import sqlalchemy as sa

from buildstock_query.db_schema.db_schema_model import Structure
from buildstock_query.query_core import QueryException, build_ts_baseline_join_condition


def _tables(bs_columns: list[str], ts_columns: list[str]) -> tuple[sa.Table, sa.Table]:
    metadata = sa.MetaData()
    bs = sa.Table("run_baseline", metadata, *[sa.Column(name, sa.String) for name in bs_columns])
    ts = sa.Table("run_timeseries", metadata, *[sa.Column(name, sa.String) for name in ts_columns])
    return bs, ts


class TestBuildTsBaselineJoinCondition:
    def test_joins_on_building_id_when_no_geography_partition(self) -> None:
        """The usual one-row-per-building publication keeps the plain building id join."""
        bs, ts = _tables(["bldg_id", "in.state"], ["bldg_id", "state"])

        condition = build_ts_baseline_join_condition(
            bs_table=bs, ts_table=ts, building_id_column="bldg_id", geography_partition=None
        )

        rendered = str(condition)
        assert "run_baseline.bldg_id = run_timeseries.bldg_id" in rendered
        assert "state" not in rendered

    def test_adds_geography_to_the_join_when_configured(self) -> None:
        """One row per (building, geography) needs geography in the join as well.

        Without it every geography's timeseries pairs with every geography's weight, so
        each geography's totals come out inflated.
        """
        bs, ts = _tables(["bldg_id", "in.state"], ["bldg_id", "state"])

        condition = build_ts_baseline_join_condition(
            bs_table=bs,
            ts_table=ts,
            building_id_column="bldg_id",
            geography_partition="state",
            characteristics_prefix="in.",
        )

        rendered = str(condition)
        assert "run_baseline.bldg_id = run_timeseries.bldg_id" in rendered
        assert '"in.state"' in rendered or "in.state" in rendered
        assert "run_timeseries.state" in rendered

    def test_respects_the_characteristics_prefix(self) -> None:
        """The baseline names the column behind its characteristics prefix."""
        bs, ts = _tables(["building_id", "build_existing_model.state"], ["building_id", "state"])

        condition = build_ts_baseline_join_condition(
            bs_table=bs,
            ts_table=ts,
            building_id_column="building_id",
            geography_partition="state",
            characteristics_prefix="build_existing_model.",
        )

        assert "build_existing_model.state" in str(condition)

    @pytest.mark.parametrize(
        "bs_columns, ts_columns",
        [
            (["bldg_id"], ["bldg_id", "state"]),  # missing from the baseline
            (["bldg_id", "in.state"], ["bldg_id"]),  # missing from the timeseries
        ],
    )
    def test_raises_when_the_geography_column_is_missing(self, bs_columns, ts_columns) -> None:
        """Refuse rather than fall back: a silent fallback would double count."""
        bs, ts = _tables(bs_columns, ts_columns)

        with pytest.raises(QueryException, match="geography_partition"):
            build_ts_baseline_join_condition(
                bs_table=bs,
                ts_table=ts,
                building_id_column="bldg_id",
                geography_partition="state",
                characteristics_prefix="in.",
            )


class TestStructureSchema:
    def test_geography_partition_defaults_to_none(self) -> None:
        """Existing schemas keep the building-id-only join without any change."""
        assert Structure.model_validate({"inapplicables_have_ts": False}).geography_partition is None

    def test_geography_partition_round_trips(self) -> None:
        structure = Structure.model_validate({"inapplicables_have_ts": True, "geography_partition": "state"})
        assert structure.geography_partition == "state"
