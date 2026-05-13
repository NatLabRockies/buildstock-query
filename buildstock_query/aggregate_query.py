from __future__ import annotations

import datetime
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
import sqlalchemy as sa
from pydantic import Field
from sqlalchemy.sql import func as safunc, visitors
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.util import ClauseAdapter

from buildstock_query import main
from buildstock_query.schema.helpers import gather_params
from buildstock_query.schema.query_params import Query
from buildstock_query.schema.utilities import (
    ColumnExpression,
    ColumnReference,
    RestrictTuple,
    SelectQuery,
    SqlExpression,
    SqlFrom,
    SqlFunction,
    SqlLabel,
    SqlPredicate,
    TableReference,
    typed_literal,
    validate_arguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FUELS = ["electricity", "natural_gas", "propane", "fuel_oil", "coal", "wood_cord", "wood_pellets"]
JoinSpec = tuple[TableReference, ColumnReference, ColumnReference]
WeightExpression = SqlExpression | int | float | None


class UnsupportedQueryShape(NotImplementedError):
    """Raised when the requested query shape is known to be unsupported on the
    current schema. Caught by the snapshot harness and treated as a skip rather
    than a failure.
    """


def _expression_for(column: ColumnExpression) -> SqlExpression:
    """Return the SQL expression behind a label, or the column itself."""
    return column.element if isinstance(column, SqlLabel) else column


def _column_or_expression(table: QuerySideTable, column: ColumnExpression) -> SqlExpression:
    """Return a table-bound column when available, otherwise the expression."""
    return table.c.get(column.name, column)


def _rebind_column_to_table(column: ColumnExpression, table: QuerySideTable) -> SqlExpression:
    """Rebind a labeled or plain column expression to a compatible table."""
    if not isinstance(column, SqlLabel):
        return _column_or_expression(table, column)

    if column.name in getattr(table, "c", {}):
        return table.c[column.name]

    adapted = ClauseAdapter(cast(SqlFrom, table), adapt_on_names=True).traverse(column.element)
    return adapted.label(column.name)


def _classify_enduse_source(
    enduse: ColumnExpression, timeseries_table: SqlFrom
) -> Literal["ts_only", "pure_bs", "mixed"]:
    """Classify whether an enduse expression reads timeseries, metadata, or both."""
    target = _expression_for(enduse)
    ts_refs, other_refs = [], []

    def visit_column(elem: object) -> None:
        if not isinstance(elem, Column):
            return
        table = getattr(elem, "table", None)
        if table is timeseries_table:
            ts_refs.append(elem)
        elif table is not None:
            other_refs.append(elem)

    visitors.traverse(target, {}, {"column": visit_column})
    if ts_refs and other_refs:
        return "mixed"
    if other_refs:
        return "pure_bs"
    return "ts_only"


class AggregateSideView:
    """Column adapter for one output side of a timeseries aggregate."""

    def __init__(
        self,
        ts_subq: SqlFrom,
        prefix: str,
        ts_enduses: Sequence[ColumnExpression],
        group_cols: Sequence[ColumnExpression],
        metadata_subquery: SqlFrom,
        metadata_enduses: Sequence[ColumnExpression],
    ) -> None:
        """Build the lookup of projected columns for one aggregate side."""
        self._cols_by_name = {}
        for enduse in ts_enduses:
            self._cols_by_name[enduse.name] = ts_subq.c[f"{prefix}__{enduse.name}"]
        for enduse in metadata_enduses:
            if enduse.name in metadata_subquery.c:
                self._cols_by_name[enduse.name] = metadata_subquery.c[enduse.name]
        for column in group_cols:
            if column.name not in self._cols_by_name:
                self._cols_by_name[column.name] = ts_subq.c[column.name]
        if "_inner_rows" in ts_subq.c:
            self._cols_by_name["_inner_rows"] = ts_subq.c["_inner_rows"]

    @property
    def c(self) -> dict[str, SqlExpression]:
        return self._cols_by_name


QuerySideTable = SqlFrom | AggregateSideView


@dataclass
class QueryTableContext:
    baseline_side: QuerySideTable
    upgrade_side: QuerySideTable
    from_clause: SqlFrom
    metadata_alias: SqlFrom
    group_by: list[SqlExpression]
    metadata_restrict: list[RestrictTuple]
    extra_restrict: list[RestrictTuple]
    extra_avoid: list[RestrictTuple]
    total_weight: WeightExpression
    agg_weight: WeightExpression
    pivot_bucketed_time: bool


@dataclass
class EnduseProjection:
    source: SqlExpression
    baseline: SqlExpression
    upgrade: SqlExpression
    savings: SqlExpression


@dataclass
class AverageKwTimeWindow:
    at_hour: list[float]
    interval_seconds: int
    exact_times: bool
    lower_timestamps: list[datetime.datetime]
    upper_timestamps: list[datetime.datetime]


class BuildStockAggregate:
    """A class to do aggregation queries for both timeseries and annual results."""

    def __init__(self, buildstock_query: main.BuildStockQuery) -> None:
        self._bsq = buildstock_query

    def _timeseries_key_names(
        self, ts_unique_keys: Sequence[str], timestamp_col: str, collapse_inner_time: bool
    ) -> list[str]:
        """Return the inner timeseries grouping keys in stable order."""
        partition_cols = [k for k in ts_unique_keys if k != self._bsq.building_id_column_name]
        key_names = [*partition_cols]
        if not collapse_inner_time:
            key_names.append(timestamp_col)
        key_names.append(self._bsq.building_id_column_name)
        return list(dict.fromkeys(key_names))

    def _bucketed_timestamp_expression(
        self,
        ts: SqlFrom,
        timestamp_col: str,
        timestamp_grouping_func: str | None,
        collapse_inner_time: bool,
    ) -> SqlExpression | None:
        """Return the timestamp expression used by the inner aggregate."""
        if collapse_inner_time:
            return None
        if not timestamp_grouping_func:
            return ts.c[timestamp_col]

        sim_info = self._bsq._get_simulation_info()
        raw_time = ts.c[timestamp_col]
        if sim_info.offset > 0:
            return sa.func.date_trunc(
                timestamp_grouping_func,
                sa.func.date_add(sim_info.unit, -sim_info.offset, raw_time),
            )
        return sa.func.date_trunc(timestamp_grouping_func, raw_time)

    @staticmethod
    def _split_enduses_by_source(
        enduses: Sequence[ColumnExpression], timeseries_table: SqlFrom
    ) -> tuple[list[ColumnExpression], list[ColumnExpression], list[ColumnExpression]]:
        """Partition enduses by the table sources referenced in each expression."""
        ts_only, metadata_only, mixed = [], [], []
        for enduse in enduses:
            kind = _classify_enduse_source(enduse, timeseries_table)
            if kind == "ts_only":
                ts_only.append(enduse)
            elif kind == "pure_bs":
                metadata_only.append(enduse)
            else:
                mixed.append(enduse)
        return ts_only, metadata_only, mixed

    def _build_timeseries_flat_subquery(
        self,
        *,
        ts: SqlFrom,
        metadata_table: SqlFrom,
        flat_enduses: Sequence[ColumnExpression],
        key_names: Sequence[str],
        timestamp_col: str,
        bucketed_time_expr: SqlExpression | None,
        extra_group_names: Sequence[str],
        single_upgrade: bool,
        upgrade_ids: Sequence[str],
        restrict_clauses: Sequence[SqlPredicate],
        avoid_clauses: Sequence[SqlPredicate],
        needs_metadata_join: bool,
    ) -> SqlFrom:
        """Build the row-level timeseries subquery before final aggregation."""
        select_columns = [
            ts.c[k].label(k) for k in key_names if k != timestamp_col
        ]
        if bucketed_time_expr is not None:
            select_columns.append(bucketed_time_expr.label(timestamp_col))
        select_columns.extend([ts.c[name].label(name) for name in extra_group_names])
        if not single_upgrade:
            select_columns.append(ts.c["upgrade"].label("upgrade"))
        for enduse in flat_enduses:
            select_columns.append(_expression_for(enduse).label(f"ts__{enduse.name}"))

        from_clause = ts
        if needs_metadata_join:
            from_clause = ts.join(
                metadata_table,
                self._bsq._baseline_timeseries_join_condition(metadata_table, ts),
            )

        return (
            sa.select(*select_columns)
            .select_from(from_clause)
            .where(
                ts.c["upgrade"].in_([typed_literal(ts.c["upgrade"], u) for u in upgrade_ids]),
                *restrict_clauses,
                *avoid_clauses,
            )
            .subquery("ts_flat")
        )

    def _build_timeseries_aggregate_subquery(
        self,
        *,
        ts: SqlFrom,
        ts_flat: SqlFrom,
        flat_enduses: Sequence[ColumnExpression],
        key_names: Sequence[str],
        extra_group_names: Sequence[str],
        single_upgrade: bool,
        upgrade_id: str,
    ) -> tuple[SqlFrom, list[SqlExpression], list[SqlExpression]]:
        """Aggregate the flattened timeseries rows to one row per join key."""
        group_keys = cast(list[SqlExpression], [ts_flat.c[k] for k in key_names])
        extra_group_columns = cast(list[SqlExpression], [ts_flat.c[name] for name in extra_group_names])

        enduse_columns = []
        if single_upgrade:
            for enduse in flat_enduses:
                value = ts_flat.c[f"ts__{enduse.name}"]
                enduse_columns.append(safunc.sum(value).label(f"bs__{enduse.name}"))
            inner_rows = safunc.count(sa.text("*")).label("_inner_rows")
        else:
            baseline_filter = ts_flat.c["upgrade"] == typed_literal(ts.c["upgrade"], "0")
            upgrade_filter = ts_flat.c["upgrade"] == typed_literal(ts.c["upgrade"], upgrade_id)
            for enduse in flat_enduses:
                value = ts_flat.c[f"ts__{enduse.name}"]
                enduse_columns.append(safunc.sum(value).filter(baseline_filter).label(f"bs__{enduse.name}"))
                enduse_columns.append(safunc.sum(value).filter(upgrade_filter).label(f"up__{enduse.name}"))
            inner_rows = safunc.count(sa.text("*")).filter(baseline_filter).label("_inner_rows")

        subquery = (
            sa.select(*group_keys, *extra_group_columns, *enduse_columns, inner_rows)
            .select_from(ts_flat)
            .group_by(*group_keys, *extra_group_columns)
            .subquery("ts_aggr")
        )
        return subquery, group_keys, extra_group_columns

    def _metadata_from_with_extensions(
        self, metadata_table: SqlFrom, join_list: Sequence[JoinSpec] | None
    ) -> SqlFrom:
        """Return the metadata FROM clause with optional user joins applied."""
        from_clause = metadata_table
        for new_table_name, metadata_col, new_col in (join_list or ()):
            join_table = self._bsq._get_table(new_table_name)
            if isinstance(metadata_col, str):
                metadata_side = metadata_table.c[metadata_col]
            elif isinstance(metadata_col, Column) and metadata_col.name in metadata_table.c:
                metadata_side = metadata_table.c[metadata_col.name]
            else:
                metadata_side = metadata_col
            joined_side = self._bsq._get_column(new_col, candidate_tables=[join_table])
            from_clause = from_clause.join(join_table, cast(SqlPredicate, metadata_side == joined_side))
        return from_clause

    def _build_metadata_per_building_subquery(
        self,
        *,
        metadata_table: SqlFrom,
        ts_unique_keys: Sequence[str],
        metadata_group_by: Sequence[ColumnExpression],
        metadata_only_enduses: Sequence[ColumnExpression],
        total_weight: WeightExpression,
        extra_metadata_cols: Sequence[ColumnExpression] | None,
        join_list: Sequence[tuple] | None,
        join_list_restrict: Sequence[RestrictTuple] | None,
        restrict_clauses: Sequence[SqlPredicate],
        avoid_clauses: Sequence[SqlPredicate],
    ) -> SqlFrom:
        """Build the metadata-per-building subquery joined to timeseries aggregates."""
        select_columns = [metadata_table.c[k].label(k) for k in ts_unique_keys]
        extra_group_exprs = []

        for group_col in metadata_group_by:
            if group_col.name in ts_unique_keys:
                continue
            underlying = _expression_for(group_col)
            select_columns.append(underlying.label(group_col.name))
            extra_group_exprs.append(underlying)

        weight_expr = total_weight if total_weight is not None else metadata_table.c["weight"]
        select_columns.append(safunc.sum(weight_expr).label("bldg_weight"))
        select_columns.append(safunc.count(sa.text("*")).label("tract_count"))

        for enduse in metadata_only_enduses:
            select_columns.append(
                safunc.arbitrary(_expression_for(enduse)).label(enduse.name)
            )

        for column in extra_metadata_cols or ():
            if column.name in {selected.name for selected in select_columns}:
                continue
            select_columns.append(safunc.arbitrary(column).label(column.name))

        join_restrict_clauses = (
            self._bsq._get_restrict_clauses(join_list_restrict, annual_only=True)
            if join_list_restrict else []
        )

        return (
            sa.select(*select_columns)
            .select_from(self._metadata_from_with_extensions(metadata_table, join_list))
            .where(
                self._bsq._upgrade_zero_filter(metadata_table),
                *restrict_clauses,
                *avoid_clauses,
                *join_restrict_clauses,
            )
            .group_by(
                *(metadata_table.c[k] for k in ts_unique_keys),
                *extra_group_exprs,
            )
            .subquery("bs_per_bldg")
        )

    @staticmethod
    def _remap_timeseries_group_by(
        group_by: Sequence[ColumnExpression],
        timeseries_table: SqlFrom,
        timeseries_aggregate: SqlFrom,
        metadata_per_building: SqlFrom,
    ) -> list[SqlExpression]:
        """Rebind outer group-by columns to the tables exposed by the aggregate."""
        remapped = []
        for group_col in group_by:
            if group_col.name in timeseries_table.columns:
                remapped.append(timeseries_aggregate.c[group_col.name])
            elif group_col.name in metadata_per_building.c:
                remapped.append(metadata_per_building.c[group_col.name])
            else:
                remapped.append(group_col)
        return remapped

    def _restrict_with_timeseries_applied_filter(
        self, restrict: Sequence[RestrictTuple], params: Query, upgrade_id: str
    ) -> list[RestrictTuple]:
        """Add the applied-buildings filter when a timeseries upgrade requires it."""
        metadata_restrict = list(restrict) if restrict else []
        if params.annual_only or not params.applied_only or upgrade_id == "0":
            return metadata_restrict

        key_kind: Literal["metadata", "timeseries"] = (
            "timeseries" if self._bsq.ts_table is not None else "metadata"
        )
        applied_select = self._bsq._build_applied_subquery(
            all_of=[upgrade_id], any_of=None, key_kind=key_kind
        )
        assert applied_select is not None
        metadata_restrict.append(
            self._bsq._make_applied_filter_tuple(applied_select, key_kind=key_kind)
        )
        return metadata_restrict

    def _time_grouping_position(self, params: Query) -> int:
        """Remove time aliases from group-by and return their original position."""
        time_index = len(params.group_by)
        time_aliases = {"time", self._bsq.timestamp_column_name}
        for alias in time_aliases:
            if alias in params.group_by:
                time_index = params.group_by.index(alias)
                params.group_by = [g for g in params.group_by if g not in time_aliases]
                break
        return time_index

    @staticmethod
    def _split_join_list_restricts(
        join_list: Sequence[JoinSpec] | None, extra_restrict: Sequence[RestrictTuple]
    ) -> tuple[list[RestrictTuple], list[RestrictTuple]]:
        """Separate restrictions that target explicit joined tables."""
        if not join_list or not extra_restrict:
            return [], list(extra_restrict)

        join_table_names = set()
        for join_entry in join_list:
            table_ref = join_entry[0]
            join_table_names.add(
                table_ref if isinstance(table_ref, str) else getattr(table_ref, "name", None)
            )

        join_restricts: list[RestrictTuple] = []
        remaining_restricts: list[RestrictTuple] = []
        for col_ref, values in extra_restrict:
            table = getattr(col_ref, "table", None) if isinstance(col_ref, Column) else None
            if table is not None and getattr(table, "name", None) in join_table_names:
                join_restricts.append((col_ref, values))
            else:
                remaining_restricts.append((col_ref, values))
        return join_restricts, remaining_restricts

    def _project_enduse(
        self,
        col: ColumnExpression,
        params: Query,
        upgrade_id: str,
        baseline_side: QuerySideTable,
        upgrade_side: QuerySideTable,
    ) -> EnduseProjection:
        """Build baseline, upgrade, and savings expressions for one enduse."""
        baseline_col = _column_or_expression(baseline_side, col)
        if upgrade_id == "0":
            upgrade_col = baseline_col
        elif params.annual_only and not params.applied_only:
            upgrade_col = sa.case(
                (
                    self._bsq._get_success_condition(cast(SqlFrom, upgrade_side)),
                    _rebind_column_to_table(col, upgrade_side),
                ),
                else_=baseline_col,
            )
        elif not params.annual_only and not params.applied_only and baseline_side is not upgrade_side:
            upgrade_col = safunc.coalesce(
                _rebind_column_to_table(col, upgrade_side), baseline_col
            )
        else:
            upgrade_col = _rebind_column_to_table(col, upgrade_side)

        return EnduseProjection(
            source=col,
            baseline=baseline_col,
            upgrade=upgrade_col,
            savings=safunc.coalesce(baseline_col, 0) - safunc.coalesce(upgrade_col, 0),
        )

    @staticmethod
    def _weighted_expr(expr: SqlExpression, weight: WeightExpression) -> SqlExpression:
        """Apply an aggregate weight expression when one is present."""
        return expr if weight is None else expr * weight

    def _aggregate_projection_column(
        self,
        projection: EnduseProjection,
        value: SqlExpression,
        suffix: str,
        params: Query,
        agg_func: SqlFunction,
        agg_weight: WeightExpression,
    ) -> SqlExpression:
        """Aggregate one projected value under the configured aggregate function."""
        label = self._bsq._simple_label(projection.source.name, params.agg_func)
        return agg_func(self._weighted_expr(value, agg_weight)).label(f"{label}{suffix}")

    def _nonzero_count_column(self, projection: EnduseProjection, total_weight: WeightExpression) -> SqlExpression:
        """Return the weighted count of buildings with nonzero upgrade values."""
        return safunc.sum(
            sa.case((safunc.coalesce(projection.upgrade, 0) != 0, 1), else_=0) * total_weight
        ).label(f"{self._bsq._simple_label(projection.upgrade.name)}__nonzero_units_count")

    def _projection_output_columns(
        self,
        projection: EnduseProjection,
        params: Query,
        agg_func: SqlFunction,
        agg_weight: WeightExpression,
        total_weight: WeightExpression,
    ) -> list[SqlExpression]:
        """Return all requested aggregate output columns for one projection."""
        output_columns = []
        if params.include_baseline:
            output_columns.append(
                self._aggregate_projection_column(
                    projection, projection.baseline, "__baseline", params, agg_func, agg_weight,
                )
            )
        if params.include_upgrade:
            suffix = "__upgrade" if params.include_savings or params.include_baseline else ""
            output_columns.append(
                self._aggregate_projection_column(
                    projection, projection.upgrade, suffix, params, agg_func, agg_weight,
                )
            )
            if params.get_nonzero_count and params.annual_only:
                output_columns.append(self._nonzero_count_column(projection, total_weight))
        if params.include_savings:
            output_columns.append(
                self._aggregate_projection_column(
                    projection, projection.savings, "__savings", params, agg_func, agg_weight,
                )
            )
        return output_columns

    def _enduse_output_columns(
        self,
        *,
        enduse_cols: Sequence[ColumnExpression],
        params: Query,
        upgrade_id: str,
        baseline_side: QuerySideTable,
        upgrade_side: QuerySideTable,
        agg_func: SqlFunction,
        agg_weight: WeightExpression,
        total_weight: WeightExpression,
    ) -> list[SqlExpression]:
        """Build aggregate output columns for every requested enduse."""
        output_columns = []
        for col in enduse_cols:
            projection = self._project_enduse(col, params, upgrade_id, baseline_side, upgrade_side)
            output_columns.extend(
                self._projection_output_columns(
                    projection, params, agg_func, agg_weight, total_weight,
                )
            )

            if params.get_quartiles:
                output_columns.extend(
                    self._quartile_output_columns(
                        projection, params,
                    )
                )
        return output_columns

    def _quartile_output_columns(self, projection: EnduseProjection, params: Query) -> list[SqlExpression]:
        """Return quartile arrays for requested baseline, upgrade, and savings values."""
        percentiles = [0, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 1]
        output_columns = []
        label = self._bsq._simple_label(projection.source.name, params.agg_func)

        if params.include_baseline:
            output_columns.append(
                sa.func.approx_percentile(projection.baseline, percentiles).label(f"{label}__baseline__quartiles")
            )
            output_columns.append(
                sa.func.approx_percentile(projection.baseline, percentiles).filter(
                    projection.baseline != 0
                ).label(f"{label}__baseline__nonzero_quartiles")
            )
        if params.include_upgrade:
            output_columns.append(
                sa.func.approx_percentile(projection.upgrade, percentiles).label(f"{label}__upgrade__quartiles")
            )
            output_columns.append(
                sa.func.approx_percentile(projection.upgrade, percentiles).filter(
                    projection.upgrade != 0
                ).label(f"{label}__upgrade__nonzero_quartiles")
            )
        if params.include_savings:
            output_columns.append(
                sa.func.approx_percentile(projection.savings, percentiles).label(f"{label}__savings__quartiles")
            )
            output_columns.append(
                sa.func.approx_percentile(projection.savings, percentiles).filter(
                    projection.savings != 0
                ).label(f"{label}__savings__nonzero_quartiles")
            )
        return output_columns

    def _model_count_can_use_count_star(
        self, params: Query, group_by_selection: Sequence[SqlExpression], metadata_alias: SqlFrom
    ) -> bool:
        """Return true when model_count can use count(*) instead of distinct keys."""
        if params.annual_only or "bldg_weight" not in getattr(metadata_alias, "c", {}):
            return False

        building_id = self._bsq.building_id_column_name
        outer_group_names = {getattr(group_col, "name", group_col) for group_col in group_by_selection or ()}
        partition_keys = [
            key for key in self._bsq._get_unique_keys("timeseries")
            if key != building_id
        ]
        return all(key in outer_group_names for key in partition_keys)

    def _model_count_column(self, alias: QuerySideTable, *, use_count_star: bool) -> SqlExpression:
        """Return the model_count aggregate for the current grouping shape."""
        if use_count_star:
            return safunc.count(sa.text("*")).label("model_count")
        building_id = self._bsq.building_id_column_name
        return self._bsq._count_distinct([alias.c[building_id]]).label("model_count")

    def _direct_timeseries_grouping_metrics(
        self, metadata_alias: SqlFrom, total_weight: WeightExpression, *, use_count_star: bool
    ) -> list[SqlExpression]:
        """Return grouping metrics computed directly from timeseries rows."""
        md_key_cols = [metadata_alias.c[k] for k in self._bsq.md_key]
        distinct_md_keys = self._bsq._count_distinct(md_key_cols)
        return [
            distinct_md_keys.label("metadata_rows_count"),
            self._model_count_column(metadata_alias, use_count_star=use_count_star),
            (distinct_md_keys * safunc.sum(total_weight) / safunc.sum(1)).label("units_count"),
            (safunc.sum(1) / distinct_md_keys).label("rows_per_sample"),
        ]

    def _metadata_per_building_grouping_metrics(
        self,
        baseline_side: QuerySideTable,
        metadata_alias: SqlFrom,
        *,
        include_rows_per_sample: bool,
        use_count_star: bool,
    ) -> list[SqlExpression]:
        """Return grouping metrics computed from one metadata row per building."""
        metrics = [
            safunc.sum(metadata_alias.c["tract_count"]).label("metadata_rows_count"),
            self._model_count_column(metadata_alias, use_count_star=use_count_star),
            safunc.sum(metadata_alias.c["bldg_weight"]).label("units_count"),
        ]
        if include_rows_per_sample:
            metrics.append(
                (safunc.sum(baseline_side.c["_inner_rows"]) / self._bsq._count_distinct(
                    [baseline_side.c[k] for k in self._bsq._get_unique_keys("timeseries")]
                )).label("rows_per_sample")
            )
        return metrics

    def _annual_grouping_metrics(
        self, baseline_side: QuerySideTable, total_weight: WeightExpression, *, use_count_star: bool
    ) -> list[SqlExpression]:
        """Return metadata-only grouping metrics for annual queries."""
        return [
            safunc.sum(1).label("metadata_rows_count"),
            self._model_count_column(baseline_side, use_count_star=use_count_star),
            safunc.sum(total_weight).label("units_count"),
        ]

    def _timeseries_grouping_metrics(
        self,
        baseline_side: QuerySideTable,
        metadata_alias: SqlFrom,
        total_weight: WeightExpression,
        *,
        include_rows_per_sample: bool,
        use_count_star: bool,
    ) -> list[SqlExpression]:
        """Return metrics for raw or per-building timeseries aggregates."""
        if "bldg_weight" in getattr(metadata_alias, "c", {}):
            return self._metadata_per_building_grouping_metrics(
                baseline_side,
                metadata_alias,
                include_rows_per_sample=include_rows_per_sample,
                use_count_star=use_count_star,
            )
        if include_rows_per_sample:
            return self._direct_timeseries_grouping_metrics(
                metadata_alias, total_weight, use_count_star=use_count_star,
            )
        return self._annual_grouping_metrics(
            baseline_side, total_weight, use_count_star=use_count_star,
        )

    def _grouping_metrics(
        self,
        *,
        params: Query,
        baseline_side: QuerySideTable,
        metadata_alias: SqlFrom,
        total_weight: WeightExpression,
        group_by_selection: list[SqlExpression],
        time_index: int,
        pivot_bucketed_time: bool,
    ) -> list[SqlExpression]:
        """Choose the grouping metrics for annual, direct, or pre-aggregated tables."""
        use_count_star = self._model_count_can_use_count_star(params, group_by_selection, metadata_alias)
        if params.annual_only:
            return self._annual_grouping_metrics(
                baseline_side, total_weight, use_count_star=use_count_star,
            )

        metrics = self._timeseries_grouping_metrics(
            baseline_side,
            metadata_alias,
            total_weight,
            include_rows_per_sample=params.timestamp_grouping_func is not None,
            use_count_star=use_count_star,
        )

        if params.timestamp_grouping_func:
            if params.timestamp_grouping_func == "year":
                return metrics
            self._insert_grouped_time_column(
                params, baseline_side, group_by_selection, time_index, pivot_bucketed_time,
            )
            return metrics

        time_col = baseline_side.c[self._bsq.timestamp_column_name].label(self._bsq.timestamp_column_name)
        group_by_selection.insert(time_index, time_col)
        return metrics

    def _insert_grouped_time_column(
        self,
        params: Query,
        baseline_side: QuerySideTable,
        group_by_selection: list[SqlExpression],
        time_index: int,
        pivot_bucketed_time: bool,
    ) -> None:
        """Insert the grouped timestamp expression back into group-by columns."""
        colname = self._bsq.timestamp_column_name
        time_col = baseline_side.c[colname]
        if pivot_bucketed_time:
            grouped_time = time_col.label(colname)
        else:
            sim_info = self._bsq._get_simulation_info()
            if sim_info.offset > 0:
                grouped_time = sa.func.date_trunc(
                    params.timestamp_grouping_func,
                    sa.func.date_add(sim_info.unit, -sim_info.offset, time_col),
                ).label(colname)
            else:
                grouped_time = sa.func.date_trunc(params.timestamp_grouping_func, time_col).label(colname)
        group_by_selection.insert(time_index, grouped_time)

    def _annual_table_context(
        self,
        *,
        upgrade_id: str,
        group_by_selection: list[SqlExpression],
        restrict: Sequence[RestrictTuple],
        total_weight: WeightExpression,
        agg_weight: WeightExpression,
        applied_only: bool | None,
    ) -> QueryTableContext:
        """Build the table context for an annual metadata query."""
        baseline_side, upgrade_side, from_clause = self._get_annual_metadata_sides(
            upgrade_id, applied_only,
        )
        return QueryTableContext(
            baseline_side=baseline_side,
            upgrade_side=upgrade_side,
            from_clause=from_clause,
            metadata_alias=baseline_side,
            group_by=group_by_selection,
            metadata_restrict=list(restrict),
            extra_restrict=[],
            extra_avoid=[],
            total_weight=total_weight,
            agg_weight=agg_weight,
            pivot_bucketed_time=False,
        )

    def _timeseries_table_context(
        self,
        *,
        params: Query,
        upgrade_id: str,
        enduse_cols: Sequence[ColumnExpression],
        group_by_selection: list[SqlExpression],
        restrict: Sequence[RestrictTuple],
        total_weight: WeightExpression,
        agg_weight: WeightExpression,
    ) -> QueryTableContext:
        """Build the table context for a timeseries aggregate query."""
        metadata_restrict, ts_restrict, extra_restrict = self._bsq._split_restrict(restrict)

        metadata_avoid, ts_avoid, extra_avoid = self._bsq._split_restrict(
            list(params.avoid) if params.avoid else []
        )
        upgrade_only = (
            upgrade_id != "0"
            and not params.include_savings
            and not params.include_baseline
        )
        join_list_restrict, extra_restrict = self._split_join_list_restricts(
            params.join_list, extra_restrict,
        )
        baseline_side, upgrade_side, from_clause, group_by_selection, metadata_alias = self._get_timeseries_metadata_sides(
            enduse_cols,
            upgrade_id,
            ts_restrict,
            avoid=ts_avoid,
            metadata_restrict=metadata_restrict,
            metadata_avoid=metadata_avoid,
            group_by=group_by_selection,
            upgrade_only=upgrade_only,
            timestamp_grouping_func=params.timestamp_grouping_func,
            total_weight=total_weight,
            extra_metadata_cols=[],
            join_list=params.join_list,
            join_list_restrict=join_list_restrict,
        )

        if "bldg_weight" in getattr(metadata_alias, "c", {}):
            total_weight = metadata_alias.c["bldg_weight"]
            if agg_weight is not None:
                agg_weight = total_weight

        return QueryTableContext(
            baseline_side=baseline_side,
            upgrade_side=upgrade_side,
            from_clause=from_clause,
            metadata_alias=metadata_alias,
            group_by=group_by_selection,
            metadata_restrict=metadata_restrict,
            extra_restrict=extra_restrict,
            extra_avoid=extra_avoid,
            total_weight=total_weight,
            agg_weight=agg_weight,
            pivot_bucketed_time=params.timestamp_grouping_func is not None,
        )

    def _prepare_query_table_context(
        self,
        *,
        params: Query,
        upgrade_id: str,
        enduse_cols: Sequence[ColumnExpression],
        group_by_selection: list[SqlExpression],
        metadata_restrict: Sequence[RestrictTuple],
        total_weight: WeightExpression,
        agg_weight: WeightExpression,
    ) -> QueryTableContext:
        """Choose the annual or timeseries table context for a query."""
        if params.annual_only:
            return self._annual_table_context(
                upgrade_id=upgrade_id,
                group_by_selection=group_by_selection,
                restrict=metadata_restrict,
                total_weight=total_weight,
                agg_weight=agg_weight,
                applied_only=params.applied_only,
            )
        return self._timeseries_table_context(
            params=params,
            upgrade_id=upgrade_id,
            enduse_cols=enduse_cols,
            group_by_selection=group_by_selection,
            restrict=metadata_restrict,
            total_weight=total_weight,
            agg_weight=agg_weight,
        )

    def _outer_join_list(self, params: Query, metadata_alias: SqlFrom) -> Sequence[JoinSpec]:
        """Return outer join entries that still apply after pre-aggregation."""
        if not params.annual_only and "bldg_weight" in getattr(metadata_alias, "c", {}):
            return []
        return params.join_list

    def _assemble_outer_query(
        self,
        *,
        params: Query,
        table_context: QueryTableContext,
        group_by_selection: Sequence[SqlExpression],
        grouping_metrics: Sequence[SqlExpression],
        enduse_columns: Sequence[SqlExpression],
    ) -> SelectQuery:
        """Assemble the final aggregate query from prepared query pieces."""
        query_cols = list(group_by_selection) + list(grouping_metrics) + list(enduse_columns)
        query = sa.select(*query_cols).select_from(table_context.from_clause)
        query = self._bsq._add_join(
            query,
            self._outer_join_list(params, table_context.metadata_alias),
            bs_alias=table_context.metadata_alias,
        )

        if params.annual_only:
            query = query.where(
                sa.and_(
                    self._bsq._get_success_condition(cast(SqlFrom, table_context.baseline_side)),
                    self._bsq._upgrade_zero_filter(cast(SqlFrom, table_context.baseline_side)),
                )
            )
            query = self._bsq._add_restrict(
                query, table_context.metadata_restrict, annual_only=params.annual_only,
            )

        if table_context.extra_restrict:
            query = self._bsq._add_restrict(
                query, table_context.extra_restrict, annual_only=params.annual_only,
            )

        outer_avoid = params.avoid if params.annual_only else table_context.extra_avoid
        query = self._bsq._add_avoid(query, outer_avoid, annual_only=params.annual_only)
        query = self._bsq._add_group_by(query, group_by_selection)
        query = self._bsq._add_order_by(query, group_by_selection if params.sort else [])
        return query.limit(params.limit) if params.limit else query

    def _compiled_or_executed_query(
        self, query: SelectQuery, params: Query, partition_by: Sequence[str]
    ) -> pd.DataFrame | str:
        """Return SQL text for query-only mode, otherwise execute the SQL."""
        compiled_query = self._bsq._compile(query)
        if params.unload_to:
            if partition_by:
                compiled_query = (
                    f"UNLOAD ({compiled_query}) \n TO 's3://{params.unload_to}' \n "
                    f"WITH (format = 'PARQUET', partitioned_by = ARRAY{partition_by})"
                )
            else:
                compiled_query = (
                    f"UNLOAD ({compiled_query}) \n TO 's3://{params.unload_to}' \n WITH (format = 'PARQUET')"
                )

        if params.get_query_only:
            return compiled_query
        return self._bsq.execute(compiled_query)

    def _metadata_choice_for_query(self, params: Query) -> Literal["primary", "state_agg"]:
        """Choose the metadata table that can satisfy this annual query."""
        if not params.annual_only:
            return "primary"

        time_aliases = ("time", self._bsq.timestamp_column_name)
        routing_group_by = [
            group_col for group_col in params.group_by
            if not (isinstance(group_col, str) and group_col in time_aliases)
        ]
        return self._bsq._pick_metadata_table(routing_group_by, params.restrict)

    @validate_arguments
    def _get_timeseries_metadata_sides(
        self,
        enduses: Sequence[ColumnExpression],
        upgrade_id: str,
        restrict: Sequence[RestrictTuple] = Field(default_factory=list),
        avoid: Sequence[RestrictTuple] = Field(default_factory=list),
        metadata_restrict: Sequence[RestrictTuple] = Field(default_factory=list),
        metadata_avoid: Sequence[RestrictTuple] = Field(default_factory=list),
        group_by: Sequence[ColumnExpression] = Field(default_factory=list),
        upgrade_only: bool = False,
        timestamp_grouping_func: str | None = None,
        total_weight: WeightExpression = None,
        extra_metadata_cols: Sequence[Column] | None = None,
        join_list: Sequence[JoinSpec] | None = None,
        join_list_restrict: Sequence[RestrictTuple] | None = None,
    ) -> tuple[AggregateSideView, AggregateSideView, SqlFrom, list[SqlExpression], SqlFrom]:
        """Return baseline and upgrade views over the timeseries aggregate."""
        if self._bsq.ts_table is None:
            raise ValueError("No timeseries table found in database.")

        ts = self._bsq.ts_table
        metadata_table = self._bsq.bs_table

        metadata_restrict_clauses = self._bsq._get_restrict_clauses(metadata_restrict, annual_only=True)
        metadata_avoid_clauses = self._bsq._get_avoid_clauses(metadata_avoid, annual_only=True)

        single_upgrade = upgrade_id == "0" or upgrade_only
        ts_upgrade_ids = [upgrade_id] if single_upgrade else ["0", upgrade_id]

        ts_group_by = [g for g in group_by if g.name in ts.columns]
        metadata_group_by = [g for g in group_by if g.name not in ts.columns]

        ts_unique_keys = self._bsq._get_unique_keys("timeseries")
        timestamp_col = self._bsq.timestamp_column_name
        collapse_inner_time = timestamp_grouping_func == "year"
        ts_key_names = self._timeseries_key_names(ts_unique_keys, timestamp_col, collapse_inner_time)
        ts_extra_group_names = [g.name for g in ts_group_by if g.name not in ts_key_names]

        bucketed_time_expr = self._bucketed_timestamp_expression(
            ts, timestamp_col, timestamp_grouping_func, collapse_inner_time,
        )

        ts_restrict_clauses = self._bsq._get_restrict_clauses(restrict, annual_only=False)
        ts_avoid_clauses = self._bsq._get_avoid_clauses(avoid, annual_only=False)

        ts_only_enduses, metadata_only_enduses, mixed_enduses = self._split_enduses_by_source(enduses, ts)

        flat_enduses = ts_only_enduses + mixed_enduses
        needs_metadata_join = bool(mixed_enduses)

        ts_flat_subq = self._build_timeseries_flat_subquery(
            ts=ts,
            metadata_table=metadata_table,
            flat_enduses=flat_enduses,
            key_names=ts_key_names,
            timestamp_col=timestamp_col,
            bucketed_time_expr=bucketed_time_expr,
            extra_group_names=ts_extra_group_names,
            single_upgrade=single_upgrade,
            upgrade_ids=ts_upgrade_ids,
            restrict_clauses=ts_restrict_clauses,
            avoid_clauses=ts_avoid_clauses,
            needs_metadata_join=needs_metadata_join,
        )

        ts_aggr_subq, flat_group_keys, flat_extra_group_cols = self._build_timeseries_aggregate_subquery(
            ts=ts,
            ts_flat=ts_flat_subq,
            flat_enduses=flat_enduses,
            key_names=ts_key_names,
            extra_group_names=ts_extra_group_names,
            single_upgrade=single_upgrade,
            upgrade_id=upgrade_id,
        )

        metadata_per_bldg = self._build_metadata_per_building_subquery(
            metadata_table=metadata_table,
            ts_unique_keys=ts_unique_keys,
            metadata_group_by=metadata_group_by,
            metadata_only_enduses=metadata_only_enduses,
            total_weight=total_weight,
            extra_metadata_cols=extra_metadata_cols,
            join_list=join_list,
            join_list_restrict=join_list_restrict,
            restrict_clauses=metadata_restrict_clauses,
            avoid_clauses=metadata_avoid_clauses,
        )

        metadata_join_condition = sa.and_(
            *(metadata_per_bldg.c[k] == ts_aggr_subq.c[k] for k in ts_unique_keys),
        )
        tbljoin = ts_aggr_subq.join(metadata_per_bldg, metadata_join_condition)

        passthrough_cols = flat_group_keys + flat_extra_group_cols
        baseline_view = AggregateSideView(
            ts_aggr_subq, "bs", flat_enduses, passthrough_cols, metadata_per_bldg, metadata_only_enduses
        )
        upgrade_view = baseline_view if single_upgrade else AggregateSideView(
            ts_aggr_subq, "up", flat_enduses, passthrough_cols, metadata_per_bldg, metadata_only_enduses,
        )

        remapped_group_by = self._remap_timeseries_group_by(
            group_by, ts, ts_aggr_subq, metadata_per_bldg,
        )

        return baseline_view, upgrade_view, tbljoin, remapped_group_by, metadata_per_bldg

    @validate_arguments
    def _get_annual_metadata_sides(
        self, upgrade_id: str, applied_only: bool | None
    ) -> tuple[SqlFrom, SqlFrom, SqlFrom]:
        """Return metadata row aliases and FROM handle for an annual aggregate."""
        # `self._bsq.bs_table` / `.md_table` / `.md_key` may be routed to
        # the alt metadata table by the `_routing_context` swap inside
        # `_query`. Reading from `self._bsq.*` thus inherits routing
        # automatically — no explicit threading needed here.
        bs = self._bsq.bs_table
        if upgrade_id == "0":
            # Baseline-only path: no self-join. The outer query filters to
            # upgrade=0 rows.
            return bs, bs, bs

        md_table = self._bsq.md_table
        up = md_table.alias("up")
        up_col = up.c["upgrade"]
        up_id = typed_literal(up_col, upgrade_id)
        join_cond = sa.and_(
            self._bsq._baseline_upgrade_join_condition(bs, up),
            up_col == up_id,
            self._bsq._get_success_condition(up),
        )
        if applied_only:
            tbljoin = bs.join(up, join_cond)
        else:
            tbljoin = bs.outerjoin(up, join_cond)

        return bs, up, tbljoin

    @staticmethod
    def _normalized_at_hours(at_hour: list[float] | float, at_days: Sequence[float]) -> list[float]:
        """Return one requested hour per requested simulation day."""
        if isinstance(at_hour, list):
            if len(at_hour) != len(at_days) or not at_hour:
                raise ValueError(
                    "The length of at_hour list should be the same as length of at_days list and not be empty"
                )
            return at_hour
        if isinstance(at_hour, (float, int)):
            return [at_hour] * len(at_days)
        raise ValueError("At hour should be a list or a number")

    @staticmethod
    def _lower_average_kw_timestamp(
        sim_year: int, sim_interval_seconds: int, day: float, hour: float
    ) -> datetime.datetime:
        """Return the simulation timestamp at or before a requested hour."""
        start = datetime.datetime(year=sim_year, month=1, day=1)
        return start + datetime.timedelta(
            days=day,
            seconds=sim_interval_seconds * int(hour * 3600 / sim_interval_seconds),
        )

    @staticmethod
    def _upper_average_kw_timestamp(
        sim_year: int, sim_interval_seconds: int, day: float, hour: float
    ) -> datetime.datetime:
        """Return the simulation timestamp at or after a requested hour."""
        start = datetime.datetime(year=sim_year, month=1, day=1)
        add = 0 if round(hour * 3600 % sim_interval_seconds, 2) == 0 else 1
        upper = start + datetime.timedelta(
            days=day,
            seconds=sim_interval_seconds * (int(hour * 3600 / sim_interval_seconds) + add),
        )
        if upper.year > sim_year:
            return start + datetime.timedelta(
                days=day,
                seconds=sim_interval_seconds * int(hour * 3600 / sim_interval_seconds),
            )
        return upper

    @staticmethod
    def _hours_align_with_sim_timestamps(at_hour: Sequence[float], sim_interval_seconds: int) -> bool:
        """Return true when all requested hours fall exactly on simulation steps."""
        return bool(np.all([round(h * 3600 % sim_interval_seconds, 2) == 0 for h in at_hour]))

    @staticmethod
    def _average_kw_timestamp_bounds(
        at_days: Sequence[float], at_hour: Sequence[float], sim_year: int, sim_interval_seconds: int
    ) -> tuple[list[datetime.datetime], list[datetime.datetime]]:
        """Return lower and upper timestamp bounds for average-kW interpolation."""
        lower = [
            BuildStockAggregate._lower_average_kw_timestamp(sim_year, sim_interval_seconds, d - 1, h)
            for d, h in zip(at_days, at_hour, strict=True)
        ]
        upper = [
            BuildStockAggregate._upper_average_kw_timestamp(sim_year, sim_interval_seconds, d - 1, h)
            for d, h in zip(at_days, at_hour, strict=True)
        ]
        return lower, upper

    @staticmethod
    def _average_kw_interpolation_weight(at_hour: Sequence[float], sim_interval_seconds: int) -> float:
        """Return the average interpolation weight for upper timestamps."""
        return np.mean(
            [
                offset_seconds / sim_interval_seconds
                for hour in at_hour
                if (offset_seconds := hour * 3600 % sim_interval_seconds)
            ]
        )

    def _average_kw_time_window(self, at_hour: list[float], at_days: Sequence[float]) -> AverageKwTimeWindow:
        """Return the timestamp window used for average-kW queries."""
        sim_info = self._bsq._get_simulation_info()
        lower_timestamps, upper_timestamps = self._average_kw_timestamp_bounds(
            at_days, at_hour, sim_info.year, sim_info.interval,
        )
        return AverageKwTimeWindow(
            at_hour=at_hour,
            interval_seconds=sim_info.interval,
            exact_times=self._hours_align_with_sim_timestamps(at_hour, sim_info.interval),
            lower_timestamps=lower_timestamps,
            upper_timestamps=upper_timestamps,
        )

    def _average_kw_base_query(
        self,
        *,
        enduse_cols: Sequence[ColumnExpression],
        total_weight: WeightExpression,
        kw_factor: float,
        upgrade_id: int | str,
        restrict: Sequence[RestrictTuple],
    ) -> SelectQuery:
        """Build the base query reused for lower and upper average-kW timestamps."""
        ts = self._bsq.ts_table
        if ts is None:
            raise ValueError("No timeseries table found in database.")

        enduse_selection = [
            safunc.avg(enduse * total_weight * kw_factor).label(self._bsq._simple_label(enduse.name))
            for enduse in enduse_cols
        ]
        grouping_metrics = [
            safunc.sum(1).label("metadata_rows_count"),
            safunc.sum(total_weight).label("units_count"),
        ]

        upgrade_str = "0" if upgrade_id in (None, "0") else str(upgrade_id)
        metadata_restrict, ts_restrict, extra_restrict = self._bsq._split_restrict(list(restrict))
        metadata_restrict_clauses = self._bsq._get_restrict_clauses(metadata_restrict, annual_only=True)
        ts_restrict_clauses = self._bsq._get_restrict_clauses(ts_restrict, annual_only=False)

        ts_key_cols = self._bsq.ts_key_cols
        metadata_table = self._bsq.bs_table
        query = sa.select(*ts_key_cols + grouping_metrics + enduse_selection)
        query = query.join(
            metadata_table,
            sa.and_(
                self._bsq._baseline_timeseries_join_condition(metadata_table, ts),
                self._bsq._ts_upgrade_col == typed_literal(self._bsq._ts_upgrade_col, upgrade_str),
                *metadata_restrict_clauses,
                *ts_restrict_clauses,
            ),
        )
        query = self._bsq._add_group_by(query, ts_key_cols)
        query = self._bsq._add_order_by(query, ts_key_cols)
        if extra_restrict:
            query = self._bsq._add_restrict(query, extra_restrict, annual_only=False)
        return query

    def _average_kw_query_strings(self, base_query: SelectQuery, time_window: AverageKwTimeWindow) -> list[str]:
        """Compile lower and optional upper timestamp queries for average kW."""
        lower_query = self._bsq._add_restrict(
            base_query, [(self._bsq.timestamp_column_name, time_window.lower_timestamps)]
        )
        if time_window.exact_times:
            queries = [lower_query]
        else:
            upper_query = self._bsq._add_restrict(
                base_query, [(self._bsq.timestamp_column_name, time_window.upper_timestamps)]
            )
            queries = [lower_query, upper_query]
        return [self._bsq._compile(query) for query in queries]

    def _average_kw_result(
        self, query_strs: Sequence[str], time_window: AverageKwTimeWindow, enduses: Sequence[str]
    ) -> pd.DataFrame:
        """Execute average-kW timestamp queries and interpolate when needed."""
        batch_id = self._bsq.submit_batch_query(query_strs)
        if time_window.exact_times:
            (values,) = self._bsq.get_batch_query_result(batch_id, combine=False)
            return values

        lower_values, upper_values = self._bsq.get_batch_query_result(batch_id, combine=False)
        upper_weight = self._average_kw_interpolation_weight(
            time_window.at_hour, time_window.interval_seconds,
        )
        lower_weight = 1 - upper_weight
        enduse_label_cols = [self._bsq._simple_label(enduse) for enduse in enduses]
        lower_values[enduse_label_cols] = (
            lower_values[enduse_label_cols] * lower_weight
            + upper_values[enduse_label_cols] * upper_weight
        )
        return lower_values

    @validate_arguments
    def get_building_average_kws_at(
        self,
        *,
        at_hour: list[float] | float,
        at_days: list[float],
        enduses: list[str],
        upgrade_id: int | str = "0",
        restrict: Sequence[RestrictTuple] = Field(default_factory=list),
        get_query_only: bool = False,
    ) -> pd.DataFrame | list[str]:
        """
        Aggregates the timeseries result on select enduses, for the given days and hours.
        If all of the hour(s) fall exactly on the simulation timestamps, the aggregation is done by averaging the kW at
        those time stamps. If any of the hour(s) fall in between timestamps, then the following process is followed:
            i. The average kWs is calculated for timestamps specified by the hour, or just after it. Call it upper_kw
            ii. The average kWs is calculated for timestamps specified by the hour, or just before it. Call it lower_kw
            iii. Return the interpolation between upper_kw and lower_kw based on the average location of the hour(s)
                 between the upper and lower timestamps.

        Check the argument description below to learn about additional features and options.
        Args:
            at_hour: the hour(s) at which the average kWs of buildings need to be calculated at. It can either be a
                     single number if the hour is same for all days, or a list of numbers if the kW needs to be
                     calculated for different hours for different days.

            at_days: The list of days (of year) for which the average kW is to be calculated for.

            enduses: The list of enduses for which to calculate the average kWs

            upgrade_id: Which upgrade scenario to compute against. Defaults to "0" (baseline). The TS-side join
                        constrains `ts.upgrade = upgrade_id` so the join doesn't cross-product across all upgrades
                        present in the TS table — without this filter, the scan multiplies by the number of
                        upgrades, which on OEDI is 3 TB+ per call.

            restrict: Optional WHERE clauses (e.g. `[("state", ["CO"])]`) to narrow the scan. Strongly recommended
                      on partitioned TS tables — without a state restrict, the join scans every state's partition.

            get_query_only: Skips submitting the query to Athena and just returns the query strings. Useful for batch
                            submitting multiple queries or debugging.

        Returns:
                If get_query_only is True, returns two queries that gets the KW at two timestamps that are to immediate
                    left and right of the the supplied hour.
                If get_query_only is False, returns the average KW of each building at the given hour(s) across the
                supplied days.

        """
        at_hour = self._normalized_at_hours(at_hour, at_days)
        enduse_cols = self._bsq._get_enduse_cols(enduses, table="timeseries")
        total_weight = self._bsq._get_weight([])
        time_window = self._average_kw_time_window(at_hour, at_days)
        base_query = self._average_kw_base_query(
            enduse_cols=enduse_cols,
            total_weight=total_weight,
            kw_factor=3600.0 / time_window.interval_seconds,
            upgrade_id=upgrade_id,
            restrict=restrict,
        )
        query_strs = self._average_kw_query_strings(base_query, time_window)
        if get_query_only:
            return query_strs

        return self._average_kw_result(query_strs, time_window, enduses)

    def validate_partition_by(self, partition_by: Sequence[str]) -> Sequence[str]:
        if not partition_by:
            return []
        [self._bsq._get_gcol(col) for col in partition_by]  # making sure all entries are valid
        return partition_by

    @gather_params(Query)
    def _query(
        self,
        *,
        params: Query,
    ) -> pd.DataFrame | str:
        """Validate query parameters and execute within the chosen metadata route."""
        [self._bsq._get_table(jl[0]) for jl in params.join_list]  # ingress all tables in join list

        upgrade_id = self._bsq._validate_upgrade(params.upgrade_id)
        self._bsq._validate_timeseries_upgrade_restrict(
            params.restrict,
            annual_only=params.annual_only,
            upgrade_id=upgrade_id,
        )
        md_choice = self._metadata_choice_for_query(params)
        with self._bsq._routing_context(md_choice):
            return self._query_inner(
                params=params, upgrade_id=upgrade_id,
            )

    def _query_inner(
        self,
        *,
        params: Query,
        upgrade_id: str,
    ) -> pd.DataFrame | str:
        """Build and run an aggregate query after routing has been selected."""
        metadata_restrict = self._restrict_with_timeseries_applied_filter(
            params.restrict, params, upgrade_id,
        )
        enduse_cols = self._bsq._get_enduse_cols(
            params.enduses, table="baseline" if params.annual_only else "timeseries"
        )
        partition_by = self.validate_partition_by(params.partition_by)
        total_weight = self._bsq._get_weight(params.weights)
        agg_func, agg_weight = self._bsq._get_agg_func_and_weight(params.weights, params.agg_func)
        # The library accepts both the canonical alias `"time"` and the schema's
        # actual timestamp column name (e.g. `"timestamp"` on OEDI) as a marker
        # for "insert the time column at this position". Strip whichever one the
        # user passed; the time-column expression is re-inserted later, so
        # leaving it in `group_by` would project the column twice (Athena
        # rejects with DUPLICATE_COLUMN_NAME).
        #
        # Default placement: AFTER the user's group_by columns (typically
        # state/county). Trino hashes by leftmost GROUP BY columns when
        # shuffling; leading with the partition column keeps the outer
        # aggregate aligned with the parquet's existing layout instead of
        # forcing a re-shuffle by timestamp. If the user explicitly
        # positions `"time"` in their group_by list, their position wins.
        time_indx = self._time_grouping_position(params)
        group_by_selection = self._bsq._process_groupby_cols(params.group_by, annual_only=params.annual_only)

        table_context = self._prepare_query_table_context(
            params=params,
            upgrade_id=upgrade_id,
            enduse_cols=enduse_cols,
            group_by_selection=group_by_selection,
            metadata_restrict=metadata_restrict,
            total_weight=total_weight,
            agg_weight=agg_weight,
        )
        baseline_side = table_context.baseline_side
        upgrade_side = table_context.upgrade_side
        metadata_alias = table_context.metadata_alias
        group_by_selection = table_context.group_by
        total_weight = table_context.total_weight
        agg_weight = table_context.agg_weight

        enduse_columns = self._enduse_output_columns(
            enduse_cols=enduse_cols,
            params=params,
            upgrade_id=upgrade_id,
            baseline_side=baseline_side,
            upgrade_side=upgrade_side,
            agg_func=agg_func,
            agg_weight=agg_weight,
            total_weight=total_weight,
        )

        grouping_metrics = self._grouping_metrics(
            params=params,
            baseline_side=baseline_side,
            metadata_alias=metadata_alias,
            total_weight=total_weight,
            group_by_selection=group_by_selection,
            time_index=time_indx,
            pivot_bucketed_time=table_context.pivot_bucketed_time,
        )

        query = self._assemble_outer_query(
            params=params,
            table_context=table_context,
            group_by_selection=group_by_selection,
            grouping_metrics=grouping_metrics,
            enduse_columns=enduse_columns,
        )
        return self._compiled_or_executed_query(query, params, partition_by)
