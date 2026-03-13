from buildstock_query import main
from buildstock_query.schema.query_params import SavingsQuery
from buildstock_query.schema.helpers import gather_params
from buildstock_query.schema.utilities import validate_arguments
import pandas as pd
from sqlalchemy import func as safunc
import sqlalchemy as sa
from typing import Union
from collections.abc import Sequence
from buildstock_query.schema.utilities import AnyColType
from pydantic import Field
from typing_extensions import deprecated


class BuildStockSavings:
    """Class for doing savings query (both timeseries and annual)."""

    _bsq: "main.BuildStockQuery"

    def __init__(self, buildstock_query: "main.BuildStockQuery") -> None:
        self._bsq = buildstock_query

    def _validate_partition_by(self, partition_by: Sequence[str]):
        if not partition_by:
            return []
        [self._bsq._get_gcol(col) for col in partition_by]  # making sure all entries are valid
        return partition_by

    @validate_arguments
    def __get_timeseries_bs_up_table(
        self,
        enduses: Sequence[AnyColType],
        upgrade_id: str,
        applied_only: bool,
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
        ts_group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list),
    ):
        if self._bsq.ts_table is None:
            raise ValueError("No timeseries table found in database.")

        ts = self._bsq.ts_table
        base = self._bsq.bs_table
        sa_ts_cols = [ts.c[self._bsq.building_id_column_name], ts.c[self._bsq.timestamp_column_name], *ts_group_by]
        sa_ts_cols.extend(enduses)
        ucol = self._bsq._ts_upgrade_col
        ts_b = self._bsq._add_restrict(sa.select(*sa_ts_cols), [[ucol, ("0")], *restrict]).alias("ts_b")
        ts_u = self._bsq._add_restrict(sa.select(*sa_ts_cols), [[ucol, (upgrade_id)], *restrict]).alias("ts_u")

        if applied_only:
            tbljoin = ts_b.join(
                ts_u,
                sa.and_(
                    ts_b.c[self._bsq.building_id_column_name] == ts_u.c[self._bsq.building_id_column_name],
                    ts_b.c[self._bsq.timestamp_column_name] == ts_u.c[self._bsq.timestamp_column_name],
                ),
            ).join(base, ts_b.c[self._bsq.building_id_column_name] == base.c[self._bsq.building_id_column_name])
        else:
            tbljoin = ts_b.outerjoin(
                ts_u,
                sa.and_(
                    ts_b.c[self._bsq.building_id_column_name] == ts_u.c[self._bsq.building_id_column_name],
                    ts_b.c[self._bsq.timestamp_column_name] == ts_u.c[self._bsq.timestamp_column_name],
                ),
            ).join(base, ts_b.c[self._bsq.building_id_column_name] == base.c[self._bsq.building_id_column_name])
        return ts_b, ts_u, tbljoin

    @validate_arguments
    def __get_annual_bs_up_table(self, upgrade_id: str, applied_only: bool):
        if self._bsq.up_table is None:
            raise ValueError("No upgrades table found in database.")
        if applied_only:
            tbljoin = self._bsq.bs_table.join(
                self._bsq.up_table,
                sa.and_(
                    self._bsq.bs_table.c[self._bsq.building_id_column_name]
                    == self._bsq.up_table.c[self._bsq.building_id_column_name],
                    self._bsq._up_upgrade_col == upgrade_id,
                    self._bsq._up_successful_condition,
                ),
            )
        else:
            tbljoin = self._bsq.bs_table.outerjoin(
                self._bsq.up_table,
                sa.and_(
                    self._bsq.bs_table.c[self._bsq.building_id_column_name]
                    == self._bsq.up_table.c[self._bsq.building_id_column_name],
                    self._bsq._up_upgrade_col == upgrade_id,
                    self._bsq._up_successful_condition,
                ),
            )

        return self._bsq.bs_table, self._bsq.up_table, tbljoin

    @gather_params(SavingsQuery)
    @deprecated("Please use my_run.query with include_savings=True.")
    def savings_shape(
        self,
        *,
        params: SavingsQuery,
    ) -> pd.DataFrame | str:
        [self._bsq._get_table(jl[0]) for jl in params.join_list]  # ingress all tables in join list

        if params.agg_func != "sum":
            raise ValueError("Only 'sum' is supported for savings_shape")

        upgrade_id = self._bsq._validate_upgrade(params.upgrade_id)
        if params.timestamp_grouping_func and params.timestamp_grouping_func not in ["hour", "day", "month"]:
            raise ValueError("timestamp_grouping_func must be one of ['hour', 'day', 'month']")

        enduse_cols = self._bsq._get_enduse_cols(
            params.enduses, table="baseline" if params.annual_only else "timeseries"
        )
        partition_by = self._validate_partition_by(params.partition_by)
        total_weight = self._bsq._get_weight(params.weights)
        group_by_selection = self._bsq._process_groupby_cols(params.group_by, annual_only=params.annual_only)

        if params.annual_only:
            ts_b, ts_u, tbljoin = self.__get_annual_bs_up_table(upgrade_id, params.applied_only)
        else:
            params.restrict, ts_restrict = self._bsq._split_restrict(params.restrict)
            bs_group_by, ts_group_by = self._bsq._split_group_by(group_by_selection)
            ts_b, ts_u, tbljoin = self.__get_timeseries_bs_up_table(
                enduse_cols, upgrade_id, params.applied_only, ts_restrict, ts_group_by
            )
            ts_group_by = [ts_b.c[c.name] for c in ts_group_by]  # Refer to the columns using ts_b table
            group_by_selection = bs_group_by + ts_group_by
        query_cols = []
        for col in enduse_cols:
            if params.annual_only:
                savings_col = safunc.coalesce(ts_b.c[col.name], 0) - safunc.coalesce(
                    sa.case((self._bsq._get_success_condition(ts_u), ts_u.c[col.name]), else_=ts_b.c[col.name]), 0
                )
            else:
                savings_col = safunc.coalesce(ts_b.c[col.name], 0) - safunc.coalesce(
                    sa.case(
                        (ts_u.c[self._bsq.building_id_column_name] == None, ts_b.c[col.name]),  # noqa E711
                        else_=ts_u.c[col.name],
                    ),
                    0,
                )
            query_cols.extend(
                [
                    sa.func.sum(ts_b.c[col.name] * total_weight).label(
                        f"{self._bsq._simple_label(col.name)}__baseline"
                    ),
                    sa.func.sum(savings_col * total_weight).label(f"{self._bsq._simple_label(col.name)}__savings"),
                ]
            )
            if params.get_quartiles:
                query_cols.extend(
                    [
                        sa.func.approx_percentile(savings_col,  [0, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 1]).label(
                            f"{self._bsq._simple_label(col.name)}__savings__quartiles"
                        )
                    ]
                )

        query_cols.extend(group_by_selection)
        if params.annual_only:  # Use annual tables
            grouping_metrics_selection = [
                safunc.sum(1).label("sample_count"),
                safunc.sum(1 * total_weight).label("units_count"),
            ]
        elif params.collapse_ts:  # Use timeseries tables but collapse timeseries
            rows_per_building = self._bsq._get_rows_per_building()
            grouping_metrics_selection = [
                (safunc.sum(1) / rows_per_building).label("sample_count"),
                safunc.sum(total_weight / rows_per_building).label("units_count"),
            ]
        elif params.timestamp_grouping_func:
            colname = self._bsq.timestamp_column_name
            # sa.func.dis
            bldg_id_col = ts_b.c[self._bsq.building_id_column_name]
            grouping_metrics_selection = [
                safunc.count(sa.func.distinct(bldg_id_col)).label("sample_count"),
                (
                    safunc.count(sa.func.distinct(bldg_id_col))
                    * safunc.sum(total_weight)
                    / safunc.sum(1)
                ).label("units_count"),
                (safunc.sum(1) / safunc.count(sa.func.distinct(bldg_id_col))).label("rows_per_sample"),
            ]
            sim_info = self._bsq._get_simulation_info()
            time_col = ts_b.c[self._bsq.timestamp_column_name]
            if sim_info.offset > 0:
                # If timestamps are not period begining we should make them so for timestamp_grouping_func aggregation.
                new_col = sa.func.date_trunc(
                    params.timestamp_grouping_func, sa.func.date_add(sim_info.unit, -sim_info.offset, time_col)
                ).label(colname)
            else:
                new_col = sa.func.date_trunc(params.timestamp_grouping_func, time_col).label(colname)
            grouping_metrics_selection.insert(0, new_col)
            group_by_selection.append(new_col)
        else:
            time_col = ts_b.c[self._bsq.timestamp_column_name].label(self._bsq.timestamp_column_name)
            grouping_metrics_selection = [
                time_col,
                safunc.sum(1).label("sample_count"),
                safunc.sum(1 * total_weight).label("units_count"),
            ]
            group_by_selection.append(time_col)

        query_cols = grouping_metrics_selection + query_cols
        query = sa.select(*query_cols).select_from(tbljoin)
        query = self._bsq._add_join(query, params.join_list)
        query = self._bsq._add_restrict(query, params.restrict)
        if params.annual_only:
            query = query.where(self._bsq._bs_successful_condition)
        query = self._bsq._add_group_by(query, group_by_selection)
        query = self._bsq._add_order_by(query, group_by_selection if params.sort else [])

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
