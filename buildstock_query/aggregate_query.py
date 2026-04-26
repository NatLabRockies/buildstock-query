import sqlalchemy as sa
from sqlalchemy.sql import func as safunc
import datetime
import numpy as np
import logging
from buildstock_query import main
from buildstock_query.schema.query_params import Query
import pandas as pd
from buildstock_query.schema.helpers import gather_params
from typing import Union
from collections.abc import Sequence
from buildstock_query.schema.utilities import DBColType, RestrictTuple, typed_literal, validate_arguments
from pydantic import Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FUELS = ["electricity", "natural_gas", "propane", "fuel_oil", "coal", "wood_cord", "wood_pellets"]


class UnsupportedQueryShape(NotImplementedError):
    """Raised when the requested query shape is known to be unsupported on the
    current schema. Caught by the snapshot harness and treated as a skip rather
    than a failure.
    """


class BuildStockAggregate:
    """A class to do aggregation queries for both timeseries and annual results."""

    def __init__(self, buildstock_query: "main.BuildStockQuery") -> None:
        self._bsq = buildstock_query

    @validate_arguments
    def __get_timeseries_bs_up_table(
        self,
        enduses: Sequence[DBColType],
        upgrade_id: str,
        applied_only: bool | None,
        restrict: Sequence[RestrictTuple] = Field(default_factory=list),
        bs_restrict: Sequence[RestrictTuple] = Field(default_factory=list),
        group_by: Sequence[DBColType] = Field(default_factory=list),
        upgrade_only: bool = False,
    ):
        if self._bsq.ts_table is None:
            raise ValueError("No timeseries table found in database.")

        ts = self._bsq.ts_table
        base = self._bsq.bs_table
        ucol = self._bsq._ts_upgrade_col

        # Push any user-supplied bs_restrict (e.g. comstock `state='CO'`) into the
        # inner ts ⋈ bs join condition. Without this, Athena scans the full metadata
        # table before applying user filters — for comstock's tract-denormalized
        # metadata that's the difference between minutes and timeouts. Adding to
        # the JOIN ON clause (rather than wrapping bs in another subquery) keeps
        # the SELECT list clean and lets Athena push the predicate into the bs
        # table scan without enumerating all columns.
        bs_restrict_clauses = self._bsq._get_restrict_clauses(bs_restrict, annual_only=True)

        if upgrade_id == "0" or upgrade_only:
            # Single-upgrade path: aggregate only the requested upgrade's ts rows.
            # Used for the baseline (upgrade_id="0") and for upgrade-only queries that need
            # neither savings nor baseline values, so no ts_b/ts_u pairing is required.
            # This path also supports runs whose timeseries table lacks upgrade=0 rows.
            if self._bsq.up_table is None:
                tbljoin = ts.join(
                    base,
                    sa.and_(
                        self._bsq._baseline_timeseries_join_condition(base, ts),
                        *self._bsq._get_restrict_clauses(restrict, annual_only=True),
                        *bs_restrict_clauses,
                    ),
                )
            else:
                tbljoin = ts.join(
                    base,
                    sa.and_(
                        self._bsq._baseline_timeseries_join_condition(base, ts),
                        ucol == typed_literal(ucol, upgrade_id),
                        *self._bsq._get_restrict_clauses(restrict, annual_only=True),
                        *bs_restrict_clauses,
                    ),
                )
            return ts, ts, tbljoin, list(group_by)

        if self._bsq.buildstock_type == "comstock":
            raise UnsupportedQueryShape(
                "Timeseries upgrade-pair queries (annual_only=False, upgrade_id != '0', "
                "with baseline/savings comparison) are not yet supported on comstock — the "
                "ts-self-join shape times out on the current cluster."
            )

        # For upgrades, create subqueries with proper joins
        # Split group_by into columns from timeseries vs baseline tables
        ts_group_by = [g for g in group_by if g.name in ts.columns]
        bs_group_by = [g for g in group_by if g.name not in ts.columns]

        # Build column list for subquery
        must_have_col_names = list(
            dict.fromkeys(
                [
                    *self._bsq._get_unique_keys("timeseries"),
                    self._bsq.timestamp_column_name,
                ]
            )
        )
        must_have_cols = [ts.c[col_name] for col_name in must_have_col_names]
        ts_group_cols = [g for g in ts_group_by if g.name not in must_have_col_names]
        group_col_names = [g.name for g in ts_group_cols]
        enduse_cols = [e for e in enduses if e.name not in must_have_col_names + group_col_names]

        # Include all necessary columns in the subquery
        subquery_cols = self._bsq._unique_columns_by_name(must_have_cols + ts_group_cols + bs_group_by + enduse_cols)

        # Create subquery with proper join to baseline table; bs_restrict clauses
        # ride the JOIN ON so the metadata scan is pre-filtered.
        subquery_base = sa.select(*subquery_cols).select_from(
            ts.join(
                base,
                sa.and_(
                    self._bsq._baseline_timeseries_join_condition(base, ts),
                    *bs_restrict_clauses,
                ),
            )
        )
        ts_b = self._bsq._add_restrict(subquery_base, [[ucol, "0"], *restrict]).alias("ts_b")
        ts_u = self._bsq._add_restrict(subquery_base, [[ucol, upgrade_id], *restrict]).alias("ts_u")

        # Remap group_by columns to reference the subquery alias
        remapped_group_by = [ts_b.c[g.name] for g in group_by]

        # Create the table join
        if applied_only:
            tbljoin = ts_b.join(
                ts_u,
                self._bsq._timeseries_pair_join_condition(ts_b, ts_u),
            ).join(
                base,
                sa.and_(
                    self._bsq._baseline_timeseries_join_condition(base, ts_b),
                    *bs_restrict_clauses,
                ),
            )
        else:
            tbljoin = ts_b.outerjoin(
                ts_u,
                self._bsq._timeseries_pair_join_condition(ts_b, ts_u),
            ).join(
                base,
                sa.and_(
                    self._bsq._baseline_timeseries_join_condition(base, ts_b),
                    *bs_restrict_clauses,
                ),
            )

        return ts_b, ts_u, tbljoin, remapped_group_by

    @validate_arguments
    def __get_annual_bs_up_table(self, upgrade_id: str, applied_only: bool | None):
        if upgrade_id == "0":
            return self._bsq.bs_table, self._bsq.bs_table, self._bsq.bs_table

        if self._bsq.up_table is None:
            raise ValueError("No upgrades table found in database.")
        up_col = self._bsq._up_upgrade_col
        up_id = typed_literal(up_col, upgrade_id)
        if applied_only:
            tbljoin = self._bsq.bs_table.join(
                self._bsq.up_table,
                sa.and_(
                    self._bsq._baseline_upgrade_join_condition(),
                    up_col == up_id,
                    self._bsq._up_successful_condition,
                ),
            )
        else:
            tbljoin = self._bsq.bs_table.outerjoin(
                self._bsq.up_table,
                sa.and_(
                    self._bsq._baseline_upgrade_join_condition(),
                    up_col == up_id,
                    self._bsq._up_successful_condition,
                ),
            )

        return self._bsq.bs_table, self._bsq.up_table, tbljoin

    @validate_arguments
    def get_building_average_kws_at(
        self,
        *,
        at_hour: Union[list[float], float],
        at_days: list[float],
        enduses: list[str],
        get_query_only: bool = False,
    ):
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

            get_query_only: Skips submitting the query to Athena and just returns the query strings. Useful for batch
                            submitting multiple queries or debugging.

        Returns:
                If get_query_only is True, returns two queries that gets the KW at two timestamps that are to immediate
                    left and right of the the supplied hour.
                If get_query_only is False, returns the average KW of each building at the given hour(s) across the
                supplied days.

        """
        if isinstance(at_hour, list):
            if len(at_hour) != len(at_days) or not at_hour:
                raise ValueError(
                    "The length of at_hour list should be the same as length of at_days list and not be empty"
                )
        elif isinstance(at_hour, (float, int)):
            at_hour = [at_hour] * len(at_days)
        else:
            raise ValueError("At hour should be a list or a number")

        enduse_cols = self._bsq._get_enduse_cols(enduses, table="timeseries")
        total_weight = self._bsq._get_weight([])

        sim_info = self._bsq._get_simulation_info()
        sim_year, sim_interval_seconds = sim_info.year, sim_info.interval
        kw_factor = 3600.0 / sim_interval_seconds

        enduse_selection = [
            safunc.avg(enduse * total_weight * kw_factor).label(self._bsq._simple_label(enduse.name))
            for enduse in enduse_cols
        ]
        grouping_metrics_selection = [
            safunc.sum(1).label("sample_count"),
            safunc.sum(total_weight).label("units_count"),
        ]

        def get_upper_timestamps(day, hour):
            new_dt = datetime.datetime(year=sim_year, month=1, day=1)

            if round(hour * 3600 % sim_interval_seconds, 2) == 0:
                # if the hour falls exactly on the simulation timestamp, use the same timestamp
                # for both lower and upper
                add = 0
            else:
                add = 1

            upper_dt = new_dt + datetime.timedelta(
                days=day, seconds=sim_interval_seconds * (int(hour * 3600 / sim_interval_seconds) + add)
            )
            if upper_dt.year > sim_year:
                upper_dt = new_dt + datetime.timedelta(
                    days=day, seconds=sim_interval_seconds * (int(hour * 3600 / sim_interval_seconds))
                )
            return upper_dt

        def get_lower_timestamps(day, hour):
            new_dt = datetime.datetime(year=sim_year, month=1, day=1)
            lower_dt = new_dt + datetime.timedelta(
                days=day, seconds=sim_interval_seconds * int(hour * 3600 / sim_interval_seconds)
            )
            return lower_dt

        # check if the supplied hours fall exactly on the simulation timestamps
        exact_times = np.all([round(h * 3600 % sim_interval_seconds, 2) == 0 for h in at_hour])
        lower_timestamps = [get_lower_timestamps(d - 1, h) for d, h in zip(at_days, at_hour)]
        upper_timestamps = [get_upper_timestamps(d - 1, h) for d, h in zip(at_days, at_hour)]

        ts_key_cols = self._bsq.ts_key_cols
        query = sa.select(*ts_key_cols + grouping_metrics_selection + enduse_selection)
        query = query.join(self._bsq.bs_table, self._bsq._baseline_timeseries_join_condition())
        query = self._bsq._add_group_by(query, ts_key_cols)
        query = self._bsq._add_order_by(query, ts_key_cols)

        lower_val_query = self._bsq._add_restrict(query, [(self._bsq.timestamp_column_name, lower_timestamps)])
        upper_val_query = self._bsq._add_restrict(query, [(self._bsq.timestamp_column_name, upper_timestamps)])

        if exact_times:
            # only one query is sufficient if the hours fall in exact timestamps
            queries = [lower_val_query]
        else:
            queries = [lower_val_query, upper_val_query]

        query_strs = [self._bsq._compile(q) for q in queries]
        if get_query_only:
            return query_strs

        batch_id = self._bsq.submit_batch_query(query_strs)
        if exact_times:
            (vals,) = self._bsq.get_batch_query_result(batch_id, combine=False)
            return vals
        else:
            lower_vals, upper_vals = self._bsq.get_batch_query_result(batch_id, combine=False)
            avg_upper_weight = np.mean(
                [
                    min_of_hour / sim_interval_seconds
                    for hour in at_hour
                    if (min_of_hour := hour * 3600 % sim_interval_seconds)
                ]
            )
            avg_lower_weight = 1 - avg_upper_weight
            # modify the lower vals to make it weighted average of upper and lower vals
            lower_vals[enduses] = lower_vals[enduses] * avg_lower_weight + upper_vals[enduses] * avg_upper_weight
            return lower_vals

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
    ) -> Union[pd.DataFrame, str]:
        [self._bsq._get_table(jl[0]) for jl in params.join_list]  # ingress all tables in join list

        upgrade_id = self._bsq._validate_upgrade(params.upgrade_id)
        self._bsq._validate_timeseries_upgrade_restrict(
            params.restrict,
            annual_only=params.annual_only,
            upgrade_id=upgrade_id,
        )
        bs_restrict = self._bsq._add_applied_in_restrict(
            params.restrict,
            applied_in=params.applied_in,
            annual_only=params.annual_only,
        )
        enduse_cols = self._bsq._get_enduse_cols(
            params.enduses, table="baseline" if params.annual_only else "timeseries"
        )
        partition_by = self.validate_partition_by(params.partition_by)
        total_weight = self._bsq._get_weight(params.weights)
        agg_func, agg_weight = self._bsq._get_agg_func_and_weight(params.weights, params.agg_func)
        time_indx = 0
        if "time" in params.group_by:  # time will be added as necessary later
            time_indx = params.group_by.index("time")
            params.group_by = [g for g in params.group_by if g != "time"]
        group_by_selection = self._bsq._process_groupby_cols(params.group_by, annual_only=params.annual_only)

        if params.annual_only:
            bs_tbl, up_tbl, tbljoin = self.__get_annual_bs_up_table(upgrade_id, params.applied_only)
        else:
            bs_restrict, ts_restrict = self._bsq._split_restrict(bs_restrict)
            # When the caller wants only upgrade values (no savings, no baseline), skip the
            # ts_b/ts_u subquery pairing. Required for applied_only=True since the column
            # expressions for applied_only reference only up_tbl; also lets runs without
            # upgrade=0 timeseries rows aggregate upgrade data directly.
            upgrade_only = (
                upgrade_id != "0"
                and params.applied_only
                and not params.include_savings
                and not params.include_baseline
            )
            bs_tbl, up_tbl, tbljoin, group_by_selection = self.__get_timeseries_bs_up_table(
                enduse_cols, upgrade_id, params.applied_only, ts_restrict,
                bs_restrict=bs_restrict, group_by=group_by_selection,
                upgrade_only=upgrade_only,
            )

        def get_col(tbl, col):  # column could be MappedColumn not available in tbl
            return tbl.c[col.name] if col.name in tbl.c else col

        query_cols = []
        for col in enduse_cols:
            if params.annual_only:
                baseline_col = get_col(bs_tbl, col)
                if upgrade_id != "0":
                    if params.applied_only:
                        upgrade_col = get_col(up_tbl, col)
                    else:
                        upgrade_col = sa.case(
                            (self._bsq._get_success_condition(up_tbl), get_col(up_tbl, col)), else_=baseline_col
                        )
                else:
                    upgrade_col = baseline_col
                savings_col = safunc.coalesce(baseline_col, 0) - safunc.coalesce(upgrade_col, 0)
            else:
                baseline_col = get_col(bs_tbl, col)
                if upgrade_id != "0":
                    if params.applied_only:
                        upgrade_col = get_col(up_tbl, col)
                    else:
                        upgrade_col = sa.case(
                            (up_tbl.c[self._bsq.building_id_column_name] == None, baseline_col),  # noqa: E711
                            else_=get_col(up_tbl, col),
                        )
                else:
                    upgrade_col = baseline_col
                savings_col = safunc.coalesce(baseline_col, 0) - safunc.coalesce(upgrade_col, 0)

            if params.include_baseline:
                query_cols.append(
                    agg_func(baseline_col if agg_weight is None else baseline_col * agg_weight).label(
                        f"{self._bsq._simple_label(col.name, params.agg_func)}__baseline"
                    )
                )
            if params.include_upgrade:
                suffix = "__upgrade" if params.include_savings or params.include_baseline else ""
                query_cols.append(
                    agg_func(upgrade_col if agg_weight is None else upgrade_col * agg_weight).label(
                        f"{self._bsq._simple_label(col.name, params.agg_func)}{suffix}"
                    )
                )
                if params.get_nonzero_count and params.annual_only:
                    # Nonzero count is only valid for annual queries
                    query_cols.append(
                        safunc.sum(sa.case((safunc.coalesce(upgrade_col, 0) != 0, 1), else_=0) * total_weight).label(
                            f"{self._bsq._simple_label(upgrade_col.name)}__nonzero_units_count"
                        )
                    )
            if params.include_savings:
                query_cols.append(
                    agg_func(savings_col if agg_weight is None else savings_col * agg_weight).label(
                        f"{self._bsq._simple_label(col.name, params.agg_func)}__savings"
                    )
                )

            if params.get_quartiles:
                if params.include_baseline:
                    query_cols.append(
                        sa.func.approx_percentile(baseline_col, [0, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 1]).label(
                            f"{self._bsq._simple_label(col.name, params.agg_func)}__baseline__quartiles"
                        )
                    )
                    query_cols.append(
                        sa.func.approx_percentile(baseline_col, [0, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 1]).filter(
                            baseline_col != 0
                        ).label(
                            f"{self._bsq._simple_label(col.name, params.agg_func)}__baseline__nonzero_quartiles"
                        )
                    )
                if params.include_upgrade:
                    query_cols.append(
                        sa.func.approx_percentile(upgrade_col, [0, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 1]).label(
                            f"{self._bsq._simple_label(col.name, params.agg_func)}__upgrade__quartiles"
                        )
                    )
                    query_cols.append(
                        sa.func.approx_percentile(upgrade_col, [0, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 1]).filter(
                            upgrade_col != 0
                        ).label(
                            f"{self._bsq._simple_label(col.name, params.agg_func)}__upgrade__nonzero_quartiles"
                        )
                    )
                if params.include_savings:
                    query_cols.append(
                        sa.func.approx_percentile(savings_col, [0, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 1]).label(
                            f"{self._bsq._simple_label(col.name, params.agg_func)}__savings__quartiles"
                        )
                    )
                    query_cols.append(
                        sa.func.approx_percentile(savings_col, [0, 0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98, 1]).filter(
                            savings_col != 0
                        ).label(
                            f"{self._bsq._simple_label(col.name, params.agg_func)}__savings__nonzero_quartiles"
                        )
                    )

        if params.annual_only:  # Use annual tables
            grouping_metrics_selection = [
                safunc.sum(1).label("sample_count"),
                safunc.sum(total_weight).label("units_count"),
            ]
        elif params.timestamp_grouping_func == "year":  # Use timeseries tables but collapse timeseries
            # Equivalent to dividing sum(1)/rows_per_building, but computed inline
            # from the data instead of pre-fetching rows_per_building via a heavy
            # `count(*) GROUP BY upgrade, bldg_id` over the full TS table. The
            # integrity check in report.check_ts_bs_integrity is the appropriate
            # place to spend that scan; here we trust the cadence and let Athena
            # compute counts in one pass.
            bs_key_cols = [self._bsq.bs_table.c[k] for k in self._bsq.bs_key]
            distinct_bs_keys = self._bsq._count_distinct(bs_key_cols)
            grouping_metrics_selection = [
                distinct_bs_keys.label("sample_count"),
                (distinct_bs_keys * safunc.sum(total_weight) / safunc.sum(1)).label("units_count"),
            ]
        elif params.timestamp_grouping_func:
            colname = self._bsq.timestamp_column_name
            bs_key_cols = [self._bsq.bs_table.c[k] for k in self._bsq.bs_key]
            distinct_bs_keys = self._bsq._count_distinct(bs_key_cols)
            grouping_metrics_selection = [
                distinct_bs_keys.label("sample_count"),
                (distinct_bs_keys * safunc.sum(total_weight) / safunc.sum(1)).label("units_count"),
                (safunc.sum(1) / distinct_bs_keys).label("rows_per_sample"),
            ]
            sim_info = self._bsq._get_simulation_info()
            time_col = bs_tbl.c[self._bsq.timestamp_column_name]
            if sim_info.offset > 0:
                # If timestamps are not period beginning we should make them so for timestamp_grouping_func aggregation.
                new_col = sa.func.date_trunc(
                    params.timestamp_grouping_func, sa.func.date_add(sim_info.unit, -sim_info.offset, time_col)
                ).label(colname)
            else:
                new_col = sa.func.date_trunc(params.timestamp_grouping_func, time_col).label(colname)
            group_by_selection.insert(time_indx, new_col)
        else:
            time_col = bs_tbl.c[self._bsq.timestamp_column_name].label(self._bsq.timestamp_column_name)
            grouping_metrics_selection = [
                safunc.sum(1).label("sample_count"),
                safunc.sum(total_weight).label("units_count"),
            ]
            group_by_selection.insert(time_indx, time_col)

        query_cols = list(group_by_selection) + grouping_metrics_selection + query_cols
        query = sa.select(*query_cols).select_from(tbljoin)
        query = self._bsq._add_join(query, params.join_list)
        if params.annual_only:
            query = query.where(self._bsq._bs_successful_condition)
        query = self._bsq._add_restrict(query, bs_restrict, annual_only=params.annual_only)
        query = self._bsq._add_avoid(query, params.avoid, annual_only=params.annual_only)
        query = self._bsq._add_group_by(query, group_by_selection)
        query = self._bsq._add_order_by(query, group_by_selection if params.sort else [])
        query = query.limit(params.limit) if params.limit else query

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
