import sqlalchemy as sa
from sqlalchemy.sql import func as safunc
import datetime
import numpy as np
import logging
from buildstock_query import main
from buildstock_query.schema.query_params import Query
import pandas as pd
from buildstock_query.schema.helpers import gather_params
from typing import Optional, Union
from collections.abc import Sequence
from buildstock_query.schema.utilities import DBColType, RestrictTuple, SALabel, typed_literal, validate_arguments
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
        timestamp_grouping_func: Optional[str] = None,
        total_weight=None,
    ):
        if self._bsq.ts_table is None:
            raise ValueError("No timeseries table found in database.")

        ts = self._bsq.ts_table
        base = self._bsq.bs_table  # canonical alias of md_table
        ucol = self._bsq._ts_upgrade_col

        # Push any user-supplied bs_restrict (e.g. comstock `state='CO'`) into the
        # inner ts ⋈ bs join condition. Without this, Athena scans the full metadata
        # table before applying user filters — for comstock's tract-denormalized
        # metadata that's the difference between minutes and timeouts. Adding to
        # the JOIN ON clause (rather than wrapping bs in another subquery) keeps
        # the SELECT list clean and lets Athena push the predicate into the bs
        # table scan without enumerating all columns.
        bs_restrict_clauses = self._bsq._get_restrict_clauses(bs_restrict, annual_only=True)

        # Unified two-level shape used for both single-upgrade and
        # upgrade-pair queries.
        #
        # ts_flat: per-row scalar projection. Each enduse expression
        #   (whether bare ts column or calc-col Label) is materialized as
        #   `_v__<name>`. This pushes arithmetic into the scan layer.
        # ts_aggr: per-(bldg_id, bucketed_time, state, ...) aggregate.
        #   Single-upgrade: SUM(_v__name) → bs__<name>. Upgrade-pair:
        #   SUM(...) FILTER (WHERE upgrade=0/N) → bs__<name> / up__<name>.
        # outer: JOIN to bs (once) for weights/metadata, then user's GROUP BY.
        #
        # Pre-bucketing time at ts_aggr cuts the per-bldg shuffle key
        # cardinality by 4×/96×/720×/35000× for hourly/daily/monthly/yearly.
        # The shuffle is what made the old upgrade-pair pivot time out on
        # national hourly queries and what slows down baseline TS queries
        # at the same scale.
        single_upgrade = upgrade_id == "0" or upgrade_only
        ts_upgrade_ids = [upgrade_id] if single_upgrade else ["0", upgrade_id]

        ts_group_by = [g for g in group_by if g.name in ts.columns]
        bs_group_by = [g for g in group_by if g.name not in ts.columns]

        ts_unique_keys = self._bsq._get_unique_keys("timeseries")
        timestamp_col = self._bsq.timestamp_column_name
        # Order keys for hash distribution: partition columns (typically
        # `state`) first, then timestamp, then bldg_id last. Trino hashes
        # by leftmost columns when shuffling for GROUP BY; partition-aligned
        # ordering lets it distribute work along the parquet's existing
        # layout instead of fighting it.
        partition_cols = [k for k in ts_unique_keys if k != self._bsq.building_id_column_name]
        ts_key_names = list(dict.fromkeys([
            *partition_cols,
            timestamp_col,
            self._bsq.building_id_column_name,
        ]))
        ts_extra_group_names = [g.name for g in ts_group_by if g.name not in ts_key_names]

        # Bucketed time expression — pushed into ts_flat so ts_aggr GROUPs BY
        # coarse buckets, not raw 15-min timestamps.
        if timestamp_grouping_func:
            sim_info = self._bsq._get_simulation_info()
            raw_time = ts.c[timestamp_col]
            if sim_info.offset > 0:
                bucketed_time_expr = sa.func.date_trunc(
                    timestamp_grouping_func,
                    sa.func.date_add(sim_info.unit, -sim_info.offset, raw_time),
                )
            else:
                bucketed_time_expr = sa.func.date_trunc(timestamp_grouping_func, raw_time)
        else:
            bucketed_time_expr = ts.c[timestamp_col]

        ts_restrict_clauses = self._bsq._get_restrict_clauses(restrict, annual_only=False)

        # Classify each enduse by which table(s) its leaf columns reference:
        # - ts-only: every leaf is on ts. Routed through ts_flat / ts_aggr.
        # - pure-bs: every leaf is on bs (no ts refs). Skips ts_flat entirely;
        #   projected at the outer SELECT directly. This is the right path
        #   for characteristic columns (sqft, vintage, etc.) — constant per
        #   bldg, no need to materialize per-15-min and re-aggregate.
        # - mixed: at least one ts and one bs leaf. Routed through ts_flat
        #   with bs joined in (preserves today's inner-join shape).
        from sqlalchemy.sql import visitors

        def _classify(expr):
            target = expr.element if isinstance(expr, SALabel) else expr
            ts_refs, bs_refs = [], []
            def _visit(elem):
                if isinstance(elem, sa.Column):
                    t = getattr(elem, "table", None)
                    if t is ts:
                        ts_refs.append(elem)
                    elif t is not None:
                        bs_refs.append(elem)
            visitors.traverse(target, {}, {"column": _visit})
            if ts_refs and bs_refs:
                return "mixed"
            if bs_refs:
                return "pure_bs"
            return "ts_only"

        ts_only_enduses, bs_only_enduses, mixed_enduses = [], [], []
        for e in enduses:
            kind = _classify(e)
            if kind == "ts_only":
                ts_only_enduses.append(e)
            elif kind == "pure_bs":
                bs_only_enduses.append(e)
            else:
                mixed_enduses.append(e)

        flat_enduses = ts_only_enduses + mixed_enduses
        needs_bs_in_flat = bool(mixed_enduses)

        # Innermost flat subquery: precomputed scalars per ts row. Pure-bs
        # enduses are NOT projected here — they go straight to the outer
        # SELECT. `upgrade` is projected only for the upgrade-pair case
        # (where ts_aggr uses FILTER per side); single-upgrade filters at
        # ts_flat WHERE and skips the column.
        flat_select_cols = [
            ts.c[k].label(k) for k in ts_key_names if k != timestamp_col
        ]
        flat_select_cols.append(bucketed_time_expr.label(timestamp_col))
        flat_select_cols.extend([ts.c[name].label(name) for name in ts_extra_group_names])
        if not single_upgrade:
            flat_select_cols.append(ts.c["upgrade"].label("upgrade"))
        for e in flat_enduses:
            value_expr = e.element if isinstance(e, SALabel) else e
            flat_select_cols.append(value_expr.label(f"_v__{e.name}"))

        # FROM: ts alone unless we have a mixed enduse referencing bs from
        # within an arithmetic expression. _baseline_timeseries_join_condition
        # bakes in bs.upgrade=0.
        if needs_bs_in_flat:
            flat_from = ts.join(
                base,
                self._bsq._baseline_timeseries_join_condition(base, ts),
            )
        else:
            flat_from = ts

        ts_flat_subq = (
            sa.select(*flat_select_cols)
            .select_from(flat_from)
            .where(
                ts.c["upgrade"].in_([typed_literal(ts.c["upgrade"], u) for u in ts_upgrade_ids]),
                *ts_restrict_clauses,
            )
            .subquery("ts_flat")
        )

        # ts_aggr: per-(bldg, bucket, state, ...) aggregate over flat_enduses.
        # Pure-bs enduses are not in flat_enduses, so they don't appear in
        # ts_aggr — they get projected directly at the outer SELECT.
        flat_group_keys = [ts_flat_subq.c[k] for k in ts_key_names]
        flat_extra_group_cols = [ts_flat_subq.c[name] for name in ts_extra_group_names]

        enduse_aggr_cols = []
        if single_upgrade:
            for e in flat_enduses:
                v = ts_flat_subq.c[f"_v__{e.name}"]
                enduse_aggr_cols.append(safunc.sum(v).label(f"bs__{e.name}"))
            inner_rows = safunc.count(sa.text("*")).label("_inner_rows")
        else:
            bs_filter = ts_flat_subq.c["upgrade"] == typed_literal(ts.c["upgrade"], "0")
            up_filter = ts_flat_subq.c["upgrade"] == typed_literal(ts.c["upgrade"], upgrade_id)
            for e in flat_enduses:
                v = ts_flat_subq.c[f"_v__{e.name}"]
                enduse_aggr_cols.append(safunc.sum(v).filter(bs_filter).label(f"bs__{e.name}"))
                enduse_aggr_cols.append(safunc.sum(v).filter(up_filter).label(f"up__{e.name}"))
            inner_rows = safunc.count(sa.text("*")).filter(bs_filter).label("_inner_rows")

        ts_aggr_subq = (
            sa.select(*flat_group_keys, *flat_extra_group_cols, *enduse_aggr_cols, inner_rows)
            .select_from(ts_flat_subq)
            .group_by(*flat_group_keys, *flat_extra_group_cols)
            .subquery("ts_aggr")
        )

        # Pre-aggregate bs to BUILDING grain (collapse tract fan-out).
        #
        # ComStock's `*_md_by_state_and_county_parquet` has multiple tract rows
        # per (bldg_id, state) pair. A direct `ts ⋈ bs` JOIN fans out each
        # ts/ts_aggr row by N_tracts-per-bldg, blowing up the post-join shuffle
        # for the outer aggregate (Stage 5 of a national hourly query
        # processed 6.28 B rows / 499 GB and aborted at 17h33m before this fix).
        #
        # All current outer aggregations are linear in `weight`, so we can
        # collapse the tract dimension upfront:
        #   bldg_weight        = SUM(weight)        per (bldg, state)
        #   tract_count        = COUNT(*)           per (bldg, state)
        #   bldg_<col>_weighted = SUM(<bs_col>*weight) per (bldg, state)
        # Outer aggregates that used `bs.weight` / `bs.<col> * bs.weight` /
        # `count_distinct(md_keys)` translate to references on this
        # subquery's pre-summed columns. ResStock's md is one row per bldg,
        # so the GROUP BY is a no-op there (sum of one term).
        #
        # `total_weight` was constructed above as `bs.weight × user_weights`
        # bound to bs_table; we pass it in here so its multipliers are
        # baked into bldg_weight before the outer SELECT references it.
        bs_per_bldg_cols = [base.c[k].label(k) for k in ts_unique_keys]
        # Pass through bs-side group-by columns (per-bldg constants).
        # `bs_group_by` items are labeled SA expressions from _get_gcol; the
        # label name is the simple form (no `in.` prefix), but the underlying
        # column reference is on `base`. Project them by the underlying
        # expression (via `arbitrary()` since they're per-bldg constants
        # being collapsed across tract rows) labeled with the simple name
        # so the outer code's `g.name` lookups on bs_per_bldg resolve.
        bs_group_by_extras = []
        for g in bs_group_by:
            if g.name in ts_unique_keys:
                continue
            underlying = g.element if isinstance(g, SALabel) else g
            bs_per_bldg_cols.append(safunc.arbitrary(underlying).label(g.name))
            bs_group_by_extras.append(g)
        weight_expr = total_weight if total_weight is not None else base.c["weight"]
        bs_per_bldg_cols.append(safunc.sum(weight_expr).label("bldg_weight"))
        bs_per_bldg_cols.append(safunc.count(sa.text("*")).label("tract_count"))
        # Pure-bs enduses (e.g. sqft, vintage) are per-bldg constants — pick
        # one value per bldg via `arbitrary()` (Trino's any-value aggregate).
        # The outer SELECT can then multiply by bldg_weight uniformly with
        # everything else, no special path needed.
        for e in bs_only_enduses:
            value_expr = e.element if isinstance(e, SALabel) else e
            bs_per_bldg_cols.append(
                safunc.arbitrary(value_expr).label(e.name)
            )

        bs_per_bldg = (
            sa.select(*bs_per_bldg_cols)
            .select_from(base)
            .where(
                self._bsq._upgrade_zero_filter(base),
                *bs_restrict_clauses,
            )
            .group_by(*(base.c[k] for k in ts_unique_keys))
            .subquery("bs_per_bldg")
        )

        # Outer JOIN: ts_aggr ⋈ bs_per_bldg (one row per bldg) on the ts
        # unique keys. No tract fan-out post-join.
        bs_join_cond = sa.and_(
            *(bs_per_bldg.c[k] == ts_aggr_subq.c[k] for k in ts_unique_keys),
        )
        # applied_only=True for the upgrade_only path is enforced upstream via
        # _add_applied_in_restrict (synthesizes applied_in=[upgrade_id] →
        # routes into ts_restrict at ts_flat WHERE).

        tbljoin = ts_aggr_subq.join(bs_per_bldg, bs_join_cond)

        # SideView adapter: indexes columns by enduse name across BOTH
        # ts_aggr (ts-side and mixed enduses, prefixed `bs__` / `up__`) and
        # bs_per_bldg (pure-bs enduses, projected by their original name).
        # This way `get_col(bs_tbl, e)` resolves uniformly regardless of
        # which side the enduse came from.
        class _SideView:
            """Adapter exposing aggregate-subquery columns indexed by enduse name."""
            def __init__(self, ts_subq, prefix, ts_enduses, group_cols, bs_subq, bs_enduses):
                self._cols_by_name = {}
                for e in ts_enduses:
                    self._cols_by_name[e.name] = ts_subq.c[f"{prefix}__{e.name}"]
                for e in bs_enduses:
                    if e.name in bs_subq.c:
                        self._cols_by_name[e.name] = bs_subq.c[e.name]
                for c in group_cols:
                    if c.name not in self._cols_by_name:
                        self._cols_by_name[c.name] = ts_subq.c[c.name]
                if "_inner_rows" in ts_subq.c:
                    self._cols_by_name["_inner_rows"] = ts_subq.c["_inner_rows"]

            @property
            def c(self):
                return self._cols_by_name

        passthrough_cols = flat_group_keys + flat_extra_group_cols
        ts_b = _SideView(ts_aggr_subq, "bs", flat_enduses, passthrough_cols, bs_per_bldg, bs_only_enduses)
        # Pure-bs enduses are upgrade-invariant (sqft is sqft regardless of
        # upgrade), so up-side resolves to the same bs_per_bldg column.
        ts_u = ts_b if single_upgrade else _SideView(
            ts_aggr_subq, "up", flat_enduses, passthrough_cols, bs_per_bldg, bs_only_enduses,
        )

        # Remap user's group_by:
        #   ts-side group_bys → ts_aggr_subq column
        #   bs-side group_bys → bs_per_bldg column (passed through above)
        remapped_group_by = []
        for g in group_by:
            if g.name in ts.columns:
                remapped_group_by.append(ts_aggr_subq.c[g.name])
            elif g.name in bs_per_bldg.c:
                remapped_group_by.append(bs_per_bldg.c[g.name])
            else:
                remapped_group_by.append(g)

        return ts_b, ts_u, tbljoin, remapped_group_by, bs_per_bldg

    @validate_arguments
    def __get_annual_bs_up_table(self, upgrade_id: str, applied_only: bool | None):
        bs = self._bsq.bs_table  # canonical alias
        if upgrade_id == "0":
            # Baseline-only path: no join. The caller filters to baseline rows
            # via `_md_baseline_successful_condition` in the outer WHERE.
            return bs, bs, bs

        up = self._bsq.md_table.alias("up")
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

    @validate_arguments
    def get_building_average_kws_at(
        self,
        *,
        at_hour: Union[list[float], float],
        at_days: list[float],
        enduses: list[str],
        upgrade_id: Union[int, str] = "0",
        restrict: Sequence[RestrictTuple] = Field(default_factory=list),
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
        ts = self._bsq.ts_table
        if ts is None:
            raise ValueError("No timeseries table found in database.")
        ucol = self._bsq._ts_upgrade_col

        # Constrain the TS-side upgrade in the join condition. Without this, the
        # join cross-products against every upgrade present in the TS table —
        # the bs subquery's `WHERE upgrade = ...` doesn't filter the TS scan.
        # Also push any user-supplied restrict into the bs/ts split so partition
        # filters (e.g. state='CO') ride the JOIN ON instead of the outer WHERE.
        upgrade_str = "0" if upgrade_id in (None, "0") else str(upgrade_id)
        bs_restrict_split, ts_restrict_split, extra_restrict_split = self._bsq._split_restrict(list(restrict))
        bs_restrict_clauses = self._bsq._get_restrict_clauses(bs_restrict_split, annual_only=True)
        ts_restrict_clauses = self._bsq._get_restrict_clauses(ts_restrict_split, annual_only=False)

        bs = self._bsq.bs_table  # canonical alias
        query = sa.select(*ts_key_cols + grouping_metrics_selection + enduse_selection)
        query = query.join(
            bs,
            sa.and_(
                self._bsq._baseline_timeseries_join_condition(bs, ts),
                ucol == typed_literal(ucol, upgrade_str),
                *bs_restrict_clauses,
                *ts_restrict_clauses,
            ),
        )
        query = self._bsq._add_group_by(query, ts_key_cols)
        query = self._bsq._add_order_by(query, ts_key_cols)
        if extra_restrict_split:
            query = self._bsq._add_restrict(query, extra_restrict_split, annual_only=False)

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
            # The result columns use the simple-label form (stripped of `out.`
            # prefix), not the raw enduse strings the user passed. Translate
            # before indexing so the weighted-average update lands on the right
            # columns.
            enduse_label_cols = [self._bsq._simple_label(e) for e in enduses]
            lower_vals[enduse_label_cols] = (
                lower_vals[enduse_label_cols] * avg_lower_weight
                + upper_vals[enduse_label_cols] * avg_upper_weight
            )
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
        # On TS paths, `applied_only=True` must filter the surviving md_keys to
        # buildings where the upgrade applied — the annual flow does this via the
        # md self-join on (bs.bldg_id = up.bldg_id AND up.applicability=true), but
        # the TS flow has no such join in the single-upgrade or upgrade-pair shapes.
        # Synthesize an `applied_in=[upgrade_id]` to ride the existing
        # `_get_applied_in_subquery` machinery (which enforces
        # `_md_successful_condition` on the upgrade rows). Without this filter,
        # inapplicable buildings (which have TS rows under inapplicables_have_ts)
        # would silently inflate totals across all `applied_only=True` TS queries.
        effective_applied_in = params.applied_in
        if (
            not params.annual_only
            and params.applied_only
            and upgrade_id != "0"
            and not effective_applied_in
        ):
            effective_applied_in = [upgrade_id]
        bs_restrict = self._bsq._add_applied_in_restrict(
            params.restrict,
            applied_in=effective_applied_in,
            annual_only=params.annual_only,
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
        time_indx = len(params.group_by)
        time_aliases = {"time", self._bsq.timestamp_column_name}
        for alias in time_aliases:
            if alias in params.group_by:
                time_indx = params.group_by.index(alias)
                params.group_by = [g for g in params.group_by if g not in time_aliases]
                break
        group_by_selection = self._bsq._process_groupby_cols(params.group_by, annual_only=params.annual_only)

        pivot_bucketed_time = False
        if params.annual_only:
            bs_tbl, up_tbl, tbljoin = self.__get_annual_bs_up_table(upgrade_id, params.applied_only)
            md_alias = bs_tbl  # annual: bs_tbl IS the metadata-side handle
            extra_restrict: list = []
        else:
            bs_restrict, ts_restrict, extra_restrict = self._bsq._split_restrict(bs_restrict)
            # When the caller wants only upgrade values (no savings, no baseline column),
            # skip the pivot subquery. For `applied_only=True` the only-upgrade-rows
            # behavior is the definition. For `applied_only=False`, the pivot's bs side
            # exists solely for the COALESCE fallback when a building is missing an
            # upgrade row — but `inapplicables_have_ts=True` (forced for this codebase)
            # guarantees every building has a TS row for every upgrade, so the fallback
            # never fires. Taking the single-scan path halves the TS scan and skips the
            # CASE/GROUP-BY pivot, restoring the pre-pivot timing for this shape.
            upgrade_only = (
                upgrade_id != "0"
                and not params.include_savings
                and not params.include_baseline
            )
            bs_tbl, up_tbl, tbljoin, group_by_selection, md_alias = self.__get_timeseries_bs_up_table(
                enduse_cols, upgrade_id, params.applied_only, ts_restrict,
                bs_restrict=bs_restrict, group_by=group_by_selection,
                upgrade_only=upgrade_only,
                timestamp_grouping_func=params.timestamp_grouping_func,
                total_weight=total_weight,
            )
            # md_alias is now the bs_per_bldg subquery (per-(bldg, state) row
            # with sum(weight) AS bldg_weight, count(*) AS tract_count, and
            # SUM(<col>*weight) AS _w__<name> for any pure-bs enduses). The
            # outer SELECT below references these pre-summed columns instead
            # of bs.weight / count_distinct(md_keys) / etc. directly. This
            # eliminates ComStock's tract fan-out at the post-join shuffle.
            #
            # The outer per-row weight becomes md_alias.c["bldg_weight"] —
            # already includes sample_wt × user_weights from total_weight,
            # pre-summed at building grain.
            ts_total_weight = md_alias.c["bldg_weight"]
            total_weight = ts_total_weight
            if agg_weight is not None:
                agg_weight = ts_total_weight
            # Inner ts_aggr always pre-buckets time when grouping_func is
            # set — true for both single-upgrade (upgrade_id=="0" or
            # upgrade_only) and upgrade-pair branches. The outer SELECT
            # references `ts_aggr.<timestamp>` directly (already bucketed)
            # and uses `_inner_rows` instead of raw sum(1) for the
            # rows_per_sample / units_count denominator.
            inner_bucketed_time = params.timestamp_grouping_func is not None
            # Legacy alias kept for the rest of _query() which references
            # the prior name. TODO: rename once the dust settles.
            pivot_bucketed_time = inner_bucketed_time

        def get_col(tbl, col):  # column could be MappedColumn not available in tbl
            return tbl.c[col.name] if col.name in tbl.c else col

        def rebind_to(col, target_tbl):
            """Bind an enduse expression to `target_tbl`.

            For bare columns (Column / SACol): if the column name exists on
            `target_tbl`, return that column; otherwise return the original.
            For Labels (from get_calculated_column): the underlying expression
            references columns on whichever table get_calculated_column was
            given (typically bs_tbl). Use SA's ClauseAdapter to rewrite each
            column reference to its counterpart on `target_tbl`, then re-label.
            Falls through unchanged for `_SideView` (pivot subquery columns
            already carry per-side prefix, so target_tbl's `.c[name]` lookup
            already returns the correct pivot column — no traversal needed).
            """
            if isinstance(col, SALabel):
                # _SideView adapters expose calc-col labels directly (already
                # per-side); plain Aliases / subqueries don't, so we adapt the
                # underlying expression's column refs to point at target_tbl.
                if col.name in getattr(target_tbl, "c", {}):
                    return target_tbl.c[col.name]
                from sqlalchemy.sql.util import ClauseAdapter
                # adapt_on_names=True needed because bs_tbl and up_tbl are
                # both aliases of the same md_table; SA's default
                # corresponding_column resolution doesn't bridge cross-alias
                # references (it stops at the alias boundary).
                adapted = ClauseAdapter(target_tbl, adapt_on_names=True).traverse(col.element)
                return adapted.label(col.name)
            return get_col(target_tbl, col)

        query_cols = []
        for col in enduse_cols:
            if params.annual_only:
                baseline_col = get_col(bs_tbl, col)
                if upgrade_id != "0":
                    if params.applied_only:
                        upgrade_col = rebind_to(col, up_tbl)
                    else:
                        upgrade_col = sa.case(
                            (self._bsq._get_success_condition(up_tbl), rebind_to(col, up_tbl)), else_=baseline_col
                        )
                else:
                    upgrade_col = baseline_col
                savings_col = safunc.coalesce(baseline_col, 0) - safunc.coalesce(upgrade_col, 0)
            else:
                baseline_col = get_col(bs_tbl, col)
                if upgrade_id != "0":
                    if params.applied_only or bs_tbl is up_tbl:
                        # Single-scan path (applied_only=True OR the
                        # upgrade_only short-circuit path which returns
                        # bs_tbl == up_tbl == ts): the upgrade col is just
                        # the ts row's value, no COALESCE fallback needed.
                        upgrade_col = rebind_to(col, up_tbl)
                    else:
                        # Pivot path: per-(bldg_id, timestamp) row, up_<col>
                        # is NULL when the upgrade didn't produce a row for
                        # this bldg. Fall back to baseline via COALESCE.
                        upgrade_col = safunc.coalesce(rebind_to(col, up_tbl), baseline_col)
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
            # md_alias is bs_per_bldg (one row per (bldg, state)). With
            # ts_aggr also pre-aggregating to per-(bldg, bucket, state),
            # the outer JOIN gives one row per (bldg, bucket, state). Per
            # outer (state, year) group:
            #   sample_count = sum(tract_count)  # collapses tract fan-out
            #                                       to original distinct count
            #   units_count  = sum(bldg_weight)
            #   rows_per_sample = sum(_inner_rows) / count(distinct bldgs)
            sample_count_expr = safunc.sum(md_alias.c["tract_count"])
            ts_row_count_for_cadence = safunc.sum(bs_tbl.c["_inner_rows"])
            distinct_bldgs = self._bsq._count_distinct(
                [bs_tbl.c[k] for k in self._bsq._get_unique_keys("timeseries")]
            )
            grouping_metrics_selection = [
                sample_count_expr.label("sample_count"),
                safunc.sum(md_alias.c["bldg_weight"]).label("units_count"),
                (ts_row_count_for_cadence / distinct_bldgs).label("rows_per_sample"),
            ]
        elif params.timestamp_grouping_func:
            colname = self._bsq.timestamp_column_name
            sample_count_expr = safunc.sum(md_alias.c["tract_count"])
            ts_row_count_for_cadence = safunc.sum(bs_tbl.c["_inner_rows"])
            distinct_bldgs = self._bsq._count_distinct(
                [bs_tbl.c[k] for k in self._bsq._get_unique_keys("timeseries")]
            )
            grouping_metrics_selection = [
                sample_count_expr.label("sample_count"),
                safunc.sum(md_alias.c["bldg_weight"]).label("units_count"),
                (ts_row_count_for_cadence / distinct_bldgs).label("rows_per_sample"),
            ]
            time_col = bs_tbl.c[self._bsq.timestamp_column_name]
            if pivot_bucketed_time:
                # Pivot subquery already date_trunc'd the time at the inner
                # GROUP BY (the perf-critical optimization). The outer SELECT
                # just references the bucketed column directly.
                new_col = time_col.label(colname)
            else:
                sim_info = self._bsq._get_simulation_info()
                if sim_info.offset > 0:
                    # If timestamps are not period beginning we should make them so for timestamp_grouping_func aggregation.
                    new_col = sa.func.date_trunc(
                        params.timestamp_grouping_func, sa.func.date_add(sim_info.unit, -sim_info.offset, time_col)
                    ).label(colname)
                else:
                    new_col = sa.func.date_trunc(params.timestamp_grouping_func, time_col).label(colname)
            group_by_selection.insert(time_indx, new_col)
        else:
            # Raw 15-min TS output (no timestamp_grouping_func). The outer
            # SELECT references the ts_aggr (or pivot subquery) timestamp
            # column directly. units_count uses bs_per_bldg's pre-summed
            # weight; sample_count counts tract rows via tract_count.
            time_col = bs_tbl.c[self._bsq.timestamp_column_name].label(self._bsq.timestamp_column_name)
            grouping_metrics_selection = [
                safunc.sum(md_alias.c["tract_count"]).label("sample_count"),
                safunc.sum(md_alias.c["bldg_weight"]).label("units_count"),
            ]
            group_by_selection.insert(time_indx, time_col)

        query_cols = list(group_by_selection) + grouping_metrics_selection + query_cols
        query = sa.select(*query_cols).select_from(tbljoin)
        query = self._bsq._add_join(query, params.join_list)
        if params.annual_only:
            # Successful baseline rows on the bs alias that's in the FROM. For
            # upgrade-pair queries the join's ON already enforces bs.upgrade=0
            # — Trino dedupes the duplicate predicate at planning time.
            query = query.where(
                sa.and_(
                    self._bsq._get_success_condition(bs_tbl),
                    self._bsq._upgrade_zero_filter(bs_tbl),
                )
            )
            # Annual queries have no inner join helper to fold bs_restrict into,
            # so the outer WHERE is the only place to apply it.
            query = self._bsq._add_restrict(query, bs_restrict, annual_only=params.annual_only)
        # TS queries fold bs_restrict into the inner ts ⋈ bs JOIN ON inside
        # __get_timeseries_bs_up_table, so adding it again here would just
        # produce duplicate predicates that Trino has to dedupe.
        # Restricts on join_list tables (e.g. utility eiaid_weights.eiaid) didn't
        # land on bs or ts — they go to the outer WHERE after _add_join has
        # introduced their referenced tables.
        if extra_restrict:
            query = self._bsq._add_restrict(query, extra_restrict, annual_only=params.annual_only)
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
