from buildstock_query import main
import logging
from typing import List, Any, Tuple, Optional, Union, Literal
from collections.abc import Sequence
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.sql import functions as safunc
from collections import defaultdict
from buildstock_query.schema.query_params import UtilityTSQuery, TSQuery
from buildstock_query.schema.helpers import gather_params
from buildstock_query.schema.utilities import AnyColType, AnyTableType, MappedColumn, validate_arguments
from buildstock_query.helpers import read_csv
from pydantic import Field, BaseModel, ValidationError


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeTuple(BaseModel):
    # has to be between 1 and 12
    month: int = Field(..., ge=1, le=12)
    is_weekend: int = Field(..., ge=0, le=1)
    hour: int = Field(..., ge=0, le=23)

    def __hash__(self):
        return hash((self.month, self.is_weekend, self.hour))


class TOURate(BaseModel):
    data: dict[TimeTuple, float] = Field(..., examples=[{TimeTuple(month=1, is_weekend=0, hour=3): 0.5}])
    raw_dict: dict[tuple[int, int, int], float] = Field(..., examples=[{(1, 0, 3): 0.5}])

    def __init__(self, rate_dict: dict[tuple[int, int, int], float]):
        data: dict[TimeTuple, float] = {}
        for key, value in rate_dict.items():
            try:
                data[TimeTuple(month=key[0], is_weekend=key[1], hour=key[2])] = value
            except ValidationError as e:
                raise ValueError(
                    f"Invalid key {key} in rate_dict. Make sure the keys are"
                    " (month (1 to 12), is_weekend (0 or 1), hour_of_day (0 to 23)"
                ) from e
        super().__init__(data=data, raw_dict=rate_dict)


class BuildStockUtility:
    """
    Class to perform electric utility centric queries.
    """

    def __init__(
        self, buildstock_query: "main.BuildStockQuery", eia_mapping_year: int = 2018, eia_mapping_version: int = 1
    ):
        """
        Class to perform electric utility centric queries
        Args:
            eia_mapping_year: The year of the EIA form 861 service territory map to use when mapping to utility \
                              service territories. Currently, only 2018 and 2012 are valid years.
            eia_mapping_version: The EIA mapping version to use.
        """
        self._bsq = buildstock_query
        self._agg = buildstock_query.agg
        self._group_query_id = 0
        self.eia_mapping_year = eia_mapping_year
        self.eia_mapping_version = eia_mapping_version
        self._cache: dict = defaultdict()

    def _aggregate_ts_by_map(
        self,
        map_table_name: str,
        baseline_column_name: str,
        map_column_name: str,
        id_column: str,
        id_list: Sequence[Any],
        params: UtilityTSQuery,
    ):
        new_table = self._bsq._get_table(map_table_name)
        new_column = self._bsq._get_column(map_column_name, candidate_tables=[new_table])
        baseline_column = self._bsq._get_column(baseline_column_name, candidate_tables=[self._bsq.bs_table])
        params.group_by = [new_table.c[id_column], *params.group_by]
        params.weights = [*params.weights, new_table.c["weight"]]
        params.join_list = [(new_table, baseline_column, new_column), *params.join_list]
        logger.info(f"Will submit request for {id_list}")
        gs = params.query_group_size
        id_list_batches = [id_list[i: i + gs] for i in range(0, len(id_list), gs)]
        results_array = []
        for current_ids in id_list_batches:
            if len(current_ids) == 1:
                current_ids = current_ids[0]
            new_params = TSQuery(
                enduses=params.enduses,
                group_by=params.group_by,
                upgrade_id=params.upgrade_id,
                sort=params.sort,
                join_list=params.join_list,
                weights=params.weights,
                restrict=[(new_table.c[id_column], current_ids)] + list(params.restrict),
                collapse_ts=params.collapse_ts,
                timestamp_grouping_func=params.timestamp_grouping_func,
                limit=params.limit,
                split_enduses=params.split_enduses,
                get_quartiles=params.get_quartiles,
                get_query_only=False if params.split_enduses else True,
            )
            logger.info(f"Submitting query for {current_ids}")
            result = self._agg.aggregate_timeseries(params=new_params)
            results_array.append(result)

        if params.get_query_only:
            return results_array

        if params.split_enduses:
            # In this case, the results_array will contain the result dataframes
            logger.info("Concatenating the results from all IDs")
            all_dfs = pd.concat(results_array)
            return all_dfs
        else:
            # In this case, results_array will contain the queries
            batch_query_id = self._bsq.submit_batch_query(results_array)
            return self._bsq.get_batch_query_result(batch_id=batch_query_id)

    def get_eiaid_map(self) -> tuple[str, str, str]:
        if self.eia_mapping_version == 1:
            map_table_name = "eiaid_weights"
            map_baseline_column = f"{self._bsq._char_prefix}county"
            map_eiaid_column = self._bsq.db_schema.column_names.map_eiaid_column
        elif self.eia_mapping_version == 2:
            map_table_name = "v2_eiaid_weights"
            map_baseline_column = f"{self._bsq._char_prefix}county"
            map_eiaid_column = self._bsq.db_schema.column_names.map_eiaid_column
        elif self.eia_mapping_version == 3:
            map_table_name = "v3_eiaid_weights_%d" % (self.eia_mapping_year)
            map_baseline_column = f"{self._bsq._char_prefix}county"
            map_eiaid_column = self._bsq.db_schema.column_names.map_eiaid_column
        else:
            raise ValueError("Invalid mapping_version")

        if map_eiaid_column is None:
            raise ValueError("map_eiaid_column is None. Check the TOML configuration or how it's being loaded.")

        return map_table_name, map_baseline_column, map_eiaid_column

    @gather_params(UtilityTSQuery)
    def aggregate_ts_by_eiaid(self, params: UtilityTSQuery):
        """
        Aggregates the timeseries result, grouping by utilities.
        Args:
            eiaid_list: The list of utility ids (EIAID) assigned by EIA.
            enduses: The list of enduses to aggregate
            group_by: Additional columns to group the aggregation by
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            get_query_only: If set to true, returns the list of queries to run instead of the result.
            query_group_size: The number of eiaids to be grouped together when running athena queries. This should be
                              used as large as possible that doesn't result in query timeout.
            split_endues: Query each enduses separately to spread load on Athena

        Returns:
            Pandas dataframe with the aggregated timeseries and the requested enduses grouped by utilities
        """
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        if not params.enduses:
            raise ValueError("Need to provide enduses")
        id_column = "eiaid"

        if params.query_group_size is None:
            params.query_group_size = min(100, len(params.eiaid_list))

        return self._aggregate_ts_by_map(
            map_table_name=eiaid_map_table_name,
            baseline_column_name=map_baseline_column,
            map_column_name=map_eiaid_column,
            id_column=id_column,
            id_list=params.eiaid_list,
            params=params,
        )

    @validate_arguments
    def aggregate_unit_counts_by_eiaid(
        self,
        *,
        eiaid_list: Optional[list[str]] = None,
        group_by: list[Union[AnyColType, tuple[str, str]]] = [],
        get_query_only: bool = False,
    ):
        """
        Returns the counts of the number of dwelling units, grouping by eiaid and other additional group_by columns if
        provided.
        Args:
            eiaid_list: Optional list of utility ids (EIAID) to aggregate for. If not provided, all the eiaids will be
                        used.
            group_by: Additional columns to group by
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            get_query_only: If set to true, returns the query instead of the result

        Returns:
            Pandas dataframe with the units counts
        """
        logger.info("Aggregating unit counts by eiaid")
        group_by = group_by or []
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        group_by = [] if group_by is None else group_by
        eiaid_col = self._bsq._get_column("eiaid", [eiaid_map_table_name])
        restrict = [(eiaid_col, eiaid_list)] if eiaid_list else []
        weight_col = ("weight", eiaid_map_table_name)
        result = self._bsq.query(
            annual_only=True,
            enduses=[],
            group_by=[eiaid_col] + group_by,
            sort=True,
            join_list=[(eiaid_map_table_name, map_baseline_column, map_eiaid_column)],
            weights=[weight_col],
            restrict=restrict,
            get_query_only=get_query_only,
        )
        return result

    @validate_arguments
    def aggregate_annual_by_eiaid(
        self,
        enduses: Sequence[AnyColType],
        group_by: Optional[List[str]] = None,
        get_query_only: bool = False,
        get_nonzero_count: bool = False,
    ):
        """
        Aggregates the annual consumption in the baseline table, grouping by all the utilities
        Args:
            enduses: The list of enduses to aggregate
            group_by: Additional columns to group the aggregation by
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            get_query_only: If set to true, returns the list of queries to run instead of the result.
        Returns:
            Pandas dataframe with the annual sum of the requested enduses, grouped by utilities
        """
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        join_list = [(eiaid_map_table_name, map_baseline_column, map_eiaid_column)]
        group_by = [] if group_by is None else group_by
        group_by_cols = self._bsq._process_groupby_cols(group_by, annual_only=True)
        eiaid_col = self._bsq._get_column("eiaid", [eiaid_map_table_name])
        weight_col = ("weight", eiaid_map_table_name)
        result = self._bsq.query(
            enduses=enduses,
            group_by=[eiaid_col] + group_by_cols,
            join_list=join_list,
            weights=[weight_col],
            sort=True,
            get_query_only=get_query_only,
            get_nonzero_count=get_nonzero_count,
        )
        return result

    @validate_arguments
    def get_filtered_results_csv_by_eiaid(self, eiaids: List[str], get_query_only: bool = False):
        """
        Returns a portion of the results csvs, which belongs to given list of utilities
        Args:
            eiaids: The eiaid list of utitlies
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            get_query_only: If set to true, returns the list of queries to run instead of the result.

        Returns:
            Pandas dataframe that is a subset of the results csv, that belongs to provided list of utilities
        """
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        query = sa.select("*").select_from(self._bsq.bs_table)
        query = self._bsq._add_join(query, [(eiaid_map_table_name, map_baseline_column, map_eiaid_column)])
        query = self._bsq._add_restrict(query, [(self._bsq._get_column("eiaid", [eiaid_map_table_name]), eiaids)])
        query = query.where(self._bsq._get_column("weight", [eiaid_map_table_name]) > 0)
        if get_query_only:
            return self._bsq._compile(query)
        res = self._bsq.execute(query)
        return res

    @validate_arguments
    def get_eiaids(self, restrict: Optional[List[Tuple[str, List]]] = None) -> list[str]:
        """
        Returns the list of eiaids
        Args:
            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
        Returns:
            Pandas dataframe consisting of the eiaids belonging to the provided list of locations.
        """
        restrict = list(restrict) if restrict else []
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        eiaid_col = self._bsq._get_column("eiaid", [eiaid_map_table_name])
        if "eiaids" in self._cache:
            if self._bsq.db_name + "/" + eiaid_map_table_name in self._cache["eiaids"]:
                return self._cache["eiaids"][self._bsq.db_name + "/" + eiaid_map_table_name]
        else:
            self._cache["eiaids"] = {}

        join_list = [(eiaid_map_table_name, map_baseline_column, map_eiaid_column)]
        weight_col = ("weight", eiaid_map_table_name)
        annual_agg = self._bsq.query(
            annual_only=True, enduses=[], group_by=[eiaid_col], restrict=restrict, join_list=join_list,
            weights=[weight_col], sort=True
        )
        self._cache["eiaids"] = list(annual_agg["eiaid"].to_numpy(dtype=str).tolist())
        return self._cache["eiaids"]

    @validate_arguments
    def get_buildings_by_eiaids(self, eiaids: List[str], get_query_only: bool = False):
        """
        Returns the list of buildings belonging to the given list of utilities.
        Args:
            eiaids: list of utility EIAIDs
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            get_query_only: If set to true, returns the query string instead of the result

        Returns:
            Pandas dataframe consisting of the building ids belonging to the provided list of utilities.

        """
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        query = sa.select(*[self._bsq.bs_bldgid_column.distinct()])
        query = self._bsq._add_join(query, [(eiaid_map_table_name, map_baseline_column, map_eiaid_column)])
        query = self._bsq._add_restrict(query, [(self._bsq._get_column("eiaid", [eiaid_map_table_name]), eiaids)])
        query = query.where(self._bsq._get_column("weight", [eiaid_map_table_name]) > 0)
        if get_query_only:
            return self._bsq._compile(query)
        res = self._bsq.execute(query)
        return res

    @validate_arguments
    def get_locations_by_eiaids(self, eiaids: List[str], get_query_only: bool = False):
        """
        Returns the list of locations/counties (depends on mapping version) belonging to a given list of utilities.
        Args:
            eiaids: list of utility EIAIDs
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
            get_query_only: If set to true, returns the query string instead of the result

        Returns:
            Pandas dataframe consisting of the locations (for version 1) or counties (for version 2) belonging to the
            provided list of utilities.

        """
        eiaid_map_table_name, map_baseline_column, map_eiaid_column = self.get_eiaid_map()
        eiaid_map_table = self._bsq._get_table(eiaid_map_table_name)
        query = sa.select(*[eiaid_map_table.c[map_eiaid_column].distinct()])
        query = self._bsq._add_restrict(query, [(self._bsq._get_column("eiaid", [eiaid_map_table_name]), eiaids)])
        query = query.where(eiaid_map_table.c["weight"] > 0)
        if get_query_only:
            return self._bsq._compile(query)
        res = self._bsq.execute(query)
        return list(res[map_eiaid_column].values)

    def get_rate_map(self, weekend_csv_path: str, weekday_csv_path: str) -> dict[tuple[int, int, int], float]:
        def read_rate_file(file_path: str) -> pd.DataFrame:
            df = read_csv(file_path)
            if len(df) != 12:
                raise ValueError(f"Invalid number of rows in {file_path}. Expected 12, got {len(df)}")
            if len(df.columns) != 25:
                raise ValueError(f"Invalid number of columns in {file_path}. Expected 25, got {len(df.columns)}")
            if df.columns[0] != "month":
                raise ValueError(f"Invalid column names in {file_path}. Expected first column to be 'month'")
            df = df.set_index("month")
            df.index = pd.Index(range(1, 13), name="month")
            df.columns = pd.Index(range(24), name="Hour")
            return df

        weekday_rate = read_rate_file(weekday_csv_path)
        weekend_rate = read_rate_file(weekend_csv_path)
        weekday_rate["weekend"] = 0
        weekend_rate["weekend"] = 1
        full_rate = pd.concat([weekday_rate, weekend_rate])
        full_rate = full_rate.reset_index().melt(
            id_vars=["month", "weekend"], value_vars=range(24), var_name="hour", value_name="rate"
        )
        rate_map = full_rate.set_index(["month", "weekend", "hour"])["rate"].to_dict()
        return rate_map

    @validate_arguments
    def calculate_tou_bill(
        self,
        *,
        rate_map: Union[tuple[str, str], dict[tuple[int, int, int], float]],
        meter_col: Optional[Union[AnyColType, tuple[AnyColType, ...]]] = None,
        group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list),
        upgrade_id: Union[int, str] = "0",
        sort: bool = True,
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = Field(default_factory=list),
        weights: Sequence[Union[str, tuple]] = Field(default_factory=list),
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
        collapse_ts: bool = False,
        timestamp_grouping_func: Optional[Literal["month", "day", "hour"]] = "month",
        limit: Optional[int] = None,
        get_query_only: bool = False,
    ):
        if isinstance(rate_map, tuple):
            rate_map = self.get_rate_map(*rate_map)
        user_rate = TOURate(rate_map)
        if self._bsq.ts_table is None:
            raise ValueError("No timeseries table found")

        TOU_enduse = {}
        if meter_col is None:
            TOU_enduse["fuel_use__electricity__net__kwh__TOU"] = self._bsq.ts_table.c[
                "fuel_use__electricity__total__kwh"
            ] + safunc.coalesce(self._bsq.ts_table.c["end_use__electricity__pv__kwh"], 0)
        elif isinstance(meter_col, tuple):
            for col in meter_col:
                TOU_enduse[f"{col}__TOU"] = self._bsq._get_column(col, [self._bsq.ts_table])
        else:
            TOU_enduse[f"{meter_col}__TOU"] = self._bsq._get_column(meter_col, [self._bsq.ts_table])

        month_col, is_weekend_col, hour_col = (
            self._bsq._get_special_column(col) for col in ("month", "is_weekend", "hour")
        )
        rate_col = MappedColumn(
            bsq=self._bsq, name="tou_rate", mapping_dict=user_rate.raw_dict, key=(month_col, is_weekend_col, hour_col)
        )

        enduses_list = []
        for col in TOU_enduse:
            enduses_list.append((TOU_enduse[col] * rate_col / 100).label(f"{col}__dollars"))

        ts_query = TSQuery(
            enduses=enduses_list,
            group_by=group_by,
            upgrade_id=str(upgrade_id),
            sort=sort,
            join_list=join_list,
            weights=weights,
            restrict=restrict,
            collapse_ts=collapse_ts,
            timestamp_grouping_func=timestamp_grouping_func,
            limit=limit,
            get_query_only=get_query_only,
        )
        return self._agg.aggregate_timeseries(params=ts_query)
