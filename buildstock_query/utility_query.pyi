import buildstock_query.main as main
import logging
from typing import Any, Optional, Union, Sequence
import pandas as pd
from buildstock_query.schema.query_params import UtilityTSQuery
from buildstock_query.schema.utilities import AnyColType, AnyTableType
import typing
from typing import Literal
from pydantic import Field
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BuildStockUtility:
    def __init__(self, buildstock_query: 'main.BuildStockQuery',
                 eia_mapping_year: int = 2018, eia_mapping_version: int = 1) -> None:
        ...

    def _aggregate_ts_by_map(self,
                             map_table_name: str,
                             baseline_column_name: str,
                             map_column_name: str,
                             id_column: str,
                             id_list: Sequence[Any],
                             params: UtilityTSQuery):
        ...

    def get_eiaid_map(self) -> tuple[str, str, str]:
        ...

    @typing.overload
    def aggregate_ts_by_eiaid(self, *,
                              enduses: Sequence[AnyColType],
                              eiaid_list: Sequence[str],
                              get_query_only: Literal[True],
                              group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list),
                              upgrade_id: Union[int, str] = '0',
                              sort: bool = True,
                              join_list: Sequence[tuple[AnyTableType, AnyColType,
                                                        AnyColType]] = Field(default_factory=list),
                              weights: Sequence[Union[str, tuple]] = [],
                              restrict: Sequence[tuple[str, Union[str, int, Sequence[int], Sequence[str]]]] = [],
                              split_enduses: bool = False,
                              collapse_ts: bool = False,
                              timestamp_grouping_func: Optional[Literal["month", "day", "hour"]] = None,
                              query_group_size: int = 20,
                              limit: Optional[int] = None,
                              ) -> str:
        ...

    @typing.overload
    def aggregate_ts_by_eiaid(self, *,
                              enduses: Sequence[AnyColType],
                              eiaid_list: Sequence[str],
                              group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list),
                              upgrade_id: Union[int, str] = '0',
                              sort: bool = True,
                              join_list: Sequence[tuple[AnyTableType, AnyColType,
                                                        AnyColType]] = Field(default_factory=list),
                              weights: Sequence[Union[str, tuple]] = [],
                              restrict: Sequence[tuple[str, Union[str, int, Sequence[int], Sequence[str]]]] = [],
                              split_enduses: bool = False,
                              collapse_ts: bool = False,
                              timestamp_grouping_func: Optional[Literal["month", "day", "hour"]] = None,
                              get_query_only: Literal[False] = False,
                              query_group_size: int = 20,
                              limit: Optional[int] = None,
                              ) -> pd.DataFrame:
        ...

    @typing.overload
    def aggregate_ts_by_eiaid(self, *,
                              enduses: Sequence[AnyColType],
                              eiaid_list: Sequence[str],
                              get_query_only: bool,
                              group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list),
                              upgrade_id: Union[int, str] = '0',
                              sort: bool = True,
                              join_list: Sequence[tuple[AnyTableType, AnyColType,
                                                        AnyColType]] = Field(default_factory=list),
                              weights: Sequence[Union[str, tuple]] = [],
                              restrict: Sequence[tuple[str, Union[str, int, Sequence[int], Sequence[str]]]] = [],
                              split_enduses: bool = False,
                              collapse_ts: bool = False,
                              timestamp_grouping_func: Optional[Literal["month", "day", "hour"]] = None,
                              query_group_size: int = 20,
                              limit: Optional[int] = None,
                              ) -> Union[str, pd.DataFrame]:
        """
        Aggregates the timeseries result, grouping by utilities.
        Args:
            enduses: The list of enduses to aggregate.
            eiaid_list: The list of utility ids (EIAID) assigned by EIA.
            group_by: Additional columns to group the aggregation by
            upgrade_id: The upgrade id to filter the results to
            sort: If set to true, sorts the results by the eiaid
            join_list: The list of joins to be performed on the query. Each join should be specified as a list of
                          form [(table_name, join_condition, join_type)]
            weights: The list of weights to be applied to the enduses. Each weight should be specified as a list of
                        form ['weight_column' or ('weight_column', 'weight_table')]
            restrict: The list of restrictions to be applied to the query. Each restriction should be specified as a
                            list of form [('column_name', restric_list / restric_value)]
            split_endues: If true, query each enduses separately to spread load on Athena.
            collapse_ts: If true, collapse the timeseries (i.e. sum them up) into a single row.
            get_query_only: If set to true, returns the list of queries to run instead of the result.
            timestamp_grouping_func: The function to be used to group the timeseries. If None, the timeseries are
            query_group_size: The number of eiaids to be grouped together when running athena queries. This should be
                              used as large as possible that doesn't result in query timeout.
            limit: The number of rows to limit the query to.

        Returns:
            Pandas dataframe with the aggregated timeseries and the requested enduses grouped by utilities
        """
        ...

    @typing.overload
    def aggregate_ts_by_eiaid(self, *,
                              params: UtilityTSQuery,
                              ) -> Union[str, pd.DataFrame]:
        ...

    @typing.overload
    def aggregate_unit_counts_by_eiaid(self, *, eiaid_list: Optional[Sequence[str]] = None,
                                       get_query_only: Literal[True],
                                       group_by: Sequence[Union[AnyColType,
                                                                tuple[str, str]]] = Field(default_factory=list),
                                       ) -> str:
        ...

    @typing.overload
    def aggregate_unit_counts_by_eiaid(self, *, eiaid_list: Optional[Sequence[str]] = None,
                                       get_query_only: Literal[False] = False,
                                       group_by: Sequence[Union[AnyColType,
                                                                tuple[str, str]]] = Field(default_factory=list),
                                       ) -> pd.DataFrame:
        ...

    @typing.overload
    def aggregate_unit_counts_by_eiaid(self, *, eiaid_list: Optional[Sequence[str]] = None,
                                       get_query_only: bool = False,
                                       group_by: Sequence[Union[AnyColType,
                                                                tuple[str, str]]] = Field(default_factory=list),
                                       ) -> Union[pd.DataFrame, str]:
        ...

    @typing.overload
    def aggregate_annual_by_eiaid(self, enduses: Sequence[AnyColType],
                                  get_query_only: Literal[True],
                                  group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list),
                                  get_nonzero_count: bool = False,
                                  ) -> str:
        ...

    @typing.overload
    def aggregate_annual_by_eiaid(self, enduses: Sequence[AnyColType],
                                  group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list),
                                  get_nonzero_count: bool = False,
                                  get_query_only: Literal[False] = False) -> pd.DataFrame:
        ...

    @typing.overload
    def aggregate_annual_by_eiaid(self, enduses: Sequence[AnyColType],
                                  get_query_only: bool,
                                  group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list),
                                  get_nonzero_count: bool = False,
                                  ) -> Union[str, pd.DataFrame]:
        ...

    @typing.overload
    def get_filtered_results_csv_by_eiaid(self, eiaids: Sequence[str],
                                          get_query_only: Literal[True]) -> str:
        ...

    @typing.overload
    def get_filtered_results_csv_by_eiaid(self, eiaids: Sequence[str],
                                          get_query_only: Literal[False] = False) -> pd.DataFrame:
        ...

    @typing.overload
    def get_filtered_results_csv_by_eiaid(
            self, eiaids: Sequence[str], get_query_only: bool) -> Union[str, pd.DataFrame]:
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
        ...

    def get_eiaids(self,
                   restrict: Sequence[tuple[AnyColType,
                                            Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list)
                   ) -> list[str]:
        """
        Returns the list of building
        Args:
            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            mapping_version: Version of eiaid mapping to use. After the spatial refactor upgrade, version two
                             should be used
        Returns:
            Pandas dataframe consisting of the eiaids belonging to the provided list of locations.
        """
        ...

    @typing.overload
    def get_buildings_by_eiaids(self, eiaids: Sequence[str], get_query_only: Literal[True]) -> str:
        ...

    @typing.overload
    def get_buildings_by_eiaids(self, eiaids: Sequence[str], get_query_only: Literal[False] = False) -> pd.DataFrame:
        ...

    @typing.overload
    def get_buildings_by_eiaids(self, eiaids: Sequence[str], get_query_only: bool) -> Union[str, pd.DataFrame]:
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
        ...

    @typing.overload
    def get_locations_by_eiaids(self, eiaids: Sequence[str], get_query_only: Literal[True]) -> str:
        ...

    @typing.overload
    def get_locations_by_eiaids(self, eiaids: Sequence[str], get_query_only: Literal[False] = False) -> list[str]:
        ...

    @typing.overload
    def get_locations_by_eiaids(self, eiaids: Sequence[str], get_query_only: bool) -> Union[str, list[str]]:
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
        ...

    @typing.overload
    def calculate_tou_bill(self, *,
                           rate_map: Union[tuple[str, str], dict[tuple[int, int, int], float]],
                           get_query_only: Literal[True],
                           meter_col: Optional[Union[AnyColType, Sequence[AnyColType]]] = None,
                           group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list),
                           upgrade_id: Union[int, str] = '0',
                           sort: bool = True,
                           join_list: Sequence[tuple[AnyTableType, AnyColType,
                                                     AnyColType]] = Field(default_factory=list),
                           weights: Sequence[Union[str, tuple]] = Field(default_factory=list),
                           restrict: Sequence[
                               tuple[AnyColType, Union[str, int,
                                                       Sequence[Union[int, str]]]]] = Field(default_factory=list),
                           collapse_ts: bool = False,
                           timestamp_grouping_func: Optional[Literal["month", "day", "hour"]] = "month",
                           limit: Optional[int] = None,
                           ) -> str:
        ...

    @typing.overload
    def calculate_tou_bill(self, *,
                           rate_map: Union[tuple[str, str], dict[tuple[int, int, int], float]],
                           meter_col: Optional[Union[AnyColType, Sequence[AnyColType]]] = None,
                           group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list),
                           upgrade_id: Union[int, str] = '0',
                           sort: bool = True,
                           join_list: Sequence[tuple[AnyTableType, AnyColType,
                                                     AnyColType]] = Field(default_factory=list),
                           weights: Sequence[Union[str, tuple]] = Field(default_factory=list),
                           restrict: Sequence[
                               tuple[AnyColType, Union[str, int,
                                                       Sequence[Union[int, str]]]]] = Field(default_factory=list),
                           collapse_ts: bool = False,
                           timestamp_grouping_func: Optional[Literal["month", "day", "hour"]] = "month",
                           limit: Optional[int] = None,
                           get_query_only: Literal[False] = False
                           ) -> pd.DataFrame:
        ...

    @typing.overload
    def calculate_tou_bill(self, *,
                           rate_map: Union[tuple[str, str], dict[tuple[int, int, int], float]],
                           get_query_only: bool,
                           meter_col: Optional[Union[AnyColType, Sequence[AnyColType]]] = None,
                           group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list),
                           upgrade_id: Union[int, str] = '0',
                           sort: bool = True,
                           join_list: Sequence[tuple[AnyTableType, AnyColType,
                                                     AnyColType]] = Field(default_factory=list),
                           weights: Sequence[Union[str, tuple]] = Field(default_factory=list),
                           restrict: Sequence[
                               tuple[AnyColType, Union[str, int,
                                                       Sequence[Union[int, str]]]]] = Field(default_factory=list),
                           collapse_ts: bool = False,
                           timestamp_grouping_func: Optional[Literal["month", "day", "hour"]] = "month",
                           limit: Optional[int] = None,
                           ) -> Union[str, pd.DataFrame]:
        """Calculates the dollar cost of electricity for a given time of use rate schedule. Currently, makes use of the
        the net_electricity_kwh enduse to make the calculation and assumes that the rate schedule is for a single
        symetrical rate. The rate schedule is specified by the rate_map argument.

        Args:
            rate_map (Union[tuple[str, str], dict[tuple[int, int, int], float]]): The rate schedule to use. This can be
                specified as a dictionary or as a tuple of strings. If a dictionary is used, the keys should be tuples
                of the form (month, is_weekend_day, hour) and the values should be the rate in cents per kilowatt hour.
                If a tuple of strings is used, the first string should be the path of the csv containing the rate
                schedule for weekday and the second string should be the path of the csv containing rate for weekend.
                The csv files need to have 12 rows excluding the header and 24 columns excluding the first column.
                The first column header should be "month" and should have values from 1 to 12. The other column headers
                are numbered 0 through 23 corresponding to hour of the day. Each row should have a the electricity cost
                in cents per kilowatt hour for the corresponding month and hour.
                Example CSV file below:
                month,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
                1, 12, 12, 12, 12, 12, 12, 12, 12, 12,  12,	12,	12,	12,	15,	15,	19,	19,	19,	19,	12,	12,	12,	12,	12
                2, 12, 12, 12, 12, 12, 12, 12, 12, 12,  12,	12,	12,	12,	15,	15,	19,	19,	19,	19,	12,	12,	12,	12,	12
                ...
                12, 12, 12, 12, 12, 12, 12, 12, 12, 12,  12, 12, 12, 12, 15, 15, 19, 19, 19, 19, 12, 12, 12, 12, 12

            meter_col (Optional[Union[AnyColType, Sequence[AnyColType]]], optional): The enduse column to use for the
                the bill calculation. If not provided, use the sum of total_electricity_kwh and pv_electricity_kwh. If
                a tuple of columns is provided, the sum of the columns is used for bill calculation.

            group_by (Sequence[Union[AnyColType, tuple[str, str]]], optional): List of columns to group the results by.
                Can be specified as a list of column names

            upgrade_id (Union[int, str], optional): The upgrade run to query. Defaults to '0' corresponding to baseline.

            sort (bool, optional): Whether to sort the result. Defaults to True. Can be made false for faster queries.

            join_list (Sequence[tuple[AnyTableType, AnyColType, AnyColType]], optional): Any additional table to
                join to. Advanced feature, typically not used.

            weights (Sequence[Union[str, tuple]], optional): Any additional weights to apply. Advanced feature,
                typically not used.

            restrict (Sequence[ tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]], optional):
                The list of where condition to restrict the results to. It should be specified as a list of tuple.
                    Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
                The column can be directly specified instead of using a string. For example:
                    `restrict = [(my_run.ts_bldgid_column, (1, 2, 3))]`
                In the above, my_run is a BuildStockQuery object and ts_bldgid_column is a building_id column of the 
                timeseries table.

            collapse_ts (bool, optional): Whether to converge all timestamps to get annual result. Defaults to False.

            timestamp_grouping_func (Literal["month", "day", "hour"]), optional): The function to use to
                group the timestamps. Defaults to "month" to get monthly bill. Can be set to None to get the bill
                for each timestamp, or to 'hour' to get hourly bill. Use `collapse_ts = True` to get annual bill.

            limit (Optional[int], optional): To limit result to certain number of rows. Defaults to None. Useful for
                debugging.

            get_query_only (bool, optional): If set to true, returns query string instead of running the query.
                Defaults to False.
        """
        ...
