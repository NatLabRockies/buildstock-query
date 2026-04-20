import pandas as pd
from typing import Sequence, Union
from buildstock_query.schema.query_params import SavingsQuery
from buildstock_query.schema.utilities import AnyColType, AnyTableType
import buildstock_query.main as main
from typing import Optional
from pydantic import Field
from typing import Literal
import typing
from typing_extensions import deprecated


class BuildStockSavings:
    """Class for doing savings query (both timeseries and annual).
    """

    def __init__(self, buildstock_query: 'main.BuildStockQuery') -> None:
        ...

    def _validate_partition_by(self, partition_by: list[str]):
        ...

    def __get_timeseries_bs_up_table(self,
                                     enduses: Sequence[AnyColType],
                                     upgrade_id: Union[int, str],
                                     applied_only: bool,
                                     restrict:
                                     Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]
                                              ] = Field(default_factory=list),
                                     ts_group_by: Sequence[Union[AnyColType, tuple[str, str]]
                                                           ] = Field(default_factory=list)):
        ...

    def __get_annual_bs_up_table(self, upgrade_id: Union[int, str], applied_only: bool) -> ...:
        ...

    @typing.overload
    @deprecated("Please use my_run.query with include_savings=True.")
    def savings_shape(
        self, *,
        get_query_only: Literal[True],
        upgrade_id: Union[int, str],
        enduses:  Sequence[AnyColType],
        group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list),
        annual_only: bool = True,
        sort: bool = True,
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = Field(default_factory=list),
        weights: Sequence[Union[str, tuple]] = Field(default_factory=list),
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]
                           ] = Field(default_factory=list),
        applied_only: bool = False,
        get_quartiles: bool = False,
        unload_to: str = '',
        partition_by: Optional[Sequence[str]] = None,
        collapse_ts: bool = False,
        timestamp_grouping_func: Optional[str] = None,
    ) -> str:
        ...

    @typing.overload
    @deprecated("Please use my_run.query with include_savings=True.")
    def savings_shape(
        self, *,
        upgrade_id: Union[int, str],
        get_query_only: Literal[False] = False,
        enduses:  Sequence[AnyColType],
        group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list),
        annual_only: bool = True,
        sort: bool = True,
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = Field(default_factory=list),
        weights: Sequence[Union[str, tuple]] = Field(default_factory=list),
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]
                           ] = Field(default_factory=list),
        applied_only: bool = False,
        get_quartiles: bool = False,
        unload_to: str = '',
        partition_by: Optional[Sequence[str]] = None,
        collapse_ts: bool = False,
        timestamp_grouping_func: Optional[str] = None,
    ) -> pd.DataFrame:
        ...

    @typing.overload
    @deprecated("Please use my_run.query with include_savings=True.")
    def savings_shape(
        self, *,
        get_query_only: bool,
        upgrade_id: Union[int, str],
        enduses:  Sequence[AnyColType],
        group_by: Sequence[Union[AnyColType, tuple[str, str]]] = Field(default_factory=list),
        annual_only: bool = True,
        sort: bool = True,
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = Field(default_factory=list),
        weights: Sequence[Union[str, tuple]] = Field(default_factory=list),
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]
                           ] = Field(default_factory=list),
        applied_only: bool = False,
        get_quartiles: bool = False,
        unload_to: str = '',
        partition_by: Optional[Sequence[str]] = None,
        collapse_ts: bool = False,
        timestamp_grouping_func: Optional[str] = None,
    ) -> Union[str, pd.DataFrame]:
        """Calculate savings shape for an upgrade
        Args:
            upgrade_id: id of the upgrade scenario from the ResStock analysis
            enduses: Enduses to query, defaults to ['fuel_use__electricity__total']
            group_by: Building characteristics columns to group by, defaults to []
            annual_only: If true, calculates only the annual savings using baseline and upgrades table
            sort: Whether the result should be sorted. Sorting takes extra time.
            join_list: Additional table to join to baseline table to perform operation. All the inputs (`enduses`,
                  `group_by` etc) can use columns from these additional tables. It should be specified as a list of
                  tuples.
                  Example: `[(new_table_name, baseline_column_name, new_column_name), ...]`
                        where baseline_column_name and new_column_name are the columns on which the new_table
                        should be joined to baseline table.
            applied_only: Calculate savings shape based on only buildings to which the upgrade applied
            weights: The additional columns to use as weight. The "build_existing_model.sample_weight" is already used.
                     It is specified as either list of string or list of tuples. When only string is used, the string
                     is the column name, when tuple is passed, the second element is the table name.

            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`

            get_query_only: Skips submitting the query to Athena and just returns the query string. Useful for batch
                            submitting multiple queries or debugging
            get_quartiles: If true, return the following quartiles in addition to the sum for each enduses:
                           [0, 0.02, .25, .5, .75, .98, 1]. The 0% quartile is the minimum and the 100% quartile
                           is the maximum.
            unload_to: Writes the ouput of the query to this location in s3. Consider using run_async = True with this
                       to unload multiple queries simulataneuosly
            partition_by: List of columns to partition when writing to s3. To be used with unload_to.
            collapse_ts: Only used when annual_only=False. When collapse_ts=True, the timeseries values are summed into
                         a single annual value. Useful for quality checking and comparing with annual values.
            timestamp_grouping_func: One of 'hour', 'day' or 'month' or None. If provided, perform timeseries
                        aggregation of specified granularity.
         Returns:
                if get_query_only is True, returns the query_string, otherwise returns a pandas dataframe
        """
        ...

    @typing.overload
    @deprecated("Please use my_run.query with include_savings=True.")
    def savings_shape(self, *, params: SavingsQuery) -> Union[str, pd.DataFrame]:
        ...
