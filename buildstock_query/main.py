import sqlalchemy as sa
from sqlalchemy.sql import func as safunc
from sqlalchemy.sql import sqltypes
from typing import Union
from collections.abc import Sequence
import logging
import re
from buildstock_query.tools import UpgradesAnalyzer
from buildstock_query.query_core import QueryCore
import pandas as pd
from pydantic import Field
from typing import Optional, Literal
from typing_extensions import assert_never
import typing
from datetime import datetime
from buildstock_query.schema.run_params import BSQParams
from buildstock_query.schema.utilities import DBColType, SALabel, AnyColType, AnyTableType
from buildstock_query.schema.utilities import validate_arguments
from buildstock_query.schema.utilities import MappedColumn
from buildstock_query.schema.query_params import Query

import os
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FUELS = ["electricity", "natural_gas", "propane", "fuel_oil", "coal", "wood_cord", "wood_pellets"]


@dataclass
class SimInfo:
    year: int
    interval: int
    offset: int
    unit: str


class BuildStockQuery(QueryCore):
    @validate_arguments
    def __init__(
        self,
        workgroup: str,
        db_name: str,
        table_name: Union[str, tuple[str, Optional[str], Optional[str]]],
        db_schema: Optional[str] = None,
        buildstock_type: Literal["resstock", "comstock"] = "resstock",
        sample_weight_override: Optional[Union[int, float]] = None,
        region_name: str = "us-west-2",
        execution_history: Optional[str] = None,
        skip_reports: bool = False,
        athena_query_reuse: bool = True,
        **kwargs,
    ) -> None:
        """A class to run Athena queries for BuildStock runs and download results as pandas DataFrame.

        Args:
            workgroup (str): The workgroup for athena. The cost will be charged based on workgroup.
            db_name (str): The athena database name
            buildstock_type (str, optional): 'resstock' or 'comstock' runs. Defaults to 'resstock'
            table_name (str or Union[str, tuple[str, Optional[str], Optional[str]]]): If a single string is provided,
            say, 'mfm_run', then it must correspond to tables in athena named mfm_run_baseline and optionally
            mfm_run_timeseries and mf_run_upgrades. Or, tuple of three elements can be provided for the table names
            for baseline, timeseries and upgrade. Timeseries and upgrade can be None if no such table exist.
            db_schema (str, optional): The database structure in Athena is different between ResStock and ComStock run.
                It is also different between the version in OEDI and default version from BuildStockBatch. This argument
                controls the assumed schema. Allowed values are 'resstock_default', 'resstock_oedi', 'comstock_default'
                and 'comstock_oedi'. Defaults to 'resstock_default' for resstock and 'comstock_default' for comstock.
            sample_weight_override (str, optional): Specify a custom sample_weight. Otherwise, the default is 1 for
                ComStock and uses sample_weight in the run for ResStock.
            region_name (str, optional): the AWS region where the database exists. Defaults to 'us-west-2'.
            execution_history (str, optional): A temporary file to record which execution is run by the user,
                to help stop them. Will use .execution_history if not supplied. Generally, not required to supply a
                custom filename.
            skip_reports (bool, optional): If true, skips report printing during initialization. If False (default),
                prints report from `buildstock_query.report_query.BuildStockReport.get_success_report`.
            athena_query_reuse (bool, optional): When true, Athena will make use of its built-in 7 day query cache.
                When false, it will not. Defaults to True. One use case to set this to False is when you have modified
                the underlying s3 data or glue schema and want to make sure you are not using the cached results.
            kargs: Any other extra keyword argument supported by the QueryCore can be supplied here
        """
        db_schema = db_schema or f"{buildstock_type}_default"
        self.params = BSQParams(
            workgroup=workgroup,
            db_name=db_name,
            buildstock_type=buildstock_type,
            table_name=table_name,
            db_schema=db_schema,
            sample_weight_override=sample_weight_override,
            region_name=region_name,
            execution_history=execution_history,
            athena_query_reuse=athena_query_reuse,
        )
        self._run_params = self.params.get_run_params()
        super(BuildStockQuery, self).__init__(params=self._run_params)
        from buildstock_query.report_query import BuildStockReport
        from buildstock_query.aggregate_query import BuildStockAggregate
        from buildstock_query.savings_query import BuildStockSavings
        from buildstock_query.utility_query import BuildStockUtility
        #: `buildstock_query.report_query.BuildStockReport` object to perform report queries
        self.report: BuildStockReport = BuildStockReport(self)
        #: `buildstock_query.aggregate_query.BuildStockAggregate` object to perform aggregate queries
        self.agg: BuildStockAggregate = BuildStockAggregate(self)
        #: `buildstock_query.savings_query.BuildStockSavings` object to perform savings queries
        self.savings = BuildStockSavings(self)
        #: `buildstock_query.utility_query.BuildStockUtility` object to perform utility queries
        self.utility = BuildStockUtility(self)

        self._char_prefix = self.db_schema.column_prefix.characteristics
        self._out_prefix = self.db_schema.column_prefix.output

        if not skip_reports:
            logger.info("Getting Success counts...")
            print(self.report.get_success_report())
            if self.ts_table is not None:
                self.report.check_ts_bs_integrity()
            self.save_cache()

    def get_buildstock_df(self) -> pd.DataFrame:
        """Returns the building characteristics data by querying Athena tables using the same format as that produced
        by the sampler and written as buildstock.csv. It only includes buildings with successful simulation.
        Returns:
            pd.DataFrame: The buildstock.csv dataframe.
        """
        results_df = self.get_results_csv_full()
        results_df = results_df[
            results_df[self.db_schema.column_names.completed_status].astype(str).str.lower()
            == self.db_schema.completion_values.success.lower()
        ]
        buildstock_cols = [c for c in results_df.columns if c.startswith(self._char_prefix)]
        buildstock_df = results_df[buildstock_cols]
        buildstock_cols = [
            "".join(c.split(".")[1:]).replace("_", " ")
            for c in buildstock_df.columns
            if c.startswith(self._char_prefix)
        ]
        buildstock_df.columns = buildstock_cols
        return buildstock_df

    @validate_arguments
    def get_upgrades_analyzer(
        self,
        *,
        opt_sat_file: str,
        yaml_file: Optional[str] = None,
        filter_yaml_file: Optional[str] = None,
        upgrade_names: Optional[dict[int, str]] = None,
    ) -> UpgradesAnalyzer:
        """
        Initialize the analyzer instance.
        Args:
            opt_sat_file (str): The path to the option saturation file.
            yaml_file (str): The path to the yaml file.
            filter_yaml_file (str): The path to the filter yaml file.
            upgrade_names (dict[int, str]): A dictionary of upgrade number to upgrade name. This
                needs to be provided if only the filter_yaml_file is provided.
        """

        buildstock_df = self.get_buildstock_df()
        if yaml_file is None and upgrade_names is None:
            upgrade_names = self.get_upgrade_names()
        ua = UpgradesAnalyzer(
            buildstock=buildstock_df,
            yaml_file=yaml_file,
            opt_sat_file=opt_sat_file,
            filter_yaml_file=filter_yaml_file,
            upgrade_names=upgrade_names,
        )
        return ua

    @typing.overload
    def get_upgrade_names(self, get_query_only: Literal[False] = False) -> dict: ...

    @typing.overload
    def get_upgrade_names(self, get_query_only: Literal[True]) -> str: ...

    @validate_arguments
    def get_upgrade_names(self, get_query_only: bool = False) -> Union[str, dict]:
        if self.up_table is None:
            raise ValueError("This run has no upgrades")
        upgrade_table = self.up_table
        query = f"""
            Select cast(upgrade as integer) as upgrade, arbitrary("apply_upgrade.upgrade_name") as upgrade_name
            from {upgrade_table}
            group by 1 order by 1
        """
        if get_query_only:
            return query
        up_name_dict = self.execute(query).set_index("upgrade").to_dict()["upgrade_name"]
        return up_name_dict

    @typing.overload
    def _get_rows_per_building(self, get_query_only: Literal[False] = False) -> int: ...

    @typing.overload
    def _get_rows_per_building(self, get_query_only: Literal[True]) -> str: ...

    @validate_arguments
    def _get_rows_per_building(self, get_query_only: bool = False) -> Union[int, str]:
        select_cols = []
        if self.up_table is not None and self.ts_table is not None:
            select_cols.append(self.ts_table.c["upgrade"])
        select_cols.extend((self.ts_bldgid_column, safunc.count().label("row_count")))
        ts_query = sa.select(*select_cols)
        if self.up_table is not None:
            ts_query = ts_query.group_by(sa.text("1"), sa.text("2"))
        else:
            ts_query = ts_query.group_by(sa.text("1"))

        if get_query_only:
            return self._compile(ts_query)
        df = self.execute(ts_query)
        if (df["row_count"] == df["row_count"][0]).all():  # verify all buildings got same number of rows
            return df["row_count"][0]
        else:
            raise ValueError("Not all buildings have same number of rows.")

    @validate_arguments
    def get_distinct_vals(
        self, column: str, table_name: Optional[str], get_query_only: bool = False
    ) -> Union[str, pd.Series]:
        """
            Find distinct vals.
        Args:
            column (str): The column in the table for which distinct vals is needed.
            table_name (str, optional): The table in athena. Defaults to baseline table.
            get_query_only (bool, optional): If true, only returns the SQL query. Defaults to False.

        Returns:
            pd.Series: The distinct vals.
        """
        table_name = self.bs_table.name if table_name is None else table_name
        tbl = self._get_table(table_name)
        query = sa.select(tbl.c[column]).distinct()
        if get_query_only:
            return self._compile(query)

        r = self.execute(query, run_async=False)
        return r[column]

    @validate_arguments
    def get_distinct_count(
        self, column: str, table_name: Optional[str] = None, get_query_only: bool = False
    ) -> Union[pd.DataFrame, str]:
        """
            Find distinct counts.
        Args:
            column (str): The column in the table for which distinct counts is needed.
            table_name (str, optional): The table in athena. Defaults to baseline table.
            get_query_only (bool, optional): If true, only returns the SQL query. Defaults to False.

        Returns:
            pd.Series: The distinct counts.
        """
        tbl = self.bs_table if table_name is None else self._get_table(table_name)
        query = sa.select(
            tbl.c[column], safunc.sum(1).label("sample_count"), safunc.sum(self.sample_wt).label("weighted_count")
        )
        query = query.group_by(tbl.c[column]).order_by(tbl.c[column])
        if get_query_only:
            return self._compile(query)

        r = self.execute(query, run_async=False)
        return r

    @typing.overload
    def get_results_csv(
        self,
        *,
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
        get_query_only: Literal[False] = False,
    ) -> pd.DataFrame: ...

    @typing.overload
    def get_results_csv(
        self,
        *,
        get_query_only: Literal[True],
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
    ) -> str: ...

    @typing.overload
    def get_results_csv(
        self,
        *,
        get_query_only: bool,
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
    ) -> Union[str, pd.DataFrame]: ...

    @validate_arguments
    def get_results_csv(
        self,
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
        get_query_only: bool = False,
    ) -> Union[pd.DataFrame, str]:
        """
        Returns the results_csv table for the BuildStock run
        Args:
            restrict (list[Tuple[str, Union[List, str, int]]], optional): The list of where condition to restrict the
                results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            get_query_only (bool): If set to true, returns the list of queries to run instead of the result.

        Returns:
            Pandas dataframe that is a subset of the results csv, that belongs to provided list of utilities
        """
        restrict = list(restrict) if restrict else []
        query = sa.select("*").select_from(self.bs_table)
        query = self._add_restrict(query, restrict, bs_only=True)
        compiled_query = self._compile(query)
        if get_query_only:
            return compiled_query
        self._session_queries.add(compiled_query)
        if compiled_query in self._query_cache:
            return self._query_cache[compiled_query].copy().set_index(self.bs_bldgid_column.name)
        logger.info("Making results_csv query ...")
        result = self.execute(query)
        return result.set_index(self.bs_bldgid_column.name)

    def _download_results_csv(self) -> str:
        """Downloads the results csv from s3 and returns the path to the downloaded file.
        Returns:
            str: The path to the downloaded file.
        """
        local_copy_path = self.cache_folder / f"{self.db_name}_{self.bs_table.name}.parquet"
        if os.path.exists(local_copy_path):
            return local_copy_path

        if isinstance(self.table_name, str):
            db_table_name = f"{self.table_name}{self.db_schema.table_suffix.baseline}"
        else:
            db_table_name = self.table_name[0]
        baseline_path = self._aws_glue.get_table(DatabaseName=self.db_name, Name=db_table_name)["Table"][
            "StorageDescriptor"
        ]["Location"]
        bucket = baseline_path.split("/")[2]
        key = "/".join(baseline_path.split("/")[3:])
        s3_data = self._aws_s3.list_objects(Bucket=bucket, Prefix=key)

        if "Contents" not in s3_data:
            raise ValueError(f"Results parquet not found in s3 at {baseline_path}")
        matching_files = [
            path["Key"]
            for path in s3_data["Contents"]
            if "up00.parquet" in path["Key"] or "baseline.parquet" in path["Key"]
        ]

        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple results parquet found in s3 at {baseline_path} for baseline."
                f"These files matched: {matching_files}"
            )
        if len(matching_files) == 0:
            raise ValueError(
                f"No results parquet found in s3 at {baseline_path} for baseline."
                f"Here are the files: {[content[0]['Key'] for content in s3_data['Contents']]}"
            )

        self._aws_s3.download_file(bucket, matching_files[0], local_copy_path)
        return local_copy_path

    def get_results_csv_full(self) -> pd.DataFrame:
        """Returns the full results csv table. This is the same as get_results_csv without any restrictions. It uses
        the stored parquet files in s3 to download the results which is faster than querying athena.
        Returns:
            pd.DataFrame: The full results csv.
        """
        local_copy_path = self._download_results_csv()
        df = pd.read_parquet(local_copy_path)
        if df.index.name != self.bs_bldgid_column.name:
            df = df.set_index(self.bs_bldgid_column.name)
        return df

    @typing.overload
    def get_upgrades_csv(
        self,
        *,
        get_query_only: Literal[False] = False,
        upgrade_id: Union[int, str] = "0",
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
    ) -> pd.DataFrame: ...

    @typing.overload
    def get_upgrades_csv(
        self,
        *,
        get_query_only: Literal[True],
        upgrade_id: Union[int, str] = "0",
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
    ) -> str: ...

    @typing.overload
    def get_upgrades_csv(
        self,
        *,
        get_query_only: bool,
        upgrade_id: Union[int, str] = "0",
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
    ) -> Union[pd.DataFrame, str]: ...

    @validate_arguments
    def get_upgrades_csv(
        self,
        *,
        upgrade_id: Union[str, int] = "0",
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
        get_query_only: bool = False,
    ) -> Union[pd.DataFrame, str]:
        """
        Returns the results_csv table for the BuildStock run for an upgrade.
        Args:
            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            get_query_only: If set to true, returns the list of queries to run instead of the result.

        Returns:
            Pandas dataframe that is a subset of the results csv, that belongs to provided list of utilities
        """
        restrict = list(restrict) if restrict else []
        query = sa.select("*").select_from(self.up_table)
        if upgrade_id:
            if self.up_table is None:
                raise ValueError("This run has no upgrades")
            query = query.where(self.up_table.c["upgrade"] == str(upgrade_id))

        query = self._add_restrict(query, restrict, bs_only=True)
        compiled_query = self._compile(query)
        if get_query_only:
            return compiled_query
        self._session_queries.add(compiled_query)
        if compiled_query in self._query_cache:
            return self._query_cache[compiled_query].copy().set_index(self.bs_bldgid_column.name)
        logger.info("Making results_csv query for upgrade ...")
        return self.execute(query).set_index(self.bs_bldgid_column.name)

    def _download_upgrades_csv(self, upgrade_id: Union[int, str]) -> str:
        """Downloads the upgrades csv from s3 and returns the path to the downloaded file."""
        if self.up_table is None:
            raise ValueError("This run has no upgrades")

        available_upgrades = list(self.get_available_upgrades())
        available_upgrades.remove("0")
        if isinstance(upgrade_id, int):
            upgrade_id = f"{upgrade_id:02}"

        if str(upgrade_id) not in available_upgrades:
            raise ValueError(f"Upgrade {upgrade_id} not found")

        local_copy_path = self.cache_folder / f"{self.db_name}_{self.up_table.name}_{upgrade_id}.parquet"
        if os.path.exists(local_copy_path):
            return local_copy_path

        if isinstance(self.table_name, str):
            db_table_name = f"{self.table_name}{self.db_schema.table_suffix.upgrades}"
        else:
            db_table_name = self.table_name[2]
        upgrades_path = self._aws_glue.get_table(DatabaseName=self.db_name, Name=db_table_name)["Table"][
            "StorageDescriptor"
        ]["Location"]
        bucket = upgrades_path.split("/")[2]
        key = "/".join(upgrades_path.split("/")[3:])
        s3_data = self._aws_s3.list_objects(Bucket=bucket, Prefix=key)

        if "Contents" not in s3_data:
            raise ValueError(f"Results parquet not found in s3 at {upgrades_path}")

        # out of the contents find the key with name matching the pattern results_up{upgrade_id}.parquet
        def is_match(upgrade_id, key):
            try:
                upgrade_id = int(upgrade_id)
                alternative_id = f"{upgrade_id:02}"
            except ValueError:
                alternative_id = str(upgrade_id)
            for prefix in ["up", "upgrade"]:
                if f"{prefix}{upgrade_id}.parquet" in key or f"{prefix}{alternative_id}.parquet" in key:
                    return True
            return False

        matching_files = [path["Key"] for path in s3_data["Contents"] if is_match(upgrade_id, path["Key"])]

        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple results parquet found in s3 at {upgrades_path} for upgrade {upgrade_id}."
                f"These files matched: {matching_files}"
            )
        if len(matching_files) == 0:
            raise ValueError(
                f"No results parquet found in s3 at {upgrades_path} for upgrade {upgrade_id}."
                f"Here are the files: {[content[0]['Key'] for content in s3_data['Contents']]}"
            )

        self._aws_s3.download_file(bucket, matching_files[0], local_copy_path)
        return local_copy_path

    def get_upgrades_csv_full(self, upgrade_id: Union[int, str]) -> pd.DataFrame:
        """Returns the full results csv table for upgrades. This is the same as get_upgrades_csv without any
        restrictions. It uses the stored parquet files in s3 to download the results which is faster than querying
        athena.
        """
        local_copy_path = self._download_upgrades_csv(upgrade_id)
        df = pd.read_parquet(local_copy_path)
        if df.index.name != self.up_bldgid_column.name:
            df = df.set_index(self.up_bldgid_column.name)
        if "upgrade" not in df.columns:
            df.insert(0, "upgrade", upgrade_id)
        return df

    @typing.overload
    def get_building_ids(
        self,
        *,
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
        get_query_only: Literal[False] = False,
    ) -> pd.DataFrame: ...

    @typing.overload
    def get_building_ids(
        self,
        *,
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
        get_query_only: Literal[True],
    ) -> str: ...

    @typing.overload
    def get_building_ids(
        self,
        *,
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
        get_query_only: bool,
    ) -> Union[pd.DataFrame, str]: ...

    @validate_arguments
    def get_building_ids(
        self,
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
        get_query_only: bool = False,
    ) -> Union[str, pd.DataFrame]:
        """
        Returns the list of buildings based on the restrict list
        Args:
            restrict (list[Tuple[str, List]], optional): The list of where condition to restrict the results to. It
                    should be specified as a list of tuple.
                    Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            get_query_only (bool): If set to true, returns the query string instead of the result. Default is False.

        Returns:
            Pandas dataframe consisting of the building ids belonging to the provided list of locations.

        """
        restrict = list(restrict) if restrict else []
        query = sa.select(self.bs_bldgid_column)
        query = self._add_restrict(query, restrict, bs_only=True)
        if get_query_only:
            return self._compile(query)
        return self.execute(query)

    @typing.overload
    def _get_simulation_info(self, get_query_only: Literal[False] = False) -> SimInfo: ...

    @typing.overload
    def _get_simulation_info(self, get_query_only: Literal[True]) -> str: ...

    @validate_arguments
    def _get_simulation_info(self, get_query_only: bool = False) -> Union[str, SimInfo]:
        # find the simulation time interval
        query0 = sa.select(self.ts_bldgid_column, self._ts_upgrade_col).limit(1)  # get a building id and upgrade
        bldg_df = self.execute(query0)
        bldg_id = bldg_df.values[0][0]
        upgrade_id = bldg_df.values[0][1]
        query1 = sa.select(self.timestamp_column.distinct().label(self.timestamp_column_name)).where(
            self.ts_bldgid_column == bldg_id
        )
        if self.up_table is not None:
            query1 = query1.where(self._ts_upgrade_col == upgrade_id)
        query1 = query1.order_by(self.timestamp_column).limit(2)
        if get_query_only:
            return self._compile(query1)

        two_times = self.execute(query1)
        time1 = two_times[self.timestamp_column_name].iloc[0]
        time2 = two_times[self.timestamp_column_name].iloc[1]
        sim_year = time1.year
        reference_time = datetime(year=sim_year, month=1, day=1)
        sim_interval_seconds = int((time2 - time1).total_seconds())
        start_offset_seconds = int((time1 - reference_time).total_seconds())
        if sim_interval_seconds >= 28 * 24 * 60 * 60:  # 28 days or more means monthly resolution
            assert start_offset_seconds in [0, 31 * 24 * 60 * 60]
            interval = 1
            offset = start_offset_seconds // (31 * 24 * 60 * 60)
            unit = "month"
        else:
            interval = sim_interval_seconds
            offset = start_offset_seconds
            unit = "second"
        assert offset in [0, interval]
        return SimInfo(sim_year, interval, offset, unit)

    def _get_special_column(
        self, column_type: Literal["month", "day", "hour", "is_weekend", "day_of_week"]
    ) -> DBColType:
        sim_info = self._get_simulation_info()
        if sim_info.offset > 0:
            # If timestamps are not period beginning we should make them so we get proper values of special columns.
            time_col = sa.func.date_add(sim_info.unit, -sim_info.offset, self.timestamp_column)
        else:
            time_col = self.timestamp_column

        if column_type == "month":
            return sa.func.month(time_col).label("month")
        elif column_type == "day":
            return sa.func.day(time_col).label("day")
        elif column_type == "hour":
            return sa.func.hour(time_col).label("hour")
        elif column_type == "day_of_week":
            return sa.func.day_of_week(time_col).label("day_of_week")
        elif column_type == "is_weekend":
            return sa.cast(sa.func.day_of_week(time_col).in_([6, 7]), sa.Integer).label("is_weekend")
        else:
            assert_never(column_type)
            raise ValueError(f"Unknown special column type: {column_type}")

    def _get_gcol(
        self, column: AnyColType, tables: Sequence[AnyTableType] | None = None
    ) -> DBColType:  # gcol => group by col
        """Get a DB column for the purpose of grouping. If the provided column doesn't exist as is,
        tries to get the column by prepending self._char_prefix."""

        if isinstance(column, sa.Column):
            return column.label(self._simple_label(column.name))  # already a col

        if isinstance(column, SALabel):
            return column

        if isinstance(column, MappedColumn):
            return sa.literal(column).label(self._simple_label(column.name))

        if isinstance(column, str):
            try:
                return self._get_column(column, tables).label(self._simple_label(column))
            except (ValueError, KeyError):
                if column.startswith(self._char_prefix):
                    new_name = column.removeprefix(self._char_prefix)
                    return self._get_column(new_name, tables).label(column)
                else:
                    new_name = f"{self._char_prefix}{column}"
                    return self._get_column(new_name, tables).label(column)

        raise ValueError(f"Invalid column name type {column}: {type(column)}")

    def _get_enduse_cols(self, enduses: Sequence[AnyColType], table="baseline") -> Sequence[DBColType]:
        tbls_dict = {"baseline": self.bs_table, "upgrade": self.up_table, "timeseries": self.ts_table}
        tbl = tbls_dict[table]
        enduse_cols: list[DBColType] = []
        for enduse in enduses:
            if isinstance(enduse, (sa.Column, SALabel)):
                enduse_cols.append(enduse)
            elif isinstance(enduse, str):
                try:
                    enduse_cols.append(tbl.c[enduse])
                except KeyError as err:
                    if table in ["baseline", "upgrade"]:
                        enduse_cols.append(tbl.c[f"{self._out_prefix}{enduse}"])
                    else:
                        raise ValueError(f"Invalid enduse column names for {table} table") from err
            elif isinstance(enduse, MappedColumn):
                enduse_cols.append(sa.literal(enduse).label(enduse.name))
            else:
                assert_never(enduse)
        return enduse_cols

    def get_groupby_cols(self) -> list[str]:
        """Find list of building characteristics that can be used for grouping.

        Returns:
            list[str]: List of building characteristics.
        """
        cols = {y.removeprefix(self._char_prefix) for y in self.bs_table.c.keys() if y.startswith(self._char_prefix)}
        return list(cols)

    def _validate_group_by(self, group_by: Sequence[Union[str, tuple[str, str]]]):
        valid_groupby_cols = self.get_groupby_cols()
        group_by_cols = [g[0] if isinstance(g, tuple) else g for g in group_by]
        if not set(group_by_cols).issubset(valid_groupby_cols):
            invalid_cols = ", ".join(f'"{x}"' for x in set(group_by).difference(valid_groupby_cols))
            raise ValueError(f"The following are not valid columns in the database: {invalid_cols}")
        return group_by
        # TODO: intelligently select groupby columns order by cardinality (most to least groups) for
        # performance

    def get_available_upgrades(self) -> Sequence[str]:
        """Get the available upgrade scenarios and their identifier numbers.
        Returns:
            list: List of upgrades
        """
        return list([str(u) for u in self.report.get_success_report().index])

    def _validate_upgrade(self, upgrade_id: Union[int, str]) -> str:
        upgrade_id = "0" if upgrade_id in (None, "0") else str(upgrade_id)
        available_upgrades = self.get_available_upgrades() or ["0"]
        if upgrade_id not in set(available_upgrades):
            raise ValueError(
                f"`upgrade_id` = {upgrade_id} is not a valid upgrade.It doesn't exist or have no successful run"
            )
        return str(upgrade_id)

    def _split_restrict(self, restrict):
        # Some cols like "state" might be available in both ts and bs table
        bs_restrict = []  # restrict to apply to baseline table
        ts_restrict = []  # restrict to apply to timeseries table
        for col, restrict_vals in restrict:
            if self.ts_table is not None and col in self.ts_table.columns:  # prioritize ts table
                ts_restrict.append([self.ts_table.c[col], restrict_vals])
            else:
                bs_restrict.append([self._get_gcol(col), restrict_vals])
        return bs_restrict, ts_restrict

    def _split_group_by(self, processed_group_by: list[DBColType]):
        # Some cols like "state" might be available in both ts and bs table
        ts_group_by: list[DBColType] = []  # restrict to apply to baseline table
        bs_group_by: list[DBColType] = []  # restrict to apply to timeseries table
        for g in processed_group_by:
            if self.ts_table is not None and g.name in self.ts_table.columns:
                ts_group_by.append(g)
            else:
                bs_group_by.append(g)
        return bs_group_by, ts_group_by

    def _clean_group_by(self, group_by):
        """
        :param group_by: The group_by list
        :return: cleaned version of group_by
        Sometimes, it is necessary to include the table name in the group_by column. For example, a group_by could be
        ['time', '"res_national_53_2018_baseline"."build_existing_model.state"']. This is necessary if the another table
        (such as correction factors table) that has the same column ("build_existing_model.state") as the baseline
        table. However, the query result will not include the table name in columns, so it is necessary to transform
        the group_by to a cleaner version (['time', 'build_existing_model.state']).
        Othertimes, quotes are used in group_by columns, such as ['"time"'], but the query result will not contain the
        quote so it is necessary to remove the quote.
        Some other time, a group_by column is specified as a tuple of column and a as name. For example, group_by can
        contain [('month(time)', 'MOY')], in this case, we want to convert it into just 'MOY' since that is what will be
        present in the returned query.
        """
        new_group_by = []
        for col in group_by:
            if isinstance(col, tuple):
                new_group_by.append(col[1])
                continue

            if match := re.search(r'"[\w\.]*"\."([\w\.]*)"', col) or re.search(r'"([\w\.]*)"', col):
                new_group_by.append(match.group(1))
            else:
                new_group_by.append(col)
        return new_group_by

    def _process_groupby_cols(self, group_by, annual_only=False) -> list[DBColType]:
        if not group_by:
            return []
        tables = [self.bs_table, self.up_table]
        if not annual_only:
            tables.insert(0, self.ts_table)
        return [self._get_gcol(entry, tables) for entry in group_by]

    def _get_simulation_timesteps_count(self):
        # find the simulation time interval
        query = sa.select(self.ts_bldgid_column, safunc.sum(1).label("count"))
        query = query.group_by(self.ts_bldgid_column)
        sim_timesteps_count = self.execute(query)
        bld0_step_count = sim_timesteps_count["count"].iloc[0]
        n_buildings_with_same_count = sum(sim_timesteps_count["count"] == bld0_step_count)
        if n_buildings_with_same_count != len(sim_timesteps_count):
            logger.warning(
                "Not all buildings have the same number of timestamps. This can cause wrong"
                "scaled_units_count and other problems."
            )

        return bld0_step_count

    @typing.overload
    def get_buildings_by_locations(
        self, location_col: str, locations: list[str], get_query_only: Literal[False] = False
    ) -> pd.DataFrame: ...

    @typing.overload
    def get_buildings_by_locations(
        self, location_col: str, locations: list[str], get_query_only: Literal[True]
    ) -> str: ...

    @typing.overload
    def get_buildings_by_locations(
        self, location_col: str, locations: list[str], get_query_only: bool
    ) -> Union[str, pd.DataFrame]: ...

    @validate_arguments
    def get_buildings_by_locations(
        self, location_col: str, locations: list[str], get_query_only: bool = False
    ) -> Union[str, pd.DataFrame]:
        """
        Returns the list of buildings belonging to given list of locations.
        Args:
            location_col: The column used for "build_existing_model.county" etc
            locations: list of `build_existing_model.location' strings
            get_query_only: If set to true, returns the query string instead of the result

        Returns:
            Pandas dataframe consisting of the building ids belonging to the provided list of locations.

        """
        query = sa.select(self.bs_bldgid_column)
        query = query.where(self._get_column(location_col).in_(locations))
        query = self._add_order_by(query, [self.bs_bldgid_column])
        if get_query_only:
            return self._compile(query)
        res = self.execute(query)
        return res

    @property
    def _bs_completed_status_col(self):
        if not isinstance(self.bs_table.c[self.db_schema.column_names.completed_status].type, sqltypes.String):
            return sa.cast(self.bs_table.c[self.db_schema.column_names.completed_status], sa.String).label(
                "completed_status"
            )
        else:
            return self.bs_table.c[self.db_schema.column_names.completed_status]

    @property
    def _up_completed_status_col(self):
        if self.up_table is None:
            raise ValueError("No upgrades table")
        if not isinstance(self.up_table.c[self.db_schema.column_names.completed_status].type, sqltypes.String):
            return sa.cast(self.up_table.c[self.db_schema.column_names.completed_status], sa.String).label(
                "completed_status"
            )
        else:
            return self.up_table.c[self.db_schema.column_names.completed_status]

    @property
    def _bs_successful_condition(self):
        return self._bs_completed_status_col == self.db_schema.completion_values.success

    @property
    def _up_successful_condition(self):
        return self._up_completed_status_col == self.db_schema.completion_values.success

    @property
    def _ts_upgrade_col(self):
        if not isinstance(self.ts_table.c["upgrade"].type, sqltypes.String):
            return sa.cast(self.ts_table.c["upgrade"], sa.String).label("upgrade")
        else:
            return self.ts_table.c["upgrade"]

    @property
    def _up_upgrade_col(self):
        if self.up_table is None:
            raise ValueError("No upgrades table")
        if not isinstance(self.up_table.c["upgrade"].type, sqltypes.String):
            return sa.cast(self.up_table.c["upgrade"], sa.String).label("upgrade")
        else:
            return self.up_table.c["upgrade"]

    def _get_completed_status_col(self, table: AnyTableType):
        if not isinstance(table.c[self.db_schema.column_names.completed_status].type, sqltypes.String):
            return sa.cast(table.c[self.db_schema.column_names.completed_status], sa.String).label("completed_status")
        else:
            return table.c[self.db_schema.column_names.completed_status]

    def _get_success_condition(self, table: AnyTableType):
        return self._get_completed_status_col(table) == self.db_schema.completion_values.success

    @typing.overload
    def query(
        self,
        *,
        get_query_only: Literal[True],
        upgrade_id: int | str = "0",
        enduses: Sequence[AnyColType],
        group_by: Sequence[AnyColType | tuple[str, str]] = Field(default_factory=list),
        annual_only: bool = True,
        include_upgrade: bool = True,
        include_savings: bool = False,
        include_baseline: bool = False,
        sort: bool = True,
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = Field(default_factory=list),
        weights: Sequence[str | tuple] = Field(default_factory=list),
        restrict: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = Field(default_factory=list),
        avoid: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = Field(default_factory=list),
        applied_only: bool = False,
        get_quartiles: bool = False,
        get_nonzero_count: bool = False,
        unload_to: str = "",
        partition_by: Sequence[str] | None = None,
        timestamp_grouping_func: str | None = None,
        limit: int | None = None,
        agg_func: str | None = "sum",
    ) -> str: ...

    @typing.overload
    def query(
        self,
        *,
        upgrade_id: int | str = "0",
        get_query_only: Literal[False] = False,
        enduses: Sequence[AnyColType],
        group_by: Sequence[AnyColType | tuple[str, str]] = Field(default_factory=list),
        annual_only: bool = True,
        include_upgrade: bool = True,
        include_savings: bool = False,
        include_baseline: bool = False,
        sort: bool = True,
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = Field(default_factory=list),
        weights: Sequence[str | tuple] = Field(default_factory=list),
        restrict: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = Field(default_factory=list),
        avoid: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = Field(default_factory=list),
        applied_only: bool = False,
        get_quartiles: bool = False,
        get_nonzero_count: bool = False,
        unload_to: str = "",
        partition_by: Sequence[str] | None = None,
        timestamp_grouping_func: str | None = None,
        limit: int | None = None,
        agg_func: str | None = "sum",
    ) -> pd.DataFrame: ...

    @typing.overload
    def query(
        self,
        *,
        get_query_only: bool,
        upgrade_id: int | str = "0",
        enduses: Sequence[AnyColType],
        group_by: Sequence[AnyColType | tuple[str, str]] = Field(default_factory=list),
        annual_only: bool = True,
        include_upgrade: bool = True,
        include_savings: bool = False,
        include_baseline: bool = False,
        sort: bool = True,
        join_list: Sequence[tuple[AnyTableType, AnyColType, AnyColType]] = Field(default_factory=list),
        weights: Sequence[str | tuple] = Field(default_factory=list),
        restrict: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = Field(default_factory=list),
        avoid: Sequence[tuple[AnyColType, str | int | Sequence[int | str]]] = Field(default_factory=list),
        applied_only: bool = False,
        get_quartiles: bool = False,
        get_nonzero_count: bool = False,
        unload_to: str = "",
        partition_by: Sequence[str] | None = None,
        timestamp_grouping_func: str | None = None,
        limit: int | None = None,
        agg_func: str | None = "sum",
    ) -> str | pd.DataFrame: ...

    @typing.overload
    def query(self, *, params: Query) -> str | pd.DataFrame: ...

    def query(self, *args, **kwargs) -> str | pd.DataFrame:
        """Query the run to obtain either the results dataframe or the query string.
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
            unload_to: Writes the output of the query to this location in s3. Consider using run_async = True with this
                       to unload multiple queries simulataneuosly
            partition_by: List of columns to partition when writing to s3. To be used with unload_to.
            timestamp_grouping_func: One of 'hour', 'day' or 'month' or 'year' or None. If provided, perform timeseries
                        aggregation of specified granularity. For 'year' - it collapses the timeseries into a single
                        annual value. Useful for quality checking or finding the annual max and min.
         Returns:
                if get_query_only is True, returns the query_string, otherwise returns a pandas dataframe
        """
        # TODO: Replace with contents of agg._query(*args, **kwargs) when aggregate_query module is deprecated
        # or implement via a Mixin
        return self.agg._query(*args, **kwargs)
