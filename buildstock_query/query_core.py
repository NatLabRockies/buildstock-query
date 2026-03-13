import boto3
import contextlib
import pathlib
from pyathena.connection import Connection
from pyathena.error import OperationalError
from pyathena.sqlalchemy.base import AthenaDialect
import sqlalchemy as sa
from sqlalchemy.sql import func as safunc
from pyathena.pandas.async_cursor import AsyncPandasCursor
from pyathena.pandas.cursor import PandasCursor
import os
from typing import Union, Optional, Literal, Callable
from collections.abc import Sequence
import typing
import time
import logging
from threading import Thread
from botocore.exceptions import ClientError
import pandas as pd
import datetime
import numpy as np
from collections import OrderedDict
import types
from buildstock_query.helpers import CachedFutureDf, AthenaFutureDf, DataExistsException, CustomCompiler
from buildstock_query.helpers import save_pickle, load_pickle, read_csv
from typing import TypedDict, NewType
from botocore.config import Config
import urllib3
from buildstock_query.schema.run_params import RunParams
from buildstock_query.db_schema.db_schema_model import DBSchema
from buildstock_query.schema.utilities import (
    DBColType,
    SACol,
    AnyColType,
    AnyTableType,
    MappedColumn,
    SALabel,
    DBTableType,
    validate_arguments
)
import hashlib
import toml

urllib3.disable_warnings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FUELS = ["electricity", "natural_gas", "propane", "fuel_oil", "coal", "wood_cord", "wood_pellets"]


class QueryException(Exception):
    pass


ExeId = NewType("ExeId", str)


class BatchQueryStatusMap(TypedDict):
    to_submit_ids: list[int]
    all_ids: list[int]
    submitted_ids: list[int]
    submitted_execution_ids: list[ExeId]
    submitted_queries: list[str]
    queries_futures: list[Union[CachedFutureDf, AthenaFutureDf]]


class BatchQueryReportMap(TypedDict):
    submitted: int
    running: int
    pending: int
    completed: int
    failed: int


class QueryCore:
    def __init__(self, *, params: RunParams) -> None:
        """
        Base class to run common Athena queries for BuildStock runs and download results as pandas dataFrame
        Usually, you should just use BuildStockQuery. This class is useuful if you want to extend the functionality
        for Athena tables that are not part of ResStock or ComStock runs.
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
            sample_weight (str, optional): Specify a custom sample_weight. Otherwise, the default is 1 for ComStock and
                uses sample_weight in the run for ResStock.
            region_name (str, optional): the AWS region where the database exists. Defaults to 'us-west-2'.
            execution_history (str, optional): A temporary file to record which execution is run by the user,
                to help stop them. Will use .execution_history if not supplied. Generally, not required to supply a
                custom filename.
            athena_query_reuse (bool, optional): When true, Athena will make use of its built-in 7 day query cache.
                When false, it will not. Defaults to True. One use case to set this to False is when you have modified
                the underlying s3 data or glue schema and want to make sure you are not using the cached results.
        """
        logger.info(f"Loading {params.table_name} ...")
        self.run_params = params
        self.workgroup = params.workgroup
        self.buildstock_type = params.buildstock_type
        self._query_cache: dict[str, pd.DataFrame] = {}  # {"query": query_result_df} to cache queries
        self._session_queries: set[str] = set()  # Set of all queries that is run in current session.

        self._aws_s3 = boto3.client("s3")
        self._aws_athena = boto3.client("athena", region_name=params.region_name)
        self._aws_glue = boto3.client("glue", region_name=params.region_name)
        self._conn = Connection(
            work_group=params.workgroup,
            region_name=params.region_name,
            cursor_class=PandasCursor,
            schema_name=params.db_name,
            config=Config(max_pool_connections=20),
        )
        self._async_conn = Connection(
            work_group=params.workgroup,
            region_name=params.region_name,
            cursor_class=AsyncPandasCursor,
            schema_name=params.db_name,
            config=Config(max_pool_connections=20),
        )

        self.db_name = params.db_name
        self.region_name = params.region_name

        self._tables: dict[str, sa.Table] = OrderedDict()  # Internal record of tables
        self._meta = sa.MetaData()

        self._batch_query_status_map: dict[int, BatchQueryStatusMap] = {}
        self._batch_query_id = 0
        db_schema_file = os.path.join(os.path.dirname(__file__), "db_schema", f"{params.db_schema}.toml")
        db_schema_dict = toml.load(db_schema_file)
        self.db_schema = DBSchema.model_validate(db_schema_dict)
        self.db_col_name = self.db_schema.column_names
        self.timestamp_column_name = self.db_col_name.timestamp
        self.building_id_column_name = self.db_col_name.building_id
        self.sample_weight = (
            params.sample_weight_override
            if params.sample_weight_override is not None
            else self.db_col_name.sample_weight
        )
        self.table_name = params.table_name
        self.cache_folder = pathlib.Path(params.cache_folder)
        self.athena_query_reuse = params.athena_query_reuse
        os.makedirs(self.cache_folder, exist_ok=True)
        self._initialize_tables()
        self._initialize_book_keeping(params.execution_history)

        with contextlib.suppress(FileNotFoundError):
            self.load_cache()

    @staticmethod
    def _get_compact_cache_name(table_name: str) -> str:
        table_name = str(table_name)
        if len(table_name) > 64:
            return hashlib.sha256(table_name.encode()).hexdigest()
        else:
            return table_name

    def _get_cache_file_path(self) -> pathlib.Path:
        return self.cache_folder / f"{self._get_compact_cache_name(self.table_name)}_query_cache.pkl"

    @validate_arguments
    def load_cache(self, path: Optional[str] = None):
        """Read and update query cache from pickle file.

        Args:
            path (str, optional): The path to the pickle file. If not provided, reads from current directory.
        """
        pickle_path = pathlib.Path(path) if path else self._get_cache_file_path()
        before_count = len(self._query_cache)
        saved_cache = load_pickle(pickle_path)
        logger.info(f"{len(saved_cache)} queries cache read from {path}.")
        self._query_cache.update(saved_cache)
        self.last_saved_queries = set(saved_cache)
        after_count = len(self._query_cache)
        if diff := after_count - before_count:
            logger.info(f"{diff} queries cache is updated.")
        else:
            logger.info("Cache already upto date.")

    @validate_arguments
    def save_cache(self, path: Optional[str] = None, trim_excess: bool = False):
        """Saves queries cache to a pickle file. It is good idea to run this after making queries so that on the next
        session these queries won't have to be run on Athena and can be directly loaded from the file.

        Args:
            path (str, optional): The path to the pickle file. If not provided, the file will be saved on the current
            directory.
            trim_excess (bool, optional): If true, any queries in the cache that is not run in current session will be
            removed before saving it to file. This is useful if the cache has accumulated a bunch of stray queries over
            several sessions that are no longer used. Defaults to False.
        """
        cached_queries = set(self._query_cache)
        if self.last_saved_queries == cached_queries:
            logger.info("No new queries to save.")
            return

        pickle_path = pathlib.Path(path) if path else self._get_cache_file_path()
        if trim_excess:
            if excess_queries := [key for key in self._query_cache if key not in self._session_queries]:
                for query in excess_queries:
                    del self._query_cache[query]
                logger.info(f"{len(excess_queries)} excess queries removed from cache.")
        self.last_saved_queries = cached_queries
        save_pickle(pickle_path, self._query_cache)
        logger.info(f"{len(self._query_cache)} queries cache saved to {pickle_path}")

    def _initialize_tables(self):
        self.bs_table, self.ts_table, self.up_table = self._get_tables(self.table_name)

        self.bs_bldgid_column = self.bs_table.c[self.building_id_column_name]
        if self.ts_table is not None:
            self.timestamp_column = self.ts_table.c[self.timestamp_column_name]
            self.ts_bldgid_column = self.ts_table.c[self.building_id_column_name]
        if self.up_table is not None:
            self.up_bldgid_column = self.up_table.c[self.building_id_column_name]
        self.sample_wt = self._get_sample_weight(self.sample_weight)

    def _get_sample_weight(self, sample_weight):
        if not sample_weight:
            return sa.literal(1)
        elif isinstance(sample_weight, str):
            try:
                return self.bs_table.c[sample_weight]
            except ValueError:
                logger.error("Sample weight column not found. Using weight of 1.")
                return sa.literal(1)
        elif isinstance(sample_weight, (int, float)):
            return sa.literal(sample_weight)
        else:
            raise ValueError("Invalid value for sample_weight")

    @typing.overload
    def _get_table(self, table_name: AnyTableType, missing_ok: Literal[True]) -> Optional[sa.Table]: ...

    @typing.overload
    def _get_table(self, table_name: AnyTableType, missing_ok: Literal[False] = False) -> sa.Table: ...

    @validate_arguments
    def _get_table(self, table_name: AnyTableType, missing_ok: bool = False) -> Optional[DBTableType]:
        if not isinstance(table_name, str):
            return table_name  # already a table

        try:
            return self._tables.setdefault(table_name, sa.Table(table_name, self._meta, autoload_with=self._engine))
        except sa.exc.NoSuchTableError:  # type: ignore
            if missing_ok:
                logger.warning(f"No {table_name} table is present.")
                return None
            else:
                raise

    @validate_arguments
    def _get_column(
        self, column_name: AnyColType, candidate_tables: Sequence[AnyTableType | None] | None = None
    ) -> DBColType:
        if isinstance(column_name, SACol):
            return column_name.label(self._simple_label(column_name.name))  # already a col

        if isinstance(column_name, SALabel):
            return column_name

        if isinstance(column_name, MappedColumn):
            return sa.literal(column_name).label(self._simple_label(column_name.name))

        if candidate_tables is not None:
            search_tables = [self._get_table(table) for table in candidate_tables if table is not None]
        else:
            search_tables = [
                self._get_table(tbl) for tbl in [self.bs_table, self.up_table, self.ts_table] if tbl is not None
            ]
        valid_tables = []
        for tbl in search_tables:
            if column_name in tbl.columns:
                valid_tables.append(tbl)
        if not valid_tables:
            valid_tables += [table for _, table in self._tables.items() if column_name in table.columns]
        if not valid_tables:
            raise ValueError(f"Column {column_name} not found in any tables {[t.name for t in self._tables.values()]}")
        if len(valid_tables) > 1:
            logger.warning(
                f"Column {column_name} found in multiple tables {[t.name for t in valid_tables]}. "
                f"Using {valid_tables[0].name}"
            )
        return valid_tables[0].c[column_name]

    def _get_tables(self, table_name: Union[str, tuple[str, Optional[str], Optional[str]]]):
        self._engine = self._create_athena_engine(
            region_name=self.region_name, database=self.db_name, workgroup=self.workgroup
        )
        if isinstance(table_name, str):
            baseline_table = self._get_table(f"{table_name}{self.db_schema.table_suffix.baseline}")
            ts_table = self._get_table(f"{table_name}{self.db_schema.table_suffix.timeseries}", missing_ok=True)
            if self.db_schema.table_suffix.upgrades == self.db_schema.table_suffix.baseline:
                upgrade_table = (
                    sa.select(baseline_table)
                    .where(sa.cast(baseline_table.c["upgrade"], sa.String) != "0")
                    .alias("upgrade")
                )
                baseline_table = (
                    sa.select(baseline_table)
                    .where(sa.cast(baseline_table.c["upgrade"], sa.String) == "0")
                    .alias("baseline")
                )
            else:
                upgrade_table = self._get_table(f"{table_name}{self.db_schema.table_suffix.upgrades}", missing_ok=True)
        else:
            baseline_table = self._get_table(f"{table_name[0]}")
            ts_table = self._get_table(f"{table_name[1]}", missing_ok=True) if table_name[1] else None
            if table_name[2] == table_name[0]:
                upgrade_table = (
                    sa.select(baseline_table)
                    .where(sa.cast(baseline_table.c["upgrade"], sa.String) != "0")
                    .alias("upgrade")
                )
                baseline_table = (
                    sa.select(baseline_table)
                    .where(sa.cast(baseline_table.c["upgrade"], sa.String) == "0")
                    .alias("baseline")
                )
            else:
                upgrade_table = self._get_table(f"{table_name[2]}", missing_ok=True) if table_name[2] else None
        return baseline_table, ts_table, upgrade_table

    def _initialize_book_keeping(self, execution_history):
        self._execution_history_file = execution_history or self.cache_folder / ".execution_history"
        self.execution_cost = {"GB": 0, "Dollars": 0}  # Tracks the cost of current session. Only used for Athena query
        self.seen_execution_ids = set()  # set to prevent double counting same execution id
        self.last_saved_queries = set()
        if os.path.exists(self._execution_history_file):
            with open(self._execution_history_file) as f:
                existing_entries = f.readlines()
            valid_entries = []
            for entry in existing_entries:
                with contextlib.suppress(ValueError, TypeError):
                    entry_time, _ = entry.split(",")
                    if time.time() - float(entry_time) < 24 * 60 * 60:  # discard history if more than a day old
                        valid_entries += entry
            with open(self._execution_history_file, "w") as f:
                f.writelines(valid_entries)

    @property
    def _execution_ids_history(self):
        exe_ids: list[ExeId] = []
        if os.path.exists(self._execution_history_file):
            with open(self._execution_history_file) as f:
                for line in f:
                    _, exe_id = line.split(",")
                    exe_ids.append(ExeId(exe_id.strip()))
        return exe_ids

    def _create_athena_engine(self, region_name: str, database: str, workgroup: str) -> sa.engine.Engine:
        connect_args = {"cursor_class": PandasCursor, "work_group": workgroup}
        engine = sa.create_engine(
            f"awsathena+rest://:@athena.{region_name}.amazonaws.com:443/{database}", connect_args=connect_args
        )
        return engine

    @validate_arguments
    def delete_table(self, table_name: str):
        """
        Function to delete athena table.
        :param table_name: Athena table name
        :return:
        """
        delete_table_query = f"""DROP TABLE {self.db_name}.{table_name};"""
        result, reason = self.execute_raw(delete_table_query)
        if result.upper() == "SUCCEEDED":
            return "SUCCEEDED"
        else:
            raise QueryException(f"Deleting it failed. Reason: {reason}")

    @validate_arguments
    def add_table(
        self, table_name: str, table_df: pd.DataFrame, s3_bucket: str, s3_prefix: str, override: bool = False
    ):
        """
        Function to add a table in s3.
        :param table_name: The name of the table
        :param table_df: The pandas dataframe to use as table data
        :param s3_bucket: s3 bucket name
        :param s3_prefix: s3 prefix to save the table to.
        :param override: Whether to override existing table.
        :return:
        """
        s3_location = s3_bucket + "/" + s3_prefix
        s3_data = self._aws_s3.list_objects(Bucket=s3_bucket, Prefix=f"{s3_prefix}/{table_name}")
        if "Contents" in s3_data and override is False:
            raise DataExistsException("Table already exists", f"s3://{s3_location}/{table_name}/{table_name}.csv")
        if "Contents" in s3_data:
            existing_objects = [{"Key": el["Key"]} for el in s3_data["Contents"]]
            print(f"The following existing objects is being delete and replaced: {existing_objects}")
            print(f"Saving s3://{s3_location}/{table_name}/{table_name}.parquet)")
            self._aws_s3.delete_objects(Bucket=s3_bucket, Delete={"Objects": existing_objects})
        print(f"Saving factors to s3 in s3://{s3_location}/{table_name}/{table_name}.parquet")
        # table_df.to_parquet(f's3://{s3_location}/{table_name}/{table_name}.parquet', index=False)
        self._aws_s3.put_object(
            Body=table_df.to_parquet(index=False),
            Bucket=s3_bucket,
            Key=f"{s3_prefix}/{table_name}/{table_name}.parquet",
        )
        print("Saving Done.")

        format_list = []
        for column_name, dtype in table_df.dtypes.items():
            if np.issubdtype(dtype, np.integer):
                col_type = "int"
            elif np.issubdtype(dtype, np.floating):
                col_type = "double"
            elif np.issubdtype(dtype, np.datetime64):
                col_type = "timestamp"
            else:
                col_type = "string"
            format_list.append(f"`{column_name}` {col_type}")

        column_formats = ",".join(format_list)

        table_create_query = f"""
        CREATE EXTERNAL TABLE {self.db_name}.{table_name} ({column_formats})
        STORED AS PARQUET
        LOCATION 's3://{s3_location}/{table_name}/'
        TBLPROPERTIES ('has_encrypted_data'='false');
        """

        print(f"Running create table query.\n {table_create_query}")
        result, reason = self.execute_raw(table_create_query)
        if result.lower() == "failed" and "alreadyexists" in reason.lower():
            if not override:
                existing_data = read_csv(f"s3://{s3_location}/{table_name}/{table_name}.csv")
                raise DataExistsException("Table already exists", existing_data)
            print(f"There was existing table {table_name} in Athena which was deleted and recreated.")
            delete_table_query = f"""
            DROP TABLE {self.db_name}.{table_name};
            """
            result, reason = self.execute_raw(delete_table_query)
            if result.upper() != "SUCCEEDED":
                raise QueryException(
                    f"There was an existing table named {table_name}. Deleting it failed. Reason: {reason}"
                )
            result, reason = self.execute_raw(table_create_query)
            if result.upper() == "SUCCEEDED":
                return "SUCCEEDED"
            else:
                raise QueryException(
                    f"There was an existing table named {table_name} which is now successfully "
                    f"deleted but new table failed to be created. Reason: {reason}"
                )
        elif result.upper() == "SUCCEEDED":
            return "SUCCEEDED"
        else:
            raise QueryException(f"Failed to create the table. Reason: {reason}")

    @validate_arguments
    def execute_raw(self, query, db: Optional[str] = None, run_async: bool = False):
        """
        Directly executes the supplied query in Athena.
        :param query:
        :param db:
        :param run_async:
        :return:
        """
        if not db:
            db = self.db_name

        response = self._aws_athena.start_query_execution(
            QueryString=query, QueryExecutionContext={"Database": db}, WorkGroup=self.workgroup
        )
        query_execution_id = ExeId(response["QueryExecutionId"])

        if run_async:
            return query_execution_id
        start_time = time.time()
        while time.time() - start_time < 30 * 60:  # 30 minute timeout
            query_stat = self._aws_athena.get_query_execution(QueryExecutionId=query_execution_id)
            if query_stat["QueryExecution"]["Status"]["State"].lower() not in ["pending", "running", "queued"]:
                reason = query_stat["QueryExecution"]["Status"].get("StateChangeReason", "")
                return query_stat["QueryExecution"]["Status"]["State"], reason
            time.sleep(1)

        raise TimeoutError("Query failed to complete within 30 mins.")

    def _save_execution_id(self, execution_id):
        with open(self._execution_history_file, "a") as f:
            f.write(f"{time.time()},{execution_id}\n")

    def _log_execution_cost(self, execution_id: ExeId):
        if execution_id == "CACHED":
            # Can't log cost for cached query
            return
        res = self._aws_athena.get_query_execution(QueryExecutionId=execution_id)
        scanned_GB = res["QueryExecution"]["Statistics"]["DataScannedInBytes"] / 1e9
        cost = scanned_GB * 5 / 1e3  # 5$ per TB scanned
        if execution_id not in self.seen_execution_ids:
            self.execution_cost["Dollars"] += cost
            self.execution_cost["GB"] += scanned_GB
            self.seen_execution_ids.add(execution_id)

        logger.info(
            f"{execution_id} cost {scanned_GB:.1f} GB (${cost:.1f}). Session total:"
            f" {self.execution_cost['GB']:.1f} GB (${self.execution_cost['Dollars']:.1f})"
        )

    def _compile(self, query) -> str:
        compiled_query = CustomCompiler(AthenaDialect(), query).process(query, literal_binds=True)
        return compiled_query

    @typing.overload
    def execute(self, query, *, run_async: Literal[False] = False) -> pd.DataFrame: ...

    @typing.overload
    def execute(
        self,
        query,
        *,
        run_async: Literal[True],
    ) -> Union[tuple[Literal["CACHED"], CachedFutureDf], tuple[ExeId, AthenaFutureDf]]: ...

    @validate_arguments
    def execute(
        self, query, run_async: bool = False
    ) -> Union[pd.DataFrame, tuple[Literal["CACHED"], CachedFutureDf], tuple[ExeId, AthenaFutureDf]]:
        """
        Executes a query
        Args:
            query: The SQL query to run in Athena
            run_async: Whether to wait until the query completes (run_async=False) or return immediately
            (run_async=True).

        Returns:
            if run_async is False, returns the results dataframe.
            if run_async is  True, returns the query_execution_id, futures
        """
        if not isinstance(query, str):
            query = self._compile(query)

        self._session_queries.add(query)
        if run_async:
            if query in self._query_cache:
                return "CACHED", CachedFutureDf(self._query_cache[query].copy())
            # in case of asynchronous run, you get the execution id and futures object
            exe_id, result_future = self._async_conn.cursor().execute(
                query, result_reuse_enable=self.athena_query_reuse, result_reuse_minutes=60 * 24 * 7, na_values=[""]
            )  # type: ignore
            exe_id = ExeId(exe_id)

            def get_pandas(future):
                res = future.result()
                if res.state != "SUCCEEDED":
                    raise OperationalError(f"{res.state}: {res.state_change_reason}")
                if query in self._query_cache:
                    return self._query_cache[query]
                return res.as_pandas()

            result_future.as_pandas = types.MethodType(get_pandas, result_future)
            result_future.add_done_callback(lambda f: self._query_cache.update({query: f.as_pandas()}))
            self._save_execution_id(exe_id)
            return exe_id, AthenaFutureDf(result_future)
        else:
            if query not in self._query_cache:
                self._query_cache[query] = (
                    self._conn.cursor()
                    .execute(
                        query,
                        result_reuse_enable=self.athena_query_reuse,
                        result_reuse_minutes=60 * 24 * 7,
                    )
                    .as_pandas()
                )
            return self._query_cache[query].copy()

    def print_all_batch_query_status(self) -> None:
        """Prints the status of all batch queries."""
        for count in self._batch_query_status_map.keys():
            print(f"Query {count}: {self.get_batch_query_report(count)}\n")

    @validate_arguments
    def stop_batch_query(self, batch_id: int) -> None:
        """
        Stops all the queries running under a batch query
        Args:
            batch_id: The batch_id of the batch_query. Returned by :py:submit_batch_query

        Returns:
            None
        """
        if batch_id not in self._batch_query_status_map:
            raise ValueError("Batch id not found")
        self._batch_query_status_map[batch_id]["to_submit_ids"].clear()
        for exec_id in self._batch_query_status_map[batch_id]["submitted_execution_ids"]:
            self.stop_query(exec_id)

    @validate_arguments
    def get_failed_queries(self, batch_id: int) -> tuple[Sequence[ExeId], Sequence[str]]:
        """_summary_

        Args:
            batch_id (int): Batch query id returned by :py:submit_batch_query

        Returns:
            _type_: tuple of list of failed query execution ids and list of failed queries
        """
        stats = self._batch_query_status_map.get(batch_id, None)
        failed_query_ids: list[ExeId] = []
        failed_queries: list[str] = []
        if stats:
            for i, exe_id in enumerate(stats["submitted_execution_ids"]):
                completion_stat = self.get_query_status(exe_id)
                if completion_stat in ["FAILED", "CANCELLED"]:
                    failed_query_ids.append(exe_id)
                    failed_queries.append(stats["submitted_queries"][i])
        return failed_query_ids, failed_queries

    @validate_arguments
    def print_failed_query_errors(self, batch_id: int) -> None:
        """Print the error messages for all queries that failed in batch query.

        Args:
            batch_id (int): Batch query id
        """
        failed_ids, failed_queries = self.get_failed_queries(batch_id)
        for exe_id, query in zip(failed_ids, failed_queries):
            print(
                f"Query id: {exe_id}. \n Query string: {query}. Query Ended with: {self.get_query_status(exe_id)}"
                f"\nError: {self.get_query_error(exe_id)}\n"
            )

    @validate_arguments
    def get_ids_for_failed_queries(self, batch_id: int) -> Sequence[str]:
        """Returns the list of execution ids for failed queries in batch query.

        Args:
            batch_id (int): batch query id

        Returns:
            Sequence[str]: List of failed execution ids.
        """
        failed_ids = []
        for i, exe_id in enumerate(self._batch_query_status_map[batch_id]["submitted_execution_ids"]):
            completion_stat = self.get_query_status(exe_id)
            if completion_stat in ["FAILED", "CANCELLED"]:
                failed_ids.append(exe_id)
        return failed_ids

    @validate_arguments
    def get_batch_query_report(self, batch_id: int) -> BatchQueryReportMap:
        """
        Returns the status of the queries running under a batch query.
        Args:
            batch_id: The batch_id of the batch_query.

        Returns:
            A dictionary detailing status of the queries.
        """
        if not (stats := self._batch_query_status_map.get(batch_id, None)):
            raise ValueError(f"{batch_id=} not found.")
        success_count = 0
        fail_count = 0
        running_count = 0
        other = 0
        for exe_id in stats["submitted_execution_ids"]:
            if exe_id == "CACHED":
                completion_stat = "SUCCEEDED"
            else:
                completion_stat = self.get_query_status(exe_id)
            if completion_stat == "RUNNING":
                running_count += 1
            elif completion_stat == "SUCCEEDED":
                success_count += 1
            elif completion_stat in ["FAILED", "CANCELLED"]:
                fail_count += 1
            else:
                # for example: QUEUED
                other += 1

        result: BatchQueryReportMap = {
            "submitted": len(stats["submitted_ids"]),
            "running": running_count,
            "pending": len(stats["to_submit_ids"]) + other,
            "completed": success_count,
            "failed": fail_count,
        }

        return result

    @validate_arguments
    def did_batch_query_complete(self, batch_id: int):
        """
        Checks if all the queries in a batch query has completed or not.
        Args:
            batch_id: The batch_id for the batch_query

        Returns:
            True or False
        """
        status = self.get_batch_query_report(batch_id)
        if status["pending"] > 0 or status["running"] > 0:
            return False
        else:
            return True

    @validate_arguments
    def wait_for_batch_query(self, batch_id: int):
        """Waits until batch query completes.

        Args:
            batch_id (int): The batch query id.
        """
        sleep_time = 0.5  # start here and keep doubling until max_sleep_time
        max_sleep_time = 20
        while True:
            last_time = time.time()
            last_report = None
            report = self.get_batch_query_report(batch_id)
            if time.time() - last_time > 60 or last_report is None or report != last_report:
                logger.info(report)
                last_report = report
                last_time = time.time()
            if report["pending"] == 0 and report["running"] == 0:
                break
            time.sleep(sleep_time)
            sleep_time = min(sleep_time * 2, max_sleep_time)

    @typing.overload
    def get_batch_query_result(
        self, batch_id: int, *, no_block: bool = False, combine: Literal[True] = True
    ) -> pd.DataFrame: ...

    @typing.overload
    def get_batch_query_result(
        self, batch_id: int, *, no_block: bool = False, combine: Literal[False]
    ) -> list[pd.DataFrame]: ...

    @validate_arguments
    def get_batch_query_result(self, batch_id: int, *, combine: bool = True, no_block: bool = False):
        """
        Concatenates and returns the results of all the queries of a batchquery
        Args:
            batch_id (int): The batch_id for the batch_query
            no_block (bool): Whether to wait until all queries have completed or return immediately. If you use
                            no_block = true and the batch hasn't completed, it will throw BatchStillRunning exception.
            combine: Whether to combine the individual query result into a single dataframe

        Returns:
            The concatenated dataframe of the results of all the queries in a batch query.

        """
        if no_block and self.did_batch_query_complete(batch_id) is False:
            raise QueryException("Batch query not completed yet.")

        self.wait_for_batch_query(batch_id)
        logger.info("Batch query completed. ")
        report = self.get_batch_query_report(batch_id)
        query_exe_ids = self._batch_query_status_map[batch_id]["submitted_execution_ids"]
        query_futures = self._batch_query_status_map[batch_id]["queries_futures"]
        if report["failed"] > 0:
            logger.warning(f"{report['failed']} queries failed. Redoing them")
            failed_ids, failed_queries = self.get_failed_queries(batch_id)
            new_batch_id = self.submit_batch_query(failed_queries)
            new_exe_ids = self._batch_query_status_map[new_batch_id]["submitted_execution_ids"]

            self.wait_for_batch_query(new_batch_id)
            new_exe_ids_map = {entry[0]: entry[1] for entry in zip(failed_ids, new_exe_ids)}

            new_report = self.get_batch_query_report(new_batch_id)
            if new_report["failed"] > 0:
                self.print_failed_query_errors(new_batch_id)
                raise QueryException("Queries failed again. Sorry!")
            logger.info("The queries succeeded this time. Gathering all the results.")
            # replace the old failed exe_ids with new successful exe_ids
            for indx, old_exe_id in enumerate(query_exe_ids):
                query_exe_ids[indx] = new_exe_ids_map.get(old_exe_id, old_exe_id)

        if len(query_exe_ids) == 0:
            raise ValueError("No query was submitted successfully")
        res_df_array: list[pd.DataFrame] = []
        for index, exe_id in enumerate(query_exe_ids):
            df = query_futures[index].as_pandas().copy()
            if combine:
                if len(df) > 0:
                    df["query_id"] = index
            logger.info(f"Got result from Query [{index}] ({exe_id})")
            self._log_execution_cost(exe_id)
            res_df_array.append(df)
        if not combine:
            return res_df_array
        logger.info("Concatenating the results.")
        # return res_df_array
        return pd.concat(res_df_array)

    @validate_arguments
    def submit_batch_query(self, queries: Sequence[str]):
        """
        Submit multiple related queries
        Args:
            queries: List of queries to submit. Setting `get_query_only` flag while making calls to aggregation
                    functions is easiest way to obtain queries.
        Returns:
            An integer representing the batch_query id. The id can be used with other batch_query functions.
        """
        queries = list(queries)
        to_submit_ids = list(range(len(queries)))
        id_list = list(to_submit_ids)  # make a copy
        submitted_ids: list[int] = []
        submitted_execution_ids: list[ExeId] = []
        submitted_queries: list[str] = []
        queries_futures: list = []
        self._batch_query_id += 1
        batch_query_id = self._batch_query_id
        self._batch_query_status_map[batch_query_id] = {
            "to_submit_ids": to_submit_ids,
            "all_ids": list(id_list),
            "submitted_ids": submitted_ids,
            "submitted_execution_ids": submitted_execution_ids,
            "submitted_queries": submitted_queries,
            "queries_futures": queries_futures,
        }

        def run_queries():
            while to_submit_ids:
                current_id = to_submit_ids[0]  # get the first one
                current_query = queries[0]
                try:
                    execution_id, future = self.execute(current_query, run_async=True)
                    logger.info(f"Submitted queries[{current_id}] ({execution_id})")
                    to_submit_ids.pop(0)  # if query queued successfully, remove it from the list
                    queries.pop(0)
                    submitted_ids.append(current_id)
                    submitted_execution_ids.append(ExeId(execution_id))
                    submitted_queries.append(current_query)
                    queries_futures.append(future)
                except ClientError as e:
                    if e.response["Error"]["Code"] == "TooManyRequestsException":
                        logger.info("Athena complained about too many requests. Waiting for a minute.")
                        time.sleep(60)  # wait for a minute before submitting another query
                    elif e.response["Error"]["Code"] == "InvalidRequestException":
                        logger.info(f"Queries[{current_id}] is Invalid: {e.response['Message']} \n {current_query}")
                        to_submit_ids.pop(0)  # query failed, so remove it from the list
                        queries.pop(0)
                        raise
                    else:
                        raise

        query_runner = Thread(target=run_queries)
        query_runner.start()
        return batch_query_id

    def _get_query_result(self, query_id):
        return self.get_athena_query_result(execution_id=query_id)

    @validate_arguments
    def get_athena_query_result(self, execution_id: ExeId, timeout_minutes: int = 30) -> pd.DataFrame:
        """Returns the query result

        Args:
            execution_id (str): Query execution id.
            timeout_minutes (int, optional): Timeout in minutes to wait for query to finish. Defaults to 30.

        Raises:
            QueryException: If query fails for some reason.

        Returns:
            pd.DataFrame: Query result as dataframe.
        """
        t = time.time()
        while time.time() - t < timeout_minutes * 60:
            stat = self.get_query_status(execution_id)
            if stat.upper() == "SUCCEEDED":
                result = self.get_result_from_s3(execution_id)
                self._log_execution_cost(execution_id)
                return result
            elif stat.upper() == "FAILED":
                error = self.get_query_error(execution_id)
                raise QueryException(error)
            else:
                logger.info(f"Query status is {stat}")
                time.sleep(30)

        raise QueryException(f"Query timed-out. {self.get_query_status(execution_id)}")

    @validate_arguments
    def get_result_from_s3(self, query_execution_id: ExeId) -> pd.DataFrame:
        """Returns query result from s3 location.

        Args:
            query_execution_id (str): The query execution ID

        Raises:
            QueryException: If query had failed.

        Returns:
            pd.DataFrame: The query result.
        """
        query_status = self.get_query_status(query_execution_id)
        if query_status == "SUCCEEDED":
            path = self.get_query_output_location(query_execution_id)
            bucket = path.split("/")[2]
            key = "/".join(path.split("/")[3:])
            response = self._aws_s3.get_object(Bucket=bucket, Key=key)
            df = read_csv(response["Body"])
            return df
        # If failed, return error message
        elif query_status == "FAILED":
            raise QueryException(self.get_query_error(query_execution_id))
        elif query_status in ["RUNNING", "QUEUED", "PENDING"]:
            raise QueryException(f"Query still {query_status}")
        else:
            raise QueryException(f"Query has unknown status {query_status}")

    @validate_arguments
    def get_query_output_location(self, query_id: ExeId) -> str:
        """Get query output location in s3.

        Args:
            query_id (str): Query execution id.

        Returns:
            str: The query location in s3.
        """
        stat = self._aws_athena.get_query_execution(QueryExecutionId=query_id)
        output_path = stat["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
        return output_path

    @validate_arguments
    def get_query_status(self, query_id: ExeId) -> str:
        """Get status of the query

        Args:
            query_id (str): Query execution id

        Returns:
            str: Status of the query.
        """
        stat = self._aws_athena.get_query_execution(QueryExecutionId=query_id)
        return stat["QueryExecution"]["Status"]["State"]

    @validate_arguments
    def get_query_error(self, query_id: ExeId) -> str:
        """Returns the error message if query has failed.

        Args:
            query_id (str): Query execution id.

        Returns:
            str: Error message for the query.
        """
        stat = self._aws_athena.get_query_execution(QueryExecutionId=query_id)
        return stat["QueryExecution"]["Status"]["StateChangeReason"]

    def get_all_running_queries(self) -> list[ExeId]:
        """
        Gives the list of all running queries (for this instance)

        Return:
            List of query execution ids of all the queries that are currently running in Athena.
        """
        exe_ids = self._aws_athena.list_query_executions(WorkGroup=self.workgroup)["QueryExecutionIds"]
        exe_ids = [ExeId(i) for i in exe_ids]

        running_ids = [i for i in exe_ids if i in self._execution_ids_history and self.get_query_status(i) == "RUNNING"]
        return running_ids

    def stop_all_queries(self) -> None:
        """
        Stops all queries that are running in Athena for this instance.
        Returns:
            Nothing

        """
        for count, stat in self._batch_query_status_map.items():
            stat["to_submit_ids"].clear()

        running_ids = self.get_all_running_queries()
        for i in running_ids:
            self.stop_query(execution_id=i)

        logger.info(f"Stopped {len(running_ids)} queries")

    @validate_arguments
    def stop_query(self, execution_id: ExeId) -> str:
        """
        Stops a running query.
        Args:
            execution_id: The execution id of the query being run.
        Returns:
        """
        return self._aws_athena.stop_query_execution(QueryExecutionId=execution_id)

    @validate_arguments
    def get_cols(self, table: AnyTableType, fuel_type=None) -> Sequence[DBColType]:
        """
        Returns the columns of for a particular table.
        Args:
            table: Name of the table. One of 'baseline' or 'timeseries'
            fuel_type: Get only the columns for this fuel_type ('electricity', 'gas' etc)

        Returns:
            A list of column names as a list of strings.
        """
        table = self._get_table(table)
        if table == self.ts_table and self.ts_table is not None:
            cols = [c for c in self.ts_table.columns]
            if fuel_type:
                cols = [c for c in cols if c.name not in [self.ts_bldgid_column.name, self.timestamp_column.name]]
                cols = [c for c in cols if fuel_type in c.name]
            return cols
        elif table in ["baseline", "bs"]:
            cols = [c for c in self.bs_table.columns]
            if fuel_type:
                cols = [c for c in cols if "simulation_output_report" in c.name]
                cols = [c for c in cols if fuel_type in c.name]
            return cols
        else:
            tbl = self._get_table(table)
            return [col for col in tbl.columns]

    def _simple_label(self, label: str, agg_func: Optional[Union[Callable, str]] = None):
        label = label.removeprefix(self.db_schema.column_prefix.characteristics)
        label = label.removeprefix(self.db_schema.column_prefix.output)

        if callable(agg_func):
            label += f"__{agg_func.__name__}"
        elif isinstance(agg_func, str) and agg_func != "sum":
            label += f"__{agg_func}"
        return label

    def _get_restrict_clauses(self, restrict, bs_only=False):
        clauses = []
        if bs_only:
            candidate_tables = [tbl for tbl in (self.bs_table, self.up_table) if tbl is not None]
        else:
            candidate_tables = [tbl for tbl in (self.ts_table, self.bs_table, self.up_table) if tbl is not None]

        for col_str, criteria in restrict:
            col = self._get_column(col_str, candidate_tables=candidate_tables)
            if isinstance(criteria, (list, tuple)):
                if len(criteria) > 1:
                    clauses.append(col.in_(criteria))
                elif len(criteria) == 1:
                    clauses.append(col == criteria[0])
                else:
                    raise ValueError(f"Invalid criteria {criteria}")
            else:
                clauses.append(col == criteria)
        return clauses

    def _add_restrict(self, query, restrict, *, bs_only=False):
        if not restrict:
            return query
        restrict_clauses = self._get_restrict_clauses(restrict, bs_only=bs_only)
        query = query.where(*restrict_clauses)
        return query

    def _add_avoid(self, query, avoid, *, bs_only=False):
        if not avoid:
            return query
        if bs_only:
            candidate_tables = [tbl for tbl in (self.bs_table, self.up_table) if tbl is not None]
        else:
            candidate_tables = [tbl for tbl in (self.ts_table, self.bs_table, self.up_table) if tbl is not None]
        where_clauses = []
        for col_str, criteria in avoid:
            col = self._get_column(col_str, candidate_tables=candidate_tables)
            if isinstance(criteria, (list, tuple)):
                if len(criteria) > 1:
                    where_clauses.append(col.not_in(criteria))
                elif len(criteria) == 1:
                    where_clauses.append(col != criteria[0])
                else:
                    raise ValueError(f"Invalid criteria {criteria}")
            else:
                where_clauses.append(col != criteria)
        query = query.where(*where_clauses)
        return query

    def _get_name(self, col):
        if isinstance(col, tuple):
            return col[1]
        if isinstance(col, str):
            return col
        if isinstance(col, (sa.Column, SALabel)):
            return col.name
        raise ValueError(f"Can't get name for {col} of type {type(col)}")

    def _add_join(self, query, join_list):
        for new_table_name, baseline_column_name, new_column_name in join_list:
            new_tbl = self._get_table(new_table_name)
            baseline_column = self._get_column(baseline_column_name, candidate_tables=[self.bs_table])
            new_column = self._get_column(new_column_name, candidate_tables=[new_tbl])
            query = query.join(new_tbl, baseline_column == new_column)
        return query

    def _add_group_by(self, query, group_by_selection):
        if group_by_selection:
            selected_cols = list(query.selected_columns)
            a = [sa.text(str(selected_cols.index(g) + 1)) for g in group_by_selection]
            query = query.group_by(*a)
        return query

    def _add_order_by(self, query, order_by_selection):
        if order_by_selection:
            selected_cols = list(query.selected_columns)
            a = [sa.text(str(selected_cols.index(g) + 1)) for g in order_by_selection]
            query = query.order_by(*a)
        return query

    def _get_weight(self, weights):
        total_weight = self.sample_wt
        for weight_col in weights:
            if isinstance(weight_col, tuple):
                tbl = self._get_table(weight_col[1])
                total_weight *= tbl.c[weight_col[0]]
            else:
                total_weight *= self._get_column(weight_col)
        return total_weight

    def _get_agg_func_and_weight(self, weights, agg_func=None):
        # from: https://trino.io/docs/current/functions.html
        if agg_func is None or agg_func == "sum":
            return safunc.sum, self._get_weight(weights)
        if agg_func == "count":
            return safunc.count, 1
        if agg_func in {"mean", "avg"}:
            return safunc.avg, 1
        if agg_func == "max":
            return safunc.max, 1
        if agg_func == "min":
            return safunc.min, 1
        if agg_func == "arbitrary":
            return safunc.arbitrary, None
        if agg_func == "stddev_pop":
            return safunc.stddev_pop, 1
        if agg_func == "stddev_samp":
            return safunc.stddev_samp, 1
        if agg_func == "var_pop":
            return safunc.var_pop, 1
        if agg_func == "var_samp":
            return safunc.var_samp, 1
        if agg_func == "count_if":
            return safunc.count_if, None
        if agg_func == "array_agg":
            return safunc.array_agg, None
        raise ValueError(f"agg_func {agg_func} is not supported")

    def delete_everything(self):
        """Deletes the athena tables and data in s3 for the run."""
        info = self._aws_glue.get_table(DatabaseName=self.db_name, Name=self.bs_table.name)
        self.pth = pathlib.Path(info["Table"]["StorageDescriptor"]["Location"]).parent
        tables_to_delete = [self.bs_table.name]
        if self.ts_table is not None:
            tables_to_delete.append(self.ts_table.name)
        if self.up_table is not None:
            tables_to_delete.append(self.up_table.name)
        print(f"Will delete the following tables {tables_to_delete} and the {self.pth} folder")
        while True:
            curtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            confirm = input(f"Enter {curtime} to confirm.")
            if confirm == "":
                print("Abandoned the idea.")
                break
            if confirm != curtime:
                print(f"Please pass {curtime} as confirmation to confirm you want to delete everything.")
                continue
            print("Proceeding with delete ...")
            self._aws_glue.batch_delete_table(DatabaseName=self.db_name, TablesToDelete=tables_to_delete)
            print("Deleted the table from athena, now will delete the data in s3")
            s3 = boto3.resource("s3")
            bucket = s3.Bucket(self.pth.parts[1])  # type: ignore
            prefix = str(pathlib.Path(*self.pth.parts[2:]))
            total_files = [file.key for file in bucket.objects.filter(Prefix=prefix)]
            print(f"There are {len(total_files)} files to be deleted. Deleting them now")
            bucket.objects.filter(Prefix=prefix).delete()
            print("Delete from s3 completed")
            break
