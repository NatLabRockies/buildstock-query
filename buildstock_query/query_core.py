import contextlib
import datetime
import json as _json_module
import logging
import os
import pathlib
import re
import time
import typing
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from threading import Thread
from typing import Literal, NewType, Protocol, TypedDict

import boto3
import numpy as np
import pandas as pd
import sqlalchemy as sa
import toml
import urllib3
from botocore.config import Config
from botocore.exceptions import ClientError
from pyathena.connection import Connection
from pyathena.pandas.async_cursor import AsyncPandasCursor
from pyathena.pandas.cursor import PandasCursor
from pyathena.sqlalchemy.base import AthenaDialect
from sqlalchemy.sql import func as safunc
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.schema import Column, Table

from buildstock_query.db_schema.db_schema_model import DBSchema
from buildstock_query.helpers import AthenaFutureDf, CachedFutureDf, CustomCompiler, DataExistsException, read_csv
from buildstock_query.query_filters import QueryFilterMixin
from buildstock_query.schema.run_params import RunParams
from buildstock_query.schema.utilities import (
    ColumnExpression,
    ColumnReference,
    MappedColumn,
    SelectQuery,
    SqlColumn,
    SqlExpression,
    SqlFrom,
    SqlFunction,
    SqlLabel,
    TableHandle,
    TableReference,
    WeightSpec,
    typed_literal,
    validate_arguments,
)
from buildstock_query.sql_cache import SqlCache, hash_sql, normalize_sql

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
    queries_futures: list[CachedFutureDf | AthenaFutureDf]
    max_threads: int | None


class BatchQueryReportMap(TypedDict):
    submitted: int
    running: int
    pending: int
    completed: int
    failed: int


class AsyncDataFrameFuture(Protocol):
    as_df: object


class QueryCore(QueryFilterMixin):
    def __init__(self, *, params: RunParams) -> None:
        """
        Base class to run common Athena queries for BuildStock runs and download results as pandas dataFrame
        Usually, you should just use BuildStockQuery. This class is useuful if you want to extend the functionality
        for Athena tables that are not part of ResStock or ComStock runs.
        Args:
            workgroup (str): The workgroup for athena. The cost will be charged based on workgroup.
            db_name (str): The athena database name
            buildstock_type (str, optional): 'resstock' or 'comstock' runs. Defaults to 'resstock'
            table_name (str or tuple[str, Optional[str]]): If a single string is provided, say, 'mfm_run', then it
                must correspond to tables in athena formed by appending the schema's
                `[table_suffix].annual_and_metadata` and `.timeseries` to it. Or, a tuple `(annual_and_metadata,
                timeseries)` can be provided directly. Timeseries may be None if no such table exists.
            db_schema (str, optional): The database structure in Athena is different between ResStock and ComStock
                run. It is also different between the version in OEDI and default version from BuildStockBatch. This
                argument controls the assumed schema. Allowed values are TOML files in the db_schema/ folder
                (e.g. 'resstock_oedi', 'comstock_oedi_state_and_county').
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

        # pool matches the download_metadata_and_annual_results thread pool (10 workers) with
        # headroom for incidental per-request metadata calls, so parallel downloads don't
        # churn the HTTPS pool and spam "Connection pool is full" warnings.
        self._aws_s3 = boto3.client("s3", config=Config(max_pool_connections=32))
        self._aws_athena = boto3.client("athena", region_name=params.region_name)
        self._aws_glue = boto3.client("glue", region_name=params.region_name)
        self._async_conn = Connection(
            work_group=params.workgroup,
            region_name=params.region_name,
            cursor_class=AsyncPandasCursor,
            schema_name=params.db_name,
            config=Config(max_pool_connections=20),
        )

        self.db_name = params.db_name
        self.region_name = params.region_name

        self._tables: dict[str, Table] = OrderedDict()  # Internal record of tables
        self._meta = sa.MetaData()

        self._batch_query_status_map: dict[int, BatchQueryStatusMap] = {}
        self._batch_query_id = 0
        if isinstance(params.db_schema, dict):
            db_schema_dict = params.db_schema
        else:
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
        self._cache = SqlCache(self.cache_folder)
        self._initialize_tables()
        self._initialize_book_keeping(params.execution_history)

    def _initialize_tables(self) -> None:
        """Load table handles and initialize stable metadata aliases."""
        self.md_table: Table
        self.ts_table: Table | None
        self.md_table, self.ts_table = self._get_tables(self.table_name)
        # `bs_table` is a stable alias of md_table that callers use as the
        # canonical metadata-side handle in outer queries. Keeping a single
        # named alias (not constructing fresh `md.alias(...)` per call) lets
        # things like `self.sample_wt = bs_table.c["weight"]` and
        # `self.md_key_cols = [bs_table.c[k] for k in md_key]` bind once at
        # init time and remain valid in any query that selects through the
        # bs alias. Self-join sites construct an additional `md.alias("up")`
        # locally for the upgrade-side row set.
        self.bs_table: TableHandle = self.md_table.alias("bs")
        # Alt metadata table for the state-aggregate routing path (set
        # by `_get_tables` when the schema declares
        # `table_suffix.annual_and_metadata_state_agg`). Most callers
        # ignore this; only `_pick_metadata_table` and the `_query`
        # path that consumes its output reference it. The shared alias
        # name "bs" matches the primary table so generated SQL is
        # interchangeable when routed.
        self.md_table_state_agg: Table | None = getattr(self, "_md_table_state_agg_raw", None)
        if self.md_table_state_agg is not None:
            self.bs_table_state_agg: TableHandle | None = self.md_table_state_agg.alias("bs")
            self.md_state_agg_key: tuple[str, ...] = tuple(
                self._get_unique_keys("metadata_state_agg")
            )
        else:
            self.bs_table_state_agg = None
            self.md_state_agg_key = ()

        self.md_bldgid_column = self.bs_table.c[self.building_id_column_name]
        if self.ts_table is not None:
            self.timestamp_column = self.ts_table.c[self.timestamp_column_name]
            self.ts_bldgid_column = self.ts_table.c[self.building_id_column_name]

        self.md_key: tuple[str, ...] = tuple(self._get_unique_keys("metadata"))
        self.ts_key: tuple[str, ...] = tuple(self._get_unique_keys("timeseries"))

        self.sample_wt = self._get_sample_weight(self.sample_weight)

    @property
    def md_key_cols(self) -> list[ColumnExpression]:
        return [self.bs_table.c[k] for k in self.md_key]

    @property
    def ts_key_cols(self) -> list[ColumnExpression]:
        if self.ts_table is None:
            raise ValueError("No timeseries table is available.")
        return [self.ts_table.c[k] for k in self.ts_key]

    @staticmethod
    def _unique_columns_by_name(columns: Sequence[ColumnExpression]) -> list[ColumnExpression]:
        """Return columns de-duplicated by SQL label name."""
        unique_columns: list[ColumnExpression] = []
        seen_names = set()
        for column in columns:
            if column.name in seen_names:
                continue
            seen_names.add(column.name)
            unique_columns.append(column)
        return unique_columns

    def _get_unique_keys(
        self, kind: Literal["metadata", "timeseries", "metadata_state_agg"]
    ) -> list[str]:
        """Return configured unique-key columns for a schema table kind."""
        # When routing is active (`_routing_context` swapped self.md_table
        # to the alt), `kind="metadata"` should return the alt-table's
        # narrower key. Detect by table identity rather than a separate
        # flag — this keeps the routing visibility consistent with how
        # other helpers detect it (via `self.bs_table` / `self.md_table`).
        if (
            kind == "metadata"
            and getattr(self, "md_table_state_agg", None) is not None
            and self.md_table is self.md_table_state_agg
        ):
            kind = "metadata_state_agg"
        configured_keys = getattr(self.db_schema.unique_keys, kind, None)
        return configured_keys or [self.building_id_column_name]

    @contextlib.contextmanager
    def _routing_context(self, md_choice: Literal["primary", "state_agg"]) -> typing.Iterator[None]:
        """Temporarily route metadata helpers to the selected metadata table."""
        state_md_table = self.md_table_state_agg
        state_bs_table = self.bs_table_state_agg
        if md_choice == "primary" or state_md_table is None or state_bs_table is None:
            yield
            return
        # Save originals.
        prev_md_table = self.md_table
        prev_bs_table = self.bs_table
        prev_md_key = self.md_key
        prev_sample_wt = self.sample_wt
        prev_md_bldgid = self.md_bldgid_column
        # Swap to alt.
        self.md_table = state_md_table
        self.bs_table = state_bs_table
        self.md_key = self.md_state_agg_key
        # Re-derive bound expressions on the alt alias.
        self.sample_wt = self._get_sample_weight(self.sample_weight)
        self.md_bldgid_column = self.bs_table.c[self.building_id_column_name]
        try:
            yield
        finally:
            self.md_table = prev_md_table
            self.bs_table = prev_bs_table
            self.md_key = prev_md_key
            self.sample_wt = prev_sample_wt
            self.md_bldgid_column = prev_md_bldgid

    def _join_condition(
        self,
        left_table: TableReference,
        right_table: TableReference,
        kind: Literal["metadata", "timeseries"],
        extra_keys: Sequence[str] = (),
    ) -> ColumnElement:
        """Return equality predicates for shared unique-key columns."""
        keys = list(dict.fromkeys([*self._get_unique_keys(kind), *extra_keys]))
        left = self._get_table(left_table)
        right = self._get_table(right_table)
        return sa.and_(*(left.c[key] == right.c[key] for key in keys))

    def _baseline_timeseries_join_condition(
        self,
        metadata_baseline: TableReference,
        timeseries_table: TableReference,
    ) -> ColumnElement:
        """Return the metadata-to-timeseries baseline join predicate."""
        return sa.and_(
            self._join_condition(metadata_baseline, timeseries_table, "timeseries"),
            self._upgrade_zero_filter(metadata_baseline),
        )

    def _baseline_upgrade_join_condition(
        self,
        baseline_side: TableReference,
        upgrade_side: TableReference,
    ) -> ColumnElement:
        """Return the metadata baseline-to-upgrade self-join predicate."""
        return sa.and_(
            self._join_condition(baseline_side, upgrade_side, "metadata"),
            self._upgrade_zero_filter(baseline_side),
        )

    def _upgrade_zero_filter(self, table: TableReference) -> ColumnElement:
        """Return the predicate that selects baseline upgrade rows."""
        upgrade_col = self._get_table(table).c["upgrade"]
        return upgrade_col == typed_literal(upgrade_col, "0")

    def _md_baseline_filter(self, table: TableReference | None = None) -> ColumnElement:
        """Return the baseline predicate for the active metadata alias."""
        return self._upgrade_zero_filter(table if table is not None else self.bs_table)

    def _timeseries_pair_join_condition(
        self,
        left_timeseries_table: TableReference,
        right_timeseries_table: TableReference,
    ) -> ColumnElement:
        """Return the timeseries-to-timeseries join predicate."""
        return self._join_condition(
            left_timeseries_table,
            right_timeseries_table,
            "timeseries",
            [self.timestamp_column_name],
        )

    @staticmethod
    def _count_distinct(columns: Sequence[ColumnExpression]) -> ColumnElement:
        """Return a count-distinct expression for one or more columns."""
        if len(columns) == 1:
            return safunc.count(safunc.distinct(columns[0]))
        return safunc.count(sa.distinct(sa.tuple_(*columns)))

    @staticmethod
    def _scalar_or_tuple(row: Sequence[object]) -> object:
        """Return a scalar for one-column rows, otherwise a tuple."""
        return row[0] if len(row) == 1 else tuple(row)

    def _get_sample_weight(self, sample_weight: str | int | float | None) -> SqlExpression:
        """Return the sample-weight expression configured for this run."""
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
    def _get_table(self, table_name: TableHandle, missing_ok: bool = False) -> TableHandle: ...

    @typing.overload
    def _get_table(self, table_name: str, missing_ok: Literal[True]) -> Table | None: ...

    @typing.overload
    def _get_table(self, table_name: str, missing_ok: Literal[False] = False) -> Table: ...

    @validate_arguments
    def _get_table(self, table_name: TableReference, missing_ok: bool = False) -> TableHandle | None:
        """Resolve a table name to a SQLAlchemy table handle."""
        if not isinstance(table_name, str):
            return table_name  # already a table

        try:
            return self._tables.setdefault(table_name, Table(table_name, self._meta, autoload_with=self._engine))
        except sa.exc.NoSuchTableError:  # type: ignore
            if missing_ok:
                logger.warning(f"No {table_name} table is present.")
                return None
            else:
                raise

    @validate_arguments
    def _get_column(
        self, column_name: ColumnReference,
        candidate_tables: Sequence[TableReference | None] | None = None,
        annual_only: bool = False,
    ) -> ColumnExpression:
        """Resolve a user column reference against candidate query tables."""
        if isinstance(column_name, SqlColumn):
            return column_name.label(self._simple_label(column_name.name))  # already a col

        if isinstance(column_name, SqlLabel):
            return column_name

        if isinstance(column_name, MappedColumn):
            return sa.literal(column_name).label(self._simple_label(column_name.name))

        if not candidate_tables:
            # Resolve annual columns against the metadata alias that appears in
            # aggregate FROM clauses. Timeseries queries may also bind to the
            # physical timeseries table.
            if annual_only:
                candidate_tables = (self.bs_table,)
            else:
                candidate_tables = (self.bs_table, self.ts_table)

        search_tables = [self._get_table(table) for table in candidate_tables if table is not None]
        char_prefix = self.db_schema.column_prefix.characteristics
        names_to_try = [column_name]
        if column_name.startswith(char_prefix):
            names_to_try.append(column_name.removeprefix(char_prefix))
        else:
            names_to_try.append(f"{char_prefix}{column_name}")

        for attempt_name in names_to_try:
            valid_tables = [tbl for tbl in search_tables if attempt_name in tbl.columns]
            if valid_tables:
                if len(valid_tables) > 1:
                    table_names = [getattr(table, "name", "<anonymous>") for table in valid_tables]
                    logger.warning(
                        f"Column {attempt_name} found in multiple tables {table_names}. "
                        f"Using {getattr(valid_tables[0], 'name', '<anonymous>')}"
                    )
                return valid_tables[0].c[attempt_name]
        table_names = [getattr(table, "name", "<anonymous>") for table in search_tables]
        raise ValueError(
            f"Column {column_name} not found in any tables {table_names} "
            f"(also tried {names_to_try[1]!r})"
        )

    def _get_subquery_table(
        self, source_table: TableHandle, where_clause: ColumnElement, alias_name: str
    ) -> TableHandle:
        """Return a named subquery preserving the source table columns."""
        raw_subquery = sa.select("*").select_from(source_table).where(where_clause)
        return sa.text(self._compile(raw_subquery)).columns(*source_table.c).subquery(alias_name)

    def _get_tables(self, table_name: str | tuple[str, str | None]) -> tuple[Table, Table | None]:
        """Resolve the underlying physical tables for this run.

        Always returns `(md_table, ts_table)`. When the schema declares
        `table_suffix.annual_and_metadata_state_agg`, the alt metadata
        table is also loaded and stored at `self.md_table_state_agg`
        for the routing-aware path; otherwise that attribute is None.

        For tuple `table_name`, the entries are
        `(annual_and_metadata, timeseries)`. The alt table can't be
        named via the tuple form; only the suffix path supports it.
        """
        self._engine = self._create_athena_engine(
            region_name=self.region_name, database=self.db_name, workgroup=self.workgroup
        )

        suffix = self.db_schema.table_suffix

        if isinstance(table_name, str):
            md_table_name = f"{table_name}{suffix.annual_and_metadata}"
            ts_table_name = f"{table_name}{suffix.timeseries}"
            md_state_agg_table_name = (
                f"{table_name}{suffix.annual_and_metadata_state_agg}"
                if suffix.annual_and_metadata_state_agg else None
            )
        else:
            md_table_name = table_name[0]
            ts_table_name = table_name[1] if len(table_name) > 1 else None
            md_state_agg_table_name = None

        md_table = self._get_table(md_table_name)
        ts_table = self._get_table(ts_table_name, missing_ok=True) if ts_table_name else None
        # Stash the alt table on the instance — _initialize_tables will
        # turn it into a stable alias. Done here so the network round-
        # trip is in the same place as the other autoload calls.
        self._md_table_state_agg_raw = (
            self._get_table(md_state_agg_table_name) if md_state_agg_table_name else None
        )

        return md_table, ts_table

    def _initialize_book_keeping(self, execution_history: str | pathlib.Path | None) -> None:
        """Initialize execution-history tracking for this query session."""
        self._execution_history_file = execution_history or self.cache_folder / ".execution_history"
        self.execution_cost = {"GB": 0, "Dollars": 0}  # Tracks the cost of current session. Only used for Athena query
        self.seen_execution_ids = set()  # set to prevent double counting same execution id
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
    def _execution_ids_history(self) -> list[ExeId]:
        """Return recent Athena execution ids tracked by this instance."""
        exe_ids: list[ExeId] = []
        if os.path.exists(self._execution_history_file):
            with open(self._execution_history_file) as f:
                for line in f:
                    _, exe_id = line.split(",")
                    exe_ids.append(ExeId(exe_id.strip()))
        return exe_ids

    def _create_athena_engine(self, region_name: str, database: str, workgroup: str) -> sa.engine.Engine:
        """Create the SQLAlchemy Athena engine for this run."""
        connect_args = {"cursor_class": PandasCursor, "work_group": workgroup}
        engine = sa.create_engine(
            f"awsathena+rest://:@athena.{region_name}.amazonaws.com:443/{database}", connect_args=connect_args
        )
        return engine

    @validate_arguments
    def delete_table(self, table_name: str) -> str:
        """Delete one Athena table by name."""
        delete_table_query = f"""DROP TABLE {self.db_name}.{table_name};"""
        result, reason = self.execute_raw(delete_table_query)
        if result.upper() == "SUCCEEDED":
            return "SUCCEEDED"
        else:
            raise QueryException(f"Deleting it failed. Reason: {reason}")

    @validate_arguments
    def add_table(
        self, table_name: str, table_df: pd.DataFrame, s3_bucket: str, s3_prefix: str, override: bool = False
    ) -> str:
        """Upload a dataframe to S3 and register it as an Athena table."""
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
    def execute_raw(self, query: str, db: str | None = None, run_async: bool = False) -> tuple[str, str] | ExeId:
        """Execute raw SQL through Athena and return status or execution id."""
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

    def _save_execution_id(self, execution_id: ExeId) -> None:
        """Append one Athena execution id to the local history file."""
        with open(self._execution_history_file, "a") as f:
            f.write(f"{time.time()},{execution_id}\n")

    def _log_execution_cost(self, execution_id: ExeId, sql: str | None = None) -> None:
        """Log Athena execution cost and cache metadata sidecars."""
        if execution_id == "CACHED":
            # Can't log cost for cached query
            return
        res = self._aws_athena.get_query_execution(QueryExecutionId=execution_id)
        qe = res["QueryExecution"]
        stats = qe["Statistics"]
        scanned_GB = stats["DataScannedInBytes"] / 1e9
        cost = scanned_GB * 5 / 1e3  # 5$ per TB scanned
        if execution_id not in self.seen_execution_ids:
            self.execution_cost["Dollars"] += cost
            self.execution_cost["GB"] += scanned_GB
            self.seen_execution_ids.add(execution_id)

        # Persist the full QueryExecution dict alongside the query result so
        # future analyses can pull whatever Athena reports without re-fetching.
        # Keyed by the same hash as the .sql / .parquet sidecars.
        if sql is not None:
            self._cache.put_metadata(sql, qe)

        logger.info(
            f"{execution_id} cost {scanned_GB:.2f} GB (${cost:.2f}). Session total:"
            f" {self.execution_cost['GB']:.2f} GB (${self.execution_cost['Dollars']:.2f})"
        )

    _UNLOAD_RE = re.compile(r"^\s*UNLOAD\s*\((.*)\)\s*TO\s*'", re.DOTALL | re.IGNORECASE)

    def build_query_metadata_index(self) -> dict[str, dict]:
        """Walk this workgroup's Athena history once and return
        `{hash_sql(inner_select): full_QueryExecution_dict}`.

        For each historical query that's an UNLOAD wrapping a SELECT, the
        index records the EARLIEST non-cache-hit successful execution
        (DataScannedInBytes > 0). The full QueryExecution dict is preserved
        so callers can pull whatever Athena reports — Statistics,
        EngineVersion, ResultReuseInformation, etc. Athena history retains
        ~45 days; older snapshots have no entry.
        """
        # hash → (submitted_ts, full_qe_dict). On a duplicate hash, keep the
        # earliest submitted_ts so we capture the cold-cache cost.
        index: dict[str, tuple[float, dict]] = {}
        paginator = self._aws_athena.get_paginator("list_query_executions")
        for page in paginator.paginate(WorkGroup=self.workgroup):
            ids = page.get("QueryExecutionIds", [])
            if not ids:
                continue
            for chunk_start in range(0, len(ids), 50):
                chunk = ids[chunk_start:chunk_start + 50]
                resp = self._aws_athena.batch_get_query_execution(QueryExecutionIds=chunk)
                for qe in resp.get("QueryExecutions", []):
                    query_text = qe.get("Query", "")
                    if "bsq_athena_unload_results" not in query_text:
                        continue
                    m = self._UNLOAD_RE.match(query_text)
                    if not m:
                        continue
                    status = qe.get("Status", {})
                    if status.get("State") != "SUCCEEDED":
                        continue
                    stats = qe.get("Statistics", {})
                    scanned_bytes = stats.get("DataScannedInBytes") or 0
                    if scanned_bytes <= 0:
                        continue
                    submitted = status.get("SubmissionDateTime")
                    if submitted is None:
                        continue
                    submitted_ts = submitted.timestamp()
                    key = hash_sql(m.group(1))
                    existing = index.get(key)
                    if existing is None or submitted_ts < existing[0]:
                        index[key] = (submitted_ts, qe)
        return {k: v[1] for k, v in index.items()}

    def backfill_cache_metadata(self, cache_dir: pathlib.Path | str | None = None) -> tuple[int, int]:
        """Walk Athena history once and write `<hash>.json` sidecars into the
        cache directory for any cached SQL that doesn't have one yet.

        `cache_dir` defaults to this BSQ's cache folder. The lookup matches
        a cached SQL by its hash against the inner-SELECT hash of every
        historical UNLOAD execution — so it works whether the cached SQL
        was originally executed in this session or weeks ago.

        Returns (filled, skipped) — count of metadata files written and
        cached entries whose execution wasn't found in history.
        """
        cache_root = pathlib.Path(cache_dir) if cache_dir else self._cache.root
        # Find cached entries lacking metadata
        missing_hashes: list[str] = []
        for parquet in cache_root.glob("*.parquet"):
            h = parquet.stem
            if not (cache_root / f"{h}.json").exists():
                missing_hashes.append(h)
        if not missing_hashes:
            return 0, 0
        index = self.build_query_metadata_index()
        filled = 0
        skipped = 0
        for h in missing_hashes:
            qe = index.get(h)
            if qe is None:
                skipped += 1
                continue
            (cache_root / f"{h}.json").write_text(_json_module.dumps(qe, indent=2, default=str))
            filled += 1
        return filled, skipped

    def get_query_cost_from_history(self, sql: str) -> dict | None:
        """Look up Athena execution metadata for a single SQL by walking
        history. Returns the full QueryExecution dict (or None if not found).

        Convenience wrapper for one-shot lookups. To backfill many snapshots,
        prefer `backfill_cache_metadata()` or `build_query_metadata_index()`
        — those walk history once instead of N times.
        """
        target_hash = hash_sql(sql)
        index = self.build_query_metadata_index()
        return index.get(target_hash)

    def _compile(self, query: object) -> str:
        """Compile a SQLAlchemy query to normalized Athena SQL."""
        compiled_query = CustomCompiler(AthenaDialect(), query).process(query, literal_binds=True)
        # Normalize whitespace at compile time so every consumer sees the same
        # canonical form: cache filename hash, S3 unload-path hash, snapshot
        # `<hash>.sql` content, and Athena query history all match for the
        # same logical SQL. Without this, cache lookups by literal SQL string
        # would miss across whitespace variations, and history-search helpers
        # would need to re-normalize on the way in.
        return normalize_sql(compiled_query)

    def _get_unload_result(self, execution_id: ExeId, result_location: str) -> pd.DataFrame:
        """Wait for an UNLOAD query and read its parquet result."""
        t = time.time()
        tick = 0
        timeout_minutes = 30
        while time.time() - t < timeout_minutes * 60:
            stat = self.get_query_status(execution_id)
            if (
                stat.upper() == "SUCCEEDED"
                or (stat.upper() == "FAILED"
                and "HIVE_PATH_ALREADY_EXISTS" in self.get_query_error(execution_id))
            ):
                try:
                    df = pd.read_parquet(result_location)
                except FileNotFoundError:  # empty result
                    # SUCCEEDED + empty destination = UNLOAD wrote zero files, result is genuinely empty.
                    # Drop an _EMPTY sentinel so future runs can recognize this as a cache hit instead of
                    # re-executing the (possibly expensive) query.
                    self._write_empty_marker(result_location)
                    df = pd.DataFrame()
                return df
            elif stat.upper() in ["FAILED", "CANCELLED"]:
                error = self.get_query_error(execution_id)
                raise QueryException(error)
            else:
                tick += 1
                if tick >= 30:
                    logger.info(f"Query is {stat}")
                    tick = 0
                time.sleep(1)
        raise TimeoutError("Query failed to complete within 30 mins.")

    _EMPTY_MARKER_KEY = "_EMPTY"

    def _write_empty_marker(self, result_location: str) -> None:
        """Write a 0-byte _EMPTY sentinel inside `result_location` to cache a zero-row UNLOAD."""
        if not result_location.startswith("s3://"):
            return
        bucket_name, prefix = result_location.replace("s3://", "").split("/", 1)
        marker_key = prefix.rstrip("/") + "/" + self._EMPTY_MARKER_KEY
        try:
            self._aws_s3.put_object(Bucket=bucket_name, Key=marker_key, Body=b"")
        except ClientError as e:
            logger.warning("Could not write _EMPTY marker to %s: %s", result_location, e)

    def _get_query_result_location(self, result_path: str) -> str | None:
        """Check if the UNLOAD result already exists in S3.

        Args:
            result_path (str): The S3 path where the UNLOAD result would be stored.
        Returns:
            Optional[str]: The S3 path to the result if it exists, otherwise None. When the
                cached result is a zero-row UNLOAD, returns a path ending in "/<folder>/_EMPTY"
                (caller must recognize this sentinel and return an empty DataFrame without
                calling read_parquet).
        """
        bucket_name, prefix = result_path.replace("s3://", "").split("/", 1)
        normalized_prefix = prefix.rstrip("/") + "/"
        try:
            paginator = self._aws_s3.get_paginator("list_objects_v2")
            folders: dict[str, datetime.datetime] = {}
            empty_folders: set[str] = set()
            for page in paginator.paginate(Bucket=bucket_name, Prefix=normalized_prefix):
                for obj in page.get("Contents", []):
                    key = obj.get("Key", "")
                    if not key.startswith(normalized_prefix):
                        continue
                    remainder = key[len(normalized_prefix):]
                    if not remainder or "/" not in remainder:
                        continue

                    folder, _, basename = remainder.partition("/")
                    if not folder:
                        continue
                    if basename == self._EMPTY_MARKER_KEY:
                        empty_folders.add(folder)
                        continue
                    last_modified = obj.get("LastModified")
                    if last_modified is None:
                        continue
                    current = folders.get(folder)
                    if current is None or last_modified > current:
                        folders[folder] = last_modified

            if folders:
                chosen_folder = max(folders.items(), key=lambda item: (item[1], item[0]))[0]
                if len(folders) > 1:
                    logger.warning(
                        "Multiple cached UNLOAD result folders found for prefix %s; using newest folder %s.",
                        normalized_prefix,
                        chosen_folder,
                    )
                return f"s3://{bucket_name}/{normalized_prefix}{chosen_folder}/"
            if empty_folders:
                chosen_folder = sorted(empty_folders)[0]
                return f"s3://{bucket_name}/{normalized_prefix}{chosen_folder}/{self._EMPTY_MARKER_KEY}"
            return None
        except ClientError as e:
            logger.error(f"Error accessing S3: {e}")
            return None

    @typing.overload
    def execute(self, query: str | SelectQuery, *, run_async: Literal[False] = False) -> pd.DataFrame: ...

    @typing.overload
    def execute(
        self,
        query: str | SelectQuery,
        *,
        run_async: Literal[True],
    ) -> tuple[Literal["CACHED"], CachedFutureDf] | tuple[ExeId, AthenaFutureDf]: ...

    @validate_arguments
    def execute(
        self, query: str | SelectQuery, run_async: bool = False,
    ) -> pd.DataFrame | tuple[Literal["CACHED"], CachedFutureDf] | tuple[ExeId, AthenaFutureDf]:
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

        cached = self._cache.get(query)
        if cached is not None:
            if run_async:
                return "CACHED", CachedFutureDf(cached)
            return cached

        # `query` here is already whitespace-normalized (see `_compile`), so
        # `hash_sql` and `sha256(query.encode())` are equivalent. The S3 unload
        # path embeds this hash, which is the same as the snapshot cache
        # `<hash>.sql` filename — letting the cost-history helper find a query's
        # past Athena execution by substring-searching history for `/<hash>/`.
        query_hash = hash_sql(query)
        result_path = f"s3://{self.run_params.query_unload_s3_bucket}/bsq_athena_unload_results/{query_hash}"
        # check if result already exists in s3
        if (result_location := self._get_query_result_location(result_path)):
            if result_location.endswith("/" + self._EMPTY_MARKER_KEY):
                df = pd.DataFrame()
            else:
                df = pd.read_parquet(result_location)
            self._cache.put(query, df)
            if run_async:
                return "CACHED", CachedFutureDf(df.copy())
            return df.copy()
        else:
            result_location = f"{result_path}/{uuid.uuid4()}/"  # unique path to avoid collision

        if not query.startswith("UNLOAD"):
            unload_query = (
                f"UNLOAD ({query}) \n TO '{result_location}' \n WITH (format = 'PARQUET')"
            )
        else:
            unload_query = query

        exe_id, result_future = self._async_conn.cursor().execute(
            unload_query, result_reuse_enable=self.athena_query_reuse, result_reuse_minutes=60 * 24 * 7, na_values=[""]
        )
        exe_id = ExeId(exe_id)

        def load_async_dataframe() -> pd.DataFrame:
            """Load or compute the async UNLOAD dataframe result."""
            cached_inner = self._cache.get(query)
            if cached_inner is not None:
                return cached_inner
            df_inner = self._get_unload_result(exe_id, result_location)
            self._cache.put(query, df_inner)
            self._log_execution_cost(exe_id, sql=query)
            return df_inner.copy()

        if run_async:
            typing.cast(AsyncDataFrameFuture, result_future).as_df = load_async_dataframe
            self._save_execution_id(exe_id)
            return exe_id, AthenaFutureDf(result_future)

        df = self._get_unload_result(exe_id, result_location)
        self._cache.put(query, df)
        self._log_execution_cost(exe_id, sql=query)
        return df.copy()

    def print_all_batch_query_status(self) -> None:
        """Prints the status of all batch queries."""
        for count in self._batch_query_status_map:
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
                if exe_id == "CACHED":
                    continue
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
        for exe_id, query in zip(failed_ids, failed_queries, strict=True):
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
        for exe_id in self._batch_query_status_map[batch_id]["submitted_execution_ids"]:
            if exe_id == "CACHED":
                continue
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
                query_error = self.get_query_error(exe_id)
                if "HIVE_PATH_ALREADY_EXISTS" in query_error:
                    # consider it a success - we will read the existing data
                    success_count += 1
                else:
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
    def did_batch_query_complete(self, batch_id: int) -> bool:
        """Return true when a batch has no pending or running queries."""
        status = self.get_batch_query_report(batch_id)
        return status["pending"] == 0 and status["running"] == 0

    @validate_arguments
    def wait_for_batch_query(self, batch_id: int) -> None:
        """Block until all queries in a batch have completed."""
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
    def get_batch_query_result(
        self, batch_id: int, *, combine: bool = True, no_block: bool = False
    ) -> pd.DataFrame | list[pd.DataFrame]:
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
            original_max_threads = self._batch_query_status_map.get(batch_id, {}).get("max_threads")
            new_batch_id = self.submit_batch_query(failed_queries, max_threads=original_max_threads)
            new_exe_ids = self._batch_query_status_map[new_batch_id]["submitted_execution_ids"]

            self.wait_for_batch_query(new_batch_id)
            new_exe_ids_map = dict(zip(failed_ids, new_exe_ids, strict=True))

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
        submitted_queries = self._batch_query_status_map[batch_id]["submitted_queries"]
        res_df_array: list[pd.DataFrame] = []
        for index, exe_id in enumerate(query_exe_ids):
            df = query_futures[index].as_pandas()
            if combine and len(df) > 0:
                df["query_id"] = index
            logger.info(f"Got result from Query [{index}] ({exe_id})")
            self._log_execution_cost(exe_id, sql=submitted_queries[index])
            res_df_array.append(df)
        if not combine:
            return res_df_array
        logger.info("Concatenating the results.")
        # return res_df_array
        return pd.concat(res_df_array)

    @validate_arguments
    def submit_batch_query(self, queries: Sequence[str], *, max_threads: int | None = None) -> int:
        """
        Submit multiple related queries
        Args:
            queries: List of queries to submit. Setting `get_query_only` flag while making calls to aggregation
                    functions is easiest way to obtain queries.
            max_threads: Maximum number of queries to have running concurrently. Defaults to None (no limit).
        Returns:
            An integer representing the batch_query id. The id can be used with other batch_query functions.
        """
        queries = list(queries)
        if max_threads is not None and max_threads < 1:
            raise ValueError("max_threads must be a positive integer.")
        max_threads = max_threads or len(queries)
        to_submit_ids = list(range(len(queries)))
        id_list = list(to_submit_ids)  # make a copy
        submitted_ids: list[int] = []
        submitted_execution_ids: list[ExeId] = []
        submitted_queries: list[str] = []
        queries_futures: list[CachedFutureDf | AthenaFutureDf] = []
        self._batch_query_id += 1
        batch_query_id = self._batch_query_id
        self._batch_query_status_map[batch_query_id] = {
            "to_submit_ids": to_submit_ids,
            "all_ids": list(id_list),
            "submitted_ids": submitted_ids,
            "submitted_execution_ids": submitted_execution_ids,
            "submitted_queries": submitted_queries,
            "queries_futures": queries_futures,
            "max_threads": max_threads,
        }

        def running_queries_count() -> int:
            return sum(1 for future in queries_futures if not future.done())

        def run_queries() -> None:
            """Submit queued queries while respecting the max-thread limit."""
            while to_submit_ids:
                while running_queries_count() >= max_threads:
                    time.sleep(5)
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

    def _get_query_result(self, query_id: ExeId) -> pd.DataFrame:
        """Return the dataframe result for one Athena query id."""
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
        tick = 0
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
                tick += 1
                if tick >= 30:
                    logger.info(f"Query is {stat}")
                    tick = 0
                time.sleep(1)

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
            full_path = f"s3://{bucket}/{key}/"
            df = pd.read_parquet(full_path)
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
        for stat in self._batch_query_status_map.values():
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
    def get_cols(self, table: TableReference, fuel_type: str | None = None) -> Sequence[ColumnExpression]:
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
            cols = list(self.ts_table.columns)
            if fuel_type:
                cols = [c for c in cols if c.name not in [self.ts_bldgid_column.name, self.timestamp_column.name]]
                cols = [c for c in cols if fuel_type in c.name]
            return cols
        elif table in ["baseline", "bs", "metadata", "md"]:
            cols = list(self.md_table.columns)
            if fuel_type:
                cols = [c for c in cols if "simulation_output_report" in c.name]
                cols = [c for c in cols if fuel_type in c.name]
            return cols
        else:
            tbl = self._get_table(table)
            return list(tbl.columns)

    def _simple_label(self, label: str, agg_func: str | None = None) -> str:
        """Return a display label stripped of configured column prefixes."""
        if not self.run_params.keep_column_prefix:
            label = label.removeprefix(self.db_schema.column_prefix.characteristics)
            label = label.removeprefix(self.db_schema.column_prefix.output)
        if agg_func and agg_func != "sum":
            label += f"__{agg_func}"
        return label

    def _get_name(self, col: object) -> str:
        """Return the output name for a group-by or enduse reference."""
        if isinstance(col, tuple) and len(col) > 1 and isinstance(col[1], str):
            return col[1]
        if isinstance(col, str):
            return col
        if isinstance(col, (Column, SqlLabel)):
            return col.name
        raise ValueError(f"Can't get name for {col} of type {type(col)}")

    def _add_join(
        self,
        query: SelectQuery,
        join_list: Sequence[tuple[TableReference, ColumnReference, ColumnReference]],
        bs_alias: SqlFrom | None = None,
    ) -> SelectQuery:
        """Apply configured joins to a query using the active metadata alias."""
        # `bs_alias` overrides which "bs side" the join's left key resolves
        # against. Defaults to the canonical self.bs_table. TS queries pass
        # `bs_per_bldg` (the per-bldg pre-aggregated subquery that replaces
        # bs in the outer FROM) so the JOIN ON references resolve to the
        # subquery's projected columns rather than the original bs alias
        # (which isn't in the outer FROM after the bs_per_bldg refactor).
        bs_for_join = bs_alias if bs_alias is not None else self.bs_table
        for new_table_name, baseline_column_name, new_column_name in join_list:
            new_tbl = self._get_table(new_table_name)
            # Resolve the bs-side column. baseline_column_name can be a
            # string (column name) or an SA Column. For both we look it up
            # by name on bs_for_join when possible — this lets the
            # bs_per_bldg subquery substitute for the canonical bs alias.
            ref_name = (
                baseline_column_name if isinstance(baseline_column_name, str)
                else getattr(baseline_column_name, "name", None)
            )
            if ref_name and ref_name in bs_for_join.c:
                baseline_column = bs_for_join.c[ref_name]
            else:
                baseline_column = self._get_column(baseline_column_name, candidate_tables=[self.bs_table])
            new_column = self._get_column(new_column_name, candidate_tables=[new_tbl])
            query = query.join(new_tbl, baseline_column == new_column)
        return query

    def _add_group_by(self, query: SelectQuery, group_by_selection: Sequence[SqlExpression]) -> SelectQuery:
        """Apply positional GROUP BY expressions for selected columns."""
        if group_by_selection:
            selected_cols = list(query.selected_columns)
            a = [sa.text(str(selected_cols.index(g) + 1)) for g in group_by_selection]
            query = query.group_by(*a)
        return query

    def _add_order_by(self, query: SelectQuery, order_by_selection: Sequence[SqlExpression]) -> SelectQuery:
        """Apply positional ORDER BY expressions for selected columns."""
        if order_by_selection:
            selected_cols = list(query.selected_columns)
            a = [sa.text(str(selected_cols.index(g) + 1)) for g in order_by_selection]
            query = query.order_by(*a)
        return query

    def _get_weight(self, weights: Sequence[WeightSpec]) -> SqlExpression:
        """Return the multiplicative weight expression for a query."""
        total_weight = self.sample_wt
        for weight_col in weights:
            if isinstance(weight_col, tuple):
                table_ref = typing.cast(TableReference, weight_col[1])
                column_name = typing.cast(str, weight_col[0])
                tbl = self._get_table(table_ref)
                total_weight *= tbl.c[column_name]
            else:
                total_weight *= self._get_column(weight_col, [self.bs_table])
        return total_weight

    def _get_agg_func_and_weight(
        self, weights: Sequence[WeightSpec], agg_func: str | None = None
    ) -> tuple[SqlFunction, SqlExpression | int | None]:
        """Return the SQL aggregate function and weight expression."""
        # from: https://trino.io/docs/current/functions.html
        if agg_func is None or agg_func == "sum":
            return typing.cast(SqlFunction, safunc.sum), self._get_weight(weights)
        if agg_func == "count":
            return typing.cast(SqlFunction, safunc.count), 1
        if agg_func in {"mean", "avg"}:
            return typing.cast(SqlFunction, safunc.avg), 1
        if agg_func == "max":
            return typing.cast(SqlFunction, safunc.max), 1
        if agg_func == "min":
            return typing.cast(SqlFunction, safunc.min), 1
        if agg_func == "arbitrary":
            return typing.cast(SqlFunction, safunc.arbitrary), None
        if agg_func == "stddev_pop":
            return typing.cast(SqlFunction, safunc.stddev_pop), 1
        if agg_func == "stddev_samp":
            return typing.cast(SqlFunction, safunc.stddev_samp), 1
        if agg_func == "var_pop":
            return typing.cast(SqlFunction, safunc.var_pop), 1
        if agg_func == "var_samp":
            return typing.cast(SqlFunction, safunc.var_samp), 1
        if agg_func == "count_if":
            return typing.cast(SqlFunction, safunc.count_if), None
        if agg_func == "array_agg":
            return typing.cast(SqlFunction, safunc.array_agg), None
        raise ValueError(f"agg_func {agg_func} is not supported")

    def delete_everything(self) -> None:
        """Deletes the athena tables and data in s3 for the run."""
        # Metadata aliases expose role names; md_table has the real Athena name.
        info = self._aws_glue.get_table(DatabaseName=self.db_name, Name=self.md_table.name)
        self.pth = pathlib.Path(info["Table"]["StorageDescriptor"]["Location"]).parent
        tables_to_delete = [self.md_table.name]
        if self.ts_table is not None:
            tables_to_delete.append(self.ts_table.name)
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
            bucket = s3.Bucket(self.pth.parts[1])
            prefix = str(pathlib.Path(*self.pth.parts[2:]))
            total_files = [file.key for file in bucket.objects.filter(Prefix=prefix)]
            print(f"There are {len(total_files)} files to be deleted. Deleting them now")
            bucket.objects.filter(Prefix=prefix).delete()
            print("Delete from s3 completed")
            break
