import sqlalchemy as sa
from sqlalchemy.sql import func as safunc
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
from buildstock_query.schema.utilities import DBColType, SALabel, SACol, AnyColType, AnyTableType, RestrictTuple
from buildstock_query.schema.utilities import validate_arguments, typed_literal
from buildstock_query.schema.utilities import MappedColumn
from buildstock_query.schema.query_params import Query

import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from tqdm.auto import tqdm

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
        table_name: Union[str, tuple[str, Optional[str]]],
        db_schema: Optional[str | dict] = None,
        buildstock_type: Literal["resstock", "comstock"] = "resstock",
        sample_weight_override: Optional[Union[int, float]] = None,
        region_name: str = "us-west-2",
        execution_history: Optional[str] = None,
        skip_reports: bool = False,
        athena_query_reuse: bool = True,
        query_unload_s3_bucket: str = "resstock-core",
        cache_folder: str = ".bsq_cache",
    ) -> None:
        """A class to run Athena queries for BuildStock runs and download results as pandas DataFrame.

        Args:
            workgroup (str): The workgroup for athena. The cost will be charged based on workgroup.
            db_name (str): The athena database name
            buildstock_type (str, optional): 'resstock' or 'comstock' runs. Defaults to 'resstock'
            table_name (str or tuple[str, Optional[str]]): If a single string is provided, say, 'mfm_run', it must
            correspond to tables in athena whose names are formed by appending the schema's
            `[table_suffix].annual_and_metadata` and `.timeseries` to it. Or, a tuple `(annual_and_metadata_name,
            timeseries_name)` can be provided to override that derivation. The timeseries entry may be None when no
            timeseries table exists.
            db_schema (str | dict, optional): The database structure in Athena is different between ResStock and
                ComStock run. It is also different between the version in OEDI and default version from
                BuildStockBatch. This argument controls the assumed schema. Allowed values are whatever files exist
                in db_schema folder. Defaults to 'resstock_default' for resstock and 'comstock_default' for comstock.
                Can also pass a dict obtained from parsing the schema file. eg: toml.load("db_schema_file.toml").
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
            query_unload_s3_bucket (str, optional): The s3 bucket to use for unloading athena query results.
                Defaults to 'resstock-core'.
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
            query_unload_s3_bucket=query_unload_s3_bucket,
            cache_folder=cache_folder,
        )
        self._run_params = self.params.get_run_params()
        super(BuildStockQuery, self).__init__(params=self._run_params)
        from buildstock_query.report_query import BuildStockReport
        from buildstock_query.aggregate_query import BuildStockAggregate
        from buildstock_query.utility_query import BuildStockUtility
        #: `buildstock_query.report_query.BuildStockReport` object to perform report queries
        self.report: BuildStockReport = BuildStockReport(self)
        #: `buildstock_query.aggregate_query.BuildStockAggregate` object to perform aggregate queries
        self.agg: BuildStockAggregate = BuildStockAggregate(self)
        #: `buildstock_query.utility_query.BuildStockUtility` object to perform utility queries
        self.utility = BuildStockUtility(self)

        self._char_prefix = self.db_schema.column_prefix.characteristics
        self._out_prefix = self.db_schema.column_prefix.output

        if not skip_reports:
            logger.info("Getting Success counts...")
            print(self.report.get_success_report())
            if self.ts_table is not None:
                self.report.check_ts_bs_integrity()

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
        """Return a dict of {upgrade_id: upgrade_name} for all upgrades in the run.

        The column carrying the human-readable upgrade name is configured per
        schema via `column_names.upgrade_name` in the TOML. Classic schemas
        default to `apply_upgrade.upgrade_name`; OEDI ComStock overrides to
        `in.upgrade_name`. If the configured column doesn't actually exist on
        the upgrade table (e.g. OEDI ResStock, where the names live in run
        config rather than the Athena tables), the name field degrades to NULL
        for every upgrade — the returned dict still has one entry per upgrade
        so downstream iteration keeps working regardless of schema.
        """
        upgrade_col = self.md_table.c["upgrade"]
        upgrade_name_col_name = self.db_schema.column_names.upgrade_name
        has_name_col = upgrade_name_col_name in self.md_table.c
        if has_name_col:
            upgrade_name_col = self.md_table.c[upgrade_name_col_name]
            name_select = safunc.arbitrary(upgrade_name_col).label("upgrade_name")
        else:
            # Schema configures a name column but the upgrade table doesn't
            # actually have it (e.g. OEDI ResStock). Project a literal NULL
            # labeled `upgrade_name` so the result shape stays the same as
            # the classic-schema path.
            name_select = sa.cast(sa.null(), sa.String).label("upgrade_name")
        query = (
            sa.select(
                sa.cast(upgrade_col, sa.Integer).label("upgrade"),
                name_select,
            )
            .select_from(self.md_table)  # explicit FROM matches the column binds
            # Exclude baseline rows from the upgrade names listing. The unified
            # annual_and_metadata table has upgrade=0 baseline rows that pre-2-table-
            # pivot lived on a separate parquet — historical callers expect this
            # method to return upgrades only (1+).
            .where(upgrade_col != typed_literal(upgrade_col, "0"))
            .group_by(sa.literal_column("1"))
            .order_by(sa.literal_column("1"))
        )
        if get_query_only:
            return self._compile(query)
        up_name_dict = self.execute(query).set_index("upgrade").to_dict()["upgrade_name"]
        return up_name_dict

    @typing.overload
    def _get_rows_per_building(self, get_query_only: Literal[False] = False) -> int: ...

    @typing.overload
    def _get_rows_per_building(self, get_query_only: Literal[True]) -> str: ...

    @validate_arguments
    def _get_rows_per_building(self, get_query_only: bool = False) -> Union[int, str]:
        if self.ts_table is None:
            raise ValueError("No timeseries table is available.")
        ts_join_keys = self._get_unique_keys("timeseries")
        group_cols: list = [self.ts_table.c["upgrade"]]
        group_cols.extend(self.ts_table.c[key] for key in ts_join_keys)
        select_cols = [*group_cols, safunc.count().label("row_count")]
        ts_query = sa.select(*select_cols)
        ts_query = ts_query.group_by(*(sa.text(str(i + 1)) for i in range(len(group_cols))))

        if get_query_only:
            return self._compile(ts_query)
        df = self.execute(ts_query)
        if (df["row_count"] == df["row_count"][0]).all():
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
        # Default to the unified metadata table when table_name is None.
        defaulted = table_name is None
        tbl = self.md_table if defaulted else self._get_table(table_name)
        query = sa.select(tbl.c[column]).select_from(tbl).distinct()
        if defaulted:
            # Restrict to baseline rows so the result matches the legacy
            # baseline-only contract.
            query = query.where(tbl.c["upgrade"] == typed_literal(tbl.c["upgrade"], "0"))
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
        # When table_name is None, use the canonical bs_table alias so column
        # references in the SELECT (e.g. self.sample_wt → bs_table.weight)
        # bind to the same table that's in the FROM. Selecting from md_table
        # directly would cause SA to auto-add bs_table as a comma-join (and
        # potentially produce a duplicate `upgrade` column on SELECT *).
        # When the user passes an explicit table_name that's the unified
        # metadata table, route it through the bs_table alias for the same
        # reason — users naturally pass the real Athena table name.
        if table_name is None or self._get_table(table_name) is self.md_table:
            tbl = self.bs_table
        else:
            tbl = self._get_table(table_name)
        # Rebind sample_wt to whichever table is actually in scope. The cached
        # `self.sample_wt` was bound to bs_table at init; if the user passed an
        # auxiliary table that also has a "weight" column, use that one to
        # avoid pulling bs_table into the FROM.
        if isinstance(self.sample_wt, sa.Column) and self.sample_wt.name in tbl.c:
            sample_wt = tbl.c[self.sample_wt.name]
        else:
            sample_wt = self.sample_wt
        query = sa.select(
            tbl.c[column], safunc.sum(1).label("metadata_rows_count"), safunc.sum(sample_wt).label("weighted_count")
        ).select_from(tbl)
        if table_name is None or tbl is self.bs_table:
            # Default-table case (or user-passed-md): restrict to baseline rows
            # so the count matches the legacy baseline-only contract.
            query = query.where(self._md_baseline_filter(tbl))
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
        # Select through the canonical bs_table alias so any restrict column
        # references (resolved via _get_column → bs_table.c[...]) bind to the
        # alias that's in the FROM. Selecting from md_table directly would
        # produce a comma-join + duplicate `upgrade` column in `SELECT *`.
        query = sa.select("*").select_from(self.bs_table).where(self._md_baseline_filter())
        query = self._add_restrict(query, restrict, annual_only=True)
        compiled_query = self._compile(query)
        if get_query_only:
            return compiled_query
        logger.info("Making results_csv query ...")
        return self.execute(query).set_index(list(self.md_key))

    def _s3_list_all(self, bucket: str, prefix: str) -> list[dict]:
        """Return all S3 objects under `prefix` by paginating list_objects_v2."""
        paginator = self._aws_s3.get_paginator("list_objects_v2")
        contents: list[dict] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            contents.extend(page.get("Contents", []))
        return contents

    @staticmethod
    def _upgrade_file_variants(upgrade_id: Union[str, int]) -> list[str]:
        """Return the set of filename tokens that would mark a parquet as belonging to
        the given upgrade.

        For baseline (upgrade 0): baseline.parquet, up00.parquet, up0.parquet, upgrade00.parquet,
            upgrade0.parquet.
        For upgrade N: up{N}.parquet, up{N:02}.parquet, upgrade{N}.parquet, upgrade{N:02}.parquet.
        """
        try:
            num = int(upgrade_id)
        except (TypeError, ValueError):
            num = None
        tokens: list[str] = []
        if num is not None:
            short = str(num)
            padded = f"{num:02d}"
            for p in ("up", "upgrade"):
                tokens.append(f"{p}{short}.parquet")
                if padded != short:
                    tokens.append(f"{p}{padded}.parquet")
            if num == 0:
                tokens.append("baseline.parquet")
        else:
            s = str(upgrade_id)
            for p in ("up", "upgrade"):
                tokens.append(f"{p}{s}.parquet")
        # preserve order while dedup
        return list(dict.fromkeys(tokens))

    def download_metadata_and_annual_results(
        self,
        upgrade_id: Union[str, int] = "0",
        folder: Optional[Union[str, pathlib.Path]] = None,
    ) -> pathlib.Path:
        """Download all annual-results parquet files for a given upgrade from S3.

        The Glue-registered table for metadata lives at `s3://<bucket>/<key>/...`. Many runs
        store their parquet inside Hive-style partition subfolders (e.g. `state=AK/county=.../`),
        each partition holding one parquet per upgrade. This method recursively walks the glue
        location and downloads every parquet whose filename ends with one of the known
        upgrade-specific tokens (see `_upgrade_file_variants`).

        Files already present locally are skipped. Downloads are done via a thread pool
        (size = min(10, N files)). Local layout mirrors the S3 layout under `folder`.

        Args:
            upgrade_id: 0/"0" for baseline, else the upgrade number.
            folder: Destination root; defaults to `cache_folder/metadata_and_annual_results/`.

        Returns:
            The local destination folder (pathlib.Path).
        """
        upgrade_id_str = str(upgrade_id)
        if folder is None:
            folder = self.cache_folder / "metadata_and_annual_results"
        folder = pathlib.Path(folder)
        # nest per-upgrade so baseline (upgrade_id="0") and upgrade-N live in separate subdirs.
        # Callers want to pd.read_parquet(folder) and get only that upgrade's rows.
        upgrade_root = folder / f"upgrade={upgrade_id_str}"

        # The unified annual_and_metadata parquet holds rows for every upgrade,
        # baseline included; the per-upgrade selection happens via the file-name
        # token filter in `_upgrade_file_variants` below.
        if isinstance(self.table_name, str):
            db_table_name = f"{self.table_name}{self.db_schema.table_suffix.annual_and_metadata}"
        else:
            db_table_name = self.table_name[0]

        table_loc = self._aws_glue.get_table(DatabaseName=self.db_name, Name=db_table_name)["Table"][
            "StorageDescriptor"
        ]["Location"]
        bucket = table_loc.split("/")[2]
        key_prefix = "/".join(table_loc.split("/")[3:])
        if not key_prefix.endswith("/"):
            key_prefix += "/"

        tokens = self._upgrade_file_variants(upgrade_id_str)
        contents = self._s3_list_all(bucket, key_prefix)
        if not contents:
            raise ValueError(f"No parquet files found in s3://{bucket}/{key_prefix}")

        def matches(path_key: str) -> bool:
            basename = path_key.rsplit("/", 1)[-1]
            return any(basename.endswith(tok) for tok in tokens)

        matching_keys = [obj["Key"] for obj in contents if matches(obj["Key"])]
        if not matching_keys:
            sample = [obj["Key"] for obj in contents[:10]]
            raise ValueError(
                f"No results parquet matching upgrade={upgrade_id_str} found in s3://{bucket}/{key_prefix}. "
                f"Looked for filenames ending in {tokens}. Example files: {sample}"
            )

        # group-by-directory uniqueness guard: in any single S3 "folder", we should have at
        # most one file per upgrade. Multiple matches in the same folder means ambiguity.
        by_dir: dict[str, list[str]] = {}
        for k in matching_keys:
            d = k.rsplit("/", 1)[0]
            by_dir.setdefault(d, []).append(k)
        ambiguous = {d: ks for d, ks in by_dir.items() if len(ks) > 1}
        if ambiguous:
            raise ValueError(
                f"Multiple parquet files match upgrade={upgrade_id_str} in the same S3 folder: "
                f"{ambiguous}"
            )

        tasks: list[tuple[str, pathlib.Path]] = []
        for k in matching_keys:
            rel = k[len(key_prefix):] if k.startswith(key_prefix) else k
            local_path = upgrade_root / rel
            if local_path.exists():
                continue
            tasks.append((k, local_path))

        total_matches = len(matching_keys)
        already_cached = total_matches - len(tasks)
        if not tasks:
            logger.info(
                f"All {total_matches} parquet file(s) for upgrade={upgrade_id_str} already present locally "
                f"at {upgrade_root}; skipping download."
            )
        else:
            if already_cached:
                logger.info(
                    f"{already_cached}/{total_matches} parquet file(s) for upgrade={upgrade_id_str} already "
                    f"present locally; downloading the remaining {len(tasks)}."
                )
            else:
                logger.info(
                    f"Downloading {len(tasks)} parquet file(s) for upgrade={upgrade_id_str} to {upgrade_root}."
                )
            max_workers = min(10, len(tasks))

            def _download(k_and_path):
                k, local_path = k_and_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self._aws_s3.download_file(bucket, k, str(local_path))
                return local_path

            desc = f"Downloading parquet for upgrade={upgrade_id_str}"
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_download, t) for t in tasks]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="file"):
                    fut.result()

        return upgrade_root

    def _download_results_csv(self) -> pathlib.Path:
        """Download the baseline results parquet(s). See `download_metadata_and_annual_results`."""
        return self.download_metadata_and_annual_results(upgrade_id="0")

    def get_results_csv_full(self) -> pd.DataFrame:
        """Returns the full results csv table. This is the same as get_results_csv without any restrictions. It uses
        the stored parquet files in s3 to download the results which is faster than querying athena.
        Returns:
            pd.DataFrame: The full results csv, indexed by md_key.
        """
        local_copy_path = self._download_results_csv()
        df = pd.read_parquet(local_copy_path)
        index_keys = list(self.md_key)
        if list(df.index.names) != index_keys:
            if df.index.name is not None or any(n is not None for n in df.index.names):
                df = df.reset_index()
            df = df.set_index(index_keys)
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
        # Select through the canonical bs_table alias so restrict columns
        # resolved via _get_column bind to the alias that's in the FROM (not
        # md_table directly, which would produce a comma-join).
        up_col = self.bs_table.c["upgrade"]
        query = sa.select("*").select_from(self.bs_table).where(
            up_col == typed_literal(up_col, upgrade_id)
        )

        rewritten_restrict = []
        for col, vals in restrict:
            if isinstance(col, str) and col in self.bs_table.c:
                rewritten_restrict.append((self.bs_table.c[col], vals))
            else:
                rewritten_restrict.append((col, vals))
        query = self._add_restrict(query, rewritten_restrict, annual_only=True)
        compiled_query = self._compile(query)
        if get_query_only:
            return compiled_query
        logger.info("Making results_csv query for upgrade ...")
        return self.execute(query).set_index(list(self.md_key))

    def _download_upgrades_csv(self, upgrade_id: Union[int, str]) -> pathlib.Path:
        """Download the upgrade-N results parquet(s). See `download_metadata_and_annual_results`."""
        if isinstance(upgrade_id, int):
            upgrade_id = f"{upgrade_id:02}"
        available_upgrades = list(self.get_available_upgrades())
        if "0" in available_upgrades:
            available_upgrades.remove("0")
        if str(upgrade_id) not in available_upgrades:
            raise ValueError(f"Upgrade {upgrade_id} not found")

        return self.download_metadata_and_annual_results(upgrade_id=str(upgrade_id))

    def get_upgrades_csv_full(self, upgrade_id: Union[int, str]) -> pd.DataFrame:
        """Returns the full results csv table for upgrades. This is the same as get_upgrades_csv without any
        restrictions. It uses the stored parquet files in s3 to download the results which is faster than querying
        athena. Indexed by md_key.
        """
        local_copy_path = self._download_upgrades_csv(upgrade_id)
        df = pd.read_parquet(local_copy_path)
        index_keys = list(self.md_key)
        if list(df.index.names) != index_keys:
            if df.index.name is not None or any(n is not None for n in df.index.names):
                df = df.reset_index()
            df = df.set_index(index_keys)
        if "upgrade" not in df.columns:
            df.insert(0, "upgrade", upgrade_id)
        return df

    @typing.overload
    def get_building_ids(
        self,
        *,
        restrict: Sequence[RestrictTuple] = Field(default_factory=list),
        avoid: Sequence[RestrictTuple] = Field(default_factory=list),
        get_query_only: Literal[False] = False,
    ) -> pd.DataFrame: ...

    @typing.overload
    def get_building_ids(
        self,
        *,
        restrict: Sequence[RestrictTuple] = Field(default_factory=list),
        avoid: Sequence[RestrictTuple] = Field(default_factory=list),
        get_query_only: Literal[True],
    ) -> str: ...

    @typing.overload
    def get_building_ids(
        self,
        *,
        restrict: Sequence[RestrictTuple] = Field(default_factory=list),
        avoid: Sequence[RestrictTuple] = Field(default_factory=list),
        get_query_only: bool,
    ) -> Union[pd.DataFrame, str]: ...

    @validate_arguments
    def get_building_ids(
        self,
        restrict: Sequence[RestrictTuple] = Field(default_factory=list),
        avoid: Sequence[RestrictTuple] = Field(default_factory=list),
        get_query_only: bool = False,
    ) -> Union[str, pd.DataFrame]:
        """Return the list of building keys.

        For applied-buildings filtering, compose with `get_applied_buildings_filter`:
            f = bsq.get_applied_buildings_filter(all_of=[1, 2])
            ids = bsq.get_building_ids(restrict=[f] if f else [])
            # Or to get the complement (universe \\ applied set):
            ids = bsq.get_building_ids(avoid=[f] if f else [])

        Args:
            restrict: Standard restrict list. Each entry is either a `(column, value)`
                scalar/list comparison, a `(column, subquery)` IN-subquery, or a
                `(tuple-of-columns, tuple-subquery)` composite-key membership.
            avoid: Same shape as `restrict`, but each entry becomes a NOT-IN /
                inequality predicate. Use to select buildings outside a given
                set (e.g. `avoid=[applied_filter]` returns buildings the
                upgrade did NOT apply to).
            get_query_only: If True, return the SQL string instead of executing.

        Returns:
            DataFrame of building keys (`md_key_cols`).
        """
        restrict = list(restrict) if restrict else []
        avoid = list(avoid) if avoid else []
        # md_table holds rows for every upgrade — filter to baseline so the
        # result is one row per (building × keys), not (building × upgrade × keys).
        query = sa.select(*self.md_key_cols).select_from(self.bs_table).where(self._md_baseline_filter())
        query = self._add_restrict(query, restrict, annual_only=True)
        query = self._add_avoid(query, avoid, annual_only=True)
        if get_query_only:
            return self._compile(query)
        return self.execute(query)

    @typing.overload
    def get_applied_buildings(
        self,
        *,
        any_of: Optional[Sequence[Union[str, int]]] = None,
        all_of: Optional[Sequence[Union[str, int]]] = None,
        get_query_only: Literal[False] = False,
    ) -> pd.DataFrame: ...

    @typing.overload
    def get_applied_buildings(
        self,
        *,
        any_of: Optional[Sequence[Union[str, int]]] = None,
        all_of: Optional[Sequence[Union[str, int]]] = None,
        get_query_only: Literal[True],
    ) -> str: ...

    @typing.overload
    def get_applied_buildings(
        self,
        *,
        any_of: Optional[Sequence[Union[str, int]]] = None,
        all_of: Optional[Sequence[Union[str, int]]] = None,
        get_query_only: bool,
    ) -> Union[pd.DataFrame, str]: ...

    @validate_arguments
    def get_applied_buildings(
        self,
        *,
        any_of: Optional[Sequence[Union[str, int]]] = None,
        all_of: Optional[Sequence[Union[str, int]]] = None,
        get_query_only: bool = False,
    ) -> Union[pd.DataFrame, str]:
        """Return building keys for buildings matching an applied-upgrade predicate.

        - `all_of`: must have applicability=true rows for every listed upgrade.
        - `any_of`: must have applicability=true rows for at least one listed upgrade.
        - Both: AND of the two predicates.
        - At least one of `any_of` or `all_of` must be provided.
        - Passing 0 (baseline) in either list raises ValueError.

        Args:
            any_of: list of upgrade ids — building must have applied to at least one.
            all_of: list of upgrade ids — building must have applied to all listed.
            get_query_only: If True, return the SQL string instead of executing.

        Returns:
            DataFrame of `md_key_cols` for matching buildings.
        """
        if not any_of and not all_of:
            raise ValueError("get_applied_buildings: must provide any_of or all_of")
        select = self._build_applied_subquery(any_of=any_of, all_of=all_of, key_kind="metadata")
        # _build_applied_subquery returns a Select; with at least one list non-empty
        # it cannot be None (empty-list case returns None and is rejected above).
        assert select is not None
        if get_query_only:
            return self._compile(select)
        return self.execute(select)

    def get_applied_buildings_filter(
        self,
        *,
        any_of: Optional[Sequence[Union[str, int]]] = None,
        all_of: Optional[Sequence[Union[str, int]]] = None,
    ) -> Optional[RestrictTuple]:
        """Return a `(cols_or_col, subquery)` tuple to drop into `restrict=[...]` or
        `avoid=[...]`. Returns None when both lists are empty/None.

        Typical use:
            f = bsq.get_applied_buildings_filter(all_of=[1, 2])
            df = bsq.query(..., restrict=[f, ("state", ["CO"])] if f else [("state", ["CO"])])

        See `get_applied_buildings` for predicate semantics.
        """
        select = self._build_applied_subquery(any_of=any_of, all_of=all_of, key_kind="metadata")
        if select is None:
            return None
        return self._make_applied_filter_tuple(select, key_kind="metadata")

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
        ucol = self._ts_upgrade_col
        query1 = (
            sa.select(self.timestamp_column.distinct().label(self.timestamp_column_name))
            .where(self.ts_bldgid_column == bldg_id)
            .where(ucol == typed_literal(ucol, upgrade_id))
            .order_by(self.timestamp_column)
            .limit(2)
        )
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
        self, column: AnyColType, annual_only: bool = False
    ) -> DBColType:  # gcol => group by col
        """Get a DB column for the purpose of grouping."""
        if isinstance(column, sa.Column):
            return column.label(self._simple_label(column.name))

        if isinstance(column, SALabel):
            return column

        if isinstance(column, MappedColumn):
            return sa.literal(column).label(self._simple_label(column.name))

        if isinstance(column, str):
            return self._get_column(column, annual_only=annual_only).label(self._simple_label(column))

        raise ValueError(f"Invalid column name type {column}: {type(column)}")

    def get_calculated_column(self, column_name: str, column_expr: str, table="baseline") -> DBColType:
        """
        Creates a calculated column from an arithmetic expression string.
        Column identifiers in the expression are resolved to SQLAlchemy columns via _get_enduse_cols,
        then the expression is evaluated using Python's eval with operator overloading.
        Supports +, -, *, /, parentheses, and numeric literals.
        Examples: "col1 + col2", "col1 - col2 - col3", "col1 * (col2 + col3)", "out.elec - out.gas * 2"
        Args:
            column_name: The name to label the calculated column.
            column_expr: The arithmetic expression to resolve (e.g. "col1 - col2 + col3").
            table: The table to use for column resolution. One of 'baseline', 'upgrade', or 'timeseries'.
        Returns:
            The calculated column with the specified label.
        """
        if not re.match(r'^[\w\s+\-*/().]+$', column_expr):
            raise ValueError(f"Invalid characters in column expression: {column_expr}")

        # Find all column identifiers, longest first to avoid partial replacement
        identifiers = sorted(
            set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_.]*', column_expr)),
            key=len, reverse=True,
        )

        # Replace all identifiers with placeholders and resolve to SA columns
        namespace: dict = {"__builtins__": {}}
        eval_expr = column_expr
        for idx, ident in enumerate(identifiers):
            placeholder = f"_col_{idx}"
            eval_expr = re.sub(re.escape(ident) + r'(?![\w.])', placeholder, eval_expr)
            namespace[placeholder] = self._get_enduse_cols([ident], table=table)[0]

        resolved_col = eval(eval_expr, namespace)  # noqa: S307
        return resolved_col.label(self._simple_label(column_name))

    def _get_enduse_cols(self, enduses: Sequence[AnyColType], table="baseline") -> Sequence[DBColType]:
        # "baseline" and "upgrade" both resolve to the unified metadata table —
        # the columns are the same; the per-upgrade selection happens via WHERE
        # at the call site. "timeseries" stays distinct. We bind to bs_table
        # (the canonical alias of md_table) so column references in outer
        # aggregation queries pick up the alias that's actually in the FROM.
        tbls_dict = {"baseline": self.bs_table, "upgrade": self.bs_table, "timeseries": self.ts_table}
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
        cols = {y.removeprefix(self._char_prefix) for y in self.md_table.c.keys() if y.startswith(self._char_prefix)}
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
        query = sa.select(self.md_table.c["upgrade"]).select_from(self.md_table).distinct().order_by(sa.text("1"))
        upgrades = self.execute(query)["upgrade"].dropna().map(str).to_list()
        return list(dict.fromkeys(["0", *upgrades]))

    def _validate_upgrade(self, upgrade_id: Union[int, str]) -> str:
        upgrade_id = "0" if upgrade_id in (None, "0") else str(upgrade_id)
        available_upgrades = self.get_available_upgrades() or ["0"]
        if upgrade_id not in set(available_upgrades):
            raise ValueError(
                f"`upgrade_id` = {upgrade_id} is not a valid upgrade.It doesn't exist or have no successful run"
            )
        return str(upgrade_id)

    def _split_restrict(self, restrict):
        # Some cols (e.g. comstock `state`, `upgrade`) live on both md and ts tables.
        # When that happens, restrict BOTH sides — Athena's planner often can't push
        # a ts-side filter back through the bldg_id join to the md scan, so a single-
        # sided filter leaves the metadata subquery scanning the full table.
        #
        # `extra_restrict` holds clauses whose column targets neither md nor ts —
        # typically a join_list table (e.g. `eiaid_weights.eiaid` from the utility
        # methods). These can't ride the inner ts/md join ON-clause because the
        # referenced table isn't in scope yet; they must go to the outer WHERE.
        md_restrict = []
        ts_restrict = []
        extra_restrict = []
        for col, restrict_vals in restrict:
            targets_ts = self._restrict_targets_ts(col)
            targets_md = self._restrict_targets_md(col)
            if targets_ts:
                if isinstance(col, tuple):
                    ts_restrict.append([col, restrict_vals])
                else:
                    col_name = col if isinstance(col, str) else col.name
                    ts_restrict.append([self.ts_table.c[col_name], restrict_vals])
            if targets_md:
                if isinstance(col, tuple):
                    md_restrict.append([col, restrict_vals])
                else:
                    md_restrict.append([self._get_gcol(col, annual_only=True), restrict_vals])
            if not targets_ts and not targets_md:
                extra_restrict.append([col, restrict_vals])
        return md_restrict, ts_restrict, extra_restrict

    def _restrict_targets_ts(self, col: AnyColType) -> bool:
        if self.ts_table is None:
            return False
        if isinstance(col, str):
            return col in self.ts_table.columns
        if isinstance(col, SACol):
            return getattr(col, "table", None) is self.ts_table
        if isinstance(col, SALabel):
            source_col = getattr(col, "element", None)
            return isinstance(source_col, SACol) and getattr(source_col, "table", None) is self.ts_table
        if isinstance(col, tuple) and col:
            return all(
                isinstance(c, SACol) and getattr(c, "table", None) is self.ts_table for c in col
            )
        return False

    def _restrict_targets_md(self, col: AnyColType) -> bool:
        # md_table and bs_table share columns (bs is an alias of md), so check
        # both — restrict columns may be bound to either depending on call site.
        md_handles = (self.md_table, self.bs_table)
        if isinstance(col, str):
            # Try both bare name and prefixed form. Char/output columns on
            # md often carry a prefix (e.g. `in.<name>`, `out.<name>`); a
            # user-supplied bare `<name>` should still classify as md so
            # _split_restrict can route the clause into the inner JOIN /
            # bs_per_bldg WHERE rather than the outer WHERE (which would
            # produce a comma-join against the canonical bs alias).
            if col in self.bs_table.columns:
                return True
            for prefix in (self._char_prefix, self._out_prefix):
                if f"{prefix}{col}" in self.bs_table.columns:
                    return True
            return False
        if isinstance(col, SACol):
            return getattr(col, "table", None) in md_handles
        if isinstance(col, SALabel):
            source_col = getattr(col, "element", None)
            return isinstance(source_col, SACol) and getattr(source_col, "table", None) in md_handles
        if isinstance(col, tuple) and col:
            return all(
                isinstance(c, SACol) and getattr(c, "table", None) in md_handles for c in col
            )
        return False

    def _is_timeseries_upgrade_restrict(self, col: AnyColType) -> bool:
        if self.ts_table is None:
            return False
        if isinstance(col, str):
            return col == "upgrade"

        if getattr(col, "name", None) != "upgrade":
            return False

        if isinstance(col, SACol):
            return getattr(col, "table", None) is self.ts_table

        base_columns = getattr(col, "base_columns", None)
        return bool(base_columns) and self.ts_table.c["upgrade"] in base_columns

    def _validate_timeseries_upgrade_restrict(
        self,
        restrict: Sequence[RestrictTuple],
        *,
        annual_only: bool,
        upgrade_id: str,
    ) -> None:
        if annual_only or upgrade_id == "0":
            return

        for col, _ in restrict:
            if self._is_timeseries_upgrade_restrict(col):
                raise ValueError(
                    "Use `upgrade_id` instead of a `restrict` on the timeseries `upgrade` column "
                    "for upgrade queries."
                )

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
        return [self._get_gcol(entry, annual_only=annual_only) for entry in group_by]

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
        md_key_cols = self.md_key_cols
        # md_table holds every upgrade — restrict to baseline rows so each
        # (key) appears once, not once per upgrade.
        query = sa.select(*md_key_cols).where(self._md_baseline_filter())
        query = query.where(self._get_column(location_col, [self.bs_table]).in_(locations))
        query = self._add_order_by(query, md_key_cols)
        if get_query_only:
            return self._compile(query)
        res = self.execute(query)
        return res

    @property
    def _md_completed_status_col(self):
        return self.bs_table.c[self.db_schema.column_names.completed_status]

    @property
    def _md_successful_condition(self):
        """`md.applicability=true`. No upgrade filter — callers pin
        `md.upgrade=N` explicitly when they want a specific upgrade."""
        col = self._md_completed_status_col
        return col == typed_literal(col, self.db_schema.completion_values.success)

    @property
    def _md_baseline_successful_condition(self):
        """`md.applicability=true AND md.upgrade=0` — combined helper for the
        common case "successful baseline rows", matching the legacy
        `_bs_successful_condition` semantics on the unified table."""
        return sa.and_(self._md_successful_condition, self._md_baseline_filter())

    @property
    def _ts_upgrade_col(self):
        return self.ts_table.c["upgrade"]

    @property
    def _md_upgrade_col(self):
        return self.bs_table.c["upgrade"]

    def _get_completed_status_col(self, table: AnyTableType):
        return table.c[self.db_schema.column_names.completed_status]

    def _get_success_condition(self, table: AnyTableType):
        col = self._get_completed_status_col(table)
        return col == typed_literal(col, self.db_schema.completion_values.success)

    @property
    def _state_agg_columns(self) -> set[str]:
        """Names of columns physically present on the alt metadata table.
        Empty set when the schema declares no alt table. Used by
        `_pick_metadata_table` to decide whether routing is safe — if any
        group_by or restrict column is absent here, we must scan the
        primary table instead.
        """
        if self.bs_table_state_agg is None:
            return set()
        return set(self.bs_table_state_agg.c.keys())

    def _column_name_or_none(self, col_ref) -> Optional[str]:
        """Resolve a user-facing column reference to its physical column
        name on `bs_table`, or return None if it can't be resolved
        (calculated columns, MappedColumns, etc., are not propagatable).
        Mirrors `_get_column`'s `in.` prefix logic so that user-supplied
        `state` resolves to ResStock's `in.state` and ComStock's bare
        `state` consistently.
        """
        try:
            resolved = self._get_column(col_ref, annual_only=True)
        except (ValueError, AttributeError):
            return None
        if not isinstance(resolved, sa.Column):
            return None
        if resolved.table is not self.bs_table:
            return None
        return resolved.name

    def _pick_metadata_table(
        self,
        group_by: Sequence,
        restrict: Sequence,
    ) -> Literal["primary", "state_agg"]:
        """Decide whether to scan the primary `annual_and_metadata` table
        (today's default) or the smaller `annual_and_metadata_state_agg`
        alt table for this query.

        Routing rule: pick `state_agg` iff the schema declares an alt
        table AND every column referenced by `group_by` or `restrict`
        physically exists on the alt table. The alt table omits
        finer-grain columns (county, tract gisjoin, tract demographics)
        — any reference to them disqualifies routing.

        Calculated/MappedColumn group-by entries can't be resolved to a
        single physical column; they conservatively disqualify routing.

        Returns:
          "primary" — use today's bs_table (always safe).
          "state_agg" — use bs_table_state_agg (smaller, faster when
                        eligible; see INVESTIGATION_partition_overhead.md
                        for measurements).
        """
        if self.bs_table_state_agg is None:
            return "primary"
        alt_cols = self._state_agg_columns
        # Check group_by: each entry must resolve to a column present
        # on the alt table.
        for g in group_by or ():
            if isinstance(g, str):
                # Try both the user-supplied name and the in.-prefixed
                # form, since the alt table uses the prefixed convention
                # for some schemas (e.g. ResStock: in.state).
                char_prefix = self.db_schema.column_prefix.characteristics
                if g in alt_cols:
                    continue
                if g.startswith(char_prefix) and g.removeprefix(char_prefix) in alt_cols:
                    continue
                if not g.startswith(char_prefix) and f"{char_prefix}{g}" in alt_cols:
                    continue
                return "primary"
            elif isinstance(g, sa.Column):
                if g.name not in alt_cols:
                    return "primary"
            elif isinstance(g, SALabel):
                # Calculated columns: conservatively unroutable. They
                # may reference columns from outside `bs_table` (e.g.
                # ts-side enduses) which the alt table also lacks.
                return "primary"
            elif isinstance(g, MappedColumn):
                # MappedColumns are user-supplied literal mappings —
                # they don't depend on table schema, so they're safe.
                continue
            else:
                # Unknown entry shape — be conservative.
                return "primary"
        # Check restrict: each col_ref must resolve to a bs_table column
        # that's also on the alt table.
        for entry in restrict or ():
            col_ref = entry[0] if isinstance(entry, (tuple, list)) else None
            if col_ref is None:
                continue
            # Multi-column tuple LHS (e.g. applied-buildings filter):
            # safe to route iff every component column is on the alt.
            if isinstance(col_ref, tuple):
                for c in col_ref:
                    name = c.name if isinstance(c, sa.Column) else None
                    if name is None or name not in alt_cols:
                        return "primary"
                continue
            # Single column ref: resolve through the same prefix logic.
            if isinstance(col_ref, str):
                char_prefix = self.db_schema.column_prefix.characteristics
                if col_ref in alt_cols:
                    continue
                if col_ref.startswith(char_prefix) and col_ref.removeprefix(char_prefix) in alt_cols:
                    continue
                if not col_ref.startswith(char_prefix) and f"{char_prefix}{col_ref}" in alt_cols:
                    continue
                return "primary"
            if isinstance(col_ref, sa.Column):
                if col_ref.name not in alt_cols:
                    return "primary"
                continue
            # Unknown shape — be conservative.
            return "primary"
        return "state_agg"

    def _build_applied_subquery(
        self,
        *,
        any_of: Optional[Sequence[str | int]] = None,
        all_of: Optional[Sequence[str | int]] = None,
        key_kind: Literal["metadata", "timeseries"] = "metadata",
    ):
        """Return a unique-key subquery for buildings matching the applied predicate.

        - `all_of`: must have applicability=true rows for every listed upgrade.
        - `any_of`: must have applicability=true rows for at least one listed upgrade.
        - Both: AND of the two predicates.
        - Neither: returns None.
        - 0 in either list: ValueError. Baseline isn't an "applied" upgrade.

        `key_kind` selects which unique-key columns to project — use "timeseries" when
        the subquery will filter the timeseries table (whose key may be narrower).
        """
        all_ids = self._normalize_applied_list(all_of) if all_of else []
        any_ids = self._normalize_applied_list(any_of) if any_of else []
        if not all_ids and not any_ids:
            return None

        up_col = self._md_upgrade_col
        union_ids = list(dict.fromkeys(all_ids + any_ids))
        typed_union = [typed_literal(up_col, uid) for uid in union_ids]
        key_names = self._get_unique_keys(key_kind)
        md_key_cols = [self.bs_table.c[name] for name in key_names]

        select = (
            sa.select(*md_key_cols)
            .where(
                up_col.in_(typed_union),
                self._md_successful_condition,
            )
            .group_by(*md_key_cols)
        )

        if all_ids and not any_ids:
            # all_of-only: identical SQL shape to the prior `applied_in` form.
            select = select.having(sa.func.count(sa.func.distinct(up_col)) == len(all_ids))
        elif any_ids and not all_ids:
            # any_of-only: GROUP BY + WHERE upgrade IN (...) is sufficient. A
            # surviving row means at least one matching applicable upgrade
            # existed; GROUP BY collapses to one row per key.
            pass
        else:
            # Both lists: AND the two predicates via CASE-filtered counts.
            typed_all = [typed_literal(up_col, uid) for uid in all_ids]
            typed_any = [typed_literal(up_col, uid) for uid in any_ids]
            all_case = sa.case((up_col.in_(typed_all), up_col), else_=None)
            any_case = sa.case((up_col.in_(typed_any), up_col), else_=None)
            select = select.having(
                sa.and_(
                    sa.func.count(sa.func.distinct(all_case)) == len(all_ids),
                    sa.func.count(sa.func.distinct(any_case)) >= 1,
                )
            )
        return select

    def _normalize_applied_list(self, upgrades: Sequence[str | int]) -> list[str]:
        """Validate and normalize an upgrade-id list. Rejects 0 (baseline)."""
        normalized: list[str] = []
        for raw in upgrades:
            uid = self._validate_upgrade(raw)
            if str(uid) == "0":
                raise ValueError(
                    "0 (baseline) is not a valid applied upgrade — applicability is "
                    "meaningful only for upgrades"
                )
            if uid not in normalized:
                normalized.append(uid)
        return normalized

    def _make_applied_filter_tuple(
        self,
        select,
        *,
        key_kind: Literal["metadata", "timeseries"] = "metadata",
    ) -> RestrictTuple:
        """Wrap a Select into the `(cols_or_col, subquery)` shape used by restrict/avoid.

        When `key_kind="timeseries"`, the LHS columns are bound to ts_table so
        that `_split_restrict` routes the filter to the TS-side WHERE (where
        `inapplicables_have_ts` rows would otherwise inflate totals).
        """
        key_names = self._get_unique_keys(key_kind)
        if key_kind == "timeseries" and self.ts_table is not None:
            cols = [self.ts_table.c[name] for name in key_names]
        else:
            cols = [self.bs_table.c[name] for name in key_names]
        if len(cols) == 1:
            return (cols[0], select)
        return (tuple(cols), select)

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
        restrict: Sequence[RestrictTuple] = Field(default_factory=list),
        avoid: Sequence[RestrictTuple] = Field(default_factory=list),
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
        restrict: Sequence[RestrictTuple] = Field(default_factory=list),
        avoid: Sequence[RestrictTuple] = Field(default_factory=list),
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
        restrict: Sequence[RestrictTuple] = Field(default_factory=list),
        avoid: Sequence[RestrictTuple] = Field(default_factory=list),
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

            restrict: The list of where conditions to restrict the results to. Each entry can be a scalar equality,
                      an `IN (...)` list, or a single-column SQLAlchemy subquery for `IN (subquery)`.
                      Example: `[('state', ['VA', 'AZ']), (self.ts_bldgid_column, sa.select(...)), ...]`

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
        return self.agg._query(*args, **kwargs)
