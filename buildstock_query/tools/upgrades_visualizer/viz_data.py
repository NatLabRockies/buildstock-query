from buildstock_query import BuildStockQuery, KWH2MBTU

import polars as pl
from buildstock_query.tools.upgrades_visualizer.plot_utils import PlotParams
from typing import Union
from typing import Literal
import datetime
from buildstock_query.schema.utilities import validate_arguments

num2month = {1: "January", 2: "February", 3: "March", 4: "April",
             5: "May", 6: "June", 7: "July", 8: "August",
             9: "September", 10: "October", 11: "November", 12: "December"}
fuels_types = ['electricity', 'natural_gas', 'propane', 'fuel_oil', 'coal', 'wood_cord', 'wood_pellets']


class VizData:
    @validate_arguments
    def __init__(self, opt_sat_path: str,
                 db_name: str,
                 run: Union[str, tuple[str, str]],
                 workgroup: str = 'largeee',
                 buildstock_type: Literal['resstock', 'comstock'] = 'resstock',
                 skip_init: bool = False,
                 include_monthly: bool = True,
                 upgrades_selection: set[str] = set()):
        if isinstance(run, tuple):
            # Allows for separate baseline and upgrade runs
            # In this case, run[0] is the baseline run and run[1] is the upgrade run
            self.baseline_run = BuildStockQuery(workgroup=workgroup,
                                                db_name=db_name,
                                                buildstock_type=buildstock_type,
                                                table_name=run[0],
                                                skip_reports=skip_init)
            baseline_table_name = run[0] + "_baseline"
            upgrade_table_name = run[1] + "_upgrades"
            ts_table_name = run[1] + "_timeseries"
            table = (baseline_table_name, ts_table_name, upgrade_table_name)
        else:
            # If only one run is specified, it is assumed to contain both baseline and upgrade data
            self.baseline_run = None
            table = run
        self.main_run = BuildStockQuery(workgroup=workgroup,
                                        db_name=db_name,
                                        buildstock_type=buildstock_type,
                                        table_name=table,
                                        skip_reports=skip_init)
        self.opt_sat_path = opt_sat_path
        self.upgrades_selection = upgrades_selection
        self.include_monthly = include_monthly
        if not skip_init:
            self.initialize()

    def initialize(self):
        available_upgrades = self.main_run.get_available_upgrades()
        if not self.upgrades_selection:
            self.upgrades_selection = set(available_upgrades)
        if (unavailable_upgrades := self.upgrades_selection - set(available_upgrades)):
            raise ValueError(f"Upgrades {unavailable_upgrades} is not available in the run")
        available_upgrades = self.upgrades_selection
        self.report = pl.from_pandas(self.main_run.report.get_success_report(), include_index=True)
        self.available_upgrades = list(set([str(u) for u in available_upgrades]) - {'0'})
        self.upgrade2name = {'0': "Upgrade 0: Baseline"}
        if self.available_upgrades:
            upgrade_names = self.main_run.get_upgrade_names()
            self.upgrade2name |= upgrade_names

        self.upgrade2shortname = {indx+1: f"Upgrade {indx+1}" for indx in range(len(self.available_upgrades) + 1)}
        self.chng2bldg = self.get_change2bldgs()
        self.init_annual_results()
        if self.include_monthly:
            self.init_monthly_results(self.metadata_df)
        self.all_upgrade_plotting_df = None

    def run_obj(self, upgrade: str) -> BuildStockQuery:
        if upgrade == '0' and self.baseline_run is not None:
            return self.baseline_run
        return self.main_run

    def get_change2bldgs(self):
        change_types = ("any", "no-chng", "bad-chng", "ok-chng", "true-bad-chng", "true-ok-chng")
        chng2bldg: dict[tuple[str, str], list[int]] = {}
        for chng in change_types:
            for upgrade in self.available_upgrades:
                print(f"Getting buildings for {upgrade} and {chng}")
                chng2bldg[(upgrade, chng)] = self.main_run.report.get_buildings_by_change(upgrade_id=int(upgrade),
                                                                                          change_type=chng)
        return chng2bldg

    def _get_results_csv_clean(self, upgrade: str):
        if upgrade == '0':
            res_df = pl.read_parquet(self.run_obj(upgrade)._download_results_csv())
        else:
            res_df = pl.read_parquet(self.run_obj(upgrade)._download_upgrades_csv(upgrade_id=upgrade))
        res_df = res_df.filter(pl.col(self.run_obj(upgrade).db_schema.column_names.completed_status) ==
                               self.run_obj(upgrade).db_schema.completion_values.success)
        res_df = res_df.drop([col for col in res_df.columns if
                              "applicable" in col
                              or "output_format" in col])
        res_df = res_df.rename({'upgrade_costs.upgrade_cost_usd': 'upgrade_cost_total_usd'})
        res_df = res_df.rename({x: x.split('.')[1] for x in res_df.columns if '.' in x})
        res_df = res_df.with_columns(upgrade=pl.lit(upgrade))
        res_df = res_df.with_columns(count=pl.lit(1))
        res_df = res_df.with_columns(month=pl.lit('All'))
        self.run_obj(upgrade).save_cache()
        return res_df

    def _get_metadata_df(self):
        bs_res_df = pl.read_parquet(self.run_obj('0')._download_results_csv())
        metadata_cols = [c for c in bs_res_df.columns if c.startswith(self.main_run._char_prefix)]
        metadata_df = bs_res_df.select([self.main_run.building_id_column_name] + metadata_cols)
        metadata_df = metadata_df.rename({x: x.split('.')[1] for x in metadata_df.columns if '.' in x})
        return metadata_df

    def init_annual_results(self):
        self.bs_res_df = self._get_results_csv_clean('0')
        self.metadata_df = self._get_metadata_df()
        self.sample_weight = self.metadata_df['sample_weight'][0]
        self.upgrade2res = {'0': self.bs_res_df}
        for upgrade in self.available_upgrades:
            print(f"Getting up_csv for {upgrade}")
            up_csv = self._get_results_csv_clean(upgrade)
            up_csv = up_csv.join(self.metadata_df, on='building_id')
            self.upgrade2res[upgrade] = up_csv

    def _get_ts_enduse_cols(self, upgrade: str):
        rub_obj = self.run_obj(upgrade)
        assert rub_obj.ts_table is not None, "No timeseries table found"
        all_cols = [str(col.name) for col in rub_obj.get_cols(table=rub_obj.ts_table)]
        enduse_cols = filter(lambda x: x.endswith(('_kbtu', '_kwh', 'lb')), all_cols)
        return list(enduse_cols)

    def init_monthly_results(self, metadata_df):
        self.upgrade2res_monthly: dict[str, pl.DataFrame] = {}
        for upgrade in ['0'] + self.available_upgrades:
            ts_cols = self._get_ts_enduse_cols(upgrade)
            print(f"Getting monthly results for {upgrade}")
            run_obj = self.run_obj(upgrade)
            monthly_vals_query = run_obj.agg.aggregate_timeseries(get_query_only=True,
                                                                  enduses=ts_cols,
                                                                  group_by=[run_obj.bs_bldgid_column],
                                                                  upgrade_id=upgrade,
                                                                  timestamp_grouping_func='month',
                                                                  )
            if monthly_vals_query in run_obj._query_cache:
                monthly_vals = run_obj._query_cache[monthly_vals_query].copy()
            else:
                month_year = f"{datetime.datetime.now().strftime('%b%Y')}"
                s3_unload_path = f"s3://{run_obj.params.query_unload_s3_bucket}/bsq_athena_unload_results/{month_year}/"
                pd_cursor = run_obj._conn.cursor(unload=True, s3_staging_dir=s3_unload_path).execute(
                    monthly_vals_query,
                    result_reuse_enable=True,
                    result_reuse_minutes=60 * 24 * 7)
                monthly_vals = pd_cursor.as_pandas()
                run_obj._query_cache[monthly_vals_query] = monthly_vals
            run_obj.save_cache()
            monthly_df = pl.from_pandas(monthly_vals, include_index=True)
            monthly_df = monthly_df.with_columns(pl.col('time').dt.month().alias("month"))
            monthly_df = monthly_df.with_columns(pl.col('month').replace_strict(num2month).alias("month"))
            modified_cols = []
            for col in ts_cols:
                # scale values down to per building and convert to m_btu to match with annual results
                if col.endswith('_kwh'):
                    modified_cols.append((KWH2MBTU * pl.col(col) / pl.col("units_count"))
                                         .alias(col.replace("kwh", "m_btu").replace("__", "_")))
                if col.endswith("_kbtu"):
                    modified_cols.append((0.001 * pl.col(col) / pl.col("units_count"))
                                         .alias(col.replace("kbtu", "m_btu").replace("__", "_")))
                else:
                    modified_cols.append((pl.col(col) / pl.col("units_count"))
                                         .alias(col.replace("__", "_")))
            monthly_df = monthly_df.select(['building_id', 'month'] + modified_cols
                                           + [pl.lit(upgrade).alias("upgrade")])
            monthly_df = monthly_df.join(metadata_df, on='building_id')
            self.upgrade2res_monthly[upgrade] = monthly_df

    def get_values(self,
                   upgrade: str,
                   params: PlotParams,
                   ) -> pl.DataFrame:
        df = self.upgrade2res[upgrade] if params.resolution == 'annual' else self.upgrade2res_monthly[upgrade]
        if params.filter_bldgs:
            df = df.filter(pl.col('building_id').is_in(set(params.filter_bldgs)))

        missing_cols = (pl.lit(0).alias(c) for c in params.enduses if c not in df.columns)
        df = df.with_columns(missing_cols)  # add missing cols as zero
        value_cols = [pl.sum_horizontal([pl.col(c).fill_null(0) for c in params.enduses]).alias("value")]
        other_cols = ['building_id'] + params.group_by
        if 'month' not in params.group_by:
            other_cols += ['month']
        return df.select(other_cols + value_cols)

    def get_plotting_df(self, upgrade: str,
                        params: PlotParams,) -> pl.DataFrame:
        baseline_df = self.get_values(upgrade='0', params=params)
        baseline_df = baseline_df.select("building_id", "month", pl.col("value").alias("baseline_value"))
        up_df = self.get_values(upgrade=upgrade, params=params)
        if params.change_type:
            chng_upgrade = str(params.sync_upgrade) if params.sync_upgrade else str(upgrade) if upgrade else '0'
            if chng_upgrade and chng_upgrade != '0':
                change_bldg_list = self.chng2bldg[(chng_upgrade, params.change_type)]
            else:
                change_bldg_list = []
            change_bldg_list = set(change_bldg_list).intersection(up_df['building_id'].to_list())
            up_df = up_df.filter(pl.col("building_id").is_in(change_bldg_list))
        up_df = up_df.join(baseline_df, on=('building_id', 'month'), how='left')
        if params.savings_type == "Savings":
            up_df = up_df.with_columns((pl.col("baseline_value") - pl.col("value")).alias("value"))
            up_df = up_df
        elif params.savings_type == "Percent Savings":
            up_df = up_df.with_columns((100 * (1 - pl.col("value") / pl.col("baseline_value"))).alias("value"))
            # handle case when baseline is 0
            up_df = up_df.with_columns(
                pl.when(pl.col("baseline_value") == 0)
                .then(-100)
                .otherwise(pl.col("value"))
                .alias("value"))
            up_df = up_df.with_columns(
                pl.when((pl.col("baseline_value") == 0) & (pl.col("value").is_between(-1e-6, 1e-6)))
                .then(0)
                .otherwise(pl.col("value"))
                .alias("value")
            )
        return up_df

    def get_all_cols(self, resolution: str) -> list[str]:
        if resolution == 'annual':
            return self.bs_res_df.columns
        else:
            return self.upgrade2res_monthly['0'].columns

    def get_all_end_use_cols(self, resolution: str) -> list[str]:
        all_cols = self.get_all_cols(resolution=resolution)
        all_end_use_cols = filter(lambda col: col.startswith(("end_use_", "energy_use_", "fuel_use_")), all_cols)
        return list(all_end_use_cols)

    def get_emissions_cols(self, resolution: str) -> list[str]:
        all_cols = self.get_all_cols(resolution=resolution)
        all_emissions_cols = filter(lambda c: c.startswith("emissions_"), all_cols)
        return list(all_emissions_cols)

    def get_cleaned_up_end_use_cols(self, resolution: str, fuel) -> list[str]:
        cols = []
        all_end_use_cols = self.get_all_end_use_cols(resolution=resolution)
        sep = "_"
        for c in all_end_use_cols:
            if fuel in c or fuel == 'All':
                c = c.removeprefix(f"end_use{sep}{fuel}{sep}")
                c = c.removeprefix(f"fuel_use{sep}{fuel}{sep}")
                if fuel == 'All':
                    for f in sorted(fuels_types):
                        c = c.removeprefix(f"end_use{sep}{f}{sep}")
                        c = c.removeprefix(f"fuel_use{sep}{f}{sep}")
                cols.append(c)
        no_dup_cols = {c: None for c in cols}
        return list(no_dup_cols.keys())

    def get_end_use_db_cols(self, resolution, fuel, end_use):
        all_enduses = self.get_all_end_use_cols(resolution=resolution)
        if not end_use:
            return all_enduses[0]
        valid_cols = []
        sep = "_"
        prefix = "fuel_use" if end_use.startswith("total") else "end_use"
        if fuel == 'All':
            valid_cols.extend(f"{prefix}{sep}{f}{sep}{end_use}" for f in fuels_types
                              if f"{prefix}{sep}{f}{sep}{end_use}" in all_enduses)

        else:
            valid_cols.append(f"{prefix}{sep}{fuel}{sep}{end_use}")
        return valid_cols

    def get_plotting_df_all_upgrades(self,
                                     params: PlotParams,
                                     ) -> pl.DataFrame:
        df_list = []
        for upgrade in ['0'] + self.available_upgrades:
            df = self.get_plotting_df(upgrade=upgrade, params=params)
            df = df.with_columns(pl.lit(upgrade).alias("upgrade"))
            df_list.append(df)
        return pl.concat(df_list).fill_null(0)
