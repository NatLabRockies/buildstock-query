from pandas import DataFrame
from collections import Counter, defaultdict
import sqlalchemy as sa
from sqlalchemy.sql import func as safunc
import logging
import pandas as pd
from buildstock_query.helpers import print_r, print_g
from ast import literal_eval
from functools import reduce
from buildstock_query import main
import typing
from typing import Optional, Union, Literal
from collections.abc import Hashable, Sequence
from buildstock_query.schema.utilities import AnyColType, validate_arguments
from pydantic import Field
from typing_extensions import assert_never

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
FUELS = ["electricity", "natural_gas", "propane", "fuel_oil", "coal", "wood_cord", "wood_pellets"]


class BuildStockReport:
    """Class with a collection of functions for reporting and integrity check queries."""

    def __init__(self, bsq: "main.BuildStockQuery") -> None:
        self._bsq = bsq

    def _rename_completed_status_column(self, df: DataFrame) -> DataFrame:
        df = df.rename(columns={self._bsq.db_schema.column_names.completed_status: "completed_status"})
        rev_value_map = {db_val: normal_val for normal_val, db_val in self._bsq.db_schema.completion_values}
        df["completed_status"] = df["completed_status"].map(rev_value_map)
        return df

    @typing.overload
    def _get_bs_success_report(self, get_query_only: Literal[False] = False) -> DataFrame: ...

    @typing.overload
    def _get_bs_success_report(self, get_query_only: Literal[True]) -> str: ...

    @typing.overload
    def _get_bs_success_report(self, get_query_only: bool) -> Union[DataFrame, str]: ...

    def _get_bs_success_report(self, get_query_only: bool = False):
        bs_query = sa.select(*[self._bsq._bs_completed_status_col, safunc.count().label("count")])
        bs_query = bs_query.group_by(sa.text("1"))
        if get_query_only:
            return self._bsq._compile(bs_query)
        df = self._bsq.execute(bs_query)
        df = self._rename_completed_status_column(df)
        df.insert(0, "upgrade", "0")
        return self._process_report(df)

    @typing.overload
    def _get_change_report(self, get_query_only: Literal[False] = False) -> DataFrame: ...

    @typing.overload
    def _get_change_report(self, get_query_only: Literal[True]) -> list[str]: ...

    @typing.overload
    def _get_change_report(self, get_query_only: bool) -> Union[DataFrame, list[str]]: ...

    def _get_change_report(self, get_query_only: bool = False):
        """Returns counts of buildings to which upgrade didn't do any changes on energy consumption

        Args:
            get_query_only (bool, optional): _description_. Defaults to False.
        """
        if self._bsq.up_table is None:
            raise ValueError("No upgrade table is available .")

        queries: list[str] = []
        chng_types = ["no-chng", "bad-chng", "ok-chng", "true-bad-chng", "true-ok-chng", "null", "any"]
        for ch_type in chng_types:
            up_query = sa.select(*[self._bsq.up_table.c["upgrade"], safunc.count().label("change")])
            up_query = up_query.join(self._bsq.bs_table, self._bsq.bs_bldgid_column == self._bsq.up_bldgid_column)
            conditions = self._get_change_conditions(change_type=ch_type)
            up_query = up_query.where(
                sa.and_(self._bsq._bs_successful_condition, self._bsq._up_successful_condition, conditions)
            )  # type: ignore
            up_query = up_query.group_by(sa.text("1"))
            up_query = up_query.order_by(sa.text("1"))
            queries.append(self._bsq._compile(up_query))
        if get_query_only:
            return queries
        change_df: DataFrame = pd.DataFrame()
        for chng_type, query in zip(chng_types, queries):
            df = self._bsq.execute(query)
            df.rename(columns={"change": chng_type}, inplace=True)
            # df['upgrade'] = df['upgrade'].map(int)
            df = df.set_index("upgrade").sort_index()
            change_df = change_df.join(df, how="outer") if len(change_df) > 0 else df
        change_df = change_df.fillna(0)
        for chng_type in chng_types:
            if chng_type not in change_df.columns:
                change_df[chng_type] = 0
        return change_df

    @validate_arguments
    def print_change_details(
        self,
        upgrade_id: Union[int, str],
        yml_file: str,
        opt_sat_path: str,
        change_type: Literal[
            "no-chng", "bad-chng", "ok-chng", "true-bad-chng", "true-ok-chng", "null", "any"
        ] = "no-chng",
    ):
        ua = self._bsq.get_upgrades_analyzer(yaml_file=yml_file, opt_sat_file=opt_sat_path)
        bad_bids = self.get_buildings_by_change(upgrade_id=upgrade_id, change_type=change_type)
        good_bids = self.get_buildings_by_change(upgrade_id=upgrade_id, change_type="ok-chng")
        ua.print_unique_characteristic(upgrade_id, change_type, good_bids, bad_bids)

    @typing.overload
    def _get_upgrade_buildings(
        self, *, upgrade_id: Union[int, str], trim_missing_bs: bool = True, get_query_only: Literal[False] = False
    ) -> list[int]: ...

    @typing.overload
    def _get_upgrade_buildings(
        self, *, upgrade_id: Union[int, str], get_query_only: Literal[True], trim_missing_bs: bool = True
    ) -> str: ...

    @typing.overload
    def _get_upgrade_buildings(
        self, *, upgrade_id: Union[int, str], get_query_only: bool, trim_missing_bs: bool = True
    ) -> Union[list[int], str]: ...

    def _get_upgrade_buildings(
        self, *, upgrade_id: Union[int, str], trim_missing_bs: bool = True, get_query_only: bool = False
    ):
        if self._bsq.up_table is None:
            raise ValueError("No upgrade table is available .")
        up_query = sa.select(*[self._bsq.up_bldgid_column])
        if trim_missing_bs:
            up_query = up_query.join(self._bsq.bs_table, self._bsq.bs_bldgid_column == self._bsq.up_bldgid_column)
            up_query = up_query.where(
                sa.and_(
                    self._bsq._bs_successful_condition,
                    self._bsq._up_successful_condition,
                    self._bsq.up_table.c["upgrade"] == str(upgrade_id),
                )
            )
        else:
            up_query = up_query.where(
                sa.and_(self._bsq.up_table.c["upgrade"] == str(upgrade_id), self._bsq._up_successful_condition)
            )
        if get_query_only:
            return self._bsq._compile(up_query)
        df = self._bsq.execute(up_query)
        return df[self._bsq.bs_bldgid_column.name].to_numpy(dtype="int32").tolist()

    def _get_change_conditions(self, change_type: str):
        if self._bsq.up_table is None:
            raise ValueError("No upgrade table is available .")

        threshold = 1e-3
        fuel_cols = list(
            c for c in self._bsq.db_schema.column_names.fuel_totals if c in self._bsq.up_table.columns
        )  # Look at all fuel type totals
        all_cols = list(fuel_cols)
        if self._bsq.db_schema.column_names.unmet_hours_cooling_hr:
            all_cols += [self._bsq.db_schema.column_names.unmet_hours_cooling_hr]
        if self._bsq.db_schema.column_names.unmet_hours_heating_hr:
            all_cols += [self._bsq.db_schema.column_names.unmet_hours_heating_hr]
        null_chng_conditions = sa.and_(
            *[
                sa.or_(self._bsq.up_table.c[col] == sa.null(), self._bsq.bs_table.c[col] == sa.null())
                for col in fuel_cols
            ]
        )

        no_chng_conditions = sa.and_(
            *[
                safunc.coalesce(safunc.abs(self._bsq.up_table.c[col] - self._bsq.bs_table.c[col]), 0) < threshold
                for col in fuel_cols
            ]
        )
        good_chng_conditions = sa.or_(
            *[self._bsq.bs_table.c[col] - self._bsq.up_table.c[col] >= threshold for col in fuel_cols]
        )
        opp_chng_conditions = sa.and_(
            *[
                safunc.coalesce(self._bsq.bs_table.c[col] - self._bsq.up_table.c[col], -1) < threshold
                for col in fuel_cols
            ],
            sa.not_(no_chng_conditions),
        )
        true_good_chng_conditions = sa.or_(
            *[self._bsq.bs_table.c[col] - self._bsq.up_table.c[col] >= threshold for col in all_cols]
        )
        true_opp_chng_conditions = sa.and_(
            *[
                safunc.coalesce(self._bsq.bs_table.c[col] - self._bsq.up_table.c[col], -1) < threshold
                for col in all_cols
            ],
            sa.not_(no_chng_conditions),
        )
        if change_type == "no-chng":
            conditions = no_chng_conditions
        elif change_type == "bad-chng":
            conditions = opp_chng_conditions
        elif change_type == "true-bad-chng":
            conditions = true_opp_chng_conditions
        elif change_type == "ok-chng":
            conditions = good_chng_conditions
        elif change_type == "true-ok-chng":
            conditions = true_good_chng_conditions
        elif change_type == "null":
            conditions = null_chng_conditions
        elif change_type == "any":
            conditions = sa.true
        else:
            raise ValueError(f"Invalid {change_type=}")
        return conditions

    @typing.overload
    def get_buildings_by_change(
        self,
        *,
        upgrade_id: Union[int, str],
        get_query_only: Literal[True],
        change_type: Literal[
            "no-chng", "bad-chng", "ok-chng", "true-bad-chng", "true-ok-chng", "null", "any"
        ] = "no-chng",
    ) -> str: ...

    @typing.overload
    def get_buildings_by_change(
        self,
        *,
        upgrade_id: Union[int, str],
        get_query_only: Literal[False] = False,
        change_type: Literal[
            "no-chng", "bad-chng", "ok-chng", "true-bad-chng", "true-ok-chng", "null", "any"
        ] = "no-chng",
    ) -> list[int]: ...

    @typing.overload
    def get_buildings_by_change(
        self,
        *,
        upgrade_id: Union[int, str],
        get_query_only: bool,
        change_type: Literal[
            "no-chng", "bad-chng", "ok-chng", "true-bad-chng", "true-ok-chng", "null", "any"
        ] = "no-chng",
    ) -> Union[list[int], str]: ...

    @validate_arguments
    def get_buildings_by_change(
        self,
        *,
        upgrade_id: Union[int, str],
        change_type: Literal[
            "no-chng", "bad-chng", "ok-chng", "true-bad-chng", "true-ok-chng", "null", "any"
        ] = "no-chng",
        get_query_only: bool = False,
    ):
        if self._bsq.up_table is None:
            raise ValueError("No upgrade table is available .")
        up_query = sa.select(
            *[self._bsq.bs_bldgid_column, self._bsq._bs_completed_status_col, self._bsq._up_completed_status_col]
        )
        up_query = up_query.join(self._bsq.up_table, self._bsq.bs_bldgid_column == self._bsq.up_bldgid_column)

        conditions = self._get_change_conditions(change_type)
        up_query = up_query.where(
            sa.and_(
                self._bsq._bs_successful_condition,
                self._bsq._up_successful_condition,
                self._bsq.up_table.c["upgrade"] == str(upgrade_id),
                conditions,
            )
        )  # type: ignore
        if get_query_only:
            return self._bsq._compile(up_query)
        df = self._bsq.execute(up_query)
        return df[self._bsq.bs_bldgid_column.name].to_numpy(dtype="int32").tolist()

    @typing.overload
    def _get_up_success_report(self, *, get_query_only: Literal[True], trim_missing_bs: bool = True) -> str: ...

    @typing.overload
    def _get_up_success_report(
        self, *, get_query_only: Literal[False] = False, trim_missing_bs: bool = True
    ) -> pd.DataFrame: ...

    @typing.overload
    def _get_up_success_report(
        self, *, get_query_only: bool, trim_missing_bs: bool = True
    ) -> Union[pd.DataFrame, str]: ...

    def _get_up_success_report(self, *, trim_missing_bs: bool = True, get_query_only: bool = False):
        """Get success report for upgrades

        Args:
            trim_missing_bs (bool, optional): Ignore buildings that have no successful runs in the baseline.
                Defaults to True.
            get_query_only (bool, optional): Returns query only without the result. Defaults to False.

        Returns:
            Union[str, pd.DataFrame]: If get_query_only then returns the query string. Otherwise returns the dataframe.
        """
        if self._bsq.up_table is None:
            raise ValueError("No upgrade table is available .")
        up_query = sa.select(
            *[self._bsq.up_table.c["upgrade"], self._bsq._up_completed_status_col, safunc.count().label("count")]
        )
        if trim_missing_bs:
            up_query = up_query.join(self._bsq.bs_table, self._bsq.bs_bldgid_column == self._bsq.up_bldgid_column)
            up_query = up_query.where(self._bsq._bs_successful_condition)

        up_query = up_query.group_by(sa.text("1"), sa.text("2"))
        up_query = up_query.order_by(sa.text("1"), sa.text("2"))
        if get_query_only:
            return self._bsq._compile(up_query)
        df = self._bsq.execute(up_query)
        df["upgrade"] = df["upgrade"].map(str)
        df = self._rename_completed_status_column(df)
        return self._process_report(df)

    def _process_report(self, df: DataFrame):
        # df['upgrade'] = df['upgrade'].map(int)
        pf = df.pivot(index=["upgrade"], columns=["completed_status"], values=["count"])
        pf.columns = [c[1] for c in pf.columns]
        pf["Sum"] = pf.sum(axis=1)
        for col in ["fail", "inapplicable"]:
            if col not in pf.columns:
                pf.insert(1, col, 0)
        return pf

    @typing.overload
    def _get_full_options_report(self, *, trim_missing_bs: bool, get_query_only: Literal[True]) -> str: ...

    @typing.overload
    def _get_full_options_report(
        self, *, trim_missing_bs: bool, get_query_only: Literal[False] = False
    ) -> pd.DataFrame: ...

    @typing.overload
    def _get_full_options_report(self, *, trim_missing_bs: bool, get_query_only: bool) -> Union[pd.DataFrame, str]: ...

    def _get_full_options_report(self, trim_missing_bs: bool = True, get_query_only: bool = False):
        if self._bsq.up_table is None:
            raise ValueError("No upgrade table is available .")
        opt_name_cols = [
            c
            for c in self._bsq.up_table.columns
            if c.name.startswith("upgrade_costs.option_") and c.name.endswith("name")
        ]
        query = sa.select(
            *[self._bsq.up_table.c["upgrade"],
              *opt_name_cols,
              safunc.count().label("success"),
              safunc.array_agg(self._bsq.up_bldgid_column)]
        )
        if trim_missing_bs:
            query = query.join(self._bsq.bs_table, self._bsq.bs_bldgid_column == self._bsq.up_bldgid_column)
            query = query.where(self._bsq._bs_successful_condition)
        grouping_texts = [sa.text(str(i + 1)) for i in range(1 + len(opt_name_cols))]
        query = query.group_by(*grouping_texts)
        query = query.order_by(*grouping_texts)
        if get_query_only:
            return self._bsq._compile(query)
        df = self._bsq.execute(query)
        simple_names = [f"option{i + 1}" for i in range(len(opt_name_cols))]
        df.columns = ["upgrade"] + simple_names + ["success", "applied_buildings"]
        # df['upgrade'] = df['upgrade'].map(int)
        df["applied_buildings"] = df["applied_buildings"].map(lambda x: literal_eval(x))
        applied_rows = df[simple_names].any(axis=1)  # select only rows with at least one option applied
        return df[applied_rows]

    @validate_arguments
    def get_options_report(self, trim_missing_bs: bool = True) -> pd.DataFrame:
        """Finds out the number and list of buildings each of the options applied to.

        Args:
            trim_missing_bs (bool, optional): Whether the buildings not available in baseline should be dropped.
                                              Defaults to True.
        Returns:
            pd.DataFrame: The list of options the corresponding set of building ids the option applied to.
        """
        if self._bsq.up_table is None:
            raise ValueError("No upgrade table is available .")

        full_report = self._get_full_options_report(trim_missing_bs=trim_missing_bs)
        option_cols = [c for c in full_report.columns if c.startswith("option")]
        total_counts: Counter = Counter()
        bldg_array: dict = defaultdict(list)
        for option in option_cols:
            grouped_dict = (
                full_report.groupby(["upgrade", option])
                .aggregate({"success": "sum", "applied_buildings": "sum"})
                .to_dict()
            )
            total_counts += Counter(grouped_dict["success"])
            for key, val in grouped_dict["applied_buildings"].items():
                bldg_array[key] += val

        option_df = pd.DataFrame.from_dict(
            {
                "success": total_counts,
                "applied_buildings": bldg_array,
            },
            orient="columns",
        )
        option_df["applied_buildings"] = option_df["applied_buildings"].map(lambda x: set(x))
        option_df = option_df.reset_index()
        option_df.columns = ["upgrade", "option", "success", "applied_buildings"]
        # Aggregate for upgrade
        agg = option_df.groupby("upgrade").aggregate({"applied_buildings": lambda x: reduce(set.union, x)})
        agg = agg.reset_index()
        agg.insert(1, "success", agg["applied_buildings"].map(lambda x: len(x)))
        agg.insert(0, "option", "All")
        full_df = pd.concat([option_df, agg])
        full_df = full_df.sort_values(["upgrade", "option"])
        return full_df

    @validate_arguments
    def get_option_integrity_report(self, yaml_file: str, opt_sat_path: str) -> pd.DataFrame:
        """Checks the upgrade/option spec in the buildstock configuration file against what is actually in the
        simulation result and tabulates the discrepancy.
        Args:
            yaml_file (str): The path to buildstock configuration file used to run the simulation

        Returns:
            pd.DataFrame: The report dataframe.
        """
        ua_df = self._bsq.get_upgrades_analyzer(yaml_file=yaml_file, opt_sat_file=opt_sat_path).get_report()
        ua_df['upgrade'] = ua_df['upgrade'].map(str)
        ua_df = ua_df.groupby(["upgrade", "option"]).aggregate(
            {"applicable_to": "sum", "applicable_buildings": lambda x: reduce(set.union, x)}
        )
        assert (ua_df["applicable_to"] == ua_df["applicable_buildings"].map(lambda x: len(x))).all()
        opt_report_df = self.get_options_report().fillna(0)
        opt_report_df = opt_report_df.set_index(["upgrade", "option"])
        diff_df = pd.DataFrame(index=ua_df.index)
        diff_df["applicable_buildings"] = ua_df["applicable_buildings"]
        diff_df["applied_buildings"] = opt_report_df["applied_buildings"]
        diff_df["overapplied_bldgs"] = opt_report_df["applied_buildings"] - ua_df["applicable_buildings"]
        diff_df["unapplied_bldgs"] = ua_df["applicable_buildings"] - opt_report_df["applied_buildings"]
        for col in diff_df.columns:
            diff_df[f"{col}_count"] = diff_df[col].map(lambda x: len(x) if isinstance(x, set) else 0)
        success_report_df = self.get_success_report()
        fail_report = success_report_df[["fail", "success"]].rename(
            columns={"fail": "Upgrade Failures", "success": "Upgrade Success"}
        )
        diff_df = diff_df.join(fail_report)
        return diff_df

    @validate_arguments
    def check_options_integrity(self, yaml_file: str, opt_sat_path: str) -> bool:
        """Checks the upgrade/option spec in the buildstock configuration file against what is actually in the
        simulation result and flags any discrepancy. The verificationa allows for some mismatch since some simulations
        could have failed. Unless there is a bug somewhere in buildstock workflow, integrity check should pass
        regardless of number of failures.

        Args:
            yaml_file (str): The path to buildstock configuration file used to run the simulation

        Returns:
            bool: Whether or not the integrity check passed.
        """
        intg_df = self.get_option_integrity_report(yaml_file, opt_sat_path).reset_index()
        all_intg_df = intg_df[intg_df["option"] == "All"]
        blank_opt_upgrades = all_intg_df[all_intg_df["applied_buildings_count"] < all_intg_df["Upgrade Success"]]
        assert (all_intg_df["applied_buildings_count"] >= all_intg_df["Upgrade Success"]).all()
        if len(blank_opt_upgrades) > 0:
            print_r(
                "BLANK OPTIONS: The following upgrades have fewere 'applied_buildings_count' than 'Upgrade Success'"
                "This indicates that some buildings in these upgrades didn't have any option applied"
            )
        serious = False
        for indx, row in intg_df.iterrows():
            upgrade_failures = row["Upgrade Failures"]
            applicable_count = row.applicable_buildings_count
            applied_count = row.applied_buildings_count
            unapplied_count = row.unapplied_bldgs_count
            overapplied_count = row.overapplied_bldgs_count
            if row.unapplied_bldgs_count > 0:
                if row.option == "All" and row.unapplied_bldgs_count == upgrade_failures:
                    print_r(
                        f"OPTION UNDERAPPLICATION: Upgrade {row.upgrade} was was supposed to be applied to "
                        f"{applicable_count} samples but applied to {applied_count} samples"
                    )
                    print_g(
                        f"This difference of {unapplied_count} exactly matches with {upgrade_failures}"
                        f" failures in Upgrade {row.upgrade}. It's all good."
                    )
                else:
                    print_r(
                        f"OPTION UNDERAPPLICATION: Upgrade {row.upgrade}, {row.option} didn't apply to "
                        f"{unapplied_count} samples that it was supposed to apply to."
                    )
                    if upgrade_failures > 0:
                        if unapplied_count > upgrade_failures:
                            print_r(f"{upgrade_failures} failures in Upgrade {row.upgrade} can't account for this.")
                            serious = True
                        else:
                            print_g(f"{upgrade_failures} failures in Upgrade {row.upgrade} may account for this.")
                    else:
                        serious = True

            if overapplied_count > 0:
                print_r(
                    f"OPTION OVERAPPLICATION: Upgrade {row.upgrade}, {row.option} applied to"
                    f" {unapplied_count} samples that it was supposed to NOT apply to."
                )
                serious = True
        if not serious:
            print_g("Integrity check passed.")
            return True
        else:
            print_r("Integrity check failed. Please check the serious issues above.")
            return False

    @typing.overload
    def get_success_report(
        self, *, get_query_only: Literal[True], trim_missing_bs: Union[Literal["auto"], bool] = "auto"
    ) -> tuple[str, str, list[str]]: ...

    @typing.overload
    def get_success_report(
        self, *, get_query_only: Literal[False] = False, trim_missing_bs: Union[Literal["auto"], bool] = "auto"
    ) -> pd.DataFrame: ...

    @typing.overload
    def get_success_report(
        self, *, get_query_only: bool, trim_missing_bs: Union[Literal["auto"], bool] = "auto"
    ) -> Union[pd.DataFrame, tuple[str, str, list[str]]]: ...

    @validate_arguments
    def get_success_report(self, trim_missing_bs: Union[Literal["auto"], bool] = "auto", get_query_only: bool = False):
        """Returns a basic report showing number of success and failures for each upgrade along with percentage.
        Additional information regarding number of buildings to which the upgrade applied and whether the enduses
        changed is also returned.

        Args:
            trim_missing_bs (str | bool, optional): Whether the buildings that failed in baseline should be
                                                    dropped from the upgrades. If true, all metrics is calculated after
                                                    those buildings are dropped from the upgrades. Defaults to 'auto'.
            get_query_only (bool, optional): If true, returns SQL query instead of the report. Defaults to False.

        Raises:
            ValueError: If something went wrong.

        Returns:
            pd.DateFrame: The report dataframe. The meaning of the various columns are as follows:
                **Fail**: Number of simulation that failed.
                **inapplicable**: Number of buildings to which the upgrade didn't apply (because of apply logic)
                **Success**: The number of buildings which completed simulation successfully. No simulation is run for
                    inapplicable buildings.
                **Sum**: Sum of the first three columns.
                **Applied %**: Success / Sum * 100 %.
                **no-chng** : Number of successful simulation that didn't have any change in values for any enduses.
                **bad-chng:** Number of successful simulation that had bad changes. It's considered a bad change if
                   none of the fuel has any reduction in energy consumption, and at least one fuel has an
                   increase in energy consumption.
                **ok-chng:** |set(success) - set(no-chng) - set(bad-chng)| i.e. count of successful simulation that are
                         neither no-chng nor bad-chng.
                **true-bad-chng:** Count of only those bad changes in which neither of the umnet cooling/heating hours
                               decreased. In other words, the increase in energy consumption in one of the fuel type
                               (often electricity - for electrification upgrades) didn't result in improvement of
                               cooling/heating umnet hours.
                **true-ok-chng:** Adjustment of ok-chng after using true-bad-chng instead of bad-chng
                **null**: Included for testing/integrity-checking purpose. It refers to number of buildings that are
                      are neither no-chng, not bad-chng nor ok-chng. It should always be zero.
                **any**: Sum of the no-chng + bad-chng + ok-chng. Refers to any change (including no-change).
                **x-chng %**: The percentage form of the change calculated by using success count as the base.
        """

        baseline_result = self._get_bs_success_report()

        if self._bsq.up_table is None:
            return baseline_result

        if trim_missing_bs == "auto":
            if "success" in baseline_result:
                trim = True
            else:
                logger.warning(
                    "None of the simulation was successful in baseline. The counts for upgrade will be"
                    " returned without requiring corresponding successful baseline run."
                )
                trim = False
        elif isinstance(trim_missing_bs, bool):
            trim = trim_missing_bs
        else:
            assert_never(trim_missing_bs)
            raise ValueError("trim_missing_bs must be either True/False or 'auto'.")

        if get_query_only:
            baseline_query = self._get_bs_success_report(get_query_only=True)
            upgrade_query = self._get_up_success_report(trim_missing_bs=trim, get_query_only=True)
            change_query = self._get_change_report(get_query_only=True)
            return baseline_query, upgrade_query, change_query

        upgrade_result = self._get_up_success_report(trim_missing_bs=trim).fillna(0)
        change_result = self._get_change_report().fillna(0)
        if get_query_only:
            return baseline_result, upgrade_result, change_result
        if "success" in upgrade_result.columns:
            pa = round(100 * (upgrade_result["fail"] + upgrade_result["success"]) / upgrade_result["Sum"], 1)
            upgrade_result["Applied %"] = pa

        pf = pd.concat([baseline_result, upgrade_result])
        pf = pf.join(change_result).fillna(0)
        pf["no-chng %"] = round(100 * pf["no-chng"] / pf["success"], 1)
        pf["bad-chng %"] = round(100 * pf["bad-chng"] / pf["success"], 1)
        pf["ok-chng %"] = round(100 * pf["ok-chng"] / pf["success"], 1)
        pf["true-ok-chng %"] = round(100 * pf["true-ok-chng"] / pf["success"], 1)
        pf["true-bad-chng %"] = round(100 * pf["true-bad-chng"] / pf["success"], 1)
        return pf

    @typing.overload
    def _get_ts_report(self, get_query_only: Literal[False] = False) -> DataFrame: ...

    @typing.overload
    def _get_ts_report(self, get_query_only: Literal[True]) -> str: ...

    @typing.overload
    def _get_ts_report(self, get_query_only: bool) -> Union[DataFrame, str]: ...

    def _get_ts_report(self, get_query_only: bool = False):
        if self._bsq.ts_table is None:
            raise ValueError("No upgrade table is available .")

        ts_query = sa.select(
            *[self._bsq.ts_table.c["upgrade"], safunc.count(self._bsq.ts_bldgid_column.distinct()).label("count")]
        )
        ts_query = ts_query.group_by(sa.text("1"))
        ts_query = ts_query.order_by(sa.text("1"))
        if get_query_only:
            return self._bsq._compile(ts_query)
        df = self._bsq.execute(ts_query)
        df["upgrade"] = df["upgrade"].map(str)
        df = df.set_index("upgrade")
        df = df.rename(columns={"count": "success"})
        return df

    def check_ts_bs_integrity(self) -> bool:
        """Checks the integrity between the timeseries and baseline (metadata) tables.

        Returns:
            bool: Whether or not the integrity check passed.
        """
        logger.info("Checking integrity with ts_tables ...")
        raw_ts_report = self._get_ts_report()
        raw_success_report = self.get_success_report(trim_missing_bs=False)
        if self._bsq.db_schema.structure.inapplicables_have_ts:
            bs_dict = raw_success_report[["inapplicable", "success"]].sum(axis=1).to_dict()
        else:
            bs_dict = raw_success_report["success"].to_dict()
        ts_dict = raw_ts_report.to_dict()["success"]
        check_pass = True
        for upgrade, count in ts_dict.items():
            if count != bs_dict.get(upgrade, 0):
                print_r(
                    f"Upgrade {upgrade} has {count} samples in timeseries table, but {bs_dict.get(upgrade, 0)}"
                    " samples in baseline/upgrade table."
                )
                check_pass = False
        if check_pass:
            print_g("Annual and timeseries tables are verified to have the same number of buildings.")
        try:
            rowcount = self._bsq._get_rows_per_building()
            print_g(f"All buildings are verified to have the same number of ({rowcount}) timeseries rows.")
        except ValueError:
            check_pass = False
            print_r("Different buildings have different number of timeseries rows.")
        return check_pass

    @validate_arguments
    def get_successful_simulation_count(
        self,
        *,
        restrict: Sequence[tuple[AnyColType, Union[str, int, Sequence[Union[int, str]]]]] = Field(default_factory=list),
        get_query_only: bool = False,
    ):
        """
        Returns the count of successful simulation for the given restric condition in the baseline.
        Args:
            restrict: The list of where condition to restrict the results to. It should be specified as a list of tuple.
                      Example: `[('state',['VA','AZ']), ("build_existing_model.lighting",['60% CFL']), ...]`
            get_query_only: If set to true, returns the list of queries to run instead of the result.

        Returns:
            Pandas integer counting the number of successful simulation
        """
        query = sa.select(safunc.count().label("count"))

        restrict = list(restrict) if restrict else []
        restrict.insert(
            0, (self._bsq.db_schema.column_names.completed_status, [self._bsq.db_schema.completion_values.success])
        )
        query = self._bsq._add_restrict(query, restrict, bs_only=True)
        if get_query_only:
            return self._bsq._compile(query)

        return self._bsq.execute(query)

    @typing.overload
    def get_applied_options(
        self, *, upgrade_id: Union[str, int], bldg_ids: list[int], include_base_opt: Literal[True]
    ) -> list[dict[str, str]]: ...

    @typing.overload
    def get_applied_options(
        self, *, upgrade_id: Union[str, int], bldg_ids: list[int], include_base_opt: Literal[False] = False
    ) -> list[set[str]]: ...

    @typing.overload
    def get_applied_options(
        self, *, upgrade_id: Union[str, int], bldg_ids: list[int], include_base_opt: bool
    ) -> list[Union[dict[str, str], set[str]]]: ...

    @validate_arguments
    def get_applied_options(self, upgrade_id: Union[str, int], bldg_ids: list[int], include_base_opt: bool = False):
        """Returns the list of options applied to each buildings for a given upgrade.

        Args:
            upgrade_id (int | str): The upgrade for which to find the applied options.
            bldg_ids (list[int]): List of building ids.
            include_base_opt (bool, optional): If baseline value is to be included. Defaults to False.

        Returns:
            list[set|dict]: List of options (along with baseline chars, if include_base_opt is true)
        """
        up_csv = self._bsq.get_upgrades_csv_full(upgrade_id=upgrade_id)
        rel_up_csv = up_csv.loc[bldg_ids]
        upgrade_cols = [
            key for key in up_csv.columns if key.startswith("upgrade_costs.option_") and key.endswith("_name")
        ]

        if include_base_opt:
            base_csv = self._bsq.get_results_csv_full()
            rel_base_csv = base_csv.loc[bldg_ids]
            rel_base_csv = rel_base_csv.rename(columns=lambda c: c.split(".")[1] if "." in c else c)
            char_df = (
                rel_up_csv[upgrade_cols]
                .fillna("")
                .agg(lambda x: {"_".join(v.split("|")[0].lower().split()) for v in x if v}, axis=1)
            )
            all_chars = [c for c in reduce(set.union, char_df.values) if c in set(rel_base_csv.columns)]
            char_dict: dict[Hashable, dict[str, str]] = rel_base_csv[all_chars].to_dict(orient="index")

            def add_base_chars(options: list):
                bldg_id = options[0]  # first entry is building_id
                return {
                    opt: char_dict[bldg_id].get("_".join(opt.split("|")[0].lower().split()), "") for opt in options[1:]
                }

            opt_df: pd.Series = (
                rel_up_csv[upgrade_cols].fillna("").reset_index().agg(lambda x: [v for v in x if v], axis=1)
            )
            return_val = opt_df.map(add_base_chars).to_list()
        else:
            return_val = rel_up_csv[upgrade_cols].fillna("").agg(lambda x: {v for v in x if v}, axis=1).to_list()

        return return_val

    @validate_arguments
    def get_enduses_buildings_map_by_change(
        self, upgrade_id: Union[str, int], change_type: str = "changed", bldg_list: Optional[list[int]] = None
    ) -> dict[str, pd.Index]:
        """Finds the list of enduses and the buildings that had change in the enduses for a given change type.
        Args:
            upgrade (int | stsr): The upgrade to look at.
            change_type (str, optional): The kind of change to look for. Valid values are increased, decreased and
                                         and changed. Defaults to 'changed' which includes both cases.
            bldg_list (list[int], optional): The list of buildings to narrow down to. If omitted, searches through all
                                             all the buildings in the upgrade. Defaults to None.

        Returns:
            dict[str, pd.Index]: Dict mapping enduses that had a given change and building ids showing that change.
        """
        up_csv = self._bsq.get_upgrades_csv_full(upgrade_id=upgrade_id)
        bs_csv = self._bsq.get_results_csv_full()
        if bldg_list:
            up_csv = up_csv.loc[bldg_list]
            bs_csv = bs_csv.loc[bldg_list]

        def clean_column(col: str):
            col = col.removeprefix(self._bsq._out_prefix)
            col = col.removeprefix("end_use_")
            col = col.removeprefix("fuel_use_")
            return col

        def get_pure_enduse(col):
            for fuel in FUELS:
                col = col.removeprefix(f"{fuel}_")
            return col

        end_use_cols = [c for c in up_csv.columns if ("end_use" in c) or ("fuel_use" in c) or ("unmet_hours_" in c)]
        up_csv = up_csv[end_use_cols].rename(columns=clean_column)
        bs_csv = bs_csv[end_use_cols].rename(columns=clean_column)

        up_csv = up_csv.astype('float64', errors='ignore')
        bs_csv = bs_csv.astype('float64', errors='ignore')

        pure_enduses = {get_pure_enduse(c) for c in up_csv.columns}

        def get_all_fuel_enduses(df, end_use):
            return [col for col in df.columns if col.endswith(end_use)]

        def add_all_fuel_cols(df):
            # Collect all new columns first to avoid fragmentation
            new_cols = {}
            for end_use in pure_enduses:
                new_cols[f"all_fuel_{end_use}"] = df[get_all_fuel_enduses(df, end_use)].sum(axis=1)
            # Add all columns at once
            return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        up_csv = add_all_fuel_cols(up_csv)
        bs_csv = add_all_fuel_cols(bs_csv)

        diff = up_csv - bs_csv
        enduses_df = diff.transpose()
        if change_type == "decreased":
            enduses_df = enduses_df < -1e-12
        elif change_type == "increased":
            enduses_df = enduses_df > 1e-12
        else:
            enduses_df = enduses_df.abs() > 1e-12
        change_dict = enduses_df.apply(lambda x: enduses_df.columns[x], axis=1).to_dict()
        clean_dict = {key: value for key, value in change_dict.items() if len(value) > 0}
        return clean_dict
