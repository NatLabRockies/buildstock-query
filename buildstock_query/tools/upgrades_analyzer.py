from functools import reduce
import yaml
import pandas as pd
import numpy as np
import logging
from itertools import combinations
from typing import Union, Optional
from InquirerPy import inquirer
from InquirerPy.validator import PathValidator
import os
from collections import defaultdict
from pathlib import Path
from buildstock_query.tools.logic_parser import LogicParser
from tabulate import tabulate
from buildstock_query.helpers import read_csv, load_script_defaults, save_script_defaults
from buildstock_query.file_getter import OpenOrDownload
from buildstock_query.tools.set_cover import SetCoverSolver
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_COMBINATION_REPORT_COUNT = 5  # Don't print combination report; There would be 2^n - n - 1 rows


class UpgradesAnalyzer:
    """
    Analyze the apply logic for various upgrades in the project yaml file.
    """

    def __init__(
        self,
        *,
        buildstock: Union[str, pd.DataFrame],
        yaml_file: Optional[str] = None,
        opt_sat_file: str,
        filter_yaml_file: Optional[str] = None,
        upgrade_names: Optional[dict[int, str]] = None,
    ) -> None:
        """
        Initialize the analyzer instance.
        Args:
            buildstock (Union[str, pd.DataFrame]): Either the buildstock dataframe, or path to the csv
            opt_sat_file (str): The path to the option saturation file.
            yaml_file (str): The path to the yaml file.
            filter_yaml_file (str): The path to the filter yaml file.
            upgrade_names (dict[int, str]): A dictionary of upgrade number to upgrade name. This
                needs to be provided if only the filter_yaml_file is provided.
        """
        self.parser = LogicParser(opt_sat_file, yaml_file)
        self.yaml_file = yaml_file
        self.filter_yaml_file = filter_yaml_file
        if not self.yaml_file and not self.filter_yaml_file:
            raise ValueError("Either yaml_file or filter_yaml_file must be provided")

        if self.yaml_file and upgrade_names:
            raise ValueError(
                "upgrade_names must not be provided if yaml_file is provided. It will be read from yaml file"
            )

        if not self.yaml_file and not upgrade_names:
            raise ValueError("upgrade_names must be provided if only filter_yaml_file is provided")

        self.cfg = self.get_cfg(yaml_file) if yaml_file else {}
        if not upgrade_names:
            self.upgrade_names = {
                indx + 1: upgrade["upgrade_name"] for indx, upgrade in enumerate(self.cfg["upgrades"])
            }
        else:
            self.upgrade_names = upgrade_names

        self.filter_cfg = self.get_filter_cfg(filter_yaml_file) if filter_yaml_file else {}

        if isinstance(buildstock, str):
            self.buildstock_df_original = read_csv(buildstock, dtype=str)
            self.buildstock_df = self.buildstock_df_original.copy()
            self.buildstock_df.columns = [c.lower() for c in self.buildstock_df.columns]
            self.buildstock_df.rename(columns={"building": "building_id"}, inplace=True)
            self.buildstock_df.set_index("building_id", inplace=True)
        elif isinstance(buildstock, pd.DataFrame):
            self.buildstock_df_original = buildstock.copy()
            self.buildstock_df = buildstock.reset_index().rename(columns=str.lower)
            self.buildstock_df.rename(columns={"building": "building_id"}, inplace=True)
            if "building_id" in self.buildstock_df.columns:
                self.buildstock_df.set_index("building_id", inplace=True)
            self.buildstock_df = self.buildstock_df.astype(str)
        self.total_samples = len(self.buildstock_df)
        self._logic_cache: dict = {}

    def get_cfg(self, yaml_file) -> dict:
        """Get the buildstock configuration file as a dictionary object.

        Returns:
            dict: The buildstock configuration file.
        """
        with OpenOrDownload(yaml_file) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config

    def get_filter_cfg(self, filter_yaml_file) -> dict:
        """Get the filter yaml file as a dictionary object.

        Returns:
            dict: The buildstock configuration file.
        """
        with OpenOrDownload(filter_yaml_file) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        listed_upgrades = set(upgrade["upgrade_name"] for upgrade in config.get("upgrades", []))
        missing_upgrades = set(self.upgrade_names.values()) - listed_upgrades
        new_config = {}
        all_upgrades_remove_logic = config.get("all_upgrades_remove_logic", {})
        for upgrade in config.get("upgrades", []):
            if all_upgrades_remove_logic:
                new_config[upgrade["upgrade_name"]] = {"or": [all_upgrades_remove_logic, upgrade["remove_logic"]]}
            else:
                new_config[upgrade["upgrade_name"]] = upgrade["remove_logic"]
        for upgrade_name in missing_upgrades:
            new_config[upgrade_name] = all_upgrades_remove_logic

        return new_config

    def get_filtered_bldgs(self, upgrade_name):
        """Get the boolean array of filtered buildings for a given upgrade

        Returns:
            dict: The buildstock configuration file.
        """
        filtered_bldgs = np.zeros((1, self.total_samples), dtype=bool)
        if not self.filter_yaml_file:
            return filtered_bldgs
        with OpenOrDownload(self.filter_yaml_file) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        logic = config.get("all_upgrades_remove_logic")
        for upgrade in config.get("upgrades", []):
            if upgrade["upgrade_name"] == upgrade_name:
                if logic:
                    logic = {"and": [logic, upgrade["remove_logic"]]}
                else:
                    logic = upgrade["remove_logic"]
                break
        return self._reduce_logic(logic, parent=None)

    @staticmethod
    def _get_eq_str(condition):
        para, option = UpgradesAnalyzer._get_para_option(condition)
        return f"`{para.lower()}`=='{option}'"

    @staticmethod
    def _get_para_option(condition):
        try:
            para, option = condition.split("|")
        except ValueError as e:
            raise ValueError(f"Condition {condition} is invalid") from e
        return para.lower(), option

    @staticmethod
    def get_mentioned_parameters(logic: Union[list, dict, str]) -> list:
        """
        Returns the list of all parameters referenced in a logic block. Useful for debugging

        Args:
            logic ( Union[list, dict, str]): The apply logic

        Raises:
            ValueError: If the input logic is invalid

        Returns:
            List: The list of parameters
        """
        if not logic:
            return []

        if isinstance(logic, str):
            return [UpgradesAnalyzer._get_para_option(logic)[0]]
        elif isinstance(logic, list):
            all_params = []
            for el in logic:
                all_params.extend(UpgradesAnalyzer.get_mentioned_parameters(el))
            return list(dict.fromkeys(all_params))  # remove duplicates while maintaining order
        elif isinstance(logic, dict):
            return UpgradesAnalyzer.get_mentioned_parameters(list(logic.values())[0])
        else:
            raise ValueError("Invalid logic type")

    def print_unique_characteristic(self, upgrade_num: int, name: str, base_bldg_list: list, compare_bldg_list: list):
        """Finds and prints what's unique among a list of buildings compared to baseline buildings.
           Useful for debugging why a certain set of buildings' energy consumption went up for an upgrade, for example.
        Args:
            upgrade_num (int): The upgrade for which the analysis is being done.
            name (str): Some name to identify the building set (only used for printing)
            base_bldg_list (list): The set of 'normal' buildings id to compare against.
            compare_bldg_list (list): The set of buildings whose unique characteristics is to be printed.
        """
        cfg = self.cfg
        if upgrade_num == 0:
            raise ValueError(f"Upgrades are 1-indexed. Got {upgrade_num}")

        try:
            upgrade_cfg = cfg["upgrades"][upgrade_num - 1]
        except KeyError as e:
            raise ValueError(f"Invalid upgrade {upgrade_num}. Upgrades are 1-indexed, FYI.") from e

        parameter_list = []
        for option_cfg in upgrade_cfg["options"]:
            parameter_list.append(UpgradesAnalyzer._get_para_option(option_cfg["option"])[0])
            parameter_list.extend(UpgradesAnalyzer.get_mentioned_parameters(option_cfg.get("apply_logic")))
        res_df = self.buildstock_df
        # remove duplicates (dict.fromkeys) and remove parameters not existing in buildstock_df
        parameter_list = [param for param in dict.fromkeys(parameter_list) if param in res_df.columns]
        compare_df = res_df.loc[compare_bldg_list]
        base_df = res_df.loc[base_bldg_list]
        print(f"Comparing {len(compare_df)} buildings with {len(base_df)} other buildings.")
        unique_vals_dict: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
        for col in res_df.columns:
            no_change_set = set(compare_df[col].fillna("").unique())
            other_set = set(base_df[col].fillna("").unique())
            if only_in_no_change := no_change_set - other_set:
                print(f"Only {name} buildings have {col} in {sorted(only_in_no_change)}")
                unique_vals_dict[(col,)] = {(entry,) for entry in only_in_no_change}

        if not unique_vals_dict:
            print("No 1-column unique chracteristics found.")

        for combi_size in range(2, min(len(parameter_list) + 1, 5)):
            print(f"Checking {combi_size} column combinations out of {parameter_list}")
            found_uniq_chars = 0
            for cols in combinations(parameter_list, combi_size):
                compare_tups = compare_df[list(cols)].fillna("").drop_duplicates().itertuples(index=False, name=None)
                other_tups = base_df[list(cols)].fillna("").drop_duplicates().itertuples(index=False, name=None)
                only_in_compare = set(compare_tups) - set(other_tups)

                # remove cases arisen out of uniqueness found earlier with smaller susbset of cols
                for sub_combi_size in range(1, len(cols)):
                    for sub_cols in combinations(cols, sub_combi_size):
                        if sub_cols in unique_vals_dict:
                            new_set = set()
                            for val in only_in_compare:
                                relevant_val = tuple(val[cols.index(sub_col)] for sub_col in sub_cols)
                                if relevant_val not in unique_vals_dict[sub_cols]:
                                    new_set.add(val)
                            only_in_compare = new_set

                if only_in_compare:
                    print(f"Only {name} buildings have {cols} in {sorted(only_in_compare)} \n")
                    found_uniq_chars += 1
                    unique_vals_dict[cols] = only_in_compare

            if not found_uniq_chars:
                print(f"No {combi_size}-column unique chracteristics found.")

    def _reduce_logic(self, logic, parent=None):
        cache_key = str(logic) if parent is None else parent + "[" + str(logic) + "]"
        if cache_key in self._logic_cache:
            return self._logic_cache[cache_key]

        logic_array = np.ones((1, self.total_samples), dtype=bool)
        if parent not in [None, "and", "or", "not"]:
            raise ValueError(f"Logic can only include and, or, not blocks. {parent} found in {logic}.")

        if isinstance(logic, str):
            para, opt = UpgradesAnalyzer._get_para_option(logic)
            logic_array = self.buildstock_df[para] == opt
        elif isinstance(logic, list):
            if len(logic) == 1:
                logic_array = self._reduce_logic(logic[0]).copy()
            elif parent in ["or"]:
                logic_array = reduce(
                    lambda l1, l2: l1 | self._reduce_logic(l2),
                    logic,
                    np.zeros((1, self.total_samples), dtype=bool),
                )
            else:
                logic_array = reduce(
                    lambda l1, l2: l1 & self._reduce_logic(l2),
                    logic,
                    np.ones((1, self.total_samples), dtype=bool),
                )
        elif isinstance(logic, dict):
            if len(logic) > 1:
                raise ValueError(f"Dicts cannot have more than one keys. {logic} has.")
            key = list(logic.keys())[0]
            logic_array = self._reduce_logic(logic[key], parent=key).copy()

        if parent == "not":
            return ~logic_array
        if not (isinstance(logic, str) or (isinstance(logic, list) and len(logic) == 1)):
            # Don't cache small logics - computing them again won't be too bad
            self._logic_cache[cache_key] = logic_array.copy()
        return logic_array

    def _get_application_report(self, upgrade_num, upgrade_name):
        records = []
        logger.info(f"Analyzing upgrade {upgrade_num}")
        all_applied_bldgs = np.zeros((1, self.total_samples), dtype=bool)
        all_to_remove_bldgs = np.zeros((1, self.total_samples), dtype=bool)
        package_applied_bldgs = np.ones((1, self.total_samples), dtype=bool)
        candidate_to_remove_bldgs = np.zeros((1, self.total_samples), dtype=bool)
        if self.cfg:
            upgrade = self.cfg["upgrades"][upgrade_num - 1]
        else:
            upgrade = {"upgrade_name": upgrade_name, "options": []}
            # If only filter_yaml_file is provided, we don't have the upgrade yaml. So, we need to
            # assume all the candidate to remove bldgs in the filter yaml as the final set of
            # to remove bldgs

            # don't copy, assign by reference because candidate_to_remove_bldgs is to be modified later
            # with the removal logic
            all_to_remove_bldgs = candidate_to_remove_bldgs

        if "package_apply_logic" in upgrade:
            pkg_flat_logic = UpgradesAnalyzer._normalize_lists(upgrade["package_apply_logic"])
            package_applied_bldgs = self._reduce_logic(pkg_flat_logic, parent=None)

        if remove_logic := self.filter_cfg.get(upgrade["upgrade_name"]):
            remove_logic = UpgradesAnalyzer._normalize_lists(remove_logic)
            candidate_to_remove_bldgs |= self._reduce_logic(remove_logic, parent=None)

        for opt_index, option in enumerate(upgrade["options"]):
            applied_bldgs = np.ones((1, self.total_samples), dtype=bool)
            if "apply_logic" in option:
                flat_logic = UpgradesAnalyzer._normalize_lists(option["apply_logic"])
                applied_bldgs &= self._reduce_logic(flat_logic, parent=None)
            else:
                applied_bldgs = np.ones((1, self.total_samples), dtype=bool)

            to_remove_buildings = np.ones((1, self.total_samples), dtype=bool)
            to_remove_buildings &= applied_bldgs & candidate_to_remove_bldgs
            applied_bldgs &= package_applied_bldgs
            applied_bldgs &= ~candidate_to_remove_bldgs
            count = applied_bldgs.sum()
            all_applied_bldgs |= applied_bldgs
            all_to_remove_bldgs |= to_remove_buildings
            record = {
                "upgrade": upgrade_num,
                "upgrade_name": upgrade["upgrade_name"],
                "option_num": opt_index + 1,
                "option": option["option"],
                "applicable_to": count,
                "applicable_percent": self._to_pct(count),
                "applicable_buildings": set(self.buildstock_df.loc[applied_bldgs[0]].index),
            }
            if remove_logic:
                record["removal_count"] = to_remove_buildings.sum()
                record["removal_percent"] = self._to_pct(record["removal_count"])
                record["removal_buildings"] = set(self.buildstock_df.loc[to_remove_buildings[0]].index)

            records.append(record)
        count = all_applied_bldgs.sum()
        record = {
            "upgrade": upgrade_num,
            "upgrade_name": upgrade_name,
            "option_num": -1,
            "option": "All",
            "applicable_to": count,
            "applicable_buildings": set(self.buildstock_df.loc[all_applied_bldgs[0]].index),
            "applicable_percent": self._to_pct(count),
        }
        if remove_logic:
            record["removal_count"] = all_to_remove_bldgs.sum()
            record["removal_percent"] = self._to_pct(record["removal_count"])
            record["removal_buildings"] = set(self.buildstock_df.loc[all_to_remove_bldgs[0]].index)
        records.append(record)
        return pd.DataFrame.from_records(records)

    def get_report(self, upgrade_num: Optional[int] = None) -> pd.DataFrame:
        """Analyses how many buildings various options in all the upgrades is going to apply to and returns
        a report in DataFrame format.
        Args:
            upgrade_num: Numeric index of upgrade (1-indexed). If None, all upgrades are assessed

        Returns:
            pd.DataFrame: The upgrade and options report.

        """

        self._logic_cache = {}
        if upgrade_num is not None and upgrade_num not in self.upgrade_names:
            raise ValueError(f"Invalid upgrade {upgrade_num}. Valid upgrade_num = {self.upgrade_names.keys()}.")

        record_dfs = []
        for indx, upgrade_names in self.upgrade_names.items():
            if upgrade_num is None or upgrade_num == indx:
                record_dfs.append(self._get_application_report(indx, upgrade_names))
            else:
                continue

        report_df = pd.concat(record_dfs)
        return report_df

    def get_upgraded_buildstock(self, upgrade_num):
        report_df = self.get_report(upgrade_num)
        upgrade_name = report_df["upgrade_name"].unique()[0]
        logger.info(f" * Upgraded buildstock for upgrade {upgrade_num} : {upgrade_name}")

        df = self.buildstock_df_original.copy()
        for idx, row in report_df.iterrows():
            if row["option"] == "All":
                continue
            dimension, upgrade_option = row["option"].split("|")
            apply_logic = df["Building"].isin(row["applicable_buildings"])
            # apply upgrade
            df[dimension] = np.where(apply_logic, upgrade_option, df[dimension])

        # report
        cond = report_df["option"] == "All"
        n_total = len(self.buildstock_df_original)
        n_applied = report_df.loc[cond, "applicable_to"].iloc[0]
        n_applied_pct = report_df.loc[cond, "applicable_percent"].iloc[0]
        logger.info(
            f"   Upgrade package has {len(report_df) - 1} options and "
            f"was applied to {n_applied} / {n_total} dwelling units ( {n_applied_pct} % )"
        )

        # QC
        n_diff = len(self.buildstock_df_original.compare(df)) - n_applied
        if n_diff > 0:
            raise ValueError(
                f"Relative to baseline buildstock, upgraded buildstock has {n_diff} more rows "
                "of difference than reported."
            )
        elif n_diff < 0:
            logger.warning(
                f"Relative to baseline buildstock, upgraded buildstock has {-1 * n_diff} fewer rows "
                "of difference than reported. This is okay, but indicates that some parameters are "
                "being upgraded to the same incumbent option (e.g., LEDs to LEDs). Check that this is intentional."
            )
        else:
            logger.info("No cases of parameter upgraded with incumbent option detected.")

        return df

    @staticmethod
    def _normalize_lists(logic, parent=None):
        """Any list that is not in a or block is considered to be in an and block.
           This block will normalize this pattern by adding "and" wherever required.
        Args:
            logic (_type_): Logic structure (dict, list etc)
            parent (_type_, optional): The parent of the current logic block. If it is a list, and there is no parent,
            the list will be wrapped in a and block.

        Returns:
            _type_: _description_
        """
        if isinstance(logic, list):
            # If it is a single element list, just unwrap and return
            if len(logic) == 1:
                return UpgradesAnalyzer._normalize_lists(logic[0])
            new_logic = [UpgradesAnalyzer._normalize_lists(el) for el in logic]
            return {"and": new_logic} if parent is None else new_logic
        elif isinstance(logic, dict):
            new_dict = {key: UpgradesAnalyzer._normalize_lists(value, parent=key) for key, value in logic.items()}
            return new_dict
        else:
            return logic

    def _get_options_application_count_report(self, logic_dict) -> pd.DataFrame:
        """
        For a given logic dictionary, this method will return a report df of options application.
        Example report below:
                           Applied options Applied buildings Cumulative sub Cumulative all
        Number of options
        4                    1, 10, 13, 14         75 (0.1%)      75 (0.1%)      75 (0.1%)
        4                    1, 11, 13, 14       2279 (2.3%)    2354 (2.4%)    2354 (2.4%)
        4                    1, 12, 13, 14        309 (0.3%)    2663 (2.7%)    2663 (2.7%)
        5                  1, 2, 3, 13, 14          8 (0.0%)       8 (0.0%)    2671 (2.7%)
        5                  1, 2, 4, 13, 14        158 (0.2%)     166 (0.2%)    2829 (2.8%)
        5                  1, 2, 5, 13, 14         65 (0.1%)     231 (0.2%)    2894 (2.9%)
        5                  1, 6, 7, 13, 14         23 (0.0%)     254 (0.3%)    2917 (2.9%)
        5                  1, 6, 8, 13, 14         42 (0.0%)     296 (0.3%)    2959 (3.0%)
        """

        logic_df = pd.DataFrame(logic_dict)
        nbldgs = len(logic_df)
        opts2count = (
            logic_df.apply(lambda row: tuple(indx + 1 for indx, val in enumerate(row) if val), axis=1)
            .value_counts()
            .to_dict()
        )
        cum_count_all = 0
        cum_count = defaultdict(int)
        application_report_rows = []
        for applied_opts in sorted(opts2count.keys(), key=lambda x: (len(x), x)):
            num_opt = len(applied_opts)
            if num_opt == 0:
                continue
            n_applied_bldgs = opts2count[applied_opts]
            cum_count_all += n_applied_bldgs
            cum_count[num_opt] += n_applied_bldgs
            record = {
                "Number of options": num_opt,
                "Applied options": ", ".join([f"{logic_df.columns[opt - 1]}" for opt in applied_opts]),
                "Applied buildings": f"{n_applied_bldgs} ({self._to_pct(n_applied_bldgs, nbldgs)}%)",
                "Cumulative sub": f"{cum_count[num_opt]} ({self._to_pct(cum_count[num_opt], nbldgs)}%)",
                "Cumulative all": f"{cum_count_all} ({self._to_pct(cum_count_all, nbldgs)}%)",
            }
            application_report_rows.append(record)

        assert cum_count_all <= nbldgs, "Cumulative count of options applied is more than total number of buildings."
        return pd.DataFrame(application_report_rows).set_index("Number of options")

    def _get_left_out_report_all(self, upgrade_num):
        cfg = self.cfg
        report_str = ""
        upgrade = cfg["upgrades"][upgrade_num - 1]
        ugrade_name = upgrade.get("upgrade_name")
        header = f"Left Out Report for - Upgrade{upgrade_num}:'{ugrade_name}'"
        report_str += "-" * len(header) + "\n"
        report_str += header + "\n"
        report_str += "-" * len(header) + "\n"
        logic = {"or": []}
        for opt in upgrade["options"]:
            if "apply_logic" in opt:
                logic["or"].append(self._normalize_lists(opt["apply_logic"]))
        if "package_apply_logic" in upgrade:
            logic = {"and": [logic, upgrade["package_apply_logic"]]}
        logic = {"not": logic}  # invert it

        if remove_logic := self.filter_cfg.get(upgrade["upgrade_name"]):
            logic = {"or": [logic, remove_logic]}

        logic = self.parser.normalize_logic(logic)
        logic_array, logic_str = self._get_logic_report(logic)
        footer_len = len(logic_str[-1])
        report_str += "\n".join(logic_str) + "\n"
        report_str += "-" * footer_len + "\n"
        count = logic_array.sum()
        footer_str = f"Overall Not Applied to => {count} ({self._to_pct(count)}%)."
        report_str += footer_str + "\n"
        report_str += "-" * len(footer_str) + "\n"
        return logic_array, report_str

    def get_left_out_report(self, upgrade_num: int, option_num: Optional[int] = None) -> tuple[np.ndarray, str]:
        """Prints detailed inverse report of what is left out for a particular upgrade (and optionally, an option)
        Args:
            upgrade_num (int): The 1-indexed upgrade for which to print the report.
            option_num (int, optional): The 1-indexed option number for which to print report. Defaults to None, which
                                        will print report for all options.
            normalize_logic (bool, optional): Whether to normalize the logic structure. Defaults to False.
        Returns:
            (np.ndarray, str): Returns a logic array of buildings to which the any of the option applied and report str.
        """
        cfg = self.cfg
        if upgrade_num <= 0 or upgrade_num > len(cfg["upgrades"]) + 1:
            raise ValueError(f"Invalid upgrade {upgrade_num}. Upgrade num is 1-indexed.")

        if option_num is None:
            return self._get_left_out_report_all(upgrade_num)

        self._logic_cache = {}
        if upgrade_num == 0 or option_num == 0:
            raise ValueError(f"Upgrades and options are 1-indexed.Got {upgrade_num} {option_num}")
        report_str = ""
        try:
            upgrade = cfg["upgrades"][upgrade_num - 1]
            opt = upgrade["options"][option_num - 1]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"The yaml doesn't have {upgrade_num}/{option_num} upgrade/option") from e

        ugrade_name = upgrade.get("upgrade_name")
        header = f"Left Out Report for - Upgrade{upgrade_num}:'{ugrade_name}', Option{option_num}:'{opt['option']}'"
        report_str += "-" * len(header) + "\n"
        report_str += header + "\n"
        report_str += "-" * len(header) + "\n"
        if "apply_logic" in opt and "package_apply_logic" in upgrade:
            logic = {"not": {"and": [opt["apply_logic"], upgrade["package_apply_logic"]]}}
        elif "apply_logic" in opt:
            logic = {"not": opt["apply_logic"]}
        else:
            logic = {"not": upgrade["package_apply_logic"]}

        if remove_logic := self.filter_cfg.get(upgrade["upgrade_name"]):
            logic = {"or": [logic, remove_logic]}

        logic = self.parser.normalize_logic(logic)

        logic_array, logic_str = self._get_logic_report(logic)
        footer_len = len(logic_str[-1])
        report_str += "\n".join(logic_str) + "\n"
        report_str += "-" * footer_len + "\n"
        count = logic_array.sum()
        footer_str = f"Overall Not Applied to => {count} ({self._to_pct(count)}%)."
        report_str += footer_str + "\n"
        report_str += "-" * len(footer_str) + "\n"
        return logic_array, report_str

    def get_detailed_report(
        self, upgrade_num: int, option_num: Optional[int] = None, normalize_logic: bool = False
    ) -> tuple[np.ndarray, str, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Prints detailed report for a particular upgrade (and optionally, an option)
        Args:
            upgrade_num (int): The 1-indexed upgrade for which to print the report.
            option_num (int, optional): The 1-indexed option number for which to print report. Defaults to None, which
                                        will print report for all options.
            normalize_logic (bool, optional): Whether to normalize the logic structure. Defaults to False.
        Returns:
            (np.ndarray, str, Optional[pd.DataFrame], Optional[pd.DataFrame]): Returns a tuple with 4 elements
                logic array of buildings to which the any of the option applied
                the report in string format.
                Optionally (if option_num is None), option combination report where same parameters are grouped
                Optionally (if option_num is None), option combination detailed report where options are kept separate.
        """
        cfg = self.cfg
        if upgrade_num <= 0 or upgrade_num > len(cfg["upgrades"]) + 1:
            raise ValueError(f"Invalid upgrade {upgrade_num}. Upgrade num is 1-indexed.")

        if option_num is None:
            return self._get_detailed_report_all(upgrade_num, normalize_logic=normalize_logic)

        self._logic_cache = {}
        if upgrade_num == 0 or option_num == 0:
            raise ValueError(f"Upgrades and options are 1-indexed.Got {upgrade_num} {option_num}")
        report_str = ""
        try:
            upgrade = cfg["upgrades"][upgrade_num - 1]
            opt = upgrade["options"][option_num - 1]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"The yaml doesn't have {upgrade_num}/{option_num} upgrade/option") from e

        ugrade_name = upgrade.get("upgrade_name")
        header = f"Option Apply Report for - Upgrade{upgrade_num}:'{ugrade_name}', Option{option_num}:'{opt['option']}'"
        report_str += "-" * len(header) + "\n"
        report_str += header + "\n"
        report_str += "-" * len(header) + "\n"
        if "apply_logic" in opt:
            logic = UpgradesAnalyzer._normalize_lists(opt["apply_logic"])
            logic = self.parser.normalize_logic(logic) if normalize_logic else logic
            logic_array, logic_str = self._get_logic_report(logic)
            footer_len = len(logic_str[-1])
            report_str += "\n".join(logic_str) + "\n"
            report_str += "-" * footer_len + "\n"
        else:
            logic_array = np.ones((1, self.total_samples), dtype=bool)

        if "package_apply_logic" in upgrade:
            logic = UpgradesAnalyzer._normalize_lists(upgrade["package_apply_logic"])
            logic = self.parser.normalize_logic(logic) if normalize_logic else logic
            package_logic_array, logic_str = self._get_logic_report(logic)
            footer_len = len(logic_str[-1])
            report_str += "Package Apply Logic Report" + "\n"
            report_str += "--------------------------" + "\n"
            report_str += "\n".join(logic_str) + "\n"
            report_str += "-" * footer_len + "\n"
            logic_array = logic_array & package_logic_array

        if remove_logic := self.filter_cfg.get(upgrade["upgrade_name"]):
            remove_logic = UpgradesAnalyzer._normalize_lists(remove_logic)
            remove_logic = self.parser.normalize_logic(remove_logic) if normalize_logic else remove_logic
            remove_logic_array, logic_str = self._get_logic_report(remove_logic)
            footer_len = len(logic_str[-1])
            report_str += "Remove Logic Report" + "\n"
            report_str += "-------------------" + "\n"
            report_str += "\n".join(logic_str) + "\n"
            report_str += "-" * footer_len + "\n"
            removed_after_apply = remove_logic_array & logic_array
            report_str += "Removed after applying => " + f"{removed_after_apply.sum()} "
            report_str += f"({self._to_pct(removed_after_apply.sum(), logic_array.sum())}% of applied)" + "\n"
            logic_array = logic_array & ~remove_logic_array

        count = logic_array.sum()
        footer_str = f"Overall applied to => {count} ({self._to_pct(count)}%)."
        report_str += footer_str + "\n"
        report_str += "-" * len(footer_str) + "\n"
        return logic_array, report_str, None, None

    def _get_detailed_report_all(self, upgrade_num, normalize_logic: bool = False):
        conds_dict = {}
        grouped_conds_dict = {}
        cfg = self.cfg
        report_str = ""
        n_options = len(cfg["upgrades"][upgrade_num - 1]["options"])
        or_array = np.zeros((1, self.total_samples), dtype=bool)
        and_array = np.ones((1, self.total_samples), dtype=bool)
        for option_indx in range(n_options):
            logic_array, sub_report_str, _, _ = self.get_detailed_report(
                upgrade_num, option_indx + 1, normalize_logic=normalize_logic
            )
            opt_name, _ = self._get_para_option(cfg["upgrades"][upgrade_num - 1]["options"][option_indx]["option"])
            report_str += sub_report_str + "\n"
            conds_dict[option_indx + 1] = logic_array.copy()
            if opt_name not in grouped_conds_dict:
                grouped_conds_dict[opt_name] = logic_array.copy()
            else:
                grouped_conds_dict[opt_name] |= logic_array
            or_array |= logic_array
            and_array &= logic_array
        and_count = and_array.sum()
        or_count = or_array.sum()
        report_str += f"All of the options (and-ing) were applied to: {and_count} ({self._to_pct(and_count)}%)" + "\n"
        report_str += f"Any of the options (or-ing) were applied to: {or_count} ({self._to_pct(or_count)}%)" + "\n"

        option_app_report_df = self._get_options_application_count_report(grouped_conds_dict)
        report_str += "-" * 80 + "\n"
        report_str += f"Report of how the {len(grouped_conds_dict)} options were applied to the buildings." + "\n"
        report_str += tabulate(option_app_report_df, headers="keys", tablefmt="grid", maxcolwidths=50) + "\n"

        detailed_app_report_df = self._get_options_application_count_report(conds_dict)
        report_str += "-" * 80 + "\n"
        if len(detailed_app_report_df) > 100:
            report_str += "Detailed report is skipped because of too many rows. " + "\n"
            report_str += "Ask the developer if this is useful to see" + "\n"
        else:
            report_str += f"Detailed report of how the {n_options} options were applied to the buildings." + "\n"
            report_str += tabulate(detailed_app_report_df, headers="keys", tablefmt="grid", maxcolwidths=50) + "\n"
        return or_array, report_str, option_app_report_df, detailed_app_report_df

    def _to_pct(self, count, total=None):
        total = total or self.total_samples
        return round(100 * count / total, 1)

    def _get_logic_report(self, logic, parent=None):
        logic_array = np.ones((1, self.total_samples), dtype=bool)
        logic_str = [""]
        if parent not in [None, "and", "or", "not"]:
            raise ValueError(f"Logic can only include and, or, not blocks. {parent} found in {logic}.")
        if isinstance(logic, str):
            logic_condition = UpgradesAnalyzer._get_eq_str(logic)
            logic_array = self.buildstock_df.eval(logic_condition, engine="python")
            count = logic_array.sum()
            logic_str = [logic + " => " + f"{count} ({self._to_pct(count)}%)"]
        elif isinstance(logic, list):
            if len(logic) == 1:
                logic_array, logic_str = self._get_logic_report(logic[0])
            elif parent in ["or"]:

                def reducer(l1, l2):
                    ll2 = self._get_logic_report(l2)
                    return l1[0] | ll2[0], l1[1] + ll2[1]

                logic_array, logic_str = reduce(reducer, logic, (np.zeros((1, self.total_samples), dtype=bool), []))
            else:

                def reducer(l1, l2):
                    ll2 = self._get_logic_report(l2)
                    return l1[0] & ll2[0], l1[1] + ll2[1]

                logic_array, logic_str = reduce(reducer, logic, (np.ones((1, self.total_samples), dtype=bool), []))
        elif isinstance(logic, dict):
            if len(logic) > 1:
                raise ValueError(f"Dicts cannot have more than one keys. {logic} has.")
            key = list(logic.keys())[0]
            sub_logic = self._get_logic_report(logic[key], parent=key)
            sub_logic_str = sub_logic[1]
            logic_array = sub_logic[0]
            if key == "not":
                logic_array = ~logic_array
            count = logic_array.sum()
            header_str = key + " => " + f"{count} ({self._to_pct(count)}%)"
            logic_str = [header_str] + [f"  {ls}" for ls in sub_logic_str]

        count = logic_array.sum()
        if parent is None and isinstance(logic, list) and len(logic) > 1:
            logic_str[0] = logic_str[0] + " => " + f"{count} ({self._to_pct(count)}%)"

        return logic_array, logic_str

    def save_detailed_report_all(self, file_path: str, normalize_logic: bool = False):
        """Save detailed text based upgrade report.

        Args:
            file_path (str): Output file.
        """
        cfg = self.cfg
        all_report = ""
        for upgrade in range(1, len(cfg["upgrades"]) + 1):
            logger.info(f"Getting report for upgrade {upgrade}")
            _, report, _, _ = self.get_detailed_report(upgrade, normalize_logic=normalize_logic)
            all_report += report + "\n"
        with open(file_path, "w") as file:
            file.write(all_report)

    def get_characteristics(self, *, max_cardinality: int, min_cardinality: int):
        """Get all characteristics with cardinality between min_cardinality and max_cardinality.
        Used in get_minimal_representative_buildings to ensure coverage for all characteristics
        that have reasonable cardinality (for example - try to cover all heating fuels (cardinality=7)
        but not all counties (cardinality=3140)).
        """
        characteristics = []
        for col in self.buildstock_df.columns.to_list():
            if min_cardinality <= self.buildstock_df[col].unique().size <= max_cardinality:
                characteristics.append(col)
        return characteristics

    def compute_preference_weights(self, soft_avoid_chars: list[tuple[str, str]]) -> dict[int, tuple[int, int]]:
        """Compute preference weights so buildings matching avoided chars are less likely to be picked.

        Returns negative weights for matching buildings: (-match_count, sum_of_matched_indices).
        More matches = lower weight. Among equal match counts, matching a later (less important)
        condition gives a higher weight. Buildings with no matches are omitted (implicit weight (0, 0)).

        Args:
            soft_avoid_chars: (column, value) tuples ordered by importance (first = most important to avoid).
        """
        match_count = np.zeros(len(self.buildstock_df), dtype=int)
        index_sum = np.zeros(len(self.buildstock_df), dtype=int)
        for i, (col, val) in enumerate(soft_avoid_chars):
            mask = (self.buildstock_df[col] == val).to_numpy()
            match_count += mask
            index_sum += mask * i
        has_match = match_count > 0
        return {
            bldg_id: (-int(mc), int(ix))
            for bldg_id, mc, ix, hm in zip(self.buildstock_df.index, match_count, index_sum, has_match)
            if hm
        }

    def get_minimal_representative_buildings(
        self,
        report_df: pd.DataFrame,
        previous_minimal_bldgs: list | None = None,
        include_never_upgraded=False,
        verbose=False,
        must_cover_chars=None,
        soft_avoid_chars: list[tuple[str, str]] | None = None,
    ) -> list:
        """Return a minimal set of buildings that covers all the building_groups.
        In other words, it returns a new set of buildings that has non-zero intersection with all sets.
        It uses greedy algorithm to solve the set cover problem - while not optimal (It's a NP-complete problem),
        typical input cases results in sufficiently small output set. The tie-breaking for greedy algorithm is done by
        choosing the last inserted building in the bucket queue, which is typically the building_id with highest index.
        but is not always the case. This results in a deterministic output.

        Parameters
        ----------
        building_groups : list[set[int]]
            List of set of buildings belonging to different apply_logic blocks for different upgrade and options.
            One set per option per upgrade.
        include_never_upgraded : bool, optional
            Whether to include one building from never_upgraded_buildings in the output. Defaults to False.
        must_cover_chars : list[int], optional
            List of building characteristics to guarantee coverage in the output. Defaults to None.
        verbose : bool, optional
            Whether to print verbose output. Defaults to False.
        soft_avoid_chars : list[tuple[str, str]], optional
            List of (column_name, value_to_avoid) tuples ordered by importance (first = most important to avoid).
            Buildings matching these characteristics are deprioritized during tie-breaking. Coverage guarantees
            are preserved — at least one building per characteristic value will still be included.

        Returns
        -------
        list[int]
            A minimal set of buildings so that when buildstock.csv is trimmed to include only these buildings,
            then all of the options will apply to at least one building and the output set is minimal. If
            include_never_upgraded is True, one building (of largest building_id) from never_upgraded_buildings
            is appended to the output.
        """
        # option_num = -1 corresponds to building that is union of all listed option_num. Since this is a
        # redundant group (i.e. if all options are covered, this group will be covered too), we remove it
        # for efficiency.
        must_cover_chars = must_cover_chars or []
        groups = report_df[report_df["option_num"] != -1]["applicable_buildings"].to_list()

        if include_never_upgraded:
            covered_bldgs = set.union(*groups) if groups else set()
            uncovered_buildings = set(self.buildstock_df.index) - covered_bldgs
            groups.append(list(uncovered_buildings))

        for char in must_cover_chars:
            unique_char_val = self.buildstock_df[char].unique()
            for char_val in unique_char_val:
                group = self.buildstock_df[self.buildstock_df[char] == char_val].index
                if verbose:
                    logger.info(f"Ensuring representation for {len(group)} buildings for {char} = {char_val}")
                groups.append(list(group))

        preference_weights = self.compute_preference_weights(soft_avoid_chars) if soft_avoid_chars else None
        if preference_weights and verbose:
            logger.info(
                f"Computed preference weights for {len(preference_weights)} buildings based on {soft_avoid_chars}"
            )

        solver = SetCoverSolver(groups=groups, verbose=verbose)
        return solver.find_minimal_set(current_list=previous_minimal_bldgs, preference_weights=preference_weights)

    def get_parameter_overlap_report(self, report_df: pd.DataFrame):
        """
        Check if any parameter (like "HVAC Heating Efficiency") is applied multiple times (via multiple options) to
        the same building within the same upgrade. If any such case is found, return a report with the details.
        """
        report_df = report_df.copy()
        report_df["parameter"] = report_df["option"].str.partition("|")[0]
        # down select to parameters that appear more than once per upgrade in the report
        param_count_df = report_df.groupby(["upgrade", "parameter"]).size().to_frame(name="count")
        param_count_df = param_count_df[param_count_df["count"] > 1]
        if param_count_df.empty:
            return ""
        filtered_report_df = param_count_df.join(report_df.set_index(["upgrade", "parameter"])).reset_index()
        filtered_report_df["bldg_count"] = filtered_report_df["applicable_buildings"].apply(lambda x: len(x))

        # down select to parameters that applies to at least one building more than once
        # i.e. size of set-union of applicable buildings is not equal to sum of applicable buildings
        union_df = filtered_report_df.groupby(["upgrade", "parameter"]).agg(
            {"applicable_buildings": lambda x: set.union(*x), "bldg_count": "sum"}
        )
        union_df["union_count"] = union_df["applicable_buildings"].apply(lambda x: len(x))
        problem_param_df = union_df[union_df["union_count"] != union_df["bldg_count"]]

        if problem_param_df.empty:
            return ""

        overlap_report_text = ""

        # Generate the detailed overlap report table
        for (upgrade, parameter), row in problem_param_df.iterrows():
            overlap_report_text += f"Parameter '{parameter}' in upgrade '{upgrade}' has overlapping applications\n"
            options = filtered_report_df[
                (filtered_report_df["upgrade"] == upgrade) & (filtered_report_df["parameter"] == parameter)
            ]

            # Use combinations to avoid checking both A->B and B->A
            for i, j in combinations(options.index, 2):
                option_row = options.loc[i]
                other_row = options.loc[j]

                option = f"Option {option_row['option_num']}:{option_row['option']}"
                other_option = f"Option {other_row['option_num']}:{other_row['option']}"

                option_bldgs = option_row["applicable_buildings"]
                other_bldgs = other_row["applicable_buildings"]

                overlap = option_bldgs.intersection(other_bldgs)
                assert overlap, (
                    f"No overlap found between {option} and {other_option} although "
                    "they should have overlapped based on problem_df"
                )
                example_bldgs = list(overlap)[:5]  # Show up to 5 example buildings
                overlap_report_text += f"{option} overlaps with {other_option} on {len(overlap)} buildings\n"
                overlap_report_text += f"Example buildings: {example_bldgs}\n"

                # Show the subset of buildstock_df using the example overlap buildings
                # and the relevant columns used in the apply logics
                upgrade_cfg = self.cfg["upgrades"][option_row["upgrade"] - 1]
                option_cfg = upgrade_cfg["options"][option_row["option_num"] - 1]
                other_option_cfg = upgrade_cfg["options"][other_row["option_num"] - 1]
                parameter_list = []
                parameter_list.append(UpgradesAnalyzer._get_para_option(option_cfg["option"])[0])
                parameter_list.extend(UpgradesAnalyzer.get_mentioned_parameters(option_cfg.get("apply_logic")))
                parameter_list.extend(UpgradesAnalyzer.get_mentioned_parameters(other_option_cfg.get("apply_logic")))
                parameter_list.extend(UpgradesAnalyzer.get_mentioned_parameters(upgrade_cfg.get("package_apply_logic")))
                parameter_list = list(set(parameter_list))
                example_buildstock_df = self.buildstock_df.loc[example_bldgs][parameter_list]
                overlap_report_text += tabulate(example_buildstock_df, headers="keys", tablefmt="grid", maxcolwidths=50)
                overlap_report_text += "\n"
        if overlap_report_text:
            Path("parameter_overlap_report.txt").write_text(overlap_report_text)
        return overlap_report_text


def main():
    import argparse

    print("Welcome to BuildStock Upgrade Analyzer 2025.04.24")
    parser = argparse.ArgumentParser(description="Analyze BuildStock upgrades")
    parser.add_argument("--yaml_file", help="Project configuration file (the yaml file)")
    parser.add_argument("--buildstock_file", help="Project sample file (buildstock.csv)")
    parser.add_argument("--opt_sat_file", help="Path to option_saturation.csv file")
    parser.add_argument("--output_prefix", help="Output file name prefix", default="")
    parser.add_argument(
        "--prev_minimal_buildstock_file", help="Previous minimal buildstock file to try to refine", default=None
    )
    args = parser.parse_args()

    defaults = load_script_defaults("project_info")

    # Use command line arguments if provided, otherwise prompt user
    yaml_file = args.yaml_file
    if not yaml_file:
        yaml_file = inquirer.filepath(
            message="Project configuration file (the yaml file):",
            default=defaults.get("yaml_file", ""),
            validate=PathValidator(),
        ).execute()

    buildstock_file = args.buildstock_file
    if not buildstock_file:
        buildstock_file = inquirer.filepath(
            message="Project sample file (buildstock.csv):",
            default=defaults.get("buildstock_file", ""),
            validate=PathValidator(),
        ).execute()

    opt_sat_file = args.opt_sat_file
    if not opt_sat_file:
        opt_sat_file = inquirer.filepath(
            message="Path to option_saturation.csv file",
            default=defaults.get("opt_sat_file", ""),
            validate=PathValidator(),
        ).execute()

    output_prefix = args.output_prefix
    if not output_prefix:  # Check if it's None, not empty string
        output_prefix = inquirer.text(
            message="output file name prefix:",
            default=defaults.get("output_prefix", ""),
            filter=lambda x: "" if x is None else f"{x}",
        ).execute()

    defaults.update(
        {
            "yaml_file": yaml_file,
            "buildstock_file": buildstock_file,
            "opt_sat_file": opt_sat_file,
            "output_prefix": output_prefix,
        }
    )
    save_script_defaults("project_info", defaults)
    previous_minimal_bldgs = None
    if args.prev_minimal_buildstock_file:
        try:
            previous_minimal_bldgs_df = pd.read_csv(args.prev_minimal_buildstock_file, dtype=str)
            previous_minimal_bldgs = previous_minimal_bldgs_df["Building"].tolist()
            print(
                f"Read {len(previous_minimal_bldgs)} buildings ({previous_minimal_bldgs})"
                f" from previous minimal buildstock file: {args.prev_minimal_buildstock_file}"
            )
            print("I will attempt to preserve as many of these buildings as possible in the new minimal buildstock.")
        except Exception:
            print(
                f"Failed to read previous minimal buildstock file: {args.prev_minimal_buildstock_file}\n"
                f" due to {traceback.format_exc()}\nWill create a brand new one!"
            )
    ua = UpgradesAnalyzer(yaml_file=yaml_file, buildstock=buildstock_file, opt_sat_file=opt_sat_file)
    report_df = ua.get_report()
    folder_path = Path.cwd()
    csv_name = folder_path / f"{output_prefix}option_application_report.csv"
    txt_name = folder_path / f"{output_prefix}option_application_detailed_report.txt"
    param_overlap_path = folder_path / f"{output_prefix}parameter_overlap_report.txt"
    print("Checking for parameter overlap...")
    param_overlap_report = ua.get_parameter_overlap_report(report_df)
    if param_overlap_report:
        print("Some parameters are applied multiple times to the same building within the same upgrade.")
        print(f"Please check the {param_overlap_path} file for details.")
        param_overlap_path.write_text(param_overlap_report)
    else:
        print("All good! No parameter applies multiple times in the same upgrade.")

    buildstock_name = folder_path / f"{output_prefix}minimal_buildstock.csv"
    must_cover_chars = [
        "census division",
        "vintage",
        "geometry building type recs",
        "heating fuel",
    ]
    soft_avoid_chars = [("heating fuel", "Wood")]
    additional_chars = ua.get_characteristics(min_cardinality=2, max_cardinality=10)
    must_cover_chars = must_cover_chars + [char for char in additional_chars if char not in must_cover_chars]
    minimal_bldgs = ua.get_minimal_representative_buildings(
        report_df,
        must_cover_chars=must_cover_chars,
        include_never_upgraded=True,
        previous_minimal_bldgs=previous_minimal_bldgs,
        verbose=True,
        soft_avoid_chars=soft_avoid_chars,
    )
    ua.buildstock_df_original.set_index("Building").loc[list(minimal_bldgs)].to_csv(buildstock_name)
    report_df.drop(columns=["applicable_buildings"]).to_csv(csv_name, index=False)
    ua.save_detailed_report_all(str(txt_name))
    print(f"Saved  {csv_name} and {txt_name} inside {os.getcwd()}")
    print(f"Saved minimal buildstock to {buildstock_name}")


if __name__ == "__main__":
    main()
