import itertools as it
import json
from collections import defaultdict
from functools import reduce
from typing import cast

import yaml

from buildstock_query.file_getter import OpenOrDownload
from buildstock_query.helpers import read_csv


class LogicParser:
    def __init__(self, opt_sat_path, yaml_file) -> None:
        self.opt_df = self.get_opt_df(opt_sat_path)
        self.opt_df = self.opt_df[self.opt_df["Saturation"] > 0]
        self.available_opts = self.opt_df.groupby("Parameter")['Option'].agg(set).to_dict()
        self.yaml_file = yaml_file
        if yaml_file:
            self.cfg = self.get_cfg(yaml_file)

    def get_cfg(self, yaml_file) -> dict:
        """Get the buildstock configuration file as a dictionary object.

        Returns:
            dict: The buildstock configuration file.
        """
        with OpenOrDownload(yaml_file) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config

    def get_opt_df(self, opt_sat_path):
        """Get the options saturation dataframe.

        Returns:
            pandas.DataFrame: The options saturation dataframe.
        """
        with OpenOrDownload(opt_sat_path) as f:
            opt_df = read_csv(f)
            return opt_df

    def get_apply_logics(self, upgrade_num, option_name):
        """Get the apply logic for a given upgrade number and option.

        Args:
            upgrade_num (int): The upgrade number.
            option_name (str): The option name.

        Returns:
            dict: The apply logic for the given upgrade number and option name.
        """
        config = self.cfg
        upgrade = config["upgrades"][upgrade_num - 1]
        opt2logic = {}
        for opt in upgrade["options"]:
            para, _ = self._get_para_option(opt["option"])
            if para == option_name:
                if "apply_logic" in opt and "package_apply_logic" in upgrade:
                    logic = {"and": [opt["apply_logic"], upgrade["package_apply_logic"]]}
                elif "apply_logic" in opt:
                    logic = opt["apply_logic"]
                else:
                    logic = upgrade["package_apply_logic"]
                if opt["option"] in opt2logic:
                    opt2logic[opt["option"]] = {"or": [opt2logic[opt["option"]], logic]}
                else:
                    opt2logic[opt["option"]] = logic
        return opt2logic

    def get_upgrade_options_map(self):
        """Get list of all options for all upgrades"""
        config = self.cfg
        upgrade_options_map = defaultdict(set)
        for upgrade_num, upgrade in enumerate(config["upgrades"], start=1):
            for opt in upgrade["options"]:
                para, _ = self._get_para_option(opt["option"])
                upgrade_options_map[(upgrade_num, upgrade['upgrade_name'])].add(para)
        return upgrade_options_map

    def get_overlap_report(self):
        """
        For all the option in each upgrade, verify that there are no overlapping selection.
        """
        upgrade_options_map = self.get_upgrade_options_map()
        overlap_report = ""
        for (upgrade_num, upgrade_name), options in upgrade_options_map.items():
            for option in options:
                print(f"Verifying upgrade {upgrade_num} ({upgrade_name}), {option}")
                opt2logic = self.get_apply_logics(upgrade_num, option)
                for opt1, opt2 in it.combinations(opt2logic.keys(), 2):
                    overlap = self.get_overlapping_selections(opt2logic[opt1], opt2logic[opt2])
                    if overlap:
                        overlap_report += f"Upgrade {upgrade_num} ({upgrade_name}),\
                                            has overlapping selections for {opt1} and {opt2}:\n"
                        overlap_report += json.dumps(overlap, indent=2) + "\n"
        return overlap_report

    @staticmethod
    def _get_para_option(condition):
        try:
            para, option = condition.split("|")
        except ValueError as e:
            raise ValueError(f"Condition {condition} is invalid") from e
        return para, option

    @staticmethod
    def and_dicts(dict1: dict[str, set[str]], dict2: dict[str, set[str]]):
        """
        Merge two dicts. If there is a conflict, the value in dict2 takes precedence.
        """
        # l1 = {'p1': {'o1', 'o2', 'o3'}, 'p2': {'o4', 'o5'}, 'p5': {'o9', 'o10'}}
        # l2 = {'p1': {'o3', 'o4'}, 'p2': {'o5', 'o6'}, 'p9': {'o11', 'o12'}}
        # l1 & l2 = {'p1': {'o3'}, 'p2': {'o5'}, 'p5': {'o9', 'o10'}, 'p9': {'o11', 'o12'}}
        new_dict: dict[str, set[str]] = {}
        for key in sorted(set(dict1.keys()) | set(dict2.keys())):
            if key in dict1 and key in dict2:
                new_dict[key] = dict1[key].intersection(dict2[key])
            elif key in dict1:
                new_dict[key] = dict1[key]
            else:
                new_dict[key] = dict2[key]
        return new_dict

    @staticmethod
    def and_(selections1: list[dict[str, set[str]]], selections2: list[dict[str, set[str]]]):
        """
        Merge two dicts. If there is a conflict, the value in dict2 takes precedence.
        """
        new_logic: list[dict[str, set[str]]] = []
        for dict1, dict2 in it.product(selections1, selections2):
            new_logic.append(LogicParser.and_dicts(dict1, dict2))
        return new_logic

    @staticmethod
    def _trim_selections(selections: list[dict[str, set[str]]]):
        """
        Remove any selections that are subsets of another selection or contains
        a key with no available values
        """
        new_selections = []
        keys2seen_selections: dict[tuple, list[dict[str, set[str]]]] = {}

        def is_subset(sel):
            sel_keys = tuple(sorted(sel.keys()))
            keys_combi = (keys for count in range(1, len(sel_keys) + 1) for keys in it.combinations(sel_keys, count))
            for keys in keys_combi:
                if keys in keys2seen_selections:
                    for seen_selection in keys2seen_selections[keys]:
                        if all(sel[key] <= seen_selection[key] for key in keys):
                            return True
            return False

        ordered_selections = cast(list[dict[str, set[str]]], sorted(selections, key=len))
        for selection in ordered_selections:
            if any(len(selection[key]) == 0 for key in selection):
                continue
            if not is_subset(selection):
                keys2seen_selections[tuple(sorted(selection.keys()))] = [selection]
                new_selections.append(selection)
        return new_selections

    @staticmethod
    def _compress_selections(selections: list[dict[str, set[str]]]):
        """
        If there are multiple selections with the same keys, and same values except for one key,
        merge them into one selection to reduce the number of selections.
        For example: [{'p1': {'o1', 'o2', 'o3'}, 'p2': {'o4', 'o5'}},
                      {'p1': {'o1', 'o2', 'o3'}, 'p2': {'o4', 'o6'}}] will be merged into
                     [{'p1': {'o1', 'o2', 'o3'}, 'p2': {'o4', 'o5', 'o6'}}]
        """
        keys2seen_selections: dict[tuple, list[dict[str, set[str]]]] = {}
        for sel_dict in selections:
            keys = tuple(sorted(sel_dict.keys()))
            if keys in keys2seen_selections:
                for seen_selection in keys2seen_selections[keys]:
                    matching_keys = {key for key in keys if seen_selection[key] == sel_dict[key]}
                    if len(matching_keys) == len(keys):
                        # If all keys match, we have a duplicate sel_dict
                        break
                    elif len(matching_keys) == len(keys) - 1:
                        # sel_dict has one key with different value from before. Simply update the record
                        key = next(iter(set(keys) - matching_keys))
                        seen_selection[key] |= sel_dict[key]
                        break
                else:
                    # If we did not break out of the loop, we have a new selection that can't be merged with
                    # any of the previous selections
                    keys2seen_selections[keys].append(sel_dict)
            else:
                keys2seen_selections[keys] = [sel_dict]
        return [selection for seen_selections in keys2seen_selections.values() for selection in seen_selections]

    @staticmethod
    def clean_selections(selections: list[dict[str, set[str]]]):
        """
        Repeatedly compress and trim selections until there is no change.
        """
        while True:
            new_selections = LogicParser._compress_selections(LogicParser._trim_selections(selections))
            if len(new_selections) == len(selections):
                return new_selections
            selections = new_selections

    @staticmethod
    def or_(selections1: list[dict[str, set[str]]], selections2: list[dict[str, set[str]]]):
        """
        Merge two dicts. If there is a conflict, the value in dict2 takes precedence.
        """
        return selections1 + selections2

    def inverse_selection(self, selection: dict):
        return [{k: self.available_opts[k] - v - {"Void"}} for k, v in selection.items()]

    def not_(self, logic1):
        """
        Merge two dicts. If there is a conflict, the value in dict2 takes precedence.
        """
        prarsed_logic = self.prase_logic(logic1)
        return_val = list(reduce(self.and_, [self.inverse_selection(selection) for selection in prarsed_logic]))
        return return_val

    def _normalize_lists(self, logic, parent=None):
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
                return self._normalize_lists(logic[0])
            new_logic = [self._normalize_lists(el) for el in logic]
            return {"and": new_logic} if parent is None else new_logic
        elif isinstance(logic, dict):
            new_dict = {key: self._normalize_lists(value, parent=key) for key, value in logic.items()}
            return new_dict
        else:
            return logic

    def prase_logic(self, logic):
        """
        Convert the parameter|option logic in the yaml file into the format the apply upgrade measure understands.

        :param logic: dict, list, or string with downselection logic in it
        :returns: str of logic
        """
        logic = self._normalize_lists(logic)
        if isinstance(logic, dict):
            assert (len(logic) == 1)
            key = next(iter(logic.keys()))
            val = logic[key]
            val = [val] if not isinstance(val, list) else val
            if key == 'and':
                and_result = list(reduce(self.and_, (self.prase_logic(block) for block in val)))
                return self.clean_selections(and_result)
            elif key == 'or':
                or_result = list(reduce(self.or_, (self.prase_logic(block) for block in val)))
                return self.clean_selections(or_result)
            elif key == 'not':
                and_result = list(reduce(self.and_, (self.prase_logic(block) for block in val)))
                not_val = list(reduce(self.and_, [self.inverse_selection(selection) for selection in and_result]))
                return self.clean_selections(not_val)
        elif isinstance(logic, list):
            list_val = list(reduce(self.and_, (self.prase_logic(block) for block in logic)))
            return self.clean_selections(list_val)
        elif isinstance(logic, str):
            para, option = self._get_para_option(logic)
            return [{para: {option}}]
        raise ValueError(f"Logic {logic} is invalid")

    def retrieve_logic(self, selections: list[dict[str, set[str]]]):
        """
        return back the logic from the selections
        """
        outer_dict = {'or': []}
        for selection in selections:
            inner_dict = {'and': []}
            for key, values in selection.items():
                if len(values) > 1:
                    inner_dict['and'].append({'or': [f"{key}|{value}" for value in values]})
                elif len(values) == 1:
                    inner_dict['and'].append(f"{key}|{next(iter(values))}")
                else:
                    raise ValueError(f"Selection {selection} has no valid value for {key}")
            if len(inner_dict['and']) == 1:
                outer_dict['or'].append(inner_dict['and'][0])
            else:
                outer_dict['or'].append(inner_dict)
        if len(outer_dict['or']) == 1:
            return outer_dict['or'][0]
        return outer_dict

    def normalize_logic(self, logic):
        selections = self.prase_logic(logic)
        return self.retrieve_logic(selections)

    def get_overlapping_selections(self, logic1, logic2):
        """
        Get the selections that are common to both logic1 and logic2
        """
        selections = self.prase_logic({"and": [logic1, logic2]})
        if len(selections) == 0:
            return None
        selections = self.clean_selections(selections)
        return self.retrieve_logic(selections)
