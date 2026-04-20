import pathlib
import numpy as np
from buildstock_query.tools import UpgradesAnalyzer
import pytest
import pandas as pd
from unittest.mock import patch


class TestUpgradesAnalyzer:
    # Create three different kind of UpgradesAnalyzer fixuture objects.
    # 1. with_filter: upgrades yaml and filter yaml are included
    # 2. without_filter: only upgrades yaml is included
    # 3. only_filter: only filter yaml is included
    # The test cases are parametrized to run with all three kinds of fixtures.
    # But some test functions skip the test if the fixture is not applicable.
    @pytest.fixture(
        params=["with_filter", "without_filter", "only_filter"], ids=["with_filter", "without_filter", "only_filter"]
    )
    def ua(self, request):
        folder_path = pathlib.Path(__file__).parent.resolve()

        if request.param == "with_filter":
            yaml_path = str(folder_path / "reference_files" / "res_n250_15min_v19.yml")
            upgrade_names = None
            filter_yaml_path = str(folder_path / "reference_files" / "res_n250_15min_v19_upgrades_filter.yml")
        elif request.param == "without_filter":
            yaml_path = str(folder_path / "reference_files" / "res_n250_15min_v19.yml")
            upgrade_names = None
            filter_yaml_path = None
        elif request.param == "only_filter":
            yaml_path = None
            filter_yaml_path = str(folder_path / "reference_files" / "res_n250_15min_v19_upgrades_filter.yml")
            upgrade_names = {i: f"upgrade{i}" for i in range(1, 9)}
        else:
            raise ValueError(f"Invalid request param: {request.param}")

        self.buildstock_path = folder_path / "reference_files" / "res_n250_15min_v19_buildstock.csv"
        self.opt_sat_path = folder_path / "reference_files" / "options_saturations.csv"
        ua = UpgradesAnalyzer(
            yaml_file=yaml_path,
            filter_yaml_file=filter_yaml_path,
            buildstock=str(self.buildstock_path),
            upgrade_names=upgrade_names,
            opt_sat_file=str(self.opt_sat_path),
        )
        return ua

    def test_read_cfg(self, ua, request):
        cfg = ua.cfg
        if request.node.callspec.id != "only_filter":
            assert isinstance(cfg, dict)
            assert "upgrades" in cfg
            assert "postprocessing" in cfg

    def test_read_filter_cfg(self, ua, request):
        cfg = ua.filter_cfg
        if request.node.callspec.id == "with_filter":
            assert isinstance(cfg, dict)
            assert set(ua.filter_cfg.keys()) == set(ua.upgrade_names.values())

    def test_parse_none(self, ua):
        assert "None" in ua.buildstock_df["hvac secondary heating type and fuel"].unique()

    @pytest.mark.parametrize("test_case", [0, 1])
    def test_get_para_option(self, test_case):
        test_entries = [
            ["Vintage|1980s", ("vintage", "1980s")],
            [
                "Windows|Single, Clear, Metal, Exterior Low-E Storm",
                ("windows", "Single, Clear, Metal, Exterior Low-E Storm"),
            ],
        ]
        test_inp, expected_output = test_entries[test_case]
        assert expected_output == UpgradesAnalyzer._get_para_option(test_inp)

    @pytest.mark.parametrize("test_case", [0, 1])
    def test_get_eq_str(self, test_case):
        test_entries = [
            ["Vintage|1980s", "`vintage`=='1980s'"],
            [
                "Windows|Single, Clear, Metal, Exterior Low-E Storm",
                "`windows`=='Single, Clear, Metal, Exterior Low-E Storm'",
            ],
        ]
        test_inp, expected_output = test_entries[test_case]
        assert expected_output == UpgradesAnalyzer._get_eq_str(test_inp)

    def test_get_mentioned_parameters(self):
        empty_logics = [{}, [], "", None]
        for logic in empty_logics:
            assert UpgradesAnalyzer.get_mentioned_parameters(logic) == []

        assert UpgradesAnalyzer.get_mentioned_parameters("Vintage|1980s") == ["vintage"]
        assert UpgradesAnalyzer.get_mentioned_parameters("Windows|Single, Clear, Metal, Exterior Low-E Storm") == [
            "windows"
        ]
        logic = {"and": [{"or": ["Vintage|1980s", "Vintage|1960s"]}, "Location Region|CR09"]}
        assert UpgradesAnalyzer.get_mentioned_parameters(logic) == [
            "vintage",
            "location region",
        ]

    def test_reduce_logic(self, ua):
        logic = {"and": [{"or": ["Vintage|1980s", "Vintage|1960s"]}, "Location Region|CR09"]}
        reduced_logic = ua._reduce_logic(logic)
        cond1 = ua.buildstock_df["location region"] == "CR09"
        cond2 = ua.buildstock_df["vintage"].isin(["1980s", "1960s"])
        assert (cond1 & cond2 == reduced_logic).all()

        logic = {"and": [{"or": ["Vintage|1980s", "Vintage|1960s"]}, "Vintage|1980s"]}
        reduced_logic = ua._reduce_logic(logic)
        cond1 = ua.buildstock_df["vintage"] == "1980s"
        assert (cond1 == reduced_logic).all()

        logic = {
            "and": [
                {"or": ["Vintage|1980s", "Vintage|1960s"]},
                {"and": ["Vintage|1980s", "Vintage|1960s"]},
            ]
        }
        reduced_logic = ua._reduce_logic(logic)
        cond1 = [False] * len(ua.buildstock_df)
        assert (cond1 == reduced_logic).all()

        # A list, except for that inside an `or` block is always interpreted as `and` block
        logic = {"not": ["Vintage|1980s", "Vintage|1960s"]}  # Since no buildings have both vintage, should select all
        reduced_logic = ua._reduce_logic(logic)
        cond1 = [True] * len(ua.buildstock_df)
        assert (cond1 == reduced_logic).all()

        logic = {
            "and": [
                {"or": ["Vintage|1980s", "Vintage|1960s"]},
                {"not": ["Vintage|1980s"]},
            ]
        }
        reduced_logic = ua._reduce_logic(logic)
        cond1 = ua.buildstock_df["vintage"] == "1960s"
        assert (cond1 == reduced_logic).all()

    def test_normalize_lists(self):
        logic = [["logic1", "logic2"], ["logic3", "logic4"]]
        flattened_logic = UpgradesAnalyzer._normalize_lists(logic)
        expected_logic = {"and": [{"and": ["logic1", "logic2"]}, {"and": ["logic3", "logic4"]}]}
        assert flattened_logic == expected_logic

        logic = {
            "or": [
                ["logic1", "logic2"],
                ["logic3", "logic4"],
                {"and": [["logic5"], ["logic6"]]},
                ["logic7"],
            ]
        }
        flattened_logic = UpgradesAnalyzer._normalize_lists(logic)
        expected_logic = {
            "or": [
                {"and": ["logic1", "logic2"]},
                {"and": ["logic3", "logic4"]},
                {"and": ["logic5", "logic6"]},
                "logic7",
            ]
        }
        assert expected_logic == flattened_logic

        logic = {
            "and": [
                {"or": ["Vintage|1980s", "Vintage|1960s"]},
                {"and": ["Vintage|1980s", "Vintage|1960s"]},
            ]
        }
        flattened_logic = UpgradesAnalyzer._normalize_lists(logic)
        expected_logic = logic.copy()
        assert flattened_logic == expected_logic

    def test_print_options_application_report(self, ua: UpgradesAnalyzer, capsys):
        logic_dict = {
            1: np.array([True, True, True]),
            2: np.array([True, False, True]),
            3: np.array([True, True, False]),
            4: np.array([False, False, True]),
        }  # {"opt_index": logic_array_of_applicable_buildings}
        report_df = ua._get_options_application_count_report(logic_dict)
        assert len(report_df) == 3
        assert report_df.loc[2]["Applied buildings"] == "1 (33.3%)"
        assert report_df.loc[2]["Cumulative all"] == "1 (33.3%)"
        assert report_df.loc[3]["Applied options"].iloc[0] == "1, 2, 3"
        assert report_df.loc[3]["Applied options"].iloc[1] == "1, 2, 4"
        assert report_df.loc[3]["Applied buildings"].iloc[0] == "1 (33.3%)"
        assert report_df.loc[3]["Applied buildings"].iloc[1] == "1 (33.3%)"
        assert report_df.loc[3]["Cumulative all"].iloc[0] == "2 (66.7%)"
        assert report_df.loc[3]["Cumulative all"].iloc[1] == "3 (100.0%)"

    def test_get_report_only_filter(self, ua: UpgradesAnalyzer, request):
        if request.node.callspec.id != "only_filter":
            return
        report = ua.get_report()
        assert len(report) == 8
        assert list(report["removal_count"].values) == [14, 36, 14, 14, 14, 14, 14, 14]

    def test_get_report(self, ua: UpgradesAnalyzer, request):
        if request.node.callspec.id == "only_filter":  # test if upgrades yaml is included
            return

        cfg = ua.cfg
        new_cfg = cfg.copy()
        # ua.cfg = new_cfg  # type: ignore

        new_cfg["upgrades"] = [cfg["upgrades"][0]]  # keep only one upgrade
        report_df = ua.get_report(1)
        assert (report_df["upgrade_name"] == "upgrade1").all()
        assert len(report_df) == 2
        opt1_cond = report_df["option"] == "Insulation Wall|Wood Stud, Uninsulated, R-5 Sheathing"
        logic_cond1 = ua.buildstock_df["insulation wall"] == "Wood Stud, Uninsulated"
        logic_cond2 = ua.buildstock_df["vintage"] == "1980s"

        if ua.filter_cfg:
            remove_logic = ua.buildstock_df["vintage"] == "1980s"
            combined_logic = (logic_cond1 | logic_cond2) & (~remove_logic)
        else:
            combined_logic = logic_cond1 | logic_cond2

        assert report_df[opt1_cond].applicable_to.values[0] == sum(combined_logic)
        assert report_df[report_df["option"] == "All"].applicable_to.values[0] == sum(combined_logic)

        new_cfg["upgrades"] = [cfg["upgrades"][1]]  # keep only one upgrade
        report_df = ua.get_report(2)
        assert (report_df["upgrade_name"] == "upgrade2").all()
        assert len(report_df) == 4
        opt1_cond = report_df["option"] == "Windows|Single, Clear, Metal, Exterior Low-E Storm"
        opt2_cond = report_df["option"] == "Vintage|1980s"
        opt3_cond = report_df["option"] == "Vintage|1970s"

        pkg_logic = ~ua.buildstock_df["vintage"].isin(["1990s", "2000s"])
        logic_cond1_1 = ua.buildstock_df["windows"] == "Single, Clear, Metal"
        logic_cond1_2 = ua.buildstock_df["windows"] == "Single, Clear, Metal, Exterior Clear Storm"
        opt1_logic = (logic_cond1_1 | logic_cond1_2) & pkg_logic
        logic_cond2_1 = ua.buildstock_df["vintage"] == "1960s"
        opt2_logic = logic_cond2_1 & pkg_logic
        logic_cond3_1 = ua.buildstock_df["vintage"] == "1980s"
        logic_cond3_2 = ua.buildstock_df["vintage"] == "1960s"
        logic_cond3_3 = ua.buildstock_df["location region"] == "CR09"
        opt3_logic = (logic_cond3_1 | logic_cond3_2) & logic_cond3_3 & pkg_logic
        if ua.filter_cfg:
            remove_logic = (ua.buildstock_df["vintage"] == "1980s") | (ua.buildstock_df["vintage"] == "1960s")
            opt1_logic &= ~remove_logic
            opt2_logic &= ~remove_logic
            opt3_logic &= ~remove_logic

        assert report_df[opt1_cond].applicable_to.values[0] == sum(opt1_logic)
        assert report_df[opt2_cond].applicable_to.values[0] == sum(opt2_logic)
        assert report_df[opt3_cond].applicable_to.values[0] == sum(opt3_logic)
        assert report_df[report_df["option"] == "All"].applicable_to.values[0] == sum(
            opt1_logic | opt2_logic | opt3_logic
        )

        new_cfg["upgrades"] = [cfg["upgrades"][4]]  # No apply_logic
        report_df = ua.get_report(5)
        assert (report_df["upgrade_name"] == "upgrade5").all()
        assert len(report_df) == 2
        opt1_cond = report_df["option"] == "Vintage|1980s"
        if ua.filter_cfg:
            remove_logic = ua.buildstock_df["vintage"] == "1980s"
            total_applicable_buildings = sum(~remove_logic)
            assert report_df[report_df["option"] == "All"].removal_count.values[0] == sum(remove_logic)
        else:
            total_applicable_buildings = len(ua.buildstock_df)

        assert report_df[opt1_cond].applicable_to.values[0] == total_applicable_buildings
        assert report_df[report_df["option"] == "All"].applicable_to.values[0] == total_applicable_buildings

        with pytest.raises(ValueError):
            ua.get_report(0)
            ua.get_report(len(cfg["upgrades"]) + 2)

    def test_print_detailed_report(self, ua: UpgradesAnalyzer, capsys, request):
        if request.node.callspec.id == "only_filter":  # test if upgrades yaml is included
            return

        with pytest.raises(ValueError):
            ua.get_detailed_report(0)  # upgrade 0 is invalid. It is 1-indexed

        with pytest.raises(ValueError):
            ua.get_detailed_report(1, 0)  # option 0 is invalid. It is 1-indexed

        _, report_text, _, _ = ua.get_detailed_report(1)
        assert "Option1:'Insulation Wall|Wood Stud, Uninsulated, R-5 Sheathing'" in report_text
        logic_cond1 = ua.buildstock_df["insulation wall"] == "Wood Stud, Uninsulated"
        cmp_str = f"Insulation Wall|Wood Stud, Uninsulated => {sum(logic_cond1)}"
        assert cmp_str in report_text
        logic_cond2 = ua.buildstock_df["vintage"] == "1980s"
        cmp_str = f"Vintage|1980s => {sum(logic_cond2)}"
        assert cmp_str in report_text
        assert f"or => {sum(logic_cond1 | logic_cond2)}" in report_text
        assert f"and => {sum(logic_cond1 | logic_cond2)}" in report_text

        if ua.filter_cfg:  # if filter yaml is included
            assert "Remove Logic Report" in report_text
            remove_logic = ua.buildstock_df["vintage"] == "1980s"
            combined_logic = (logic_cond1 | logic_cond2) & (~remove_logic)
            assert f"Overall applied to => {sum(combined_logic)}" in report_text
        else:
            assert f"Overall applied to => {sum(logic_cond1 | logic_cond2)}" in report_text

        _, report_text, _, _ = ua.get_detailed_report(2)
        opt1_text = "Option1:'Windows|Single, Clear, Metal, Exterior Low-E Storm'"
        opt2_text = "Option2:'Vintage|1980s'"
        opt3_text = "Option3:'Vintage|1970s'"
        assert opt1_text in report_text
        assert opt2_text in report_text
        assert opt3_text in report_text

        substr1 = report_text[report_text.index(opt1_text) : report_text.index(opt2_text)]
        assert "Package Apply Logic Report" in substr1
        if ua.filter_cfg:  # if filter yaml is included
            package_report = substr1[substr1.index("Package Apply Logic Report") : substr1.index("Remove Logic Report")]
        else:
            package_report = substr1[substr1.index("Package Apply Logic Report") :]
        main_report = substr1[: substr1.index("Package Apply Logic Report")]
        logic_opt1_1 = ua.buildstock_df["windows"] == "Single, Clear, Metal"
        assert f"Windows|Single, Clear, Metal => {sum(logic_opt1_1)}" in main_report
        logic_opt1_2 = ua.buildstock_df["windows"] == "Single, Clear, Metal, Exterior Clear Storm"
        assert f"Single, Clear, Metal, Exterior Clear Storm => {sum(logic_opt1_2)}" in main_report
        assert f"or => {sum(logic_opt1_1 | logic_opt1_2)}" in main_report
        assert f"and => {sum(logic_opt1_1 | logic_opt1_2)}" in main_report

        logic_package_1 = ua.buildstock_df["vintage"] == "1990s"
        assert f"Vintage|1990s => {sum(logic_package_1)}" in package_report
        logic_package_2 = ua.buildstock_df["vintage"] == "2000s"
        assert f"Vintage|2000s => {sum(logic_package_2)}" in package_report
        assert f"or => {sum(logic_package_1 | logic_package_2)}" in package_report
        final_package_apply_logic = ~(logic_package_1 | logic_package_2)
        assert f"not => {sum(final_package_apply_logic)}" in package_report

        overall_logic = (logic_opt1_1 | logic_opt1_2) & final_package_apply_logic

        if ua.filter_cfg:  # if filter yaml is included
            assert "Remove Logic Report" in substr1
            remove_report = substr1[substr1.index("Remove Logic Report") :]
            remove_logic = (ua.buildstock_df["vintage"] == "1980s") | (ua.buildstock_df["vintage"] == "1960s")
            assert f"or => {sum(remove_logic)}" in remove_report
            assert f"Vintage|1980s => {sum(ua.buildstock_df['vintage'] == '1980s')}" in remove_report
            assert f"Vintage|1960s => {sum(ua.buildstock_df['vintage'] == '1960s')}" in remove_report
            assert f"Removed after applying => {sum(overall_logic & remove_logic)}" in remove_report
            overall_logic &= ~remove_logic

        assert f"Overall applied to => {sum(overall_logic)}" in report_text

        _, report_text, opt_app_report_df, opt_app_detailed_report_df = ua.get_detailed_report(2)

        # verify opt_app_report_df
        if ua.filter_cfg:
            assert opt_app_report_df["Applied options"].to_list() == ["windows"]
            assert opt_app_report_df["Applied buildings"].str.split(" ").str[0].to_list() == ["28"]
        else:
            assert opt_app_report_df["Applied options"].to_list() == ["windows", "vintage", "windows, vintage"]
            assert opt_app_report_df["Applied buildings"].str.split(" ").str[0].to_list() == ["29", "20", "9"]

        # verify opt_app_detailed_report_df
        for indx, row in opt_app_detailed_report_df.iterrows():
            applied_bldgs_array = np.ones((1, ua.total_samples), dtype=bool)
            applied_options = [int(opt) for opt in row["Applied options"].split(",")]
            not_applied_options = [i for i in range(1, 4) if i not in applied_options]
            for opt in applied_options:
                logic_arr, _, _, _ = ua.get_detailed_report(2, opt)
                applied_bldgs_array &= logic_arr
            for opt in not_applied_options:
                logic_arr, _, _, _ = ua.get_detailed_report(2, opt)
                applied_bldgs_array &= ~logic_arr
            assert applied_bldgs_array.sum() == int(row["Applied buildings"].split()[0]), f"Row \n{row}\n failed"

    def test_get_logic_report(self, ua: UpgradesAnalyzer):
        for logic_cfg in ["Vintage|1980s", ["Vintage|1980s"]]:
            logic_arr = ua.buildstock_df["vintage"] == "1980s"
            logic_arr_out, logic_report = ua._get_logic_report(logic_cfg)
            assert (logic_arr == logic_arr_out).all()
            assert isinstance(logic_report, list)
            assert f"Vintage|1980s => {sum(logic_arr)}" in logic_report[0]

        for logic_cfg2 in [{"or": ["Vintage|1980s"]}, {"or": "Vintage|1980s"}]:
            logic_arr = ua.buildstock_df["vintage"] == "1980s"
            logic_arr_out, logic_report = ua._get_logic_report(logic_cfg2)
            assert (logic_arr == logic_arr_out).all()
            assert isinstance(logic_report, list)
            assert len(logic_report) == 2
            assert f"or => {sum(logic_arr)}" in logic_report[0]
            assert f"Vintage|1980s => {sum(logic_arr)}" in logic_report[1]

        logic_cfg3 = ["Vintage|1980s", "Vintage|1960s"]
        logic_arr1 = ua.buildstock_df["vintage"] == "1980s"
        logic_arr2 = ua.buildstock_df["vintage"] == "1960s"
        logic_arr_out, logic_report = ua._get_logic_report(logic_cfg3)
        assert (logic_arr1 & logic_arr2 == logic_arr_out).all()
        assert isinstance(logic_report, list)
        assert len(logic_report) == 2
        assert f"Vintage|1980s => {sum(logic_arr1)}" in logic_report[0]
        assert f"=> {sum(logic_arr1 & logic_arr2)}" in logic_report[0]  # for overall sum
        assert f"Vintage|1960s => {sum(logic_arr2)}" in logic_report[1]

        logic_cfg4 = {"and": {"or": ["Vintage|1980s", "Vintage|1960s"]}}
        logic_arr1 = ua.buildstock_df["vintage"] == "1980s"
        logic_arr2 = ua.buildstock_df["vintage"] == "1960s"
        logic_arr_out, logic_report = ua._get_logic_report(logic_cfg4)
        assert (logic_arr1 | logic_arr2 == logic_arr_out).all()
        assert isinstance(logic_report, list)
        assert len(logic_report) == 4
        assert f"and => {sum(logic_arr1 | logic_arr2)}" in logic_report[0]
        assert f"or => {sum(logic_arr1 | logic_arr2)}" in logic_report[1]
        assert f"Vintage|1980s => {sum(logic_arr1)}" in logic_report[2]
        assert f"Vintage|1960s => {sum(logic_arr2)}" in logic_report[3]

        logic_cfg5 = {"not": ["Vintage|1980s"]}
        logic_arr = ua.buildstock_df["vintage"] == "1980s"
        logic_arr_out, logic_report = ua._get_logic_report(logic_cfg5)
        assert ((~logic_arr) == logic_arr_out).all()
        assert isinstance(logic_report, list)
        assert len(logic_report) == 2
        assert f"not => {sum(~logic_arr)}" in logic_report[0]
        assert f"Vintage|1980s => {sum(logic_arr)}" in logic_report[1]

        logic_cfg6 = {"not": ["Vintage|1980s", "Vintage|1960s"]}
        logic_arr1 = ua.buildstock_df["vintage"] == "1980s"
        logic_arr2 = ua.buildstock_df["vintage"] == "1960s"
        logic_arr_out, logic_report = ua._get_logic_report(logic_cfg6)
        assert (~(logic_arr1 & logic_arr2) == logic_arr_out).all()
        assert isinstance(logic_report, list)
        assert len(logic_report) == 3
        assert f"not => {sum(~(logic_arr1 & logic_arr2))}" in logic_report[0]
        assert f"Vintage|1980s => {sum(logic_arr1)}" in logic_report[1]
        assert f"Vintage|1960s => {sum(logic_arr2)}" in logic_report[2]

        logic_cfg7 = {"not": {"or": ["Vintage|1980s", "Vintage|1960s"]}}
        logic_arr1 = ua.buildstock_df["vintage"] == "1980s"
        logic_arr2 = ua.buildstock_df["vintage"] == "1960s"
        logic_arr_or = logic_arr1 | logic_arr2
        logic_arr_out, logic_report = ua._get_logic_report(logic_cfg7)
        assert ((~logic_arr_or) == logic_arr_out).all()
        assert isinstance(logic_report, list)
        assert len(logic_report) == 4
        assert f"not => {sum(~logic_arr_or)}" in logic_report[0]
        assert f"or => {sum(logic_arr_or)}" in logic_report[1]
        assert f"Vintage|1980s => {sum(logic_arr1)}" in logic_report[2]
        assert f"Vintage|1960s => {sum(logic_arr2)}" in logic_report[3]

    def test_print_unique_characteristics(self, ua: UpgradesAnalyzer, capsys, request):
        if request.node.callspec.id == "only_filter":  # test if upgrades yaml is included
            return

        compare_bldg_list = ua.buildstock_df[ua.buildstock_df["vintage"].isin(["2000s", "1990s"])].index
        other_bldg_list = ua.buildstock_df[~ua.buildstock_df["vintage"].isin(["2000s", "1990s"])].index
        ua.print_unique_characteristic(1, "no-chng", other_bldg_list, compare_bldg_list)
        printed_text, err = capsys.readouterr()
        assert "Only no-chng buildings have vintage in ['1990s', '2000s']" in printed_text
        assert "Checking 2 column combinations out of ['insulation wall', 'vintage']" in printed_text
        assert "No 2-column unique chracteristics found."

        condition1 = ua.buildstock_df["vintage"].isin(["2000s", "1990s"])
        condition2 = ua.buildstock_df["windows"].isin(
            ["Single, Clear, Metal", "Single, Clear, Metal, Exterior Clear Storm"]
        )
        condition3 = ua.buildstock_df["location region"].isin(["CR09"])
        compare_bldg_list = ua.buildstock_df[condition1 & condition2 & condition3].index
        other_bldg_list = ua.buildstock_df[~(condition1 & condition2 & condition3)].index
        ua.print_unique_characteristic(2, "no-chng", other_bldg_list, compare_bldg_list)
        printed_text, err = capsys.readouterr()
        assert "Checking 2 column combinations out of ['windows', 'vintage', 'location region']" in printed_text
        assert "Checking 3 column combinations out of ['windows', 'vintage', 'location region']" in printed_text
        assert (
            "Only no-chng buildings have ('windows', 'vintage', 'location region') in "
            "[('Single, Clear, Metal', '1990s', 'CR09')]" in printed_text
        )

    def test_get_upgraded_buildstock(self, ua: UpgradesAnalyzer, request):
        if request.node.callspec.id == "only_filter":  # test if upgrades yaml is included
            return

        report_df = ua.get_report()
        upgrades = report_df["upgrade"].unique()
        for upg in upgrades:
            report_df_upg = report_df.loc[report_df["upgrade"] == upg]
            n_applied = report_df_upg.loc[report_df_upg["option"] == "All", "applicable_to"].iloc[0]
            df_bsl = ua.buildstock_df_original.set_index("Building")
            df_upg = ua.get_upgraded_buildstock(upg).set_index("Building")
            df_diff = df_bsl.compare(df_upg)
            n_diff = len(df_diff) - n_applied

            if n_diff < 0:
                # this means some baseline parameters are being upgraded to the same incumbent options,
                # (e.g., LED upgraded to LED)
                df_same = df_bsl[~df_bsl.index.isin(df_diff.index)]

                # find # of dwelling units that were unchanged because it already has the upgrade option
                query = []
                dimensions = set()
                for idx, row in report_df_upg.iterrows():
                    if row["option"] == "All":
                        continue
                    dim, opt = row["option"].split("|")
                    query.append(f"`{dim}` == '{opt}'")
                    dimensions.add(dim)
                query = " or ".join(query)
                dimensions = list(dimensions)

                n_unchanged = len(df_same.query(query)[dimensions])
                n_diff = abs(n_diff)
                assert n_diff == n_unchanged, (
                    f"Only {n_unchanged} dwelling units were found to be unchanged, expecting {n_diff} per report"
                )

    def test_get_minimal_representative_buildings(self):
        # Create mock UpgradesAnalyzer instance
        mock_buildstock_df = pd.DataFrame(index=list(range(1, 11)),
        columns=["windows", "vintage", "location region"],
        data=[
            ("Single, Clear, Metal", "2000s", "CR01"),
            ("Single, Clear, Metal", "2000s", "CR02"),
            ("Single, Clear, Metal", "2000s", "CR03"),
            ("Single, Clear, Metal", "2000s", "CR04"),
            ("Single, Clear, Metal", "2000s", "CR05"),
            ("Single, Clear, Metal", "2000s", "CR06"),
            ("Single, Clear, Metal", "2000s", "CR07"),
            ("Single, Clear, Metal", "2000s", "CR08"),
            ("Single, Clear, Metal", "2000s", "CR09"),
            ("Single, Clear, Metal", "2010s", "CR10"),
        ])

        # Create the analyzer with just the necessary components for this test
        with patch.object(UpgradesAnalyzer, "__init__", return_value=None):
            ua = UpgradesAnalyzer()
            ua.buildstock_df = mock_buildstock_df


        # Test case: Basic functionality
        building_groups = [
            {1, 2, 3},  # Group 1
            {2, 4, 5},  # Group 2
            {3, 5, 6},  # Group 3
            {6, 7, 8},  # Group 4
            {1, 8, 9},  # Group 5
        ]
        report_df = pd.DataFrame(
            {"applicable_buildings": building_groups, "option_num": range(1, len(building_groups) + 1)}
        )

        minimal_set = ua.get_minimal_representative_buildings(report_df, include_never_upgraded=False)
        assert isinstance(minimal_set, list)
        assert minimal_set == [8, 5, 3]

        assert [8, 5, 3, 10] == ua.get_minimal_representative_buildings(report_df, include_never_upgraded=True)

        # Test case: Empty input
        assert ua.get_minimal_representative_buildings(report_df[0:0], include_never_upgraded=False) == []
        assert ua.get_minimal_representative_buildings(report_df[0:0], include_never_upgraded=True) == [10]

        # Test case: Input with empty sets (should be ignored)
        building_groups.append(set())
        report_df = pd.DataFrame(
            {"applicable_buildings": building_groups, "option_num": range(1, len(building_groups) + 1)}
        )
        assert [8, 5, 3] == ua.get_minimal_representative_buildings(report_df)

        # Test case with must_cover_chars
        # Since we need all buildings to cover all location regions, we expect all buildings to be in the minimal set
        assert list(range(1, 11)) == sorted(ua.get_minimal_representative_buildings(report_df, must_cover_chars=["location region"]))
        # We need building 10 to cover vintage 2010s
        assert [8, 5, 3, 10] == ua.get_minimal_representative_buildings(report_df, must_cover_chars=["vintage"])
        # We don't need anything extra to cover windows
        assert [8, 5, 3] == ua.get_minimal_representative_buildings(report_df, must_cover_chars=["windows"])
        # We need building 10 to cover vintage 2010s and windows
        assert [8, 5, 3, 10] == ua.get_minimal_representative_buildings(report_df, must_cover_chars=["vintage", "windows"])


    def test_check_parameter_overlap(self, ua: UpgradesAnalyzer, request):
        if request.node.callspec.id == "only_filter":  # test if upgrades yaml is included
            return
        report_df = ua.get_report()
        overlap_text = ua.get_parameter_overlap_report(report_df)
        if ua.filter_cfg:
            assert overlap_text == ""
        else:
            assert "Option 2:Vintage|1980s overlaps with Option 3:Vintage|1970s on 12 buildings" in overlap_text
