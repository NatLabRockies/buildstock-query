from buildstock_query.tools.upgrades_visualizer.viz_data import VizData
from buildstock_query.tools.upgrades_visualizer.figure import UpgradesPlot
from buildstock_query.tools.upgrades_visualizer.plot_utils import PlotParams, SavingsTypes, ValueTypes
from buildstock_query.tools.upgrades_visualizer.upgrades_visualizer import get_app
import pathlib
import itertools as it
import pytest
from _pytest.monkeypatch import MonkeyPatch
from tests.utils import load_tbl_from_pkl


# @pytest.fixture(scope="module", autouse=True)
# def _mock_query_core():
#     """
#     Monkey-patch SQLAlchemy, boto3 and other dependencies so instead of db, data is
#     read from pkl files. The patch is automatically rolled back after each test, ensuring other test
#     modules use the real objects.
#     """
#     import buildstock_query.query_core as _qc  # local import to avoid circular issues
#     from unittest.mock import MagicMock

#     mp = MonkeyPatch()
#     mp.setattr(_qc.sa, "Table", load_tbl_from_pkl, raising=False)
#     mp.setattr(_qc.sa, "create_engine", MagicMock(), raising=False)
#     mp.setattr(_qc, "Connection", MagicMock(), raising=False)
#     mp.setattr(_qc, "boto3", MagicMock(), raising=False)
#     yield
#     mp.undo()


class TestViz:
    @pytest.fixture(scope="class")
    def viz_data(self):
        folder_path = pathlib.Path(__file__).parent.resolve()
        opt_sat_path = str(folder_path / "reference_files" / "options_saturations.csv")
        mydata = VizData(
            opt_sat_path=opt_sat_path,
            workgroup="rescore",
            db_name="largeee_test_runs",
            run=("small_run_baseline_20230810_100", "small_run_category_1_20230616"),
            buildstock_type="resstock",
            skip_init=True,
        )
        assert mydata.baseline_run is not None
        mydata.baseline_run.cache_folder = folder_path / "reference_files"
        mydata.main_run.cache_folder = folder_path / "reference_files"
        mydata.baseline_run.load_cache()
        mydata.main_run.load_cache()
        mydata.initialize()
        return mydata

    @pytest.fixture(scope="class")
    def upgrades_plot(self, viz_data) -> UpgradesPlot:
        upgrades_plot = UpgradesPlot(viz_data=viz_data)
        return upgrades_plot

    @pytest.fixture(scope="class")
    def dash_app(self, viz_data):
        dash_app = get_app(viz_data)
        return dash_app

    @pytest.mark.parametrize(
        "resolution, value_type, savings_type, upgrade, group_by",
        it.product(
            ["annual", "monthly"],
            ValueTypes,
            SavingsTypes,
            [None, 0, 1],
            [[], ["ahs_region"], ["vacancy_status", "ahs_region"], ["month", "vacancy_status", "ahs_region"]],
        ),
    )
    def test_get_plotting_df_all_upgrades(self, viz_data, resolution, value_type, savings_type, upgrade, group_by):
        all_cols = viz_data.get_all_end_use_cols(resolution)
        params = PlotParams(
            enduses=all_cols,
            savings_type=savings_type,
            value_type=value_type,
            change_type=None,
            resolution=resolution,
            group_by=group_by,
            upgrade=upgrade,
        )
        df = viz_data.get_plotting_df_all_upgrades(params)
        assert len(df) > 0

    @pytest.mark.parametrize(
        "resolution, value_type, savings_type, upgrade, num_enduse, group_by",
        it.product(
            ["annual", "monthly"],
            ValueTypes,
            SavingsTypes,
            [None, 0, 1],
            [1, 3],
            [[], ["ahs_region"], ["vacancy_status", "ahs_region"], ["month", "vacancy_status", "ahs_region"]],
        ),
    )
    def test_get_plot(
        self, upgrades_plot: UpgradesPlot, resolution, value_type, savings_type, upgrade, num_enduse, group_by
    ):
        all_enduses = upgrades_plot.viz_data.get_all_end_use_cols(resolution=resolution)
        enduses = all_enduses[:num_enduse]
        params = PlotParams(
            enduses=enduses,
            savings_type=savings_type,
            value_type=value_type,
            change_type=None,
            resolution=resolution,
            group_by=group_by,
            upgrade=upgrade,
        )
        fig, report_df = upgrades_plot.get_plot(params=params)
        assert len(report_df) > 0
        assert len(fig.data) > 0

    def test_dash_app(self, dash_app):
        assert dash_app is not None
