from tests.utils import save_ref_pkl
from buildstock_query.tools.upgrades_visualizer.viz_data import VizData
from buildstock_query.tools.upgrades_visualizer.plot_utils import PlotParams, SavingsTypes, ValueTypes
import pathlib
import itertools as it
from buildstock_query import BuildStockQuery


def save_bsq_obj(bsq_obj: BuildStockQuery, cache_name=None):
    cache_name = bsq_obj._get_compact_cache_name(bsq_obj.table_name)
    save_ref_pkl(f"{cache_name}_query_cache", bsq_obj._query_cache)
    save_ref_pkl(bsq_obj.bs_table.name, bsq_obj.bs_table)
    if bsq_obj.up_table is not None:
        save_ref_pkl(bsq_obj.up_table.name, bsq_obj.up_table)
    if bsq_obj.ts_table is not None:
        save_ref_pkl(bsq_obj.ts_table.name, bsq_obj.ts_table)


def save_viz_data_reference_data():
    folder_path = pathlib.Path(__file__).parent.resolve()
    opt_sat_path = str(folder_path / "reference_files" / "options_saturations.csv")
    viz_data = VizData(
        opt_sat_path=opt_sat_path,
        workgroup='largeee',
        db_name='largeee_test_runs',
        run=('small_run_baseline_20230810_100', 'small_run_category_1_20230616'),
        buildstock_type='resstock',
    )
    assert viz_data.baseline_run is not None
    all_cols = viz_data.get_all_end_use_cols("annual")
    for resolution, value_type, savings_type in it.product(["annual", "monthly"], ValueTypes, SavingsTypes):
        params = PlotParams(enduses=all_cols, savings_type=savings_type, value_type=value_type,
                            change_type=None, resolution=resolution,
                            upgrade=0,
                            )
        print(f"Generating table for {resolution, value_type, savings_type}")
        viz_data.get_plotting_df_all_upgrades(params)
    save_bsq_obj(viz_data.baseline_run)
    save_bsq_obj(viz_data.main_run)


if __name__ == "__main__":
    save_viz_data_reference_data()
