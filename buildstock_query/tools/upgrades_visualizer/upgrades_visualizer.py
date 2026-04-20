"""
- - - - - - - - -
Upgrades Visualizer
Experimental Stage.
:author: Rajendra.Adhikari@nrel.gov
"""

from functools import reduce
import re
from collections import defaultdict, Counter
import dash_bootstrap_components as dbc
from dash import html, ALL, dcc, ctx
import dash
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import MultiplexerTransform, DashProxy
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from InquirerPy import inquirer
from buildstock_query.tools.upgrades_visualizer.viz_data import VizData
from buildstock_query.tools.upgrades_visualizer.plot_utils import PlotParams, ValueTypes, SavingsTypes
from buildstock_query.tools.upgrades_visualizer.figure import UpgradesPlot
from buildstock_query.helpers import load_script_defaults, save_script_defaults
import polars as pl
from typing import Literal

# os.chdir("/Users/radhikar/Documents/eulpda/EULP-data-analysis/eulpda/smart_query/")
# from: https://github.com/thedirtyfew/dash-extensions/tree/1b8c6466b5b8522690442713eb421f622a1d7a59
# app = DashProxy(transforms=[
#     # TriggerTransform(),  # enable use of Trigger objects
#     MultiplexerTransform(),  # makes it possible to target an output multiple times in callbacks
#     # ServersideOutputTransform(),  # enable use of ServersideOutput objects
#     # NoOutputTransform(),  # enable callbacks without output
#     # BlockingCallbackTransform(),  # makes it possible to skip callback invocations while a callback is running
#     # LogTransform()  # makes it possible to write log messages to a Dash component
# ])
transforms = [MultiplexerTransform()]

opt_sat_path = "/Users/radhikar/Downloads/options_saturations.csv"
default_end_use = "fuel_use_electricity_total_m_btu"


def filter_cols(all_columns, prefixes=[], suffixes=[]):
    cols = []
    for col in all_columns:
        for prefix in prefixes:
            if col.startswith(prefix):
                cols.append(col)
                break
        else:
            for suffix in suffixes:
                if col.endswith(suffix):
                    cols.append(col)
                    break
    return cols


def get_int_set(input_str):
    """
        Convert "1,2,3-6,8,9" to [1, 2, 3, 4, 5, 6, 8, 9]
    """
    if not input_str:
        return set()

    pattern = r'^(\d+(-\d+)?,)*(\d+(-\d+)?)$'
    if not re.match(pattern, input_str):
        raise ValueError(f"{input_str} is not a valid pattern for list")

    result = set()
    segments = input_str.split(',')
    for segment in segments:
        if '-' in segment:
            start, end = map(int, segment.split('-'))
            result |= set(range(start, end + 1))
        else:
            result.add(int(segment))

    return result


def _get_app(opt_sat_path: str, db_name: str = 'euss-tests',
             table_name: str = 'res_test_03_2018_10k_20220607',
             workgroup: str = 'largeee',
             buildstock_type: Literal['resstock', 'comstock'] = 'resstock',
             include_monthly: bool = True,
             upgrades_selection_str: str = ''):
    viz_data = VizData(opt_sat_path=opt_sat_path, db_name=db_name,
                       run=table_name, workgroup=workgroup, buildstock_type=buildstock_type,
                       include_monthly=include_monthly,
                       upgrades_selection=get_int_set(upgrades_selection_str)
                       )
    return get_app(viz_data)


def get_app(viz_data: VizData):
    upgrades_plot = UpgradesPlot(viz_data)
    upgrade2res = viz_data.upgrade2res
    # upgrade2res_monthly = viz_data.upgrade2res_monthly
    upgrade2name = viz_data.upgrade2name
    all_cols = viz_data.upgrade2res['0'].columns
    emissions_cols = filter_cols(all_cols, suffixes=['_lb'])
    end_use_cols = filter_cols(all_cols, ["end_use_", "energy_use__", "fuel_use_"])
    water_usage_cols = filter_cols(all_cols, suffixes=["_gal"])
    load_cols = filter_cols(all_cols, ["load_", "flow_rate_", "component_load_", "hvac_design_load_"])
    peak_cols = filter_cols(all_cols, ["peak_"])
    unmet_cols = filter_cols(all_cols, ["unmet_", "resilience_battery_hr"])
    area_cols = filter_cols(all_cols, suffixes=["_ft_2", ])
    size_cols = filter_cols(all_cols, ["size_"])
    qoi_cols = filter_cols(all_cols, ["qoi_"])
    cost_cols = filter_cols(all_cols, ["upgrade_cost_"])

    build_cols = viz_data.metadata_df.columns
    listed_cols = emissions_cols + water_usage_cols + load_cols + peak_cols + unmet_cols + area_cols + size_cols +\
        qoi_cols + cost_cols + build_cols + end_use_cols
    numerical_cols = [name for name, dtype in viz_data.upgrade2res['0'].schema.items()
                      if dtype in (pl.NUMERIC_DTYPES)]
    other_cols = list(set(numerical_cols) - set(listed_cols))
    char_cols = [c.removeprefix(viz_data.main_run._char_prefix) for c in build_cols if 'applicable' not in c]
    char_cols += ['month']
    fuels_types = ['electricity', 'natural_gas', 'propane', 'fuel_oil', 'coal', 'wood_cord', 'wood_pellets']
    change_types = ["any", "no-chng", "bad-chng", "ok-chng", "true-bad-chng", "true-ok-chng"]
    download_csv_df = pl.DataFrame()

    def get_buildings(upgrade: str):
        return upgrade2res[upgrade]['building_id'].to_list()

    def get_plot(end_use, value_type='mean', savings_type='', change_type='',
                 sync_upgrade=None, filter_bldg=None, group_cols=None, report_upgrade=None,
                 resolution='annual'):
        filter_bldg = filter_bldg or []
        group_cols = group_cols or []
        sync_upgrade = sync_upgrade or 0
        report_upgrade = report_upgrade if report_upgrade else None

        params = PlotParams(enduses=end_use, value_type=ValueTypes[value_type.lower()],
                            savings_type=SavingsTypes[savings_type.lower().replace(' ', '_')],
                            change_type=change_type, sync_upgrade=sync_upgrade,
                            filter_bldgs=filter_bldg, group_by=group_cols, upgrade=report_upgrade,
                            resolution=resolution)
        return upgrades_plot.get_plot(params)

    external_script = ["https://tailwindcss.com/", {"src": "https://cdn.tailwindcss.com"}]

    app = DashProxy(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], transforms=transforms,
                    external_scripts=external_script)
    app.layout = html.Div([dbc.Container(html.Div([
        dcc.Download(id="download-dataframe-csv"),
        dbc.Row([dbc.Col(html.H1("Upgrades Visualizer"), width='auto'), dbc.Col(html.Sup("beta"))]),
        # Add a row for annual, vs monthly vs seasonal plot radio buttons
        dbc.Row([dbc.Col(dbc.Label("Resolution: "), width='auto'),
                 dbc.Col(dcc.RadioItems(["annual", "monthly"] if viz_data.include_monthly else ["annual"], "annual",
                                        inline=True, id="radio_resolution"))]),

        dbc.Row([dbc.Col(dbc.Label("Visualization Type: "), width='auto'),
                 dbc.Col(dcc.RadioItems(["Mean", "Total", "Count", "Distribution", "Scatter", "Sorted"], "Mean",
                                        id="radio_graph_type",
                                        inline=True,
                                        labelClassName="pr-2"), width='auto'),
                 dbc.Col(dbc.Label("Value Type: "), width='auto'),
                 dbc.Col(dcc.RadioItems(["Absolute", "Savings", "Percent Savings"], "Absolute",  inline=True,
                                        id='radio_savings', labelClassName="pr-2"), width='auto'),
                 ]),
        dbc.Row([dbc.Col(html.Br())]),
        dbc.Row([dbc.Col(dcc.Loading(id='graph-loader', children=[html.Div(id='loader_label')]))]),
        dbc.Row([dbc.Col(dcc.Graph(id='graph'))]),
        dbc.Row([dbc.Col(dbc.Button("Download", id='csv-download'))], justify='end'),
        dcc.Store(id='graph-data-store'),
    ])),
        dbc.Row([dbc.Col(
            dcc.Tabs(id='tab_view_type', value='energy', children=[
                dcc.Tab(id='energy_tab', label='Energy', value='energy', children=[
                    dcc.RadioItems(fuels_types + ['All'], "electricity", id='radio_fuel',  inline=True,
                                   labelClassName="pr-2")]
                        ),
                dcc.Tab(label='Water Usage', value='water', children=[]
                        ),
                dcc.Tab(label='Load', value='load', children=[]
                        ),
                dcc.Tab(label='Peak', value='peak', children=[]
                        ),
                dcc.Tab(label='Unmet Hours', value='unmet_hours', children=[]
                        ),
                dcc.Tab(label='Area', value='area', children=[]
                        ),
                dcc.Tab(label='Size', value='size', children=[]
                        ),
                dcc.Tab(label='QOI', value='qoi', children=[]
                        ),
                dcc.Tab(label='emissions', value='emissions', children=[]
                        ),
                dcc.Tab(label='Upgrade Cost', value='upgrade_cost', children=[]
                        ),
                dcc.Tab(label='Others', value='others', children=[]
                        ),
            ])
        )
        ], className="mx-5 mt-5"),
        dbc.Row([dbc.Col(dcc.Dropdown(id='dropdown_enduse'))], className="mx-5 my-1"),
        dbc.Row(
            dbc.Col([
                dbc.Row([dbc.Col(html.Div("Restrict to buildings that have "), width='auto'),
                         dbc.Col(dcc.Dropdown(change_types, "", placeholder="Select change type...",
                                              id='dropdown_chng_type'), width='2'),
                         dbc.Col(html.Div(" in "), width='auto'),
                         dbc.Col(dcc.Dropdown(id='sync_upgrade', value='',
                                              options={}))
                         ],
                        className="flex items-center"),
                dbc.Row([dbc.Col(html.Div("Select:"), style={"padding-left": "12px", "padding-right": "0px"},
                                 width='auto'),
                         dbc.Col(dcc.Dropdown(id='input_building'), width=1),
                         dbc.Col(html.Div("("), width='auto',
                                 style={"padding-left": "0px", "padding-right": "0px"}),
                         dbc.Col(dcc.Checklist(['Lock)'], [],
                                               inline=True, id='chk-lock'),
                                 width='auto', style={"padding-left": "0px", "padding-right": "0px"}),
                         dbc.Col(html.Div(" in "), style={"padding-right": "0px"}, width='auto'),
                         dbc.Col(dcc.Dropdown(id='report_upgrade', value='', placeholder="Upgrade ...",
                                              options=viz_data.upgrade2shortname), width=1),
                         dbc.Col(html.Div("grouped by:"), style={"padding-right": "0px"}, width='auto'),
                         dbc.Col(dcc.Dropdown(id='drp-group-by', options=char_cols, value=None,
                                              multi=True, placeholder="Select characteristics..."),
                                 width=3),
                         dbc.Col(dbc.Button("<= Copy", id="btn-copy", color="primary", size="sm",
                                            outline=True), class_name="centered-col", style={"padding-right": "0px"}),
                         dbc.Col(html.Div("Extra restriction: "), style={"padding-right": "0px"}, width='auto'),
                         dbc.Col(dcc.Dropdown(id='input_building2', disabled=False), width=1),
                         dbc.Col(dcc.Checklist(id='chk-graph', options=['Graph'], value=[],
                                               inline=True), width='auto'),
                         dbc.Col(dcc.Checklist(id='chk-options', options=['Options'], value=[],
                                               inline=True), width='auto'),
                         dbc.Col(dcc.Checklist(id='chk-enduses', options=['Enduses'], value=[],
                                               inline=True), width='auto'),
                         dbc.Col(dcc.Checklist(id='chk-chars', options=['Chars'], value=[],
                                               inline=True), width='auto'),
                         dbc.Col(dbc.Button("Reset", id="btn-reset", color="primary", size="sm", outline=True),
                                 width='auto'),
                         ]),
                dbc.Row([dbc.Col([
                    dbc.Row(html.Div(id="options_report_header")),
                    dbc.Row(dcc.Loading(id='opt-loader',
                                        children=html.Div(id="options_report"))),
                    dcc.Store("opt_report_store")
                ], width=5),
                    dbc.Col([
                        dbc.Row([dbc.Col(html.Div("View enduse that: "), width='auto'),
                                 dbc.Col(dcc.Dropdown(id='input_enduse_type',
                                                      options=['changed', 'increased', 'decreased', 'are almost zero'],
                                                      value='changed', clearable=False),
                                         width=2),
                                 dbc.Col(html.Div(), width='auto'),
                                 dbc.Col(html.Div("Charecteristics Report:"), width='auto'),
                                 dbc.Col(dcc.Dropdown(id='drp-char-report', options=char_cols, value=None,
                                                      multi=True, placeholder="Select characteristics...")),
                                 ]),
                        dbc.Row([dbc.Col(dcc.Loading(id='enduse_loader',
                                                     children=html.Div(id="enduse_report"))
                                         ),
                                 dbc.Col(dcc.Loading(id='char_loader',
                                                     children=html.Div(id="char_report"))
                                         ),
                                 ]),
                        dcc.Store("enduse_report_store"),
                        dcc.Store("char_report_store")
                    ], width=7)

                ]),
                dbc.Row([
                    dbc.Col(width=5),
                    dbc.Col(
                        dbc.Row([
                            dbc.Col(),
                            dbc.Col(dbc.Button("Download All Characteristics", id="btn-dwn-chars",
                                               color="primary",
                                               size="sm", outline=True), width='auto'),
                        ]), width=7)
                ])
            ]), className="mx-5 my-1"),
        html.Div(id="status_bar"),
        dcc.Download(id="download-chars-csv"),
        dcc.Store("uirevision")
        # dbc.Button("Kill me", id="button110")
    ])

    # download data with button click
    @app.callback(
        Output("download-dataframe-csv", "data"),
        Input("csv-download", "n_clicks"),
        prevent_initial_call=True)
    def download_csv(n_clicks):
        if not n_clicks:
            raise PreventUpdate()
        bldg_ids = set(download_csv_df['building_id'].to_list()) - {0}
        bdf = viz_data.upgrade2res['0'].filter(pl.col("building_id").is_in(set(bldg_ids))).select(char_cols)
        if len(bdf) > 0:
            combined_df = download_csv_df.join(bdf, on="building_id", how="left")
            return dcc.send_bytes(combined_df.write_csv, "graph_data.csv")
        else:
            return dcc.send_bytes(download_csv_df.write_csv, "graph_data.csv")

    @app.callback(
        Output("download-chars-csv", "data"),
        Input("btn-dwn-chars", "n_clicks"),
        State("input_building", "value"),
        State("input_building", "options"),
        State("input_building2", "options"),
        State('chk-chars', 'value'),
    )
    def download_char(n_clicks, bldg_id, bldg_options, bldg_options2, chk_chars):
        if not n_clicks:
            raise PreventUpdate()

        if "Chars" in chk_chars and bldg_options2:
            bldg_ids = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options2]
        else:
            bldg_ids = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options]
        bldg_ids = [int(b) for b in bldg_ids]
        bdf = viz_data.upgrade2res['0'].filter(pl.col("building_id").is_in(set(bldg_ids))).select(char_cols)
        return dcc.send_bytes(bdf.write_csv, f"chars_{n_clicks}.csv")

    def get_elligible_output_columns(category, fuel, resolution):
        if category == 'energy':
            elligible_cols = viz_data.get_cleaned_up_end_use_cols(resolution, fuel)
        elif category == 'water':
            elligible_cols = water_usage_cols if resolution == 'annual' else []
        elif category == 'load':
            elligible_cols = load_cols
        elif category == 'peak':
            elligible_cols = peak_cols if resolution == 'annual' else []
        elif category == 'unmet_hours':
            elligible_cols = unmet_cols if resolution == 'annual' else []
        elif category == 'area':
            elligible_cols = area_cols if resolution == 'annual' else []
        elif category == 'size':
            elligible_cols = size_cols if resolution == 'annual' else []
        elif category == 'qoi':
            elligible_cols = qoi_cols if resolution == 'annual' else []
        elif category == 'emissions':
            elligible_cols = emissions_cols if resolution == 'annual' else\
                viz_data.get_emissions_cols(resolution=resolution)
        elif category == 'upgrade_cost':
            elligible_cols = cost_cols if resolution == 'annual' else []
        elif category == 'others':
            elligible_cols = other_cols if resolution == 'annual' else []
        else:
            raise ValueError(f"Invalid tab {category}")
        return elligible_cols

    @app.callback(
        Output('dropdown_enduse', "options"),
        Output('dropdown_enduse', "value"),
        Input('tab_view_type', "value"),
        Input('radio_fuel', "value"),
        Input('dropdown_enduse', "value"),
        Input('radio_resolution', 'value')
    )
    def update_enduse_options(view_tab, fuel_type, current_enduse, resolution):
        elligible_cols = get_elligible_output_columns(view_tab, fuel_type, resolution)
        enduse = current_enduse if current_enduse in elligible_cols else elligible_cols[0]
        return sorted(elligible_cols), enduse

    @app.callback(
        Output("sync_upgrade", 'value'),
        Output("sync_upgrade", 'options'),
        Output("sync_upgrade", 'placeholder'),
        Input('dropdown_chng_type', "value"),
        State("sync_upgrade", "value"),
        State("sync_upgrade", "options")
    )
    def update_sync_upgrade(chng_type, current_sync_upgrade, sync_upgrade_options):
        # print(chng_type, current_sync_upgrade, sync_upgrade_options)
        if chng_type:
            return current_sync_upgrade, upgrade2name, 'respective upgrades. (Click to restrict to specific upgrade)'
        else:
            return '', {}, ' <select a change type on the left first>'

    @app.callback(
        Output('input_building', 'placeholder'),
        Input('input_building', 'options')
    )
    def update_building_placeholder(options):
        return f"{len(options)} Buidlings" if options else "0 buildings."

    @app.callback(
        Output('input_building2', 'placeholder'),
        Output('input_building2', 'value'),
        Input('input_building2', 'value'),
        Input('input_building2', 'options')
    )
    def update_building_placeholder2(value, options):
        return f"{len(options)} Buidlings" if options else "No restriction", None

    @app.callback(
        Output('input_building', 'value'),
        Output('input_building', 'options'),
        Output('report_upgrade', 'value'),
        Input('graph', "selectedData"),
        State('input_building', 'options'),
        State('report_upgrade', 'value')
    )
    def graph_click(selection_data, current_options, current_upgrade):
        if not selection_data or 'points' not in selection_data or len(selection_data['points']) < 1:
            raise PreventUpdate()

        selected_buildings = []
        selected_upgrades = []
        for point in selection_data['points']:
            if not (match := re.search(r"Building: (\d*)", point.get('hovertext', ''))):
                continue
            if bldg := match.groups()[0]:
                upgrade_match = re.search(r"Upgrade (\d*)", point.get('hovertext', ''))
                upgrade = upgrade_match.groups()[0] if upgrade_match else ''
                selected_buildings.append(bldg)
                selected_upgrades.append(upgrade)

        if not selected_buildings:
            raise PreventUpdate()

        selected_upgrade = selected_upgrades[0] or current_upgrade
        if len(selected_buildings) != 1:
            selected_buildings = list(set(selected_buildings))
            return '', selected_buildings, selected_upgrade
        current_options = current_options or selected_buildings
        return selected_buildings[0], current_options, selected_upgrade

    @app.callback(
        Output('chk-graph', 'value'),
        Output('chk-options', 'value'),
        Output('chk-enduses', 'value'),
        Output('input_building2', 'options'),
        Output("uirevision", "data"),
        Input('btn-reset', "n_clicks")
    )
    def reset(n_clicks):
        return [], [], [], [], n_clicks

    @app.callback(
        Output('input_building', 'options'),
        Input('btn-copy', "n_clicks"),
        State('input_building2', 'options'),
    )
    def copy(n_clicks, bldg_options2):
        return bldg_options2 or dash.no_update

    @app.callback(
        Output('input_building', 'value'),
        Output('input_building', 'options'),
        Output('input_building2', 'options'),
        State('input_building', 'value'),
        State('input_building', 'options'),
        State('input_building2', 'options'),
        Input('chk-lock', 'value'),
        Input('dropdown_chng_type', "value"),
        Input('sync_upgrade', 'value'),
        Input('report_upgrade', 'value'),
        Input('btn-reset', "n_clicks")
    )
    def bldg_selection(current_bldg, current_options, current_options2, chk_lock,
                       change_type, sync_upgrade, report_upgrade,
                       reset_click):

        if sync_upgrade and change_type:
            valid_bldgs = set(viz_data.chng2bldg[(sync_upgrade, change_type)])
        elif report_upgrade and change_type:
            valid_bldgs = set(viz_data.chng2bldg[(report_upgrade, change_type)])
            buildings = get_buildings(report_upgrade)
            valid_bldgs = set(buildings).intersection(valid_bldgs)
        elif report_upgrade:
            buildings = get_buildings(report_upgrade)
            valid_bldgs = set(buildings)
        else:
            valid_bldgs = set(viz_data.upgrade2res['0']['building_id'].to_list())

        base_res = upgrade2res['0'].filter(pl.col("building_id").is_in(valid_bldgs))
        valid_bldgs = list(base_res['building_id'].to_list())

        if "btn-reset" != ctx.triggered_id and current_options and len(current_options) > 0 and chk_lock:
            current_options_set = set(current_options)
            valid_bldgs = [b for b in valid_bldgs if b in current_options_set]

        valid_bldgs2 = []

        if current_bldg and (int(current_bldg) not in valid_bldgs):
            current_bldg = valid_bldgs[0] if valid_bldgs else ''
        return current_bldg, valid_bldgs, valid_bldgs2

    def get_char_choices(char):
        if char:
            res0 = upgrade2res['0']
            unique_choices = sorted(list(res0[char].unique()))
            return unique_choices, unique_choices[0]
        else:
            return [], None

    def get_action_button_pairs(id, bldg_list_dict, report_type='opt'):
        buttons = []
        bldg_str = '' if len(bldg_list_dict[id]) > 10 else " [" + ','.join([str(b) for b in bldg_list_dict[id]]) + "]"
        for type in ['check', 'cross']:
            icon_name = "akar-icons:circle-check-fill" if type == 'check' else "gridicons:cross-circle"
            button = html.Div(dmc.ActionIcon(DashIconify(icon=icon_name,
                                                         width=20 if type == "check" else 22,
                                                         height=20 if type == "check" else 22,
                                                         ),
                                             id={'index': id, 'type': f'btn-{type}', 'report_type': report_type},
                                             variant="light"),
                              id=f"div-tooltip-target-{type}-{id}")
            if type == "check":
                tooltip = dbc.Tooltip(f"Select these buildings.{bldg_str}",
                                      target=f"div-tooltip-target-{type}-{id}", delay={'show': 1000})

                col = dbc.Col(html.Div([button, tooltip]),  width='auto', class_name="col-btn-cross",
                              style={"padding-left": "0px", "padding-right": "0px"})
            else:
                tooltip = dbc.Tooltip(f"Select all except these buildings.{bldg_str}",
                                      target=f"div-tooltip-target-{type}-{id}", delay={'show': 1000})
                col = dbc.Col(html.Div([button, tooltip]),  width='auto',  class_name="col-btn-check",
                              style={"padding-left": "0px", "padding-right": "0px"})
            buttons.append(col)
        return buttons

    @app.callback(
        Output('status_bar', "children"),
        Output('input_building2', "options"),
        Input({"type": "btn-check", "index": ALL, "report_type": "opt"}, "n_clicks"),
        Input({"type": "btn-cross", "index": ALL, "report_type": "opt"}, "n_clicks"),
        State("input_building", "options"),
        State("input_building2", "options"),
        State("chk-options", "value"),
        State("opt_report_store", "data"),
    )
    def opt_check_button_click(check_clicks, cross_clicks, bldg_options, bldg_options2, chk_options, opt_report):
        triggers = dash.callback_context.triggered_prop_ids
        if len(triggers) != 1:
            raise PreventUpdate()

        if "Options" in chk_options and bldg_options2:
            bldg_list = [int(b) for b in bldg_options2]
        else:
            bldg_list = [int(b) for b in bldg_options]

        trigger_val = next(iter(triggers.values()))
        buildings = opt_report.get(trigger_val['index'], [])
        if trigger_val['type'] == 'btn-check':
            return '', [str(b) for b in buildings]

        bldg_set = set(buildings)
        except_buildings = [str(v) for v in bldg_list if int(v) not in bldg_set]
        return '', except_buildings

    @app.callback(
        Output('status_bar', "children"),
        Output('input_building2', "options"),
        Input({"type": "btn-check", "index": ALL, "report_type": "enduse"}, "n_clicks"),
        Input({"type": "btn-cross", "index": ALL, "report_type": "enduse"}, "n_clicks"),
        State("input_building", "options"),
        State("input_building2", "options"),
        State("chk-enduses", "value"),
        State("enduse_report_store", "data"),
    )
    def enduse_button_click(check_clicks, cross_clicks, bldg_options, bldg_options2, chk_enduses, opt_report):
        triggers = dash.callback_context.triggered_prop_ids
        if len(triggers) != 1:
            raise PreventUpdate()

        if "Enduses" in chk_enduses and bldg_options2:
            bldg_list = [int(b) for b in bldg_options2]
        else:
            bldg_list = [int(b) for b in bldg_options]

        trigger_val = next(iter(triggers.values()))
        buildings = opt_report.get(trigger_val['index'], [])
        if trigger_val['type'] == 'btn-check':
            return '', [str(b) for b in buildings]

        bldg_set = set(buildings)
        except_buildings = [str(v) for v in bldg_list if int(v) not in bldg_set]
        return '', except_buildings

    @app.callback(
        Output('status_bar', "children"),
        Output('input_building2', "options"),
        Input({"type": "btn-check", "index": ALL, "report_type": "char"}, "n_clicks"),
        Input({"type": "btn-cross", "index": ALL, "report_type": "char"}, "n_clicks"),
        State("input_building", "options"),
        State("input_building2", "options"),
        State('chk-chars', 'value'),
        State("char_report_store", "data"),
    )
    def char_report_button_click(check_clicks, cross_clicks, bldg_options, bldg_options2, chk_char, char_report):
        triggers = dash.callback_context.triggered_prop_ids
        if len(triggers) != 1:
            raise PreventUpdate()

        if "Chars" in chk_char and bldg_options2:
            bldg_list = [int(b) for b in bldg_options2]
        else:
            bldg_list = [int(b) for b in bldg_options]

        trigger_val = next(iter(triggers.values()))
        buildings = char_report.get(trigger_val['index'], [])
        if trigger_val['type'] == 'btn-check':
            return '', [str(b) for b in buildings]

        bldg_set = set(buildings)
        except_buildings = [str(v) for v in bldg_list if int(v) not in bldg_set]
        return '', except_buildings

    @app.callback(
        Output("options_report_header", "children"),
        Output("options_report", 'children'),
        Output("opt_report_store", "data"),
        Input('input_building', "value"),
        Input('input_building', "options"),
        Input('input_building2', "options"),
        State('report_upgrade', 'value'),
        Input('chk-options', 'value'),
    )
    def show_opt_report(bldg_id, bldg_options, bldg_options2, report_upgrade, chk_options):
        if not report_upgrade or not bldg_options:
            return "Select an upgrade to see options applied in that upgrade", [''], {}

        if dash.callback_context.triggered_id == 'input_building2' and "Options" not in chk_options:
            raise PreventUpdate()

        if "Options" in chk_options and bldg_options2:
            bldg_list = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options2]
        else:
            bldg_list = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options]
        run_obj = viz_data.run_obj(report_upgrade)
        applied_options = run_obj.report.get_applied_options(upgrade_id=int(report_upgrade),
                                                             bldg_ids=bldg_list,
                                                             include_base_opt=True)
        opt_only = [{entry.split('|')[0] for entry in opt.keys()} for opt in applied_options]
        reduced_set = list(reduce(set.union, opt_only))

        nested_dict = defaultdict(lambda: defaultdict(Counter))
        bldg_list_dict = defaultdict(list)

        for bldg_id, opt_dict in zip(bldg_list, applied_options):
            for opt_para, value in opt_dict.items():
                opt = opt_para.split('|')[0]
                para = opt_para.split('|')[1]
                nested_dict[opt][para][value] += 1
                bldg_list_dict[opt].append(bldg_id)
                bldg_list_dict[opt + "|" + para].append(bldg_id)
                bldg_list_dict[opt + "|" + para + "<-" + value].append(bldg_id)

        def get_accord_item(opt_name):
            total_count = sum(counter.total() for counter in nested_dict[opt_name].values())
            children = []
            for parameter, counter in nested_dict[opt_name].items():
                contents = []
                new_counter = Counter({"All": counter.total()})
                new_counter.update(counter)
                for from_val, count in new_counter.items():
                    if from_val == "All":
                        but_ids = f"{opt_name}|{parameter}"
                    else:
                        but_ids = f"{opt_name}|{parameter}<-{from_val}"
                    entry = dbc.Row([dbc.Col(f"<-{from_val} ({count})", width="auto"),
                                     *get_action_button_pairs(but_ids, bldg_list_dict)])
                    contents.append(entry)
                children.append(dmc.AccordionItem(contents, label=f"{parameter} ({counter.total()})"))

            accordian = dmc.Accordion(children, multiple=True)
            first_row = dbc.Row([dbc.Col(f"All ({total_count})", width="auto"),
                                 *get_action_button_pairs(opt_name, bldg_list_dict)])
            return dmc.AccordionItem([first_row, accordian], label=f"{opt_name} ({total_count})")
        if reduced_set:
            final_report = dmc.Accordion([get_accord_item(opt_name) for opt_name in reduced_set], multiple=True)
        else:
            final_report = ["No option got applied to the selected building(s)."]
        up_name = upgrade2name[report_upgrade]
        return f"Options applied in {up_name}", final_report, dict(bldg_list_dict)

    @app.callback(
        Output("enduse_report", "children"),
        Output("enduse_report_store", "data"),
        State('report_upgrade', 'value'),
        Input('input_building', "value"),
        Input('input_building', "options"),
        Input('input_building2', "options"),
        Input('input_enduse_type', 'value'),
        Input('chk-enduses', 'value'),
    )
    def show_enduse_report(report_upgrade, bldg_id, bldg_options, bldg_options2, enduse_change_type, chk_enduse):
        if not report_upgrade or not bldg_options:
            return ["Select an upgrade to see enuse report."], {}

        if dash.callback_context.triggered_id == 'input_building2' and "Enduses" not in chk_enduse:
            raise PreventUpdate()

        if "Enduses" in chk_enduse and bldg_options2:
            bldg_list = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options2]
        else:
            bldg_list = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options]

        # print(bldg_list)
        run_obj = viz_data.run_obj(report_upgrade)
        dict_changed_enduses = run_obj.report.get_enduses_buildings_map_by_change(upgrade_id=int(report_upgrade),
                                                                                  change_type=enduse_change_type,
                                                                                  bldg_list=bldg_list)
        # print(changed_enduses)

        all_changed_enduses = list(dict_changed_enduses.keys())
        if not all_changed_enduses:
            if bldg_id:
                return f'No enduse has {enduse_change_type} in building {bldg_id} ', {}
            else:
                return f'No enduse has {enduse_change_type} in any of the buildings', {}

        enduses2bldgs = defaultdict(list)
        for end_use, bldgs in dict_changed_enduses.items():
            if end_use in all_changed_enduses:
                enduses2bldgs[end_use].extend([int(bldg_id) for bldg_id in bldgs])

        fuel2bldgs = defaultdict(set)
        fuel2enduses = defaultdict(list)
        for enduse, bldgs in enduses2bldgs.items():
            for fuel in ['all_fuel'] + fuels_types:
                if fuel in enduse:
                    fuel2bldgs[fuel] |= set(bldgs)
                    fuel2enduses[fuel].append(enduse)
                    break
            else:
                fuel2bldgs['other'] |= set(bldgs)
                fuel2enduses['other'].append(enduse)

        for key, val in fuel2bldgs.items():
            fuel2bldgs[key] = list(val)

        enduses2bldgs.update(fuel2bldgs)

        def get_accord_item(fuel):
            total_count = len(enduses2bldgs[fuel])
            contents = [dbc.Row([dbc.Col(f"All {fuel} ({total_count})", width="auto"),
                                 *get_action_button_pairs(fuel, enduses2bldgs, "enduse")])]
            for enduse in fuel2enduses[fuel]:
                count = len(enduses2bldgs[enduse])
                row = dbc.Row([dbc.Col(f"{enduse} ({count})", width="auto"),
                               *get_action_button_pairs(enduse, enduses2bldgs, "enduse")])
                contents.append(row)
            return dmc.AccordionItem(contents, label=f"{fuel} ({total_count})")

        report = dmc.Accordion([get_accord_item(fuel) for fuel in fuel2enduses.keys()],  multiple=True)
        storedict = dict(enduses2bldgs)
        return report, storedict

    @app.callback(
        Output("char_report", "children"),
        Output("char_report_store", "data"),
        Input('input_building', "value"),
        Input('input_building', "options"),
        Input('input_building2', "options"),
        Input('drp-char-report', 'value'),
        Input('chk-chars', 'value'),
    )
    def show_char_report(bldg_id, bldg_options, bldg_options2, inp_char: list[str], chk_chars):
        if not (bldg_options or bldg_options2 or bldg_id):
            return [""], {}
        if not inp_char:
            return ["Select a characteristics to see its report."], {}

        if dash.callback_context.triggered_id == 'input_building2' and "Chars" not in chk_chars:
            raise PreventUpdate()

        if "Chars" in chk_chars and bldg_options2:
            bldg_list = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options2]
        else:
            bldg_list = [int(bldg_id)] if bldg_id else [int(b) for b in bldg_options]
        chars_df = viz_data.bs_res_df.filter(pl.col('building_id').is_in(
            set(bldg_list))).select(inp_char + ['building_id'])
        char2bldgs = chars_df.group_by(inp_char).agg('building_id')
        if (total_len := len(char2bldgs)) > 250:
            return [f"Sorry, this would create more than 200 ({total_len}) rows."], {}
        char_dict = {}
        total_count = 0
        contents = []
        for char_vals, group_df in chars_df.group_by(inp_char):
            bldglist = group_df['building_id'].to_list()
            but_ids = "+".join(char_vals) if isinstance(char_vals, tuple) else char_vals
            char_dict[but_ids] = [int(b) for b in bldglist]
            count = len(bldglist)
            total_count += count
            contents.append(dbc.Row([dbc.Col(f"{char_vals} ({count})", width="auto"),
                                     *get_action_button_pairs(but_ids, char_dict, "char")]))

        report = dmc.Accordion([dmc.AccordionItem(contents, label=f"{inp_char} ({total_count})")])
        return report, char_dict

    @app.callback(
        Output('graph', 'figure'),
        Output('graph', 'config'),
        Output('loader_label', "children"),
        State('tab_view_type', "value"),
        Input('drp-group-by', 'value'),
        Input('radio_fuel', "value"),
        Input('dropdown_enduse', "value"),
        Input('radio_graph_type', "value"),
        Input('radio_savings', "value"),
        Input('dropdown_chng_type', "value"),
        Input('sync_upgrade', 'value'),
        Input('input_building', 'value'),
        Input('input_building', 'options'),
        Input('input_building2', 'options'),
        Input('chk-graph', 'value'),
        State("uirevision", "data"),
        State('report_upgrade', 'value'),
        State('radio_resolution', 'value')
    )
    def update_figure(view_tab, grp_by, fuel, enduse, graph_type, savings_type, chng_type,
                      sync_upgrade, selected_bldg, bldg_options, bldg_options2, chk_graph, uirevision,
                      report_upgrade, resolution):
        nonlocal download_csv_df
        if dash.callback_context.triggered_id == 'input_building2' and "Graph" not in chk_graph:
            raise PreventUpdate()

        if "Graph" in chk_graph and bldg_options2:
            bldg_options = bldg_options2

        bldg_options = bldg_options or []
        if not enduse:
            full_name = []
        if view_tab == 'energy':
            full_name = viz_data.get_end_use_db_cols(resolution, fuel, enduse)
        else:
            full_name = [enduse]

        if selected_bldg:
            filter_bldg = [int(selected_bldg)]
        else:
            filter_bldg = [int(b) for b in bldg_options]

        new_figure, report_df = get_plot(full_name, graph_type, savings_type,
                                         chng_type, sync_upgrade, filter_bldg, grp_by, report_upgrade,
                                         resolution)

        uirevision = uirevision or "default"
        new_figure.update_layout(uirevision=uirevision)
        download_csv_df = report_df
        config = {'edits': {"titleText": True, "axisTitleText": True}, 'displayModeBar': True,
                  "modeBarButtonsToRemove": ["Zoom", "ZoomIn", "Pan", "ZoomOut", "AutoScale", "select2d"]}
        return new_figure, config, ""

    return app


def main():
    print("Welcome to Upgrades Visualizer.")
    defaults = load_script_defaults("project_info")
    opt_sat_file = inquirer.text(message="Please enter path to the options saturation csv file:",
                                 default=defaults.get("opt_sat_file", "")).execute()
    workgroup = inquirer.text(message="Please enter Athena workgroup name:",
                              default=defaults.get("workgroup", "")).execute()
    db_name = inquirer.text(message="Please enter database name "
                            "(found in postprocessing:aws:athena in the buildstock configuration file):",
                            default=defaults.get("db_name", "")).execute()
    table_name = inquirer.text(message="Please enter table name (same as output folder name; found under "
                               "output_directory in the buildstock configuration file). [Enter two names "
                               "separated by comma if baseline and upgrades are in different run]:",
                               default=defaults.get("table_name", "")
                               ).execute()
    monthly_default = defaults.get("include_monthly", True)
    default_str = "Yes" if monthly_default else "No"
    include_monthly = inquirer.confirm(f"Do you want to include monthly plots ({default_str})?",
                                       default=monthly_default,
                                       ).execute()
    upgrades_selection = inquirer.text(message="Please enter upgrade ids separated by comma and dashes "
                                       "(example: `1-3,5,7,8-9`) or leave empty to include all upgrades.",
                                       default=defaults.get("upgrades_selection", "")).execute()
    defaults.update({"opt_sat_file": opt_sat_file, "workgroup": workgroup,
                     "db_name": db_name, "table_name": table_name, "include_monthly": include_monthly,
                     "upgrades_selection": upgrades_selection})
    save_script_defaults("project_info", defaults)
    if ',' in table_name:
        table_name = table_name.split(',')
    app = _get_app(opt_sat_path=opt_sat_file,
                   workgroup=workgroup,
                   db_name=db_name,
                   table_name=table_name,
                   include_monthly=include_monthly,
                   upgrades_selection_str=upgrades_selection)
    app.run_server(debug=False, port=8006)


if __name__ == '__main__':
    main()
