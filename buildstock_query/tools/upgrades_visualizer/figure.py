import re

import plotly.graph_objects as go
import polars as pl

from buildstock_query.tools.upgrades_visualizer.plot_utils import PlotParams, ValueTypes, flatten_list, human_sort_key
from buildstock_query.tools.upgrades_visualizer.viz_data import VizData

color_list = [
    '#F7DF10',  # Interior Lighting
    '#DEC310',  # Exterior Lighting
    '#4A4D4A',  # Plug Loads
    '#29AAE7',  # Refrigerator
    '#3cb6f0',  # Extra Refrigerator
    '#59caff',  # Freezer
    '#51e889',  # Clothes Washer
    '#FF79AD',  # Clothes Dryer
    '#D3D3D3',  # Dishwasher
    '#ff2200',  # Cooking Range
    '#632C94',  # Well Pump
    '#ff7300',  # Pool/Spa Pump
    '#FFB239',  # Hot Water
    '#C0C0C0',  # Ceiling Fan
    '#FF79AD',  # Vent Fans
    '#632C94',  # HVAC Fan/Pump
    '#0071BD',  # Cooling
    '#EF1C21',  # Heating
    '#1adb61',  # Electric Vehicle
    '#4748a8'  # PV
]  # from EZVIZ.py


class UpgradesPlot:
    def __init__(self, viz_data: VizData) -> None:
        self.viz_data = viz_data

    def get_ylabel(self, end_use):
        if len(end_use) == 1:
            return end_use[0]
        pure_end_use_name = end_use[0].removeprefix("end_use_")
        pure_end_use_name = pure_end_use_name.removeprefix("fuel_use_")
        pure_end_use_name = "_".join(pure_end_use_name.split("_")[1:])
        return f"{len(end_use)}_fuels_{pure_end_use_name}"

    def explode_str(self, input_str):
        input_str = str(input_str).lower()
        month2num = {"january": 1, "february": 2, "march": 3, "april": 4,
                     "may": 5, "june": 6, "july": 7, "august": 8,
                     "september": 9, "october": 10, "november": 11, "december": 12}
        input_str = str(month2num.get(input_str, input_str))
        input_str = [
            int(x) if x and x[0] in "0123456789" else x
            for x in re.split(r"([\<\-])|([0-9]+)", input_str)
        ]

        return tuple("X" if x is None else x for x in input_str)

    def get_plot(self, params: PlotParams):
        if len(params.group_by) >= 2 or params.upgrade is not None or \
           (params.value_type in [ValueTypes.distribution, ValueTypes.scatter,
                                  ValueTypes.sorted] and len(params.group_by) >= 1):
            params.upgrade = params.upgrade if params.upgrade else '0'
            params.group_by = ['upgrade'] if not params.group_by else params.group_by
            plot_df = self.viz_data.get_plotting_df(upgrade=params.upgrade, params=params)
        else:
            params.group_by = ['upgrade', *params.group_by]
            plot_df = self.viz_data.get_plotting_df_all_upgrades(params=params)

        return self._get_plot(plot_df, params)

    def _get_plot(self, df: pl.DataFrame, params: PlotParams):
        fig = go.Figure()
        counter = 0
        counter2 = 0
        report_dfs = [pl.DataFrame()]

        if params.value_type in [ValueTypes.mean, ValueTypes.total, ValueTypes.count]:
            xtitle = ", ".join(params.group_by[1:]) if len(params.group_by) > 1 else params.group_by[0]
            ytitle = f"{self.get_ylabel(params.enduses)}_{params.value_type.value}_{params.savings_type.value}"
        elif params.value_type in [ValueTypes.distribution]:
            xtitle = ", ".join(params.group_by[1:]) if len(params.group_by) > 1 else params.group_by[0]
            ytitle = f"{self.get_ylabel(params.enduses)}_{params.savings_type.value}"
        else:
            assert params.value_type in [ValueTypes.scatter, ValueTypes.sorted]
            xtitle = "baseline_value"
            ytitle = f"{self.get_ylabel(params.enduses)}_{params.savings_type.value}"
        for raw_grp0, group_df in df.group_by(params.group_by[0], maintain_order=True):
            grp0 = raw_grp0[0] if isinstance(raw_grp0, tuple) and len(raw_grp0) == 1 else raw_grp0
            if params.upgrade is None:
                upgrade: str = '0'
            else:
                upgrade: str = str(grp0) if params.group_by[0] == 'upgrade' else params.upgrade
            yvals = []
            xvals = []
            second_groups = []
            sample_counts = []
            upgrades = []
            hovervals = []
            building_ids = []
            if len(params.group_by) > 1:
                second_plots = [(group_name, group_df) for group_name, group_df in
                                group_df.group_by(params.group_by[1:], maintain_order=True)]
            else:
                second_plots = [((), group_df)]
            for second_name, second_df in second_plots:
                name = ','.join(second_name) if second_name else str(grp0)
                count = len(second_df)
                mean = second_df['value'].mean()
                if counter >= 500:
                    yvals.append(0.1)
                    xvals.append("Too many groups")
                    sample_counts.append(0)
                    upgrades.append(upgrade)
                    hovervals.append("Too many groups")
                    building_ids.append(0)
                    second_groups.append("Too many groups")
                    grp0 = "Too many groups"
                    break
                sort_index = 0
                if params.value_type in [ValueTypes.total, ValueTypes.mean, ValueTypes.count]:
                    if params.value_type == ValueTypes.total:
                        val = second_df['value'].sum()
                    elif params.value_type == ValueTypes.count:
                        val = second_df['building_id'].n_unique()
                    else:
                        val = second_df['value'].mean()
                    val = float(val)
                    yvals.append(val)
                    xvals.append(name)
                    sample_counts.append(count)
                    upgrades.append(upgrade)
                    hovertext = f"{self.viz_data.upgrade2name.get(upgrade)}<br>{grp0}<br>{name}<br>Average {mean}."\
                        f"<br>Sample Count: {count}."
                    f"<br>Units Count: {count * self.viz_data.sample_weight}."
                    hovervals.append(hovertext)
                    second_groups.append(name)
                    building_ids.append(0)
                elif params.value_type in [ValueTypes.distribution, ValueTypes.scatter, ValueTypes.sorted]:
                    hovertext = [f'{self.viz_data.upgrade2name.get(upgrade)}<br>{grp0}<br>{name}<br>Building:'
                                 f'{bid}<br>Sample Count: {count}'
                                 for bid in second_df['building_id'].to_list()]
                    if params.value_type in [ValueTypes.distribution]:
                        xvals.extend([name] * count)
                        yvals.extend(second_df['value'].to_list())
                        second_groups.extend([name] * count)
                        sample_counts.extend([count] * count)
                        upgrades.extend([upgrade] * count)
                        hovervals.extend(hovertext)
                        building_ids.extend(second_df['building_id'].to_list())
                    else:
                        xvals.append(second_df['baseline_value'].to_list())
                        yvals.append(second_df['value'].to_list())
                        second_groups.append([name] * count)
                        sample_counts.append([count] * count)
                        upgrades.append([upgrade] * count)
                        hovervals.append(hovertext)
                        building_ids.append(second_df['building_id'].to_list())
                        sort_index = 2
                counter += 1
                counter2 += 1
            # sort the xvals using human_sort_key and yvals accordingly

            xvals, yvals, second_groups, sample_counts, hovervals, upgrades, building_ids = zip(*sorted(
                zip(xvals, yvals, second_groups, sample_counts, hovervals, upgrades, building_ids, strict=False),
                key=lambda x: human_sort_key(x[sort_index])), strict=False)
            self._add_plot(params, fig, grp0, yvals, xvals, second_groups, hovervals, len(params.group_by) > 1)
            try:
                fl = flatten_list
                if params.value_type == ValueTypes.scatter:
                    report_df = pl.DataFrame({xtitle: fl(xvals), ytitle: fl(yvals),
                                              'upgrade': [f'Upgrade {grp0}'] * len(fl(xvals)),
                                              'sample_count': fl(sample_counts), 'info': fl(hovervals),
                                              'second_group': fl(second_groups), 'building_id': fl(building_ids)})
                else:
                    report_df = pl.DataFrame({xtitle: xvals, ytitle: yvals,
                                              'upgrade': [f'Upgrade {grp0}'] * len(xvals),
                                              'sample_count': sample_counts, 'info': hovervals,
                                              'second_group': second_groups, 'building_id': building_ids})
                # sub_df = sub_df.with_columns(pl.col(ytitle).cast(pl.Float32))
                report_dfs.append(report_df)
            except Exception as e:
                print(e)
                continue

        if params.change_type:
            title = f"{params.value_type} - {params.savings_type} for {params.change_type} buildings"
        else:
            title = f'{params.value_type} - {params.savings_type} value'

        if params.group_by[0] != "upgrade":
            title = f"Upgrade {params.upgrade} - {title}"

        self._update_layout(params, fig, xtitle, ytitle, title, len(params.group_by) > 1)
        fig.data = sorted(fig.data, key=lambda x: human_sort_key(x.name) if x.name else "")

        return fig, pl.concat(report_dfs)

    def _update_layout(self, params: PlotParams, fig, xtitle, ytitle, title, multi_group=False):
        if params.value_type in [ValueTypes.mean, ValueTypes.total, ValueTypes.count]:
            fig.update_layout(yaxis_title=ytitle,
                              barmode='group',
                              xaxis_title=xtitle,
                              legend={"title": params.group_by[0]},
                              title=title)
        elif params.value_type in [ValueTypes.distribution] or params.value_type in [ValueTypes.sorted]:
            fig.update_layout(yaxis_title=ytitle,
                              boxmode="group" if multi_group else "overlay",
                              xaxis_title=xtitle,
                              title=title,
                              legend={"title": params.group_by[0]},
                              clickmode='event+select')
        elif params.value_type == ValueTypes.scatter:
            fig.update_layout(yaxis_title=ytitle,
                              title=title,
                              boxmode="group" if multi_group else "overlay",
                              legend={"title": params.group_by[0]},
                              clickmode='event+select')
            fig.add_annotation(
                {
                    'x': 0.5,
                    'y': -0.20,
                    'showarrow': False,
                    'text': xtitle,
                    'xref': "paper",
                    'yref': "paper",
                    'xanchor': "center",
                    'yanchor': "top",
                    'font': {'size': 14}
                }
            )

    def _add_plot(self, params: PlotParams, fig, grp0, yvals, xvals, second_groups, hovervals, multi_group=False):
        if params.value_type in [ValueTypes.mean, ValueTypes.total, ValueTypes.count]:
            fig.add_trace(go.Bar(
                y=yvals,
                x=xvals,
                hovertext=hovervals,
                name=f'{grp0}',
                hoverinfo="all"
            )).update_traces(
                marker={"line": {"width": 0.5, "color": "rgb(0,0,0)"}}
            )
        elif params.value_type == ValueTypes.distribution:
            fig.add_trace(go.Box(
                y=yvals,
                x=xvals,
                name=f'{grp0}',
                boxpoints='suspectedoutliers',
                boxmean=True,  # represent mean
                hovertext=hovervals,
                hoverinfo="all"
            ))
        elif params.value_type in [ValueTypes.scatter, ValueTypes.sorted]:
            if multi_group:
                self._add_multi_scatter(fig, grp0, list(zip(second_groups, xvals, yvals, hovervals, strict=False)), params.value_type)
            elif params.value_type == ValueTypes.sorted:
                yvals, hovervals = zip(*sorted(zip(yvals[0], hovervals[0], strict=False), reverse=True), strict=False)
                xvals = list(range(len(yvals)))
                fig.add_trace(go.Scatter(
                    y=yvals,
                    x=xvals,
                    name=f'{grp0}',
                    mode='lines',
                    hovertext=hovervals,
                    hoverinfo="all"))
            else:
                fig.add_trace(go.Scatter(
                    y=yvals[0],
                    x=xvals[0],
                    name=f'{grp0}',
                    mode='markers',
                    hovertext=hovervals[0],
                    hoverinfo="all"))

    def _add_multi_scatter(self, fig, grp0, plot_tuples, plot_type):
        # Define the width of each subplot based on total plots and desired gaps
        num_plots = max(len(plot_tuples), 1)
        gap_fraction = 0.05  # for a 5% gap
        subplot_width = (1.0 - (num_plots-1)*gap_fraction) / num_plots
        # color_index = 0
        # chosen_color = dict()
        # Iterate over the tuples and add scatter plots
        for index, (name, x_values, y_values, hover_values) in enumerate(plot_tuples):
            domain_start = index * (subplot_width + gap_fraction)
            domain_end = domain_start + subplot_width

            # Constructing the appropriate xaxis and yaxis keys
            scatter_xaxis_key = 'x' if index == 0 else f'x{index+1}'
            scatter_yaxis_key = 'y' if index == 0 else f'y{index+1}'

            layout_xaxis_key = f'xaxis{"" if index == 0 else index+1}'
            layout_yaxis_key = f'yaxis{"" if index == 0 else index+1}'

            mode = 'markers'
            plot_x_values = x_values
            plot_y_values = y_values
            plot_hover_values = hover_values
            if plot_type == ValueTypes.sorted:
                plot_y_values, plot_hover_values = zip(
                    *sorted(zip(plot_y_values, plot_hover_values, strict=False), reverse=True),
                    strict=False,
                )
                plot_x_values = list(range(len(plot_y_values)))
                mode = 'lines'

            fig.add_trace(go.Scatter(x=plot_x_values,
                                     y=plot_y_values,
                                     hovertext=plot_hover_values,
                                     name=grp0,
                                     legendgroup=grp0,
                                     mode=mode,
                                     showlegend=index == 0,
                                     xaxis=scatter_xaxis_key,
                                     yaxis=scatter_yaxis_key,
                                     hoverinfo="all"))

            # Update the layout with the appropriate axis properties
            fig.update_layout({
                layout_xaxis_key: {
                    'domain': [domain_start, domain_end],
                    'anchor': scatter_yaxis_key,
                    'title': str(name)
                },
                layout_yaxis_key: {
                    'anchor': scatter_xaxis_key
                }
            })
