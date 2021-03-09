"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from collections import defaultdict
import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import plotly.express as px  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from radiantkit import distance, report, stat
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


def get_axis_label(axis: str, aid: int) -> str:
    return f"{axis}{aid+1}" if aid > 0 else axis


def get_axis_range(
    trace_list: List[go.Figure], axis_type: str, axis_label: str
) -> Tuple[float, float]:
    return (
        np.min(
            [
                trace[axis_type].min()
                for trace in trace_list
                if axis_label == trace[f"{axis_type}axis"]
            ]
        ),
        np.max(
            [
                trace[axis_type].max()
                for trace in trace_list
                if axis_label == trace[f"{axis_type}axis"]
            ]
        ),
    )


def add_derivative_xaxis_to_profiles(fig: go.Figure) -> go.Figure:
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0,
        y1=0,
        xsizemode="scaled",
        ysizemode="scaled",
        line_color="#969696",
        xref="x",
        yref="y2",
        line_dash="dash",
    )
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        y0=0,
        y1=0,
        xsizemode="scaled",
        ysizemode="scaled",
        line_color="#969696",
        xref="x",
        yref="y3",
        line_dash="dash",
    )
    return fig


def add_line_trace(
    fig: go.Figure,
    x0: Optional[float],
    x1: Optional[float],
    y0: Optional[float],
    y1: Optional[float],
    line_color: str = "#969696",
    **kwargs,
) -> go.Figure:
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[y0, y1],
            mode="lines",
            line_color=line_color,
            **kwargs,
        )
    )
    return fig


class ProfileMultiConditionNorm(object):
    html_class: str = "plot-multi-condition-normalized"
    _stub: str

    def __init__(self, stub: str):
        super(ProfileMultiConditionNorm, self).__init__()
        self._stub = stub

    def __make_scatter_trace(
        self,
        channel_data: pd.DataFrame,
        pfit: Dict[str, Dict[str, Any]],
    ) -> go.Scatter:
        condition_list: List[str] = sorted(list(set(channel_data["condition"])))
        panel_data = []
        for condition_idx in range(len(condition_list)):
            condition_lab = condition_list[condition_idx]
            condition_data = channel_data.loc[
                condition_lab == channel_data["condition"], :
            ]
            assert condition_lab in pfit
            assert "pfit" in pfit[condition_lab]
            x, y = pfit[condition_lab]["pfit"].linspace(200)
            xx, yy = pfit[condition_lab]["pfit"].deriv().linspace(200)
            xxx, yyy = pfit[condition_lab]["pfit"].deriv().deriv().linspace(200)
            stat_lab = pfit[condition_lab]["stat"].value
            panel_data.extend(
                [
                    go.Scatter(
                        name=f"{condition_lab}_{stat_lab}_raw",
                        xaxis="x",
                        yaxis=get_axis_label("y", condition_idx),
                        x=condition_data["x"],
                        y=condition_data[f"{stat_lab}_raw"],
                        mode="markers",
                        legendgroup=condition_lab,
                        marker=dict(
                            size=4,
                            opacity=0.5,
                            color=px.colors.qualitative.Pastel2[condition_idx],
                        ),
                        showlegend=False,
                    ),
                    go.Scatter(
                        name=f"{condition_lab}_{stat_lab}",
                        x=x,
                        y=y,
                        xaxis="x",
                        yaxis=get_axis_label("y", condition_idx),
                        mode="lines",
                        legendgroup=condition_lab,
                        line_color=px.colors.qualitative.Dark2[condition_idx],
                    ),
                ]
            )
        return panel_data

    def __add_der_zeros(
        self, fig: go.Figure, pfit_data: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        pfit_sorted = sorted(pfit_data.items(), key=lambda x: x[0])
        for pfit_idx in range(len(pfit_sorted)):
            condition_lab, pfit = pfit_sorted[pfit_idx]
            der_roots = stat.get_radial_profile_roots(pfit["pfit"])
            for rid in range(len(der_roots)):
                if np.isnan(der_roots[rid]):
                    continue
                pid = 0
                panel_trace_y = np.concatenate(
                    [
                        p["y"]
                        for p in fig["data"]
                        if p["yaxis"] == get_axis_label("y", pid)
                    ]
                )
                fig = add_line_trace(
                    fig,
                    der_roots[rid],
                    der_roots[rid],
                    panel_trace_y.min(),
                    panel_trace_y.max(),
                    line_dash="dot" if rid == 1 else "dash",
                    line_color=px.colors.qualitative.Set2[pfit_idx],
                    legendgroup=condition_lab,
                    showlegend=False,
                    xaxis="x",
                    yaxis=get_axis_label("y", pid),
                )
        return fig

    def __secondary_yaxes_props(
        self, pfit_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        yaxes_props: Dict[str, Any] = {}
        for ii in range(1, len(pfit_data)):
            yaxes_props[get_axis_label("yaxis", ii)] = dict(
                domain=[0, 1],
                side="left",
                showgrid=False,
                zeroline=False,
                visible=False,
            )
            if "y" != get_axis_label("y", ii):
                yaxes_props[get_axis_label("yaxis", ii)]["overlaying"] = "y"
        return yaxes_props

    def __make_panel(
        self,
        data: pd.DataFrame,
        pfit_data: Dict[str, List[Dict[str, Any]]],
        stat_type: stat.ProfileStatType,
        dtype: distance.DistanceType,
    ) -> go.Figure:
        channel_lab = data["channel"].tolist()[0]
        selected_pfits: Dict[str, Dict[str, Any]] = {}
        for condition_lab, pfit_list in pfit_data.items():
            for pfit in pfit_list:
                condition = pfit["stat"] == stat_type
                condition = condition and pfit["distance_type"] == dtype.value
                condition = condition and pfit["cname"] == channel_lab
                if condition:
                    selected_pfits[
                        os.path.basename(os.path.dirname(condition_lab))
                    ] = pfit

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        plot_data = self.__make_scatter_trace(
            data,
            selected_pfits,
        )
        for panel in plot_data:
            fig.add_trace(panel)

        fig = self.__add_der_zeros(fig, selected_pfits)

        fig.update_layout(
            template="plotly_dark",
            title=f"""Signal profile (y-axis not comparable across curves)<br>
            <sub>Channel: {channel_lab}; Stat: {stat_type.value}</sub>""".replace(
                f"\n{' '*4*3}", "\n"
            ),
            xaxis=dict(title=dtype.label, anchor="y"),
            yaxis=dict(
                showgrid=True,
                zeroline=False,
                visible=False,
            ),
            **self.__secondary_yaxes_props(pfit_data),
            autosize=False,
            width=1000,
            height=500,
        )

        return fig

    def _plot(
        self, data: DefaultDict[str, Dict[str, pd.DataFrame]], *args, **kwargs
    ) -> DefaultDict[str, Dict[str, go.Figure]]:
        distance_type = distance.DistanceType.LAMINA_NORM
        fig_data: DefaultDict[str, Dict[str, go.Figure]] = defaultdict(lambda: {})
        assert "raw_data" in data
        assert "poly_fit" in data

        condition_data = []
        for dirpath, dirdata in data["raw_data"].items():
            assert isinstance(dirdata, pd.DataFrame)
            assert dirpath in data["poly_fit"]
            condition_lab = os.path.basename(os.path.dirname(dirpath))
            distdata = dirdata.loc[
                distance_type.value == dirdata["distance_type"], :
            ].copy()
            distdata["condition"] = condition_lab
            condition_data.append(distdata)

        plottable_data = pd.concat(condition_data)
        for channel_lab in list(set(plottable_data["channel"])):
            channel_data = plottable_data.loc[
                channel_lab == plottable_data["channel"], :
            ]
            for stat_type in stat.ProfileStatType:
                fig_data[self._stub][
                    f"{channel_lab}-{stat_type.value}"
                ] = self.__make_panel(
                    channel_data,
                    data["poly_fit"],
                    stat_type,
                    distance_type,
                )

        return fig_data

    def make(
        self, output_data: DefaultDict[str, Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        fig_data = self._plot(output_data)
        panels = "\n\t".join(
            [
                report.ReportBase.figure_to_html(
                    fig,
                    classes=[self._stub, f"{self.html_class}-panel", "hidden"],
                    data=dict(condition=os.path.basename(dpath)),
                )
                for dpath, fig in sorted(
                    fig_data[self._stub].items(), key=lambda x: x[0]
                )
            ]
        )
        return (panels, sorted(fig_data[self._stub].keys()))


class ProfileMultiCondition(object):
    html_class: str = "plot-multi-condition"
    _stub: str

    def __init__(self, stub: str):
        super(ProfileMultiCondition, self).__init__()
        self._stub = stub

    def __make_scatter_trace(
        self,
        channel_data: pd.DataFrame,
        pfit: Dict[str, Dict[str, Any]],
    ) -> go.Scatter:
        condition_list: List[str] = sorted(list(set(channel_data["condition"])))
        panel_data = []
        for condition_idx in range(len(condition_list)):
            condition_lab = condition_list[condition_idx]
            condition_data = channel_data.loc[
                condition_lab == channel_data["condition"], :
            ]
            assert condition_lab in pfit
            assert "pfit" in pfit[condition_lab]
            x, y = pfit[condition_lab]["pfit"].linspace(200)
            xx, yy = pfit[condition_lab]["pfit"].deriv().linspace(200)
            xxx, yyy = pfit[condition_lab]["pfit"].deriv().deriv().linspace(200)
            stat_lab = pfit[condition_lab]["stat"].value
            panel_data.extend(
                [
                    go.Scatter(
                        name=f"{condition_lab}_{stat_lab}_raw",
                        xaxis="x",
                        yaxis="y",
                        x=condition_data["x"],
                        y=condition_data[f"{stat_lab}_raw"],
                        mode="markers",
                        legendgroup=condition_lab,
                        marker=dict(
                            size=4,
                            opacity=0.5,
                            color=px.colors.qualitative.Pastel2[condition_idx],
                        ),
                        showlegend=False,
                    ),
                    go.Scatter(
                        name=f"{condition_lab}_{stat_lab}",
                        x=x,
                        y=y,
                        xaxis="x",
                        yaxis="y",
                        mode="lines",
                        legendgroup=condition_lab,
                        line_color=px.colors.qualitative.Dark2[condition_idx],
                    ),
                    go.Scatter(
                        name=f"{condition_lab}_{stat_lab}_der1",
                        x=xx,
                        y=yy,
                        xaxis="x",
                        yaxis="y2",
                        mode="lines",
                        legendgroup=condition_lab,
                        showlegend=False,
                        line_color=px.colors.qualitative.Dark2[condition_idx],
                    ),
                    go.Scatter(
                        name=f"{condition_lab}_{stat_lab}_der2",
                        x=xxx,
                        y=yyy,
                        xaxis="x",
                        yaxis="y3",
                        mode="lines",
                        legendgroup=condition_lab,
                        showlegend=False,
                        line_color=px.colors.qualitative.Dark2[condition_idx],
                    ),
                ]
            )
        return panel_data

    def __add_der_zeros(
        self, fig: go.Figure, pfit_data: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        pfit_sorted = sorted(pfit_data.items(), key=lambda x: x[0])
        for pfit_idx in range(len(pfit_sorted)):
            condition_lab, pfit = pfit_sorted[pfit_idx]
            der_roots = stat.get_radial_profile_roots(pfit["pfit"])
            for rid in range(len(der_roots)):
                if np.isnan(der_roots[rid]):
                    continue
                for pid in range(min(rid + 2, 3)):
                    panel_trace_y = np.concatenate(
                        [
                            p["y"]
                            for p in fig["data"]
                            if p["yaxis"] == get_axis_label("y", pid)
                        ]
                    )
                    fig = add_line_trace(
                        fig,
                        der_roots[rid],
                        der_roots[rid],
                        panel_trace_y.min(),
                        panel_trace_y.max(),
                        line_dash="dot" if rid == 1 else "dash",
                        line_color=px.colors.qualitative.Set2[pfit_idx],
                        legendgroup=condition_lab,
                        showlegend=False,
                        xaxis="x",
                        yaxis=get_axis_label("y", pid),
                    )
        return fig

    def __make_panel(
        self,
        data: pd.DataFrame,
        pfit_data: Dict[str, List[Dict[str, Any]]],
        stat_type: stat.ProfileStatType,
        dtype: distance.DistanceType,
    ) -> go.Figure:
        channel_lab = data["channel"].tolist()[0]
        selected_pfits: Dict[str, Dict[str, Any]] = {}
        for condition_lab, pfit_list in pfit_data.items():
            for pfit in pfit_list:
                condition = pfit["stat"] == stat_type
                condition = condition and pfit["distance_type"] == dtype.value
                condition = condition and pfit["cname"] == channel_lab
                if condition:
                    selected_pfits[
                        os.path.basename(os.path.dirname(condition_lab))
                    ] = pfit

        fig = make_subplots(rows=3, cols=1)
        plot_data = self.__make_scatter_trace(
            data,
            selected_pfits,
        )
        for panel in plot_data:
            fig.add_trace(panel)

        fig = add_derivative_xaxis_to_profiles(fig)
        fig = self.__add_der_zeros(fig, selected_pfits)

        yranges = dict(
            y=get_axis_range(plot_data, "y", "y"),
            y2=get_axis_range(plot_data, "y", "y2"),
            y3=get_axis_range(plot_data, "y", "y3"),
        )
        fig.update_layout(
            template="plotly_dark",
            title=f"""Signal profile<br>
            <sub>Channel: {channel_lab}; Stat: {stat_type.value}</sub>""".replace(
                f"\n{' '*4*3}", "\n"
            ),
            xaxis=dict(title=dtype.label, anchor="y3"),
            yaxis=dict(
                domain=[0.66, 1],
                range=yranges["y"],
                title="Intensity (a.u.)",
            ),
            yaxis2=dict(
                domain=[0.33, 0.63],
                range=yranges["y2"],
                title="1st Derivative Intensity (a.u.)",
            ),
            yaxis3=dict(
                domain=[0, 0.30],
                range=yranges["y3"],
                title="2nd Derivative Intensity (a.u.)",
            ),
            autosize=False,
            width=1000,
            height=1000,
        )

        return fig

    def _plot(
        self, data: DefaultDict[str, Dict[str, pd.DataFrame]], *args, **kwargs
    ) -> DefaultDict[str, Dict[str, go.Figure]]:
        distance_type = distance.DistanceType.LAMINA_NORM
        fig_data: DefaultDict[str, Dict[str, go.Figure]] = defaultdict(lambda: {})
        assert "raw_data" in data
        assert "poly_fit" in data

        condition_data = []
        for dirpath, dirdata in data["raw_data"].items():
            assert isinstance(dirdata, pd.DataFrame)
            assert dirpath in data["poly_fit"]
            condition_lab = os.path.basename(os.path.dirname(dirpath))
            distdata = dirdata.loc[
                distance_type.value == dirdata["distance_type"], :
            ].copy()
            distdata["condition"] = condition_lab
            condition_data.append(distdata)

        plottable_data = pd.concat(condition_data)
        for channel_lab in list(set(plottable_data["channel"])):
            channel_data = plottable_data.loc[
                channel_lab == plottable_data["channel"], :
            ]
            for stat_type in stat.ProfileStatType:
                fig_data[self._stub][
                    f"{channel_lab}-{stat_type.value}"
                ] = self.__make_panel(
                    channel_data,
                    data["poly_fit"],
                    stat_type,
                    distance_type,
                )

        return fig_data

    def make(
        self, output_data: DefaultDict[str, Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        fig_data = self._plot(output_data)
        panels = "\n\t".join(
            [
                report.ReportBase.figure_to_html(
                    fig,
                    classes=[self._stub, f"{self.html_class}-panel", "hidden"],
                    data=dict(condition=os.path.basename(dpath)),
                )
                for dpath, fig in sorted(
                    fig_data[self._stub].items(), key=lambda x: x[0]
                )
            ]
        )
        return (panels, sorted(fig_data[self._stub].keys()))


class ProfileSingleCondition(object):
    html_class: str = "plot-single-condition"
    _stub: str

    def __init__(self, stub: str):
        super(ProfileSingleCondition, self).__init__()
        self._stub = stub

    def __make_scatter_trace(
        self,
        name: str,
        data: pd.DataFrame,
        pfit_data: List[Dict[str, Any]],
    ) -> go.Scatter:
        panel_data = []
        for stat_type in stat.ProfileStatType:
            pfit = [x for x in pfit_data if x["stat"] == stat_type]
            assert 1 == len(pfit), pfit
            assert "pfit" in pfit[0]
            x, y = pfit[0]["pfit"].linspace(200)
            xx, yy = pfit[0]["pfit"].deriv().linspace(200)
            xxx, yyy = pfit[0]["pfit"].deriv().deriv().linspace(200)
            panel_data.extend(
                [
                    go.Scatter(
                        name=f"{name}_{stat_type.value}_raw",
                        xaxis="x",
                        yaxis="y",
                        x=data["x"],
                        y=data[f"{stat_type.value}_raw"],
                        mode="markers",
                        legendgroup=stat_type.value,
                        marker=dict(
                            size=4,
                            opacity=0.5,
                            color=px.colors.qualitative.Pastel2[stat_type.id],
                        ),
                        showlegend=False,
                    ),
                    go.Scatter(
                        name=f"{name}_{stat_type.value}",
                        x=x,
                        y=y,
                        xaxis="x",
                        yaxis="y",
                        mode="lines",
                        legendgroup=stat_type.value,
                        line_color=px.colors.qualitative.Dark2[stat_type.id],
                    ),
                    go.Scatter(
                        name=f"{name}_{stat_type.value}_der1",
                        x=xx,
                        y=yy,
                        xaxis="x",
                        yaxis="y2",
                        mode="lines",
                        legendgroup=stat_type.value,
                        showlegend=False,
                        line_color=px.colors.qualitative.Dark2[stat_type.id],
                    ),
                    go.Scatter(
                        name=f"{name}_{stat_type.value}_der2",
                        x=xxx,
                        y=yyy,
                        xaxis="x",
                        yaxis="y3",
                        mode="lines",
                        legendgroup=stat_type.value,
                        showlegend=False,
                        line_color=px.colors.qualitative.Dark2[stat_type.id],
                    ),
                ]
            )
        return panel_data

    def __add_der_zeros(
        self, fig: go.Figure, pfit_data: List[Dict[str, Any]]
    ) -> go.Figure:
        for pfit in pfit_data:
            der_roots = stat.get_radial_profile_roots(pfit["pfit"])
            for rid in range(len(der_roots)):
                if np.isnan(der_roots[rid]):
                    continue
                for pid in range(min(rid + 2, 3)):
                    panel_trace_y = np.concatenate(
                        [
                            p["y"]
                            for p in fig["data"]
                            if p["yaxis"] == get_axis_label("y", pid)
                        ]
                    )
                    fig = add_line_trace(
                        fig,
                        der_roots[rid],
                        der_roots[rid],
                        panel_trace_y.min(),
                        panel_trace_y.max(),
                        line_dash="dot" if rid == 1 else "dash",
                        line_color=px.colors.qualitative.Set2[pfit["stat"].id],
                        legendgroup=pfit["stat"].value,
                        showlegend=False,
                        xaxis="x",
                        yaxis=get_axis_label("y", pid),
                    )
        return fig

    def __make_panel(
        self,
        data: pd.DataFrame,
        pfit_data: List[Dict[str, Any]],
        condition_lab: str,
        channel_lab: str,
        dtype: distance.DistanceType,
    ) -> go.Figure:
        pfit = [
            x
            for x in pfit_data
            if x["cname"] == channel_lab and x["distance_type"] == dtype.value
        ]

        fig = make_subplots(rows=3, cols=1)
        plot_data = self.__make_scatter_trace(
            channel_lab,
            data.loc[channel_lab == data["channel"]],
            pfit,
        )
        for panel in plot_data:
            fig.add_trace(panel)

        fig = add_derivative_xaxis_to_profiles(fig)
        fig = self.__add_der_zeros(fig, pfit)

        yranges = dict(
            y=get_axis_range(plot_data, "y", "y"),
            y2=get_axis_range(plot_data, "y", "y2"),
            y3=get_axis_range(plot_data, "y", "y3"),
        )
        fig.update_layout(
            template="plotly_dark",
            title=f"""Signal profile<br>
            <sub>Condition: {condition_lab}; Channel: {channel_lab}</sub>""".replace(
                f"\n{' '*4*3}", "\n"
            ),
            xaxis=dict(title=dtype.label, anchor="y3"),
            yaxis=dict(
                domain=[0.66, 1],
                range=yranges["y"],
                title="Intensity (a.u.)",
            ),
            yaxis2=dict(
                domain=[0.33, 0.63],
                range=yranges["y2"],
                title="1st Derivative Intensity (a.u.)",
            ),
            yaxis3=dict(
                domain=[0, 0.30],
                range=yranges["y3"],
                title="2nd Derivative Intensity (a.u.)",
            ),
            autosize=False,
            width=1000,
            height=1000,
        )

        return fig

    def _plot(
        self, data: DefaultDict[str, Dict[str, pd.DataFrame]], *args, **kwargs
    ) -> DefaultDict[str, Dict[str, go.Figure]]:
        fig_data: DefaultDict[str, Dict[str, go.Figure]] = defaultdict(lambda: {})
        assert "raw_data" in data
        assert "poly_fit" in data

        for dirpath, dirdata in data["raw_data"].items():
            assert isinstance(dirdata, pd.DataFrame)
            assert dirpath in data["poly_fit"]
            condition_lab = os.path.basename(os.path.dirname(dirpath))
            distance_type = distance.DistanceType.LAMINA_NORM
            for channel_lab in set(dirdata["channel"]):
                distdata = dirdata.loc[distance_type.value == dirdata["distance_type"]]
                if 0 == distdata.shape[0]:
                    continue
                fig_data[self._stub][
                    f"{channel_lab}-{condition_lab}"
                ] = self.__make_panel(
                    distdata,
                    data["poly_fit"][dirpath],
                    condition_lab,
                    channel_lab,
                    distance_type,
                )

        return fig_data

    def make(
        self, output_data: DefaultDict[str, Dict[str, Any]]
    ) -> Tuple[str, List[str]]:
        fig_data = self._plot(output_data)
        panels = "\n\t".join(
            [
                report.ReportBase.figure_to_html(
                    fig,
                    classes=[self._stub, f"{self.html_class}-panel", "hidden"],
                    data=dict(condition=os.path.basename(dpath)),
                )
                for dpath, fig in sorted(
                    fig_data[self._stub].items(), key=lambda x: x[0]
                )
            ]
        )
        return (panels, sorted(fig_data[self._stub].keys()))
