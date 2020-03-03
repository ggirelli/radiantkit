'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from numpy.polynomial.polynomial import Polynomial  # type: ignore
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from radiantkit import distance, path, stat
from scipy.stats import gaussian_kde  # type: ignore
from typing import Any, Dict, List, Optional


def export(opath: str, exp_format: str = 'pdf') -> None:
    assert exp_format in ['pdf', 'png', 'jpg']
    opath = path.add_extension(opath, f".{exp_format}")
    if exp_format == 'pdf':
        pp = PdfPages(opath)
        plt.savefig(pp, format=exp_format)
        pp.close()
    else:
        plt.savefig(opath, format=exp_format)


def get_palette(N: int) -> List[str]:
    return ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]


def plot_nuclear_selection(raw_data, fit) -> go.Figure:
    ref = raw_data.loc[0, 'ref']
    size_range = fit['data']['size']['range']
    size_fit = fit['data']['size']['fit']
    isum_range = fit['data']['isum']['range']
    isum_fit = fit['data']['isum']['fit']
    npoints: int = 1000

    assert all([x in raw_data.columns
                for x in ["size", f"isum_{ref}", "pass", "label", "image"]])
    x_data = raw_data['size'].values
    y_data = raw_data[f"isum_{ref}"].values

    xdf = gaussian_kde(x_data)
    ydf = gaussian_kde(y_data)

    xx_linspace = np.linspace(x_data.min(), x_data.max(), npoints)
    yy_linspace = np.linspace(y_data.min(), y_data.max(), npoints)

    passed = raw_data['pass']
    not_passed = np.logical_not(passed)

    scatter_selected = go.Scatter(
        name="selected nuclei", x=x_data[passed], y=y_data[passed],
        mode='markers', marker=dict(size=4, opacity=.5, color="#1f78b4"),
        xaxis="x", yaxis="y",
        customdata=np.dstack((raw_data[passed]['label'],
                              raw_data[passed]['image']))[0],
        hovertemplate='Size=%{x}<br>Intensity sum=%{y}<br>'
                      + 'Label=%{customdata[0]}<br>Image="%{customdata[1]}"')
    scatter_filtered = go.Scatter(
        name="discarded nuclei", x=x_data[not_passed], y=y_data[not_passed],
        mode='markers', marker=dict(size=4, opacity=.5, color="#e31a1c"),
        xaxis="x", yaxis="y", customdata=np.dstack((
            raw_data[not_passed]['label'], raw_data[not_passed]['image']))[0],
        hovertemplate='Size=%{x}<br>Intensity sum=%{y}<br>'
                      + 'Label=%{customdata[0]}<br>Image="%{customdata[1]}"')
    contour_y = go.Scatter(
        name="intensity sum", x=ydf(yy_linspace), y=yy_linspace,
        xaxis="x2", yaxis="y", line=dict(color="#33a02c"))
    contour_x = go.Scatter(
        name="size", x=xx_linspace, y=xdf(xx_linspace),
        xaxis="x", yaxis="y3", line=dict(color="#ff7f00"))

    data = [scatter_selected, scatter_filtered, contour_x, contour_y]

    if size_fit is not None:
        data.append(go.Scatter(
            name="size_gauss1",
            x=xx_linspace, y=stat.gaussian(xx_linspace, *size_fit[0][:3]),
            xaxis="x", yaxis="y3", line=dict(color="#323232", dash="dot")))
        if stat.FitType.SOG == size_fit[1]:
            data.append(go.Scatter(
                name="size_gauss2",
                x=xx_linspace, y=stat.gaussian(xx_linspace, *size_fit[0][3:]),
                xaxis="x", yaxis="y3", line=dict(color="#999999", dash="dot")))
    if isum_fit is not None:
        data.append(go.Scatter(
            name="isum_gauss1",
            y=yy_linspace, x=stat.gaussian(yy_linspace, *isum_fit[0][:3]),
            xaxis="x2", yaxis="y", line=dict(color="#323232", dash="dot")))
        if stat.FitType.SOG == isum_fit[1]:
            data.append(go.Scatter(
                name="isum_gauss2",
                y=yy_linspace, x=stat.gaussian(yy_linspace, *isum_fit[0][3:]),
                xaxis="x2", yaxis="y", line=dict(color="#999999", dash="dot")))

    layout = go.Layout(
        xaxis=dict(domain=[.19, 1], title="size"),
        yaxis=dict(domain=[0, .82], anchor="x2", title="intensity sum",),
        xaxis2=dict(domain=[0, .18], autorange="reversed", title="density"),
        yaxis2=dict(domain=[0, .82]),
        xaxis3=dict(domain=[.19, 1]),
        yaxis3=dict(domain=[.83, 1], title="density"),
        autosize=False, width=1000, height=1000
    )

    fig = go.Figure(data=data, layout=layout)

    fig.add_shape(go.layout.Shape(
        type="line", line=dict(dash="dash", color="#969696"),
        x0=size_range[0], y0=0, x1=size_range[0], y1=y_data.max()
    ))
    fig.add_shape(go.layout.Shape(
        type="line", line=dict(dash="dash", color="#969696"),
        x0=size_range[1], y0=0, x1=size_range[1], y1=y_data.max()
    ))
    fig.add_shape(go.layout.Shape(
        type="line", line=dict(dash="dash", color="#969696"), yref="y3",
        x0=size_range[0], y0=0, x1=size_range[0], y1=xdf(x_data).max()
    ))
    fig.add_shape(go.layout.Shape(
        type="line", line=dict(dash="dash", color="#969696"), yref="y3",
        x0=size_range[1], y0=0, x1=size_range[1], y1=xdf(x_data).max()
    ))

    fig.add_shape(go.layout.Shape(
        type="line", line=dict(dash="dash", color="#969696"),
        y0=isum_range[0], x0=0, y1=isum_range[0], x1=x_data.max()
    ))
    fig.add_shape(go.layout.Shape(
        type="line", line=dict(dash="dash", color="#969696"),
        y0=isum_range[1], x0=0, y1=isum_range[1], x1=x_data.max()
    ))
    fig.add_shape(go.layout.Shape(
        type="line", line=dict(dash="dash", color="#969696"), xref="x2",
        y0=isum_range[0], x0=0, y1=isum_range[0], x1=ydf(y_data).max()
    ))
    fig.add_shape(go.layout.Shape(
        type="line", line=dict(dash="dash", color="#969696"), xref="x2",
        y0=isum_range[1], x0=0, y1=isum_range[1], x1=ydf(y_data).max()
    ))

    fig.update_layout(template="plotly_white")

    return fig


class NuclearFeaturePlotter(object):
    _obj_data: pd.DataFrame
    _spx_data: Optional[pd.DataFrame] = None
    _n_input_cols: int = 3
    _n_grid_cols: int = 3
    _root_list: List[str]
    _pal: List[str]
    _channel_list: List[str]
    _n_features: int
    _n_rows: int
    _fig: go.Figure
    _layout: Dict[str, Any]

    def __init__(self, obj_data: pd.DataFrame,
                 spx_data: Optional[pd.DataFrame]):
        super(NuclearFeaturePlotter, self).__init__()
        self._obj_data = obj_data
        self._spx_data = spx_data
        assert 'root' in self._obj_data.columns
        self._root_list = np.unique(obj_data['root'].values).tolist()
        self._pal = get_palette(len(self._root_list))
        self._channel_list = self.__get_channel_list()
        self._n_features = self._obj_data.shape[1] - self.n_input_cols
        self._n_rows = (self._n_features + len(
            self._channel_list)) // self.n_grid_cols + 1

    @property
    def nfeat(self) -> int:
        return self._n_features

    @property
    def n_input_cols(self) -> int:
        return self._n_input_cols

    @n_input_cols.setter
    def n_input_cols(self, n: int):
        if n >= 1:
            self._n_input_cols = n

    @property
    def n_grid_cols(self) -> int:
        return self._n_grid_cols

    @n_grid_cols.setter
    def n_grid_cols(self, n: int):
        if n >= 1:
            self._n_grid_cols = n

    def __get_channel_list(self) -> List[str]:
        channel_list = []
        if self._spx_data is not None:
            assert 'channel' in self._spx_data.columns
            channel_list = np.unique(self._spx_data['channel'].values)
        return channel_list

    def __add_box(self, ci, root, colname) -> None:
        self._fig.add_trace(go.Box(
            name=root, notched=True,
            y=self._obj_data[self._obj_data['root'] == root][colname].values,
            marker_color=self._pal[np.argmax(self._root_list == root)]),
            row=ci//self.n_grid_cols+1,
            col=ci % self.n_grid_cols+1)

    def __add_precomputed_box(self, spx_channel, ci, root) -> None:
        spx_root = spx_channel[spx_channel['root'] == root]
        self._fig.add_trace(
            go.Box(
                name=root, notched=False,
                y=[[spx_root['vmin'].values[0],
                    spx_root['vmax'].values[0]]],
                marker_color=self._pal[np.argmax(self._root_list == root)]),
            row=ci//self.n_grid_cols+1, col=ci % self.n_grid_cols+1)
        self._fig.update_traces(
            q1=spx_root['q1'].values, q3=spx_root['q3'].values,
            median=spx_root['median'].values,
            lowerfence=spx_root['whisk_low'].values,
            upperfence=spx_root['whisk_high'].values,
            row=ci//self.n_grid_cols+1, col=ci % self.n_grid_cols+1)

    def __init_canvas(self) -> None:
        self._fig = make_subplots(
            rows=self._n_rows, cols=self.n_grid_cols,
            horizontal_spacing=.2)
        self._layout = {}

    def __update_layout(self) -> None:
        self._fig.update_xaxes(tickangle=45)
        self._fig.update_xaxes(
            ticks="outside", tickwidth=1, ticklen=5, automargin=True)
        self._fig.update_yaxes(
            ticks="outside", tickwidth=1, ticklen=5, automargin=True)
        self._fig.update_layout(**self._layout)
        self._fig.update_layout(
            height=400*self._n_rows, width=1000, autosize=False)
        self._fig.update_layout(showlegend=False)

    def __draw_boxes(self) -> None:
        for ci in range(self.nfeat):
            colname = self._obj_data.columns[ci+self.n_input_cols]
            # collab = labels[colname] if colname in labels else colname
            for root in np.unique(self._obj_data['root'].values.tolist()):
                self.__add_box(ci, root, colname)
            # layout["yaxis" if ci == 0 else f"yaxis{ci+1}"
            #    ] = dict(title=collab)

    def __draw_precomputed_boxes(self) -> None:
        if self._spx_data is not None:
            for ci in range(self.nfeat, self.nfeat+len(self._channel_list)):
                channel = self._channel_list[ci-self.nfeat]
                spx_channel = self._spx_data[
                    self._spx_data['channel'] == channel]
                for root in spx_channel['root'].values.tolist():
                    self.__add_precomputed_box(spx_channel, ci, root)
                self._layout[f"yaxis{ci+1}"] = dict(
                    title=f'"{channel}" single pixel intensity (a.u.)')

    def plot(self) -> go.Figure:
        self.__init_canvas()
        self.__draw_boxes()
        self.__draw_precomputed_boxes()
        self.__update_layout()
        return self._fig


def plot_nuclear_features(
        obj_data, spx_data,
        n_input_cols: int = 3, n_grid_cols: int = 3
        ) -> go.Figure:
    nfp = NuclearFeaturePlotter(obj_data, spx_data)
    nfp.n_input_cols = n_input_cols
    nfp.n_grid_cols = n_grid_cols
    return nfp.plot()


def plot_profiles(**data):
    pass


# def add_profile_trace(
#         label: str, profile: Polynomial, raw_data: pd.DataFrame,
#         npoints: int = 1000, color: Optional[str] = None) -> List[go.Scatter]:
#     raw_label = f"{label}_raw"
#     assert raw_label in raw_data.columns

#     x, y = profile.linspace(npoints)
#     xx, yy = profile.deriv().linspace(npoints)
#     xxx, yyy = profile.deriv().deriv().linspace(npoints)

#     return [go.Scatter(
#                 name=f"raw_{label}", xaxis="x", yaxis="y",
#                 x=raw_data['x'].values, y=raw_data[raw_label].values,
#                 mode='markers', legendgroup=label,
#                 marker=dict(size=4, opacity=.5, color="#989898")),
#             go.Scatter(
#                 name=label, x=x, y=y, xaxis="x", yaxis="y", mode='lines',
#                 legendgroup=label, line=dict(color=color)),
#             go.Scatter(
#                 name=f"der1_{label}", x=xx, y=yy, xaxis="x", yaxis="y2",
#                 mode='lines', line=dict(color=color),
#                 legendgroup=label, showlegend=False),
#             go.Scatter(
#                 name=f"der2_{label}", x=xxx, y=yyy, xaxis="x", yaxis="y3",
#                 mode='lines', line=dict(color=color),
#                 legendgroup=label, showlegend=False)]


# def add_profile_roots(
#         label: str, profile: Polynomial, roots: Tuple[float, float],
#         yranges: Dict[str, Tuple[float, float]], npoints: int = 1000
#         ) -> List[go.Scatter]:
#     data = []

#     if roots[0] is not None:
#         data.extend([
#             go.Scatter(
#                 name=f"root_der1_{label}",
#                 x=[roots[0], roots[0]], y=yranges['y'],
#                 xaxis="x", yaxis="y", mode="lines",
#                 line=dict(color="#969696", dash="dash"), legendgroup=label),
#             go.Scatter(
#                 name=f"root_der1_{label}",
#                 x=[roots[0], roots[0]], y=yranges['y2'],
#                 xaxis="x", yaxis="y2", mode="lines",
#                 line=dict(color="#969696", dash="dash"),
#                 legendgroup=label, showlegend=False)
#         ])

#     if roots[1] is not None:
#         data.extend([
#             go.Scatter(
#                 name=f"root_der2_{label}",
#                 x=[roots[1], roots[1]], y=yranges['y'],
#                 xaxis="x", yaxis="y", mode="lines",
#                 line=dict(color="#969696", dash="dot"),
#                 legendgroup=label),
#             go.Scatter(
#                 name=f"root_der2_{label}",
#                 x=[roots[1], roots[1]], y=yranges['y2'],
#                 xaxis="x", yaxis="y2", mode="lines",
#                 line=dict(color="#969696", dash="dot"),
#                 legendgroup=label, showlegend=False),
#             go.Scatter(
#                 name=f"root_der2_{label}",
#                 x=[roots[1], roots[1]], y=yranges['y3'],
#                 xaxis="x", yaxis="y3", mode="lines",
#                 line=dict(color="#969696", dash="dot"),
#                 legendgroup=label, showlegend=False)
#         ])

#     return data


# def plot_profile(
#         dtype: str, profiles: stat.PolyFitResult, raw_data: pd.DataFrame,
#         roots: Dict[str, Tuple[float, float]]) -> go.Figure:
#     pal = get_palette(len(profiles))
#     stat_name_list = list(profiles.keys())
#     npoints = 1000
#     data = []

#     for pi in range(len(stat_name_list)):
#         data.extend(add_profile_trace(
#             stat_name_list[pi], profiles[stat_name_list[pi]],
#             raw_data, npoints, pal[pi]))

#     yranges = dict(
#         y=(np.min([trace['y'].min()
#                    for trace in data if 'y' == trace['yaxis']]),
#             np.max([trace['y'].max()
#                     for trace in data if 'y' == trace['yaxis']])),
#         y2=(np.min([trace['y'].min() for trace in data
#                     if 'y2' == trace['yaxis']]),
#             np.max([trace['y'].max() for trace in data
#                     if 'y2' == trace['yaxis']])),
#         y3=(np.min([trace['y'].min() for trace in data
#                     if 'y3' == trace['yaxis']]),
#             np.max([trace['y'].max() for trace in data
#                     if 'y3' == trace['yaxis']])))

#     for pi in range(len(stat_name_list)):
#         stat_name = stat_name_list[pi]
#         data.extend(add_profile_roots(
#             stat_name, profiles[stat_name],
#             roots[stat_name], yranges))

#     layout = go.Layout(
#         xaxis=dict(title=distance.__distance_labels__[dtype], anchor="y3"),
#         yaxis=dict(domain=[.66, 1], range=yranges['y'],
#                    title="Intensity (a.u.)"),
#         yaxis2=dict(domain=[.33, .63], range=yranges['y2'],
#                     title="1st Derivative Intensity (a.u.)"),
#         yaxis3=dict(domain=[0, .30], range=yranges['y3'],
#                     title="2nd Derivative Intensity (a.u.)"),
#         autosize=False, width=1000, height=1000
#     )

#     fig = go.Figure(
#         data=data,
#         layout=layout)

#     fig.add_shape(go.layout.Shape(
#         type="line", line=dict(dash="dash", color="#969696"),
#         y0=0, x0=0, y1=0, x1=raw_data['x'].values.max(), yref="y2"))
#     fig.add_shape(go.layout.Shape(
#         type="line", line=dict(dash="dash", color="#969696"),
#         y0=0, x0=0, y1=0, x1=raw_data['x'].values.max(), yref="y3"))

#     fig.update_layout(template="plotly_white")
#     return fig

# END =========================================================================

###############################################################################
