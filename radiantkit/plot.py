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
from typing import Any, Dict, List, Optional, Set, Tuple


BaseName = str
ChannelName = str
ChannelProfile = Dict[ChannelName, stat.PolyFitResult]

StatProfilePlots = Dict[stat.ProfileStatType, go.Figure]
ChannelProfilePlots = Dict[ChannelName, StatProfilePlots]
DistanceProfilePlots = Dict[distance.DistanceType, ChannelProfilePlots]
BaseProfilePlots = Dict[BaseName, DistanceProfilePlots]


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


class NuclearSelectionPlotter(object):
    _raw_data: pd.DataFrame
    __col_req: List[str] = ["size", "ref", "pass", "label", "image"]
    _fit_data: Dict[str, Dict[str, Any]]
    _ref: str
    _npoints: int = 1000

    def __init__(self, raw_data: pd.DataFrame,
                 fit_data: Dict[str, Dict[str, Any]]):
        super(NuclearSelectionPlotter, self).__init__()
        assert all([x in raw_data.columns for x in self.__col_req])
        self._raw_data = raw_data
        self._fit_data = fit_data
        self._ref = self._raw_data.loc[0, 'ref']
        assert f"isum_{self._ref}" in self._raw_data.columns

    @property
    def size_range(self) -> np.ndarray:
        return self._fit_data['data']['size']['range']

    @property
    def isum_range(self) -> np.ndarray:
        return self._fit_data['data']['isum']['range']

    @property
    def size_fit(self) -> Polynomial:
        return self._fit_data['data']['size']['fit']

    @property
    def isum_fit(self) -> Polynomial:
        return self._fit_data['data']['isum']['fit']

    @property
    def npoints(self) -> int:
        return self._npoints

    @npoints.setter
    def npoints(self, n: int) -> None:
        assert n > 0
        self._npoints = n

    def __prepare_data(self) -> None:
        self._x = self._raw_data['size'].values
        self._y = self._raw_data[f"isum_{self._ref}"].values

        self._xdf = gaussian_kde(self._x)
        self._ydf = gaussian_kde(self._y)

        self._x_linsp = np.linspace(self._x.min(), self._x.max(), self.npoints)
        self._y_linsp = np.linspace(self._y.min(), self._y.max(), self.npoints)

        self._passed = self._raw_data['pass']
        self._not_passed = np.logical_not(self._passed)

    def __mk_scatters(self) -> Tuple[go.Scatter, go.Scatter]:
        scatter_selected = go.Scatter(
            name="selected nuclei",
            x=self._x[self._passed], y=self._y[self._passed],
            mode='markers', marker=dict(size=4, opacity=.5, color="#1f78b4"),
            xaxis="x", yaxis="y",
            customdata=np.dstack((self._raw_data[self._passed]['label'],
                                  self._raw_data[self._passed]['image']))[0],
            hovertemplate='Size=%{x}<br>Intensity sum=%{y}<br>'
                          + 'Label=%{customdata[0]}<br>'
                          + 'Image="%{customdata[1]}"')
        scatter_filtered = go.Scatter(
            name="discarded nuclei",
            x=self._x[self._not_passed], y=self._y[self._not_passed],
            mode='markers', marker=dict(size=4, opacity=.5, color="#e31a1c"),
            xaxis="x", yaxis="y",
            customdata=np.dstack((
                self._raw_data[self._not_passed]['label'],
                self._raw_data[self._not_passed]['image']))[0],
            hovertemplate='Size=%{x}<br>Intensity sum=%{y}<br>'
                          + 'Label=%{customdata[0]}<br>'
                          + 'Image="%{customdata[1]}"')
        return (scatter_selected, scatter_filtered)

    def __mk_density_contours(self) -> Tuple[go.Scatter, go.Scatter]:
        contour_y = go.Scatter(
            name="intensity sum", x=self._ydf(self._y_linsp), y=self._y_linsp,
            xaxis="x2", yaxis="y", line=dict(color="#33a02c"))
        contour_x = go.Scatter(
            name="size", x=self._x_linsp, y=self._xdf(self._x_linsp),
            xaxis="x", yaxis="y3", line=dict(color="#ff7f00"))
        return (contour_x, contour_y)

    def __mk_fit_contours(self) -> List[go.Scatter]:
        data = []
        if self.size_fit is not None:
            data.append(go.Scatter(
                name="size_gauss1",
                x=self._x_linsp,
                y=stat.gaussian(self._x_linsp, *self.size_fit[0][:3]),
                xaxis="x", yaxis="y3",
                line=dict(color="#323232", dash="dot")))
            if stat.FitType.SOG == self.size_fit[1]:
                data.append(go.Scatter(
                    name="size_gauss2",
                    x=self._x_linsp,
                    y=stat.gaussian(self._x_linsp, *self.size_fit[0][3:]),
                    xaxis="x", yaxis="y3",
                    line=dict(color="#999999", dash="dot")))
        if self.isum_fit is not None:
            data.append(go.Scatter(
                name="isum_gauss1",
                y=self._y_linsp,
                x=stat.gaussian(self._y_linsp, *self.isum_fit[0][:3]),
                xaxis="x2", yaxis="y",
                line=dict(color="#323232", dash="dot")))
            if stat.FitType.SOG == self.isum_fit[1]:
                data.append(go.Scatter(
                    name="isum_gauss2",
                    y=self._y_linsp,
                    x=stat.gaussian(self._y_linsp, *self.isum_fit[0][3:]),
                    xaxis="x2", yaxis="y",
                    line=dict(color="#999999", dash="dot")))
        return data

    def __mk_border_lines(self, fig: go.Figure) -> go.Figure:
        fig.add_shape(go.layout.Shape(
            type="line", line=dict(dash="dash", color="#969696"),
            x0=self.size_range[0], y0=0, x1=self.size_range[0],
            y1=self._y.max()
        ))
        fig.add_shape(go.layout.Shape(
            type="line", line=dict(dash="dash", color="#969696"),
            x0=self.size_range[1], y0=0, x1=self.size_range[1],
            y1=self._y.max()
        ))
        fig.add_shape(go.layout.Shape(
            type="line", line=dict(dash="dash", color="#969696"), yref="y3",
            x0=self.size_range[0], y0=0, x1=self.size_range[0],
            y1=self._xdf(self._x).max()
        ))
        fig.add_shape(go.layout.Shape(
            type="line", line=dict(dash="dash", color="#969696"), yref="y3",
            x0=self.size_range[1], y0=0, x1=self.size_range[1],
            y1=self._xdf(self._x).max()
        ))
        fig.add_shape(go.layout.Shape(
            type="line", line=dict(dash="dash", color="#969696"),
            y0=self.isum_range[0], x0=0, y1=self.isum_range[0],
            x1=self._x.max()
        ))
        fig.add_shape(go.layout.Shape(
            type="line", line=dict(dash="dash", color="#969696"),
            y0=self.isum_range[1], x0=0, y1=self.isum_range[1],
            x1=self._x.max()
        ))
        fig.add_shape(go.layout.Shape(
            type="line", line=dict(dash="dash", color="#969696"), xref="x2",
            y0=self.isum_range[0], x0=0, y1=self.isum_range[0],
            x1=self._ydf(self._y).max()
        ))
        fig.add_shape(go.layout.Shape(
            type="line", line=dict(dash="dash", color="#969696"), xref="x2",
            y0=self.isum_range[1], x0=0, y1=self.isum_range[1],
            x1=self._ydf(self._y).max()
        ))
        return fig

    def _layout(self) -> go.Layout:
        return go.Layout(
            xaxis=dict(domain=[.19, 1], title="size"),
            yaxis=dict(domain=[0, .82], anchor="x2", title="intensity sum",),
            xaxis2=dict(domain=[0, .18],
                        autorange="reversed", title="density"),
            yaxis2=dict(domain=[0, .82]),
            xaxis3=dict(domain=[.19, 1]),
            yaxis3=dict(domain=[.83, 1], title="density"),
            autosize=False, width=1000, height=1000
        )

    def plot(self) -> go.Figure:
        self.__prepare_data()
        data = [*self.__mk_scatters(),
                *self.__mk_density_contours(),
                *self.__mk_fit_contours()]
        fig = go.Figure(data=data, layout=self._layout())
        fig = self.__mk_border_lines(fig)
        fig.update_layout(template="plotly_white")
        return fig


def plot_nuclear_selection(
        raw_data: pd.DataFrame, fit: Dict[str, Dict[str, Any]],
        npoints: int = 1000) -> go.Figure:
    nsp = NuclearSelectionPlotter(raw_data, fit)
    nsp.npoints = npoints
    return nsp.plot()


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
    def n_input_cols(self, n: int) -> None:
        if n >= 1:
            self._n_input_cols = n

    @property
    def n_grid_cols(self) -> int:
        return self._n_grid_cols

    @n_grid_cols.setter
    def n_grid_cols(self, n: int) -> None:
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
        obj_features: pd.DataFrame, spx_features: Optional[pd.DataFrame],
        n_input_cols: int = 3, n_grid_cols: int = 3
        ) -> go.Figure:
    nfp = NuclearFeaturePlotter(obj_features, spx_features)
    nfp.n_input_cols = n_input_cols
    nfp.n_grid_cols = n_grid_cols
    return nfp.plot()


class RadialProfilePlotter(object):
    _raw_data: pd.DataFrame
    __col_req: Tuple = ("x", "q1_raw", "median_raw", "mean_raw", "q3_raw",
                        "channel", "distance_type", "root", "base")
    _fit_data: Dict[BaseName, Dict[distance.DistanceType, ChannelProfile]]
    __field_req: Tuple = ("cname", "distance_type", "stat", "pfit")
    _root: Optional[str] = None
    _npoints: int = 1000
    _channel_set: Set[str]
    _stat_set: Set[stat.ProfileStatType]
    _dist_set: Set[distance.DistanceType]

    def __init__(self, raw_data: pd.DataFrame, fit_data: Dict):
        super(RadialProfilePlotter, self).__init__()
        self._raw_data = pd.DataFrame()
        self._fit_data = {}
        self._channel_set = set()
        self._stat_set = set()
        self._dist_set = set()
        self.add_new_base(raw_data, fit_data)

    @property
    def root(self):
        return self._root

    @property
    def npoints(self) -> int:
        return self._npoints

    @npoints.setter
    def npoints(self, n: int) -> None:
        assert n > 0
        self._npoints = n

    @property
    def bases(self):
        return list(self._fit_data.keys())

    @property
    def dtypes(self):
        return list(self._dist_set)

    @property
    def channels(self):
        return list(self._channel_set)

    @property
    def stats(self):
        return list(self._stat_set)

    def __add_profile_to_fit_data(self, base, profile: Dict) -> None:
        if base not in self._fit_data:
            self._fit_data[base] = {}

        dtype = distance.DistanceType(profile['distance_type'])
        if dtype not in self._fit_data[base]:
            self._fit_data[base][dtype] = {}

        cname = profile['cname']
        if cname not in self._fit_data[base][dtype]:
            self._fit_data[base][dtype][cname] = {}
        else:
            return

        self._dist_set.add(dtype)
        self._channel_set.add(cname)
        stype = stat.ProfileStatType(profile['stat'])
        self._stat_set.add(stype)
        self._fit_data[base][dtype][cname][stype] = profile['pfit']

    def add_new_base(self, raw_data: pd.DataFrame, fit_data: Dict) -> None:
        assert all([x in raw_data.columns for x in self.__col_req])
        input_root_set = np.unique(raw_data['root'])
        assert 1 == len(input_root_set), input_root_set
        assert raw_data.loc[0, 'base'] not in self.bases
        self._raw_data = pd.concat([self._raw_data, raw_data])

        if self._root is None:
            self._root = raw_data.loc[0, 'root']
        else:
            assert self._root == raw_data.loc[0, 'root']

        for profile in fit_data['data']:
            assert all([x in profile.keys() for x in self.__field_req])
        assert fit_data['root'] == self.root
        for profile in fit_data['data']:
            self.__add_profile_to_fit_data(fit_data['base'], profile)

    def __add_profile_traces(
            self, label: str, pfit: Polynomial, raw_data: pd.DataFrame,
            color: Optional[str] = None) -> List[go.Scatter]:
        raw_label = f"{label}_raw"
        assert raw_label in raw_data.columns

        x, y = pfit.linspace(self.npoints)
        xx, yy = pfit.deriv().linspace(self.npoints)
        xxx, yyy = pfit.deriv().deriv().linspace(self.npoints)

        return [go.Scatter(
                    name=f"raw_{label}", xaxis="x", yaxis="y",
                    x=raw_data['x'].values, y=raw_data[raw_label].values,
                    mode='markers', legendgroup=label,
                    marker=dict(size=4, opacity=.5, color="#989898")),
                go.Scatter(
                    name=label, x=x, y=y, xaxis="x", yaxis="y", mode='lines',
                    legendgroup=label, line=dict(color=color)),
                go.Scatter(
                    name=f"der1_{label}", x=xx, y=yy, xaxis="x", yaxis="y2",
                    mode='lines', line=dict(color=color),
                    legendgroup=label, showlegend=False),
                go.Scatter(
                    name=f"der2_{label}", x=xxx, y=yyy, xaxis="x", yaxis="y3",
                    mode='lines', line=dict(color=color),
                    legendgroup=label, showlegend=False)]

    def __add_profile_roots(
            self, label: str, pfit: Polynomial,
            yranges: Dict[str, Tuple[float, float]]
            ) -> List[go.Scatter]:
        data = []
        roots = stat.get_radial_profile_roots(pfit, self.npoints)

        if roots[0] is not None:
            data.extend([
                go.Scatter(
                    name=f"root_der1_{label}",
                    x=[roots[0], roots[0]], y=yranges['y'],
                    xaxis="x", yaxis="y", mode="lines",
                    line=dict(color="#969696", dash="dash"),
                    legendgroup=label),
                go.Scatter(
                    name=f"root_der1_{label}",
                    x=[roots[0], roots[0]], y=yranges['y2'],
                    xaxis="x", yaxis="y2", mode="lines",
                    line=dict(color="#969696", dash="dash"),
                    legendgroup=label, showlegend=False)
            ])

        if roots[1] is not None:
            data.extend([
                go.Scatter(
                    name=f"root_der2_{label}",
                    x=[roots[1], roots[1]], y=yranges['y'],
                    xaxis="x", yaxis="y", mode="lines",
                    line=dict(color="#969696", dash="dot"),
                    legendgroup=label),
                go.Scatter(
                    name=f"root_der2_{label}",
                    x=[roots[1], roots[1]], y=yranges['y2'],
                    xaxis="x", yaxis="y2", mode="lines",
                    line=dict(color="#969696", dash="dot"),
                    legendgroup=label, showlegend=False),
                go.Scatter(
                    name=f"root_der2_{label}",
                    x=[roots[1], roots[1]], y=yranges['y3'],
                    xaxis="x", yaxis="y3", mode="lines",
                    line=dict(color="#969696", dash="dot"),
                    legendgroup=label, showlegend=False)
            ])

        return data

    def plot_channel_profile(
            self, base: str, dtype: distance.DistanceType, cname: str
            ) -> go.Figure:
        assert base in self.bases
        assert dtype in self.dtypes
        assert cname in self.channels

        profiles = self._fit_data[base][dtype][cname]
        pal = get_palette(len(profiles))
        stype_list = list(profiles.keys())
        data: List[go.Scatter] = []

        query_string = f'base=="{base}" and distance_type=="{dtype.value}"'
        query_string += f' and channel=="{cname}"'
        raw_data = self._raw_data.query(query_string)

        for pi in range(len(stype_list)):
            data.extend(self.__add_profile_traces(
                stype_list[pi].value, profiles[stype_list[pi]],
                raw_data, pal[pi]))

        yranges = dict(
            y=(np.min([trace['y'].min()
                       for trace in data if 'y' == trace['yaxis']]),
                np.max([trace['y'].max()
                        for trace in data if 'y' == trace['yaxis']])),
            y2=(np.min([trace['y'].min() for trace in data
                        if 'y2' == trace['yaxis']]),
                np.max([trace['y'].max() for trace in data
                        if 'y2' == trace['yaxis']])),
            y3=(np.min([trace['y'].min() for trace in data
                        if 'y3' == trace['yaxis']]),
                np.max([trace['y'].max() for trace in data
                        if 'y3' == trace['yaxis']])))

        for pi in range(len(stype_list)):
            data.extend(self.__add_profile_roots(
                stype_list[pi].value, profiles[stype_list[pi]], yranges))

        layout = go.Layout(
            xaxis=dict(title=dtype.label, anchor="y3"),
            yaxis=dict(domain=[.66, 1], range=yranges['y'],
                       title="Intensity (a.u.)"),
            yaxis2=dict(domain=[.33, .63], range=yranges['y2'],
                        title="1st Derivative Intensity (a.u.)"),
            yaxis3=dict(domain=[0, .30], range=yranges['y3'],
                        title="2nd Derivative Intensity (a.u.)"),
            autosize=False, width=1000, height=1000
        )

        fig = go.Figure(data=data, layout=layout)
        fig.add_shape(go.layout.Shape(
            type="line", line=dict(dash="dash", color="#969696"),
            y0=0, x0=0, y1=0, x1=raw_data['x'].values.max(), yref="y2"))
        fig.add_shape(go.layout.Shape(
            type="line", line=dict(dash="dash", color="#969696"),
            y0=0, x0=0, y1=0, x1=raw_data['x'].values.max(), yref="y3"))
        fig.update_layout(template="plotly_white")
        return fig

    def plot_channel_profile_list(self) -> BaseProfilePlots:
        plots: BaseProfilePlots = {}
        for base in self.bases:
            plots[base] = {}
            for dtype in self._fit_data[base].keys():
                plots[base][dtype] = {}
                for cname in self._fit_data[base][dtype].keys():
                    plots[base][dtype][cname] = self.plot_channel_profile(
                        base, dtype, cname)
        return plots


def plot_profiles(poly_fit: Dict, raw_data: pd.DataFrame) -> BaseProfilePlots:
    rpp = RadialProfilePlotter(raw_data, poly_fit)
    return rpp.plot_channel_profile_list()
