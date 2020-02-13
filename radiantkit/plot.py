'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from matplotlib.backends.backend_pdf import PdfPages  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
from radiantkit import path as pt, stat
from scipy.stats import gaussian_kde  # type: ignore
from typing import Dict, Optional


def export(path: str, exp_format: str = 'pdf') -> None:
    assert exp_format in ['pdf', 'png', 'jpg']
    path = pt.add_extension(path, '.' + exp_format)
    if exp_format == 'pdf':
        pp = PdfPages(path)
        plt.savefig(pp, format=exp_format)
        pp.close()
    else:
        plt.savefig(path, format=exp_format)


def plot_nuclear_selection(
        data: pd.DataFrame, ref: str,
        size_range: stat.Interval, isum_range: stat.Interval,
        size_fit: Optional[stat.FitResult] = None,
        isum_fit: Optional[stat.FitResult] = None,
        npoints: int = 1000) -> go.Figure:
    assert all([x in data.columns
                for x in ["size", f"isum_{ref}", "pass", "label", "image"]])
    x_data = data['size'].values
    y_data = data[f"isum_{ref}"].values

    xdf = gaussian_kde(x_data)
    ydf = gaussian_kde(y_data)

    xx_linspace = np.linspace(x_data.min(), x_data.max(), npoints)
    yy_linspace = np.linspace(y_data.min(), y_data.max(), npoints)

    passed = data['pass']
    not_passed = np.logical_not(passed)

    scatter_selected = go.Scatter(
        name="selected nuclei", x=x_data[passed], y=y_data[passed],
        mode='markers', marker=dict(size=4, opacity=.5, color="#1f78b4"),
        xaxis="x", yaxis="y",
        customdata=np.dstack((data[passed]['label'],
                              data[passed]['image']))[0],
        hovertemplate='Size=%{x}<br>Intensity sum=%{y}<br>'
                      + 'Label=%{customdata[0]}<br>Image="%{customdata[1]}"')
    scatter_filtered = go.Scatter(
        name="discarded nuclei", x=x_data[not_passed], y=y_data[not_passed],
        mode='markers', marker=dict(size=4, opacity=.5, color="#e31a1c"),
        xaxis="x", yaxis="y", customdata=np.dstack((
            data[not_passed]['label'], data[not_passed]['image']))[0],
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


def plot_nuclear_features(
        data: pd.DataFrame, spx_data: pd.DataFrame, labels: Dict[str, str],
        n_id_cols: int = 3, n_plot_grid_col: int = 3) -> go.Figure:
    root_list = np.unique(data['root'].values)
    N = root_list.shape[0]
    pal = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

    channel_list = np.unique(spx_data['channel'].values)
    n_features = data.shape[1]-n_id_cols
    n_rows = (n_features+len(channel_list))//n_plot_grid_col+1
    fig = make_subplots(rows=n_rows, cols=n_plot_grid_col,
                        horizontal_spacing=.2)

    layout = {}
    for ci in range(n_features):
        colname = data.columns[ci+n_id_cols]
        collab = labels[colname] if colname in labels else colname

        for root in np.unique(data['root'].values):
            fig.add_trace(go.Box(
                name=root,
                y=data[data['root'] == root][colname].values,
                notched=True, marker_color=pal[np.argmax(root_list == root)]),
                row=ci//n_plot_grid_col+1, col=ci % n_plot_grid_col+1)

        layout["yaxis" if ci == 0 else f"yaxis{ci+1}"] = dict(title=collab)

    for ci in range(n_features, n_features+len(channel_list)):
        channel = channel_list[ci-n_features]
        spx_channel = spx_data[spx_data['channel'] == channel]

        for root in spx_channel['root'].values:
            spx_root = spx_channel[spx_channel['root'] == root]
            fig.add_trace(
                go.Box(
                    name=root, notched=True,
                    y=[[spx_root['vmin'].values[0],
                        spx_root['vmax'].values[0]]],
                    marker_color=pal[np.argmax(root_list == root)]),
                row=ci//n_plot_grid_col+1, col=ci % n_plot_grid_col+1)
            fig.update_traces(
                q1=spx_root['q1'], q3=spx_root['q3'],
                median=spx_root['median'],
                lowerfence=spx_root['whisk_low'],
                upperfence=spx_root['whisk_high'],
                row=ci//n_plot_grid_col+1, col=ci % n_plot_grid_col+1)

        layout[f"yaxis{ci+1}"] = dict(
            title=f'"{channel}" single pixel intensity (a.u.)')

    fig.update_xaxes(tickangle=45)
    fig.update_xaxes(ticks="outside", tickwidth=1, ticklen=5, automargin=True)
    fig.update_yaxes(ticks="outside", tickwidth=1, ticklen=5, automargin=True)
    fig.update_layout(**layout)
    fig.update_layout(height=400*n_rows, width=1000, autosize=False)
    fig.update_layout(showlegend=False)
    return fig

# END =========================================================================

###############################################################################
