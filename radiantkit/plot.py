'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from radiantkit import path as pt
from scipy.stats import gaussian_kde
from typing import Tuple

def export(path: str, exp_format: str = 'pdf') -> None:
    assert exp_format in ['pdf', 'png', 'jpg']
    path = pt.add_extension(path, '.' + exp_format)
    if exp_format == 'pdf':
        pp = PdfPages(path)
        plt.savefig(pp, format=exp_format)
        pp.close()
    else:
        plt.savefig(path, format=exp_format)

def plot_nuclear_selection(data: pd.DataFrame,
	size_range: Tuple[float], isum_range: Tuple[float]) -> go.Figure:
	assert all([x in data.columns
		for x in ["size", "isum", "pass", "label", "image"]])
	x_data = data['size'].values
	y_data = data['isum'].values

	xdf = gaussian_kde(x_data)
	ydf = gaussian_kde(y_data)

	x_data_sorted = sorted(set(x_data))
	y_data_sorted = sorted(set(y_data))

	passed = data['pass']
	not_passed = np.logical_not(passed)

	scatter_selected = go.Scatter(name="selected nuclei",
		x=x_data[passed], y=y_data[passed],
		mode='markers', marker=dict(size=4, opacity=.5, color="#1f78b4"),
		xaxis="x", yaxis="y",
		customdata=np.dstack((data[passed]['label'], data[passed]['image']))[0],
		hovertemplate='Size=%{x}<br>Intensity sum=%{y}<br>' +
			'Label=%{customdata[0]}<br>Image="%{customdata[1]}"')
	scatter_filtered = go.Scatter(name="discarded nuclei",
			x=x_data[not_passed],
			y=y_data[not_passed],
		mode='markers', marker=dict(size=4, opacity=.5, color="#e31a1c"),
		xaxis="x", yaxis="y", customdata=np.dstack((
			data[not_passed]['label'], data[not_passed]['image']))[0],
		hovertemplate='Size=%{x}<br>Intensity sum=%{y}<br>' +
			'Label=%{customdata[0]}<br>Image="%{customdata[1]}"')
	contour_y = go.Scatter(name="intensity sum",
		x=ydf(y_data_sorted), y=y_data_sorted,
		xaxis="x2", yaxis="y", line=dict(color="#33a02c"))
	contour_x = go.Scatter(name="size",
		x=x_data_sorted, y=xdf(sorted(set(x_data_sorted))),
		xaxis="x", yaxis="y3", line=dict(color="#ff7f00"))

	data = [scatter_selected, scatter_filtered, contour_x, contour_y]

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

	fig.add_shape(go.layout.Shape(type="line",
		x0=size_range[0], y0=0, x1=size_range[0], y1=y_data.max(),
	    line=dict(dash="dash", color="#969696")
	))
	fig.add_shape(go.layout.Shape(type="line",
		x0=size_range[1], y0=0, x1=size_range[1], y1=y_data.max(),
	    line=dict(dash="dash", color="#969696")
	))
	fig.add_shape(go.layout.Shape(type="line",
		x0=size_range[0], y0=0, x1=size_range[0], y1=xdf(x_data).max(),
	    line=dict(dash="dash", color="#969696"), yref="y3"
	))
	fig.add_shape(go.layout.Shape(type="line",
		x0=size_range[1], y0=0, x1=size_range[1], y1=xdf(x_data).max(),
	    line=dict(dash="dash", color="#969696"), yref="y3"
	))

	fig.add_shape(go.layout.Shape(type="line",
		y0=isum_range[0], x0=0, y1=isum_range[0], x1=x_data.max(),
	    line=dict(dash="dash", color="#969696")
	))
	fig.add_shape(go.layout.Shape(type="line",
		y0=isum_range[1], x0=0, y1=isum_range[1], x1=x_data.max(),
	    line=dict(dash="dash", color="#969696")
	))
	fig.add_shape(go.layout.Shape(type="line",
		y0=isum_range[0], x0=0, y1=isum_range[0], x1=ydf(y_data).max(),
	    line=dict(dash="dash", color="#969696"), xref="x2"
	))
	fig.add_shape(go.layout.Shape(type="line",
		y0=isum_range[1], x0=0, y1=isum_range[1], x1=ydf(y_data).max(),
	    line=dict(dash="dash", color="#969696"), xref="x2"
	))

	fig.update_layout(template="plotly_white")
	
	return fig

# END ==========================================================================

################################################################################
