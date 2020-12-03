"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import numpy as np  # type: ignore
import plotly.graph_objects as go  # type: ignore
from typing import List, Optional, Tuple


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
