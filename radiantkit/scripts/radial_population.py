"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
from collections import defaultdict
from joblib import cpu_count  # type: ignore
import logging
import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import pickle
from radiantkit import const, distance, particle, report, series, stat, string
from radiantkit.scripts.common import series as ra_series
from radiantkit.scripts.common import argtools
import re
from rich.prompt import Confirm  # type: ignore
import sys
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

__OUTPUT__: Dict[str, str] = {
    "poly_fit": "radial_population.profile.poly_fit.pkl",
    "raw_data": "radial_population.profile.raw_data.tsv",
}


def init_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1],
        description="""Generate average radial
        profiles for a cell population. Requires a folder containing tiff
        images with grayscale intensities and masks with segmented nuclei.
        We recommend deconvolving the grayscale images to obtain a better
        reconstruction of the radial profile.

        Crucial aspect and axes

        A radial profile is intended to be a curve with voxel intensity (Y) as
        a function of a distance (X). This distance can either be the distance
        of a voxel from the nuclear lamina, or from the nuclear center. Here,
        the distance from the nuclear lamina is calculated as the euclidean
        distance from the background of masks of segmented nuclei. See below,
        for multiple definitions of nuclear center, accessible via the
        --center-type parameter. The profile is also generated for a normalized
        lamina distance, obtain by dividing the absolute lamina distance of a
        voxel by the sum of absolute lamina and center distances.

        Center definitions:
        - Centroid: ...
        - Center of Mass: ...
        - Quantile: ...
        - Maxima: ...

        Bins and degree, polynomial fit

        Roots
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Generate average radial profiles for a cell population.",
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to folder containing deconvolved tiff images and masks.",
    )
    parser.add_argument(
        "ref_channel", type=str, help="Name of channel with DNA staining intensity."
    )

    parser.add_argument(
        "--output",
        type=str,
        help=f"""Path to folder where output should be written to. Defaults to
        "{const.default_subfolder}" subfolder in the input directory.""",
    )
    parser.add_argument(
        "--version", action="version", version=f"{sys.argv[0]} {const.__version__}"
    )

    critical = parser.add_argument_group("critical arguments")
    critical.add_argument(
        "--aspect",
        type=float,
        nargs=3,
        help="""Physical size
        of Z, Y and X voxel sides in nm. Default: 300.0 216.6 216.6""",
        metavar=("Z", "Y", "X"),
        default=[300.0, 216.6, 216.6],
    )
    critical.add_argument(
        "--axes",
        type=str,
        metavar="STRING",
        help="""Axes to be used for distance calculation.""",
    )
    critical.add_argument(
        "--center-type",
        type=str,
        default=distance.CenterType.get_default().name,
        choices=[t.name for t in distance.CenterType],
        help=f"""Type of center for distance normalization.
        Default: {distance.CenterType.get_default().name}""",
    )
    critical.add_argument(
        "--quantile",
        type=float,
        metavar="NUMBER",
        help=f"""Quantile used to
        identify the center when '--center-type
        {distance.CenterType.QUANTILE.name}' is used.
        A number from 0 to 1 is expected. Defaults to 1e-N where N is
        the number of axes in an image.""",
    )
    critical.add_argument(
        "--mask-prefix",
        type=str,
        metavar="TEXT",
        help="""Prefix for output binarized images name.
        Default: ''.""",
        default="",
    )
    critical.add_argument(
        "--mask-suffix",
        type=str,
        metavar="TEXT",
        help="""Suffix for output binarized images name.
        Default: 'mask'.""",
        default="mask",
    )
    critical.add_argument(
        "--bins",
        type=int,
        metavar="NUMBER",
        default=200,
        help="""Number of bins for polynomial fitting. Default: 200.""",
    )
    critical.add_argument(
        "--degree",
        type=int,
        metavar="NUMBER",
        default=5,
        help="""Degree of polynomial fitting. Default: 5.""",
    )

    pickler = parser.add_argument_group("pickle arguments")
    pickler.add_argument(
        "--pickle-name",
        type=str,
        metavar="STRING",
        help=f"""Filename for input/output pickle file.
        Default: '{const.default_pickle}'""",
        default=const.default_pickle,
    )
    pickler.add_argument(
        "--export-instance",
        action="store_const",
        dest="export_instance",
        const=True,
        default=False,
        help="Export pickled series instance.",
    )
    pickler.add_argument(
        "--import-instance",
        action="store_const",
        dest="import_instance",
        const=True,
        default=False,
        help="Unpickle instance if pickle file is found.",
    )

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument(
        "--block-side",
        type=int,
        metavar="NUMBER",
        help="""Structural element side for dilation-based background/
        foreground measurement. Should be odd. Default: 11.""",
        default=11,
    )
    advanced.add_argument(
        "--use-labels",
        action="store_const",
        dest="labeled",
        const=True,
        default=False,
        help="Use labels from masks instead of relabeling.",
    )
    advanced.add_argument(
        "--no-rescaling",
        action="store_const",
        dest="do_rescaling",
        const=False,
        default=True,
        help="Do not rescale image even if deconvolved.",
    )
    advanced.add_argument(
        "--uncompressed",
        action="store_const",
        dest="compressed",
        const=False,
        default=True,
        help="Generate uncompressed TIFF binary masks.",
    )
    advanced.add_argument(
        "--inreg",
        type=str,
        metavar="REGEXP",
        help=f"""Regular expression to identify input TIFF images.
        Must contain 'channel_name' and 'series_id' fields.
        Default: '{const.default_inreg}'""",
        default=const.default_inreg,
    )
    advanced.add_argument(
        "--threads",
        type=int,
        metavar="NUMBER",
        dest="threads",
        default=1,
        help="""Number of threads for parallelization. Default: 1""",
    )
    advanced.add_argument(
        "-y",
        "--do-all",
        action="store_const",
        const=True,
        default=False,
        help="""Do not ask for settings confirmation and proceed.""",
    )

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    args.version = const.__version__

    if args.output is None:
        args.output = os.path.join(args.input, const.default_subfolder)
    argtools.check_output_folder_path(args.output)

    assert "(?P<channel_name>" in args.inreg
    assert "(?P<series_id>" in args.inreg
    args.inreg = re.compile(args.inreg)

    args.mask_prefix = string.add_trailing_dot(args.mask_prefix)
    args.mask_suffix = string.add_leading_dot(args.mask_suffix)

    argtools.check_axes(args.axes)
    if args.center_type is distance.CenterType.QUANTILE:
        if args.quantile is not None:
            assert args.quantile > 0 and args.quantile <= 1
    args.center_type = distance.CenterType[args.center_type]

    if not 0 != args.block_side % 2:
        logging.warning(
            "changed ground block side from "
            + f"{args.block_side} to {args.block_side+1}"
        )
        args.block_side += 1

    args.threads = cpu_count() if args.threads > cpu_count() else args.threads

    return args


def print_settings(args: argparse.Namespace, clear: bool = True) -> str:
    s = f"""# Object extraction v{args.version}

    ---------- SETTING : VALUE ----------

       Input directory : '{args.input}'
      Output directory : '{args.output}'
Reference channel name : '{args.ref_channel}'

    Voxel aspect (ZYX) : {args.aspect}
                  Axes : {args.axes}
           Center type : {args.center_type}
              Quantile : {args.quantile}
                  Bins : {args.bins}
                Degree : {args.degree}

           Mask prefix : '{args.mask_prefix}'
           Mask suffix : '{args.mask_suffix}'

     Ground block side : {args.block_side}
            Use labels : {args.labeled}
               Rescale : {args.do_rescaling}
            Compressed : {args.compressed}

           Pickle name : {args.pickle_name}
         Import pickle : {args.import_instance}
         Export pickle : {args.export_instance}

               Threads : {args.threads}
                Regexp : {args.inreg.pattern}
    """
    if clear:
        print("\033[H\033[J")
    print(s)
    return s


def confirm_arguments(args: argparse.Namespace) -> None:
    # settings_string =
    print_settings(args)
    if not args.do_all:
        assert Confirm.ask("Confirm settings and proceed?")

    assert os.path.isdir(args.input), f"input folder not found: {args.input}"

    # settings_path = os.path.join(args.output, "radial_population.config.txt")
    # with open(settings_path, "w+") as OH:
    #     ggc.args.export_settings(OH, settings_string)


def export_profiles(
    args: argparse.Namespace, profiles: series.RadialProfileData
) -> None:
    raw_data_separate = []
    pfit_data_separate = []
    for cname in profiles:
        for dtype in profiles[cname]:
            raw_data_tmp = profiles[cname][dtype][1]
            raw_data_tmp["channel"] = cname
            raw_data_tmp["distance_type"] = dtype
            raw_data_separate.append(raw_data_tmp)

            for sname in profiles[cname][dtype][0]:
                pfit_data_separate.append(
                    dict(
                        cname=cname,
                        distance_type=dtype,
                        stat=sname,
                        pfit=profiles[cname][dtype][0][sname],
                    )
                )

    logging.info("exporting profile data")
    pd.concat(raw_data_separate).to_csv(
        os.path.join(args.output, __OUTPUT__["raw_data"]), sep="\t", index=False
    )
    pickle_path = os.path.join(args.output, __OUTPUT__["poly_fit"])
    with open(pickle_path, "wb") as POH:
        pickle.dump(pfit_data_separate, POH)


def run(args: argparse.Namespace) -> None:
    confirm_arguments(args)
    args, series_list = ra_series.init_series_list(args)

    logging.info("extracting nuclei")
    series_list.extract_particles(particle.Nucleus, threads=args.threads)
    logging.info(f"extracted {len(list(series_list.particles()))} nuclei")

    logging.info("generating radial profiles")
    rdc = distance.RadialDistanceCalculator(args.axes, args.center_type, args.quantile)
    profiles = series_list.get_radial_profiles(
        rdc, args.bins, args.degree, threads=args.threads
    )

    export_profiles(args, profiles)
    ra_series.pickle_series_list(args, series_list)


class Report(report.ReportBase):
    def __init__(self, *args, **kwargs):
        super(Report, self).__init__(*args, **kwargs)
        self._idx = 3.0
        self._stub = "radial_population"
        self._title = "Radiality (population)"
        self._files = {
            "poly_fit": (__OUTPUT__["poly_fit"], True, []),
            "raw_data": (__OUTPUT__["raw_data"], True, []),
        }
        self._log = {"log": ("radial_population.log.txt", False, [])}
        self._args = {"args": ("radial_population.args.pkl", False, [])}

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
                        marker=dict(size=4, opacity=0.5, color="#989898"),
                    ),
                    go.Scatter(
                        name=f"{name}_{stat_type.value}",
                        x=x,
                        y=y,
                        xaxis="x",
                        yaxis="y",
                        mode="lines",
                        legendgroup=stat_type.value,
                    ),
                    go.Scatter(
                        name=f"{name}_{stat_type.value}_der1",
                        x=xx,
                        y=yy,
                        xaxis="x",
                        yaxis="y2",
                        mode="lines",
                        legendgroup=stat_type.value,
                    ),
                    go.Scatter(
                        name=f"{name}_{stat_type.value}_der2",
                        x=xxx,
                        y=yyy,
                        xaxis="x",
                        yaxis="y3",
                        mode="lines",
                        legendgroup=stat_type.value,
                    ),
                ]
            )
        return panel_data

    def __get_axis_range(
        self, trace_list: List[go.Figure], axis_type: str, axis_label: str
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

    def __add_line_shape(
        self,
        fig: go.Figure,
        x0: Optional[float],
        x1: Optional[float],
        y0: Optional[float],
        y1: Optional[float],
        line_color: str = "#969696",
        **kwargs,
    ) -> go.Figure:
        fig.add_shape(
            type="line",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            xsizemode="scaled",
            ysizemode="scaled",
            line_color=line_color,
            **kwargs,
        )
        return fig

    def __add_derivative_xaxis(self, fig: go.Figure) -> go.Figure:
        fig = self.__add_line_shape(
            fig, 0, 1, 0, 0, xref="x", yref="y2", line_dash="dash"
        )
        fig = self.__add_line_shape(
            fig, 0, 1, 0, 0, xref="x", yref="y3", line_dash="dash"
        )
        return fig

    __panel_list: List[Dict[str, str]] = [
        {"xref": "x", "yref": "y"},
        {"xref": "x", "yref": "y2"},
        {"xref": "x", "yref": "y3"},
    ]

    def __add_der1_zeros(
        self, fig: go.Figure, pfit_data: List[Dict[str, Any]]
    ) -> go.Figure:
        for pfit in pfit_data:
            root_der1, _ = stat.get_radial_profile_roots(pfit["pfit"])
            if np.isnan(root_der1):
                continue
            for panel in self.__panel_list[:2]:
                panel_trace_y = np.concatenate(
                    [p["y"] for p in fig["data"] if p["yaxis"] == panel["yref"]]
                )
                fig = self.__add_line_shape(
                    fig,
                    root_der1,
                    root_der1,
                    panel_trace_y.min(),
                    panel_trace_y.max(),
                    **panel,
                    line_dash="dash",
                )
        return fig

    def __add_der2_zeros(
        self, fig: go.Figure, pfit_data: List[Dict[str, Any]]
    ) -> go.Figure:
        for pfit in pfit_data:
            _, root_der2 = stat.get_radial_profile_roots(pfit["pfit"])
            if np.isnan(root_der2):
                continue
            for panel in self.__panel_list:
                panel_trace_y = np.concatenate(
                    [p["y"] for p in fig["data"] if p["yaxis"] == panel["yref"]]
                )
                fig = self.__add_line_shape(
                    fig,
                    root_der2,
                    root_der2,
                    panel_trace_y.min(),
                    panel_trace_y.max(),
                    **panel,
                    line_dash="dot",
                )
        return fig

    def __make_single_panel(
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

        fig = self.__add_derivative_xaxis(fig)
        fig = self.__add_der1_zeros(fig, pfit)
        fig = self.__add_der2_zeros(fig, pfit)

        yranges = dict(
            y=self.__get_axis_range(plot_data, "y", "y"),
            y2=self.__get_axis_range(plot_data, "y", "y2"),
            y3=self.__get_axis_range(plot_data, "y", "y3"),
        )
        fig.update_layout(
            title=f"""Signal profile<br>
<sub>Condition: {condition_lab};
Channel: {channel_lab}</sub>""",
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
                ] = self.__make_single_panel(
                    distdata,
                    data["poly_fit"][dirpath],
                    condition_lab,
                    channel_lab,
                    distance_type,
                )

        return fig_data
