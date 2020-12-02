"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
from collections import defaultdict
from joblib import delayed, Parallel  # type: ignore
import itertools
import logging
import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import pickle
from radiantkit import const, io
from radiantkit.image import ImageBinary, ImageLabeled
from radiantkit.particle import NucleiList, Nucleus
from radiantkit.report import ReportBase
import radiantkit.scripts.common.series as ra_series
from radiantkit.scripts.common import argtools
from radiantkit.series import Series, SeriesList
from radiantkit import path, stat, string
import re
from rich.progress import track  # type: ignore
from rich.prompt import Confirm  # type: ignore
from scipy.stats import gaussian_kde  # type: ignore
from typing import Any, DefaultDict, Dict, List, Optional, Pattern, Tuple


def init_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1],
        description="""
Select nuclei (objects) from segmented images based on their size (volume in
3D, area in 2D) and integral of intensity from raw image.

To achieve this, the script looks for mask/raw image pairs in the input folder.
Mask images are identified by the specified prefix/suffix. For example, a pair
with suffix "mask" would be:
    [RAW] "dapi_001.tiff" and [MASK] "dapi_001.mask.tiff".

Nuclei are extracted and size and integral of intensity are calculated. Then,
their density profile is calculated across all images. A sum of Gaussian is fit
to the profiles and a range of +-k_sigma around the peak of the first Gaussian
is selected. If the fit fails, a single Gaussian is fitted and the range is
selected in the same manner around its peak. If this fit fails, the selected
range corresponds to the FWHM range around the first peak of the profiles. In
the last scenario, k_sigma is ignored.

A tabulation-separated table is generated with the nuclear features and whether
they pass the filter(s). Alongside it, an html report is generated with
interactive data visualization.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Select G1 nuclei.",
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to folder containing deconvolved tiff images and masks.",
    )
    parser.add_argument(
        "ref_channel", type=str, help="Name of channel with DNA staining intensity."
    )
    parser = argtools.add_version_argument(parser)

    critical = parser.add_argument_group("critical arguments")
    critical.add_argument(
        "--k-sigma",
        type=float,
        metavar="NUMBER",
        help="""Suffix for output binarized images name.
        Default: 2.5""",
        default=2.5,
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
        "--no-remove",
        action="store_const",
        dest="remove_labels",
        const=False,
        default=True,
        help="Do not export masks after removing discarded nuclei labels.",
    )
    advanced.add_argument(
        "--uncompressed",
        action="store_const",
        dest="compressed",
        const=False,
        default=True,
        help="Generate uncompressed TIFF binary masks.",
    )
    advanced = argtools.add_pattern_argument(advanced)
    advanced = argtools.add_threads_argument(advanced)
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
    assert "(?P<channel_name>" in args.inreg
    assert "(?P<series_id>" in args.inreg
    args.inreg = re.compile(args.inreg)

    args.mask_prefix = string.add_trailing_dot(args.mask_prefix)
    args.mask_suffix = string.add_leading_dot(args.mask_suffix)

    if not 0 != args.block_side % 2:
        logging.warning(
            "changed ground block side from "
            + f"{args.block_side} to {args.block_side+1}"
        )
        args.block_side += 1

    args.threads = argtools.check_threads(args.threads)

    return args


def print_settings(args: argparse.Namespace, clear: bool = True) -> str:
    s = f"""# Nuclei selection v{const.__version__}

    ---------- SETTING : VALUE ----------

       Input directory : '{args.input}'
      DNA channel name : '{args.ref_channel}'
               K sigma : {args.k_sigma}

           Mask prefix : '{args.mask_prefix}'
           Mask suffix : '{args.mask_suffix}'

     Ground block side : {args.block_side}
            Use labels : {args.labeled}
               Rescale : {args.do_rescaling}
         Remove labels : {args.remove_labels}
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
    print_settings(args)
    if not args.do_all:
        assert Confirm.ask("Confirm settings and proceed?")

    assert os.path.isdir(args.input), f"image folder not found: {args.input}"


def extract_passing_nuclei_per_series(
    ndata: pd.DataFrame, inreg: Pattern
) -> Dict[int, List[int]]:
    passed = ndata.loc[ndata["pass"], ["image", "label"]]
    passed["series_id"] = np.nan
    for ii in passed.index:
        image_details = path.get_image_details(passed.loc[ii, "image"], inreg)
        assert image_details is not None
        passed.loc[ii, "series_id"] = image_details[0]
    passed.drop("image", 1, inplace=True)
    passed = dict(
        [
            (sid, passed.loc[passed["series_id"] == sid, "label"].values.tolist())
            for sid in set(passed["series_id"].values)
        ]
    )
    return passed


def remove_labels_from_series_mask(
    series: Series, labels: List[int], labeled: bool, compressed: bool
) -> Series:
    if series.mask is None:
        return series

    series.mask.load_from_local()

    if labeled:
        L = series.mask.pixels
        L[np.logical_not(np.isin(L, labels))] = 0
        L = ImageLabeled(L)
        L.to_tiff(path.add_suffix(series.mask.path, "selected"), compressed)
    else:
        if isinstance(series.mask, ImageBinary):
            L = series.mask.label().pixels
        else:
            L = series.mask.pixels
        L[np.logical_not(np.isin(L, labels))] = 0
        M = ImageBinary(L)
        M.to_tiff(path.add_suffix(series.mask.path, "selected"), compressed)

    series.unload()
    return series


def remove_labels_from_series_list_masks(
    args: argparse.Namespace,
    series_list: SeriesList,
    passed: Dict[int, List[int]],
    nuclei: NucleiList,
) -> SeriesList:
    series_list.unload()
    if args.remove_labels:
        logging.info("removing discarded nuclei labels from masks")
        if 1 == args.threads:
            for s in track(series_list):
                s = remove_labels_from_series_mask(
                    s, passed[s.ID], args.labeled, args.compressed
                )
        else:
            series_list.series = Parallel(n_jobs=args.threads, verbose=11)(
                delayed(remove_labels_from_series_mask)(
                    s, passed[s.ID], args.labeled, args.compressed
                )
                for s in series_list
            )
        n_removed = len(nuclei) - len(list(itertools.chain(*passed.values())))
        logging.info(f"removed {n_removed} nuclei labels")
    return series_list


def run(args: argparse.Namespace) -> None:
    confirm_arguments(args)
    argtools.dump_args(args, "select_nuclei.args.pkl")
    io.add_log_file_handler(os.path.join(args.input, "select_nuclei.log.txt"))
    args, series_list = ra_series.init_series_list(args)

    logging.info("extracting nuclei")
    series_list.extract_particles(Nucleus, [args.ref_channel], args.threads)

    nuclei = NucleiList(list(itertools.chain(*[s.particles for s in series_list])))
    logging.info(f"extracted {len(nuclei)} nuclei.")

    logging.info("selecting G1 nuclei.")
    nuclei_data, details = nuclei.select_G1(args.k_sigma, args.ref_channel)
    passed = extract_passing_nuclei_per_series(nuclei_data, args.inreg)

    series_list = remove_labels_from_series_list_masks(
        args, series_list, passed, nuclei
    )

    np.set_printoptions(formatter={"float_kind": "{:.2E}".format})
    logging.info(f"size fit:\n{details['size']['fit']}")
    np.set_printoptions(formatter={"float_kind": "{:.2E}".format})
    logging.info(f"size range: {details['size']['range']}")
    np.set_printoptions(formatter={"float_kind": "{:.2E}".format})
    logging.info(f"intensity sum fit:\n{details['isum']['fit']}")
    np.set_printoptions(formatter={"float_kind": "{:.2E}".format})
    logging.info(f"intensity sum range: {details['isum']['range']}")

    tsv_path = os.path.join(args.input, Report().files["raw_data"][0])
    logging.info(f"writing nuclear data to:\n{tsv_path}")
    nuclei_data.to_csv(tsv_path, sep="\t", index=False)

    pkl_path = os.path.join(args.input, Report().files["fit"][0])
    logging.info(f"writing fit data to:\n{pkl_path}")
    with open(pkl_path, "wb") as POH:
        pickle.dump(details, POH)

    ra_series.pickle_series_list(args, series_list)


class Report(ReportBase):
    def __init__(self, *args, **kwargs):
        super(Report, self).__init__(*args, **kwargs)
        self._idx = 1.0
        self._stub = "select_nuclei"
        self._title = "Nuclei selection"
        self._files = {
            "raw_data": ("select_nuclei.data.tsv", True, []),
            "fit": ("select_nuclei.fit.pkl", True, []),
        }
        self._log = {"log": ("select_nuclei.log.txt", False, [])}
        self._args = {"args": ("select_nuclei.args.pkl", False, [])}

    def __make_scatter_trace(self, data: pd.DataFrame, name: str) -> go.Scatter:
        return go.Scatter(
            x=data["size"],
            y=data["isum_dapi"],
            mode="markers",
            name=name,
            xaxis="x",
            yaxis="y",
            customdata=np.dstack(
                (
                    data["label"],
                    data["image"],
                )
            )[0],
            hovertemplate="Size=%{x}<br>Intensity sum=%{y}<br>"
            + "Label=%{customdata[0]}<br>"
            + 'Image="%{customdata[1]}"',
            legendgroup=name,
        )

    def __add_density_contours(
        self, fig: go.Figure, data: pd.DataFrame, fit: Dict[str, Dict[str, Any]]
    ) -> go.Figure:
        assert "size" in data
        size_linsp = np.linspace(data["size"].min(), data["size"].max(), 200)
        size_kde = gaussian_kde(data["size"])
        fig.add_trace(
            go.Scatter(
                name="Size",
                x=size_linsp,
                y=size_kde(size_linsp),
                xaxis="x",
                yaxis="y3",
                legendgroup="Size",
            )
        )
        assert "isum_dapi" in data
        isum_linsp = np.linspace(data["isum_dapi"].min(), data["isum_dapi"].max(), 200)
        isum_kde = gaussian_kde(data["isum_dapi"])
        fig.add_trace(
            go.Scatter(
                name="Intensity sum",
                x=isum_kde(isum_linsp),
                y=isum_linsp,
                xaxis="x2",
                yaxis="y",
                legendgroup="Intensity sum",
            ),
        )
        return fig

    def __prep_fit_contours_data(
        self, data_type: str, data_series: np.ndarray, params: List[float]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, List[float]]]:
        assert data_type in ["x", "y"]
        if "x" in data_type:
            return (
                dict(x=data_series, y=stat.gaussian(data_series, *params[:3])),
                dict(x=data_series, y=stat.gaussian(data_series, *params[3:])),
                dict(xx=[params[1]], yy=[]),
            )
        else:
            return (
                dict(y=data_series, x=stat.gaussian(data_series, *params[:3])),
                dict(y=data_series, x=stat.gaussian(data_series, *params[3:])),
                dict(xx=[], yy=[params[1]]),
            )

    def __add_fit_contours(
        self,
        fig: go.Figure,
        name: str,
        data_series: np.ndarray,
        data_type: str,
        fit: Tuple[np.ndarray, stat.FitType],
        xref: str = "x",
        yref: str = "y",
    ) -> go.Figure:
        assert data_type in ["x", "y"]
        params, fit_type = fit
        data_g1, data_g2, line_data = self.__prep_fit_contours_data(
            data_type, data_series, params
        )
        if stat.FitType.FWHM != fit_type:
            fig.add_trace(
                go.Scatter(
                    name=f"{name}_gaussian_1",
                    **data_g1,
                    xaxis=xref,
                    yaxis=yref,
                    legendgroup=name,
                )
            )
            self.__add_range_lines(
                fig,
                line_data["xx"],
                line_data["yy"],
                line_props=dict(
                    line_color="#323232",
                    line_width=1,
                    line_dash="dot",
                ),
            )
        if stat.FitType.SOG == fit_type:
            fig.add_trace(
                go.Scatter(
                    name=f"{name}_gaussian_2",
                    **data_g2,
                    xaxis=xref,
                    yaxis=yref,
                    legendgroup=name,
                )
            )
        return fig

    def __add_range_lines(
        self,
        fig: go.Figure,
        xx: List[float],
        yy: List[float],
        line_props: Optional[Dict[str, Any]] = None,
    ) -> go.Figure:
        if line_props is None:
            line_props = dict(
                line_color="#323232",
                line_width=1,
                line_dash="dash",
            )
        for x0 in xx:
            fig.add_vline(x=x0, **line_props)
            fig.add_shape(
                type="line",
                x0=x0,
                x1=x0,
                y0=fig.data[2].y.min(),
                y1=fig.data[2].y.max(),
                xref="x",
                xsizemode="scaled",
                yref="y3",
                ysizemode="scaled",
                **line_props,
            )
        for y0 in yy:
            fig.add_hline(y=y0, **line_props)
            fig.add_shape(
                type="line",
                x0=fig.data[3].x.min(),
                x1=fig.data[3].x.max(),
                y0=y0,
                y1=y0,
                xref="x2",
                xsizemode="scaled",
                yref="y",
                ysizemode="scaled",
                **line_props,
            )
        return fig

    def _plot(
        self, data: DefaultDict[str, Dict[str, pd.DataFrame]], *args, **kwargs
    ) -> DefaultDict[str, Dict[str, go.Figure]]:
        logging.info(f"plotting '{self._stub}'.")
        fig_data: DefaultDict[str, Dict[str, go.Figure]] = defaultdict(lambda: {})
        assert "raw_data" in data
        assert "arg_data" in kwargs
        for dirpath, dirdata in data["raw_data"].items():
            assert isinstance(dirdata, pd.DataFrame)
            fig = go.Figure()

            fig.add_trace(
                self.__make_scatter_trace(dirdata.loc[dirdata["pass"]], "Selected")
            )
            fig.add_trace(
                self.__make_scatter_trace(
                    dirdata.loc[np.logical_not(dirdata["pass"])], "Discarded"
                )
            )

            if dirpath in data["fit"]:
                fig = self.__add_density_contours(fig, dirdata, data["fit"][dirpath])
                fig = self.__add_fit_contours(
                    fig,
                    "Size",
                    np.linspace(dirdata["size"].min(), dirdata["size"].max(), 200),
                    "x",
                    data["fit"][dirpath]["size"]["fit"],
                    "x",
                    "y3",
                )
                fig = self.__add_fit_contours(
                    fig,
                    "Intensity sum",
                    np.linspace(
                        dirdata["isum_dapi"].min(), dirdata["isum_dapi"].max(), 200
                    ),
                    "y",
                    data["fit"][dirpath]["isum"]["fit"],
                    "x2",
                    "y",
                )
                fig = self.__add_range_lines(
                    fig,
                    xx=data["fit"][dirpath]["size"]["range"],
                    yy=data["fit"][dirpath]["isum"]["range"],
                )

            fig.update_layout(
                title_text=f"""Nuclei selection<br>
<sub>Condition: {os.path.basename(dirpath)}; #nuclei: {dirdata.shape[0]};
 #selected: {dirdata['pass'].sum()}</sub>""",
                xaxis=dict(domain=[0.19, 1], title="Size"),
                yaxis=dict(
                    domain=[0, 0.82],
                    anchor="x2",
                    title="Intensity sum",
                ),
                xaxis2=dict(domain=[0, 0.18], autorange="reversed", title="Density"),
                yaxis2=dict(domain=[0, 0.82]),
                xaxis3=dict(domain=[0.19, 1]),
                yaxis3=dict(domain=[0.83, 1], title="Density"),
                autosize=False,
                width=1000,
                height=1000,
            )

            fig_data[self._stub][dirpath] = fig
        return fig_data
