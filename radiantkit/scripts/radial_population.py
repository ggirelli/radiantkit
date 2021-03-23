"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
from joblib import cpu_count  # type: ignore
import logging
import os
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import pickle
from radiantkit import const, distance, io, particle, plot, report, series, string
from radiantkit.scripts import argtools
import re
from rich.prompt import Confirm  # type: ignore
import sys
from typing import Any, DefaultDict, Dict, Optional

__OUTPUT__: Dict[str, str] = {
    "poly_fit": "radial_population.profile.poly_fit.pkl",
    "raw_data": "radial_population.profile.raw_data.tsv",
    "args": "radial_population.args.pkl",
    "log": "radial_population.log.txt",
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
            "".join(
                [
                    "changed ground block side from ",
                    f"{args.block_side} to {args.block_side+1}",
                ]
            )
        )
        args.block_side += 1

    args.threads = max(1, min(cpu_count(), args.threads))

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
    argtools.dump_args(args, __OUTPUT__["args"])
    io.add_log_file_handler(os.path.join(args.input, __OUTPUT__["log"]))
    args, series_list = series.init_series_list(args)

    logging.info("extracting nuclei")
    series_list.extract_particles(particle.Nucleus, threads=args.threads)
    logging.info(f"extracted {len(list(series_list.particles()))} nuclei")

    logging.info("generating radial profiles")
    rdc = distance.RadialDistanceCalculator(args.axes, args.center_type, args.quantile)
    profiles = series_list.get_radial_profiles(
        rdc, args.bins, args.degree, threads=args.threads
    )

    export_profiles(args, profiles)
    series.pickle_series_list(args, series_list)


class ReportRadialPopulation(report.ReportBase):
    def __init__(self, *args, **kwargs):
        super(ReportRadialPopulation, self).__init__(*args, **kwargs)
        self._idx = 3.0
        self._stub = "radial_population"
        self._title = "Radiality (population)"
        self._files = {
            "poly_fit": (__OUTPUT__["poly_fit"], True, []),
            "raw_data": (__OUTPUT__["raw_data"], True, []),
        }
        self._log = {"log": (__OUTPUT__["log"], False, [])}
        self._args = {"args": (__OUTPUT__["args"], False, [])}

    def _make_plot_page(
        self, data: DefaultDict[str, Dict[str, pd.DataFrame]]
    ) -> report.ReportPage:
        page = report.ReportPage("plot-subpage", 1)
        logging.info("Plotting multi-condition panel.")
        page.add_panel(
            plot.ProfileMultiCondition.html_class,
            "Multi-condition",
            self._make_panel_page(
                plot.ProfileMultiCondition.html_class,
                *plot.ProfileMultiCondition(self._stub).make(data),
                "Select a channel-stat to update the plot below.",
            ),
        )
        logging.info("Plotting multi-condition (normalized) panel.")
        page.add_panel(
            plot.ProfileMultiConditionNorm.html_class,
            "Multi-condition (norm)",
            self._make_panel_page(
                plot.ProfileMultiConditionNorm.html_class,
                *plot.ProfileMultiConditionNorm(self._stub).make(data),
                "Select a channel-stat to update the plot below.",
            ),
        )
        logging.info("Plotting single-condition panel.")
        page.add_panel(
            plot.ProfileSingleCondition.html_class,
            "Single condition",
            self._make_panel_page(
                plot.ProfileSingleCondition.html_class,
                *plot.ProfileSingleCondition(self._stub).make(data),
                "Select a channel-condition to update the plot below.",
            ),
        )
        return page

    def _make_html(
        self,
        fig_data: Optional[Dict[str, Dict[str, go.Figure]]] = None,
        log_data: Optional[DefaultDict[str, Dict[str, Any]]] = None,
        arg_data: Optional[DefaultDict[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        assert "output_data" in kwargs
        page = report.ReportPage(self._stub, 0)
        page.add_panel(
            "plot", "Plots", self._make_plot_page(kwargs["output_data"]).make()
        )
        if log_data is not None:
            page.add_panel("log", "Log", self._make_log_panels(log_data))
        if arg_data is not None:
            page.add_panel("arg", "Args", self._make_arg_panels(arg_data))
        return page.make()

    def make(self) -> str:
        logging.info(f"reading output files of '{self._stub}'.")
        output_data = self._read(self._search(self._files))
        logging.info(f"reading logs and args of '{self._stub}'.")
        log_data = self._read(self._search(self._log))
        arg_data = self._read(self._search(self._args))
        try:
            return self._make_html(
                fig_data=None,
                log_data=log_data,
                arg_data=arg_data,
                output_data=output_data,
            )
        except AssertionError:
            logging.warning(f"skipped '{self._stub}'.")
            return ""
