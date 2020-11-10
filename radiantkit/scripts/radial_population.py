"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
from joblib import cpu_count  # type: ignore
import logging
import os
import pandas as pd  # type: ignore
import pickle
from radiantkit import const
from radiantkit import distance, io, string
from radiantkit import particle, series
from radiantkit.scripts.common import series as ra_series
from radiantkit.scripts.common import args as ra_args
import re
from rich.logging import RichHandler  # type: ignore
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

__OUTPUT__ = {
    "poly_fit": "radial_population.profile.poly_fit.pkl",
    "raw_data": "radial_population.profile.raw_data.tsv",
}
__OUTPUT_CONDITION__ = all
__LABEL__ = "Radiality (population)"


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
    ra_args.check_output_folder_path(args.output)

    assert "(?P<channel_name>" in args.inreg
    assert "(?P<series_id>" in args.inreg
    args.inreg = re.compile(args.inreg)

    args.mask_prefix = string.add_trailing_dot(args.mask_prefix)
    args.mask_suffix = string.add_leading_dot(args.mask_suffix)

    ra_args.check_axes(args.axes)
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
        io.ask("Confirm settings and proceed?")

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
