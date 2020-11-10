"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
from joblib import cpu_count  # type: ignore
import logging
import os
from radiantkit import const
from radiantkit import particle, series
from radiantkit import string
from radiantkit.scripts.common import series as ra_series
import re
from rich.logging import RichHandler  # type: ignore
from rich.prompt import Confirm  # type: ignore
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

__OUTPUT__ = {
    "obj_features": "nuclear_features.tsv",
    "spx_features": "single_pixel_features.tsv",
}
__OUTPUT_CONDITION__ = any
__LABEL__ = "Object features"


def init_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1],
        description="""Measure objects from masks. Exports a
        tabulation-separated table containing the following features: size in
        voxels, volume in m, surface in m2, shape descriptor(s), sum and mean
        of voxel intensity values per channel. Also, for each channel, also
        provides quartiles of single voxel intensity values.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Measure objects from masks.",
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to folder containing deconvolved tiff images and masks.",
    )
    parser.add_argument(
        "ref_channel", type=str, help="Name of channel with masks to be used."
    )

    parser.add_argument(
        "--output",
        type=str,
        help=f"""Path to folder where output should be written to. Defaults to
        "{const.default_subfolder}" subfolder in the input directory.""",
    )
    parser.add_argument(
        "--export-single-voxel",
        action="store_const",
        const=True,
        default=False,
        dest="exportSingleVoxel",
        help="Export also quantiles of single voxel features.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%s %s"
        % (
            sys.argv[0],
            const.__version__,
        ),
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
    assert not os.path.isfile(args.output)
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

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

    args.threads = cpu_count() if args.threads > cpu_count() else args.threads

    return args


def print_settings(args: argparse.Namespace, clear: bool = True) -> str:
    s = f"""# Object extraction v{args.version}

    ---------- SETTING : VALUE ----------

       Input directory : '{args.input}'
      Output directory : '{args.output}'
Reference channel name : '{args.ref_channel}'
    Voxel aspect (ZYX) : {args.aspect}

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

    assert os.path.isdir(args.input), f"image folder not found: {args.input}"

    # settings_path = os.path.join(args.output, "measure_objects.config.txt")
    # with open(settings_path, "w+") as OH:
    #     ggc.args.export_settings(OH, settings_string)


def measure_object_features(
    args: argparse.Namespace, series_list: series.SeriesList
) -> None:
    feat_path = os.path.join(args.output, __OUTPUT__["obj_features"])
    logging.info(f"exporting nuclear features to '{feat_path}'")
    series_list.export_particle_features(feat_path)

    if args.exportSingleVoxel:
        feat_path = os.path.join(args.output, __OUTPUT__["spx_features"])
        logging.info(f"exporting single_pixel features to '{feat_path}'")
        single_pixel_box_data = series_list.get_particle_single_px_stats()
        single_pixel_box_data.to_csv(feat_path, index=False, sep="\t")


def run(args: argparse.Namespace) -> None:
    confirm_arguments(args)
    args, series_list = ra_series.init_series_list(args)

    logging.info("extracting nuclei")
    series_list.extract_particles(particle.Nucleus, threads=args.threads)
    logging.info(f"extracted {len(list(series_list.particles()))} nuclei")

    measure_object_features(args, series_list)

    ra_series.pickle_series_list(args, series_list)
