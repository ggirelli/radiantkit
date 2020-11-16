"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
import logging
import numpy as np  # type: ignore
import os
from radiantkit.const import __version__
from radiantkit.conversion import CziFile2
from radiantkit.exception import enable_rich_assert
import radiantkit.image as imt
from radiantkit.io import add_log_file_handler
from radiantkit.string import MultiRange
from radiantkit.string import TIFFNameTemplateFields as TNTFields
from radiantkit.string import TIFFNameTemplate as TNTemplate
from rich.logging import RichHandler  # type: ignore
from rich.progress import track  # type: ignore
import sys
from typing import Iterable, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)


@enable_rich_assert
def init_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1],
        description=f"""
Convert a czi file into single channel tiff images.

The output tiff file names follow the specified template (-T). A template is a
string including a series of "seeds" that are replaced by the corresponding
values when writing the output file. Available seeds are:
{TNTFields.CHANNEL_NAME} : channel name, lower-cased.
{TNTFields.CHANNEL_ID} : channel ID (number).
{TNTFields.SERIES_ID} : series ID (number).
{TNTFields.DIMENSIONS} : number of dimensions, followed by "D".
{TNTFields.AXES_ORDER} : axes order (e.g., "TZYX").
Leading 0s are added up to 3 digits to any ID seed.

The default template is "{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}".
Hence, when writing the 3rd series of the "a488" channel, the output file name
would be:"a488_003.tiff".

Please, remember to escape the "$" when running from command line if using
double quotes, i.e., "\\$". Alternatively, use single quotes, i.e., '$'.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Convert a czi file into single channel tiff images.",
    )

    parser.add_argument("input", type=str, help="""Path to the czi file to convert.""")

    parser.add_argument(
        "--outdir",
        metavar="DIRPATH",
        type=str,
        help="""Path to output TIFF folder. Defaults to the input file
        basename.""",
        default=None,
    )
    parser.add_argument(
        "--fields",
        metavar="STRING",
        type=str,
        help="""Extract only fields of view specified as when printing a set
        of pages. E.g., '1-2,5,8-9'.""",
        default=None,
    )
    parser.add_argument(
        "--channels",
        metavar="STRING",
        type=str,
        help="""Extract only specified channels. Specified as space-separated
        channel names. E.g., 'dapi cy5 a488'.""",
        default=None,
        nargs="+",
    )

    parser.add_argument(
        "--version", action="version", version=f"{sys.argv[0]} {__version__}"
    )

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument(
        "--template",
        metavar="STRING",
        type=str,
        help="""Template for output file name. See main description for more
        details. Default: '{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}'""",
        default=f"{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}",
    )
    advanced.add_argument(
        "--compressed",
        action="store_const",
        dest="doCompress",
        const=True,
        default=False,
        help="Write compressed TIFF as output.",
    )
    advanced.add_argument(
        "-n",
        "--dry-run",
        action="store_const",
        dest="dry",
        const=True,
        default=False,
        help="Describe input data and stop.",
    )

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


@enable_rich_assert
def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    if args.outdir is None:
        args.outdir = os.path.splitext(os.path.basename(args.input))[0]
        args.outdir = os.path.join(os.path.dirname(args.input), args.outdir)

    assert os.path.isfile(args.input), f"input file not found: {args.input}"
    assert not os.path.isfile(
        args.outdir
    ), f"output directory cannot be a file: {args.outdir}"

    if args.fields is not None:
        args.fields = MultiRange(args.fields)

    if args.channels is not None:
        args.channels = [c.lower() for c in args.channels]

    assert 0 != len(args.template)
    args.template = TNTemplate(args.template)

    return args


def check_channels(channels: List[str], czi_image: CziFile2) -> List[str]:
    if channels is None:
        channels = list(czi_image.get_channel_names())
    else:
        channels = czi_image.select_channels(channels)
        if 0 == len(channels):
            logging.error("None of the specified channels was found.")
            sys.exit()
        logging.info(f"Converting only the following channels: {channels}")
    return channels


def field_generator(
    args: argparse.Namespace, czi_image: CziFile2
) -> Iterable[Tuple[np.ndarray, str]]:
    for field_id in args.fields:
        if field_id - 1 >= czi_image.field_count():
            logging.warning(
                f"Skipped field #{field_id} "
                + "(from specified field range, "
                + "not available in czi file)."
            )
            continue
        for yieldedValue in czi_image.get_channel_pixels(args, field_id - 1):
            channel_pixels, channel_id = yieldedValue
            if not list(czi_image.get_channel_names())[channel_id] in args.channels:
                continue
            yield (
                channel_pixels,
                czi_image.get_tiff_path(args.template, channel_id, field_id - 1),
            )


def convert_to_tiff(args: argparse.Namespace, czi_image: CziFile2) -> None:
    export_total = float("inf")
    if args.fields is not None and args.channels is not None:
        export_total = len(args.fields) * len(args.channels)
    elif args.fields is not None:
        export_total = len(args.fields)
    elif args.channels is not None:
        export_total = len(args.channels)
    export_total = min(
        czi_image.field_count() * czi_image.channel_count(), export_total
    )
    for (OI, opath) in track(field_generator(args, czi_image), total=int(export_total)):
        imt.save_tiff(
            os.path.join(args.outdir, opath),
            OI.astype(imt.get_dtype(OI.max())),
            args.doCompress,
            bundle_axes="TZYX",
            resolution=(
                1e-6 / czi_image.get_axis_resolution("X"),
                1e-6 / czi_image.get_axis_resolution("Y"),
            ),
            inMicrons=True,
            ResolutionZ=czi_image.get_axis_resolution("Z") * 1e6,
        )


def check_argument_compatibility(
    args: argparse.Namespace, czi_image: CziFile2
) -> argparse.Namespace:
    assert args.template.can_export_fields(czi_image.field_count(), args.fields), (
        "when exporting more than 1 field, the template "
        + f"must include the {TNTFields.SERIES_ID} seed. "
        + f"Got '{args.template.template}' instead."
    )

    args.channels = check_channels(args.channels, czi_image)

    assert args.template.can_export_channels(
        czi_image.channel_count(), args.channels
    ), (
        "when exporting more than 1 channel, the template "
        + f"must include either {TNTFields.CHANNEL_ID} or "
        + f"{TNTFields.CHANNEL_NAME} seeds. "
        + f"Got '{args.template.template}' instead."
    )

    if args.fields is None:
        args.fields = range(czi_image.field_count())

    return args


@enable_rich_assert
def run(args: argparse.Namespace) -> None:
    czi_image = CziFile2(args.input)
    if args.dry:
        czi_image.log_details()
        sys.exit()

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
    add_log_file_handler(os.path.join(args.outdir, "nd2_to_tiff.log.txt"))

    czi_image.log_details()
    args = check_argument_compatibility(args, czi_image)
    assert not czi_image.isLive(), "time-course conversion images not implemented."

    logging.info(f"Output directory: '{args.outdir}'")
    czi_image.squeeze_axes("SCZYX")
    reordered_axes = "CZYX" if 1 == czi_image.field_count() else "SCZYX"
    czi_image.reorder_axes(reordered_axes)

    if args.fields is not None:
        args.fields = list(args.fields)
        logging.info(
            "Converting only the following fields: " + f"{[x for x in args.fields]}"
        )

    convert_to_tiff(args, czi_image)

    logging.info("Done. :thumbs_up: :smiley:")
