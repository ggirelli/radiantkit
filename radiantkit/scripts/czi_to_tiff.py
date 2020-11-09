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
import radiantkit.image as imt
from radiantkit.string import MultiRange
from radiantkit.string import TIFFNameTemplateFields as TNTFields
from radiantkit.string import TIFFNameTemplate as TNTemplate
from rich.logging import RichHandler  # type: ignore
import sys
from tqdm import tqdm  # type: ignore
from typing import Iterable, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)


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
        args.fields.zero_indexed = True

    if args.channels is not None:
        args.channels = [c.lower() for c in args.channels]

    assert 0 != len(args.template)
    args.template = TNTemplate(args.template)

    return args


def check_channels(channels: List[str], CZI: CziFile2) -> List[str]:
    if channels is None:
        channels = list(CZI.get_channel_names())
    else:
        channels = CZI.select_channels(channels)
        if 0 == len(channels):
            logging.error("None of the specified channels was found.")
            sys.exit()
        logging.info(f"Converting only the following channels: {channels}")
    return channels


def check_argument_compatibility(
    args: argparse.Namespace, CZI: CziFile2
) -> argparse.Namespace:
    if not args.template.can_export_fields(CZI.field_count(), args.fields):
        logging.critical(
            "when exporting more than 1 field, the template "
            + f"must include the {TNTFields.SERIES_ID} seed. "
            + f"Got '{args.template.template}' instead."
        )
        sys.exit()

    args.channels = check_channels(args.channels, CZI)

    if not args.template.can_export_channels(CZI.channel_count(), args.channels):
        logging.critical(
            "when exporting more than 1 channel, the template "
            + f"must include either {TNTFields.CHANNEL_ID} or "
            + f"{TNTFields.CHANNEL_NAME} seeds. "
            + f"Got '{args.template.template}' instead."
        )
        sys.exit()

    if args.fields is None:
        args.fields = range(CZI.field_count())

    return args


def field_generator(
    args: argparse.Namespace, CZI: CziFile2
) -> Iterable[Tuple[np.ndarray, str]]:
    for field_id in args.fields:
        if field_id - 1 >= CZI.field_count():
            logging.warning(
                f"Skipped field #{field_id} "
                + "(from specified field range, "
                + "not available in czi file)."
            )
            continue
        for yieldedValue in CZI.get_channel_pixels(args, field_id - 1):
            channel_pixels, channel_id = yieldedValue
            if not list(CZI.get_channel_names())[channel_id] in args.channels:
                continue
            yield (
                channel_pixels,
                CZI.get_tiff_path(args.template, channel_id, field_id - 1),
            )


def convert_to_tiff(args: argparse.Namespace, CZI: CziFile2) -> None:
    export_total = float("inf")
    if args.fields is not None and args.channels is not None:
        export_total = len(args.fields) * len(args.channels)
    elif args.fields is not None:
        export_total = len(args.fields)
    elif args.channels is not None:
        export_total = len(args.channels)
    export_total = min(CZI.field_count() * CZI.channel_count(), export_total)
    for (OI, opath) in tqdm(field_generator(args, CZI), total=export_total):
        imt.save_tiff(
            os.path.join(args.outdir, opath),
            OI,
            args.doCompress,
            dtype=imt.get_dtype(OI.max()),
            bundle_axes="TZYX",
            resolution=(
                1e-6 / CZI.get_axis_resolution("X"),
                1e-6 / CZI.get_axis_resolution("Y"),
            ),
            inMicrons=True,
            ResolutionZ=CZI.get_axis_resolution("Z") * 1e6,
        )


def run(args: argparse.Namespace) -> None:
    CZI = CziFile2(args.input)
    assert not CZI.isLive(), "time-course conversion images not implemented."
    CZI.log_details()
    if args.dry:
        sys.exit()

    args = check_argument_compatibility(args, CZI)

    logging.info(f"Output directory: '{args.outdir}'")
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    CZI.squeeze_axes("SCZYX")
    reordered_axes = "CZYX" if 1 == CZI.field_count() else "SCZYX"
    CZI.reorder_axes(reordered_axes)

    if args.fields is not None:
        args.fields = list(args.fields)
        logging.info(
            "Converting only the following fields: " + f"{[x for x in args.fields]}"
        )

    convert_to_tiff(args, CZI)
