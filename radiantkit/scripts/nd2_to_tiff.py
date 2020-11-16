"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
import logging
import numpy as np  # type: ignore
import os
import pims  # type: ignore
from radiantkit.const import __version__
from radiantkit.conversion import ND2Reader2
from radiantkit.exception import enable_rich_assert
import radiantkit.image as imt
from radiantkit.io import add_log_file_handler
from radiantkit.string import MultiRange
from radiantkit.string import TIFFNameTemplateFields as TNTFields
from radiantkit.string import TIFFNameTemplate as TNTemplate
from rich.logging import RichHandler  # type: ignore
from rich.progress import track  # type: ignore
import sys
from typing import List, Optional, Tuple

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
    Convert a nd2 file into single channel tiff images.

    In the case of 3+D images, the script also checks for consistent deltaZ
    distance across consecutive 2D slices (i.e., dZ). If the distance is consitent,
    it is used to set the tiff image dZ metadata. Otherwise, the script stops. Use
    the -Z argument to disable this check and provide a single dZ value to be used.

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
        help="Convert a nd2 file into single channel tiff images.",
    )

    parser.add_argument("input", type=str, help="""Path to the nd2 file to convert.""")

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
        "--deltaZ",
        type=float,
        metavar="FLOAT",
        help="""If provided (in um), the script does not check delta Z
        consistency and instead uses the provided one.""",
        default=None,
    )
    advanced.add_argument(
        "--template",
        metavar="STRING",
        type=str,
        help=f"""Template for output file name. See main description for more
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
        args.fields = list(MultiRange(args.fields))

    if args.channels is not None:
        args.channels = [c.lower() for c in args.channels]

    assert 0 != len(args.template)
    args.template = TNTemplate(args.template)

    return args


def get_resolution_Z(nd2_image: ND2Reader2, field_id: int, enforce: float) -> float:
    if not nd2_image.is3D():
        return 0.0

    if enforce is not None:
        return enforce

    resolutionZ = nd2_image.get_field_resolutionZ(field_id)
    assert 1 == len(
        resolutionZ
    ), f"Z resolution is not constant {resolutionZ} in field {field_id}."
    return list(resolutionZ)[0]


def get_field_from_2d_nd2(
    nd2_image: ND2Reader2, field_id: int, channel_id: int
) -> np.ndarray:
    return nd2_image[field_id][:, :, channel_id].astype(nd2_image.dtype)


def get_field_from_3d_nd2(
    nd2_image: ND2Reader2, field_id: int, channel_id: int
) -> np.ndarray:
    return nd2_image[field_id][:, :, :, channel_id].astype(nd2_image.dtype)


get_field_fun = {2: get_field_from_2d_nd2, 3: get_field_from_3d_nd2}


def export_single_channel(
    field_of_view: pims.frame.Frame,
    opath: str,
    resolution: Tuple[Tuple[float, float], float] = ((0.0, 0.0), 0.0),
    compress: bool = False,
) -> None:
    imt.save_tiff(
        opath,
        field_of_view,
        compress,
        resolution=resolution[0],
        inMicrons=True,
        ResolutionZ=resolution[1],
    )


def export_multiple_channels(
    nd2_image: ND2Reader2,
    field_id: int,
    args: argparse.Namespace,
    channels: Optional[List[str]] = None,
    resolutionZ: float = 0.0,
) -> None:
    channels = list(nd2_image.get_channel_names()) if channels is None else channels
    get_field = get_field_fun[3] if nd2_image.is3D() else get_field_fun[2]
    channels = nd2_image.select_channels(channels)
    for channel_id in range(nd2_image[field_id].shape[3]):
        channel_name = nd2_image.metadata["channels"][channel_id].lower()
        if channel_name in channels:
            export_single_channel(
                get_field(nd2_image, field_id, channel_id),
                os.path.join(
                    args.outdir,
                    nd2_image.get_tiff_path(args.template, channel_id, field_id),
                ),
                (
                    (nd2_image.xy_resolution, nd2_image.xy_resolution),
                    resolutionZ,
                ),
                args.doCompress,
            )


def export_field(
    nd2_image: ND2Reader2,
    field_id: int,
    args: argparse.Namespace,
    channels: Optional[List[str]] = None,
) -> None:
    resolutionZ = get_resolution_Z(nd2_image, field_id, args.deltaZ)
    try:
        export_multiple_channels(nd2_image, field_id, args, channels, resolutionZ)
    except ValueError as e:
        if "could not broadcast input array from shape" in e.args[0]:
            logging.error(
                f"corrupted file raised {type(e).__name__}. "
                + "At least one frame has mismatching shape."
            )
            logging.critical(f"{e.args[0]}")
            sys.exit()
        raise e


def convert_to_tiff(args: argparse.Namespace, nd2_image: ND2Reader2) -> None:
    nd2_image.iter_axes = "v"
    nd2_image.set_axes_for_bundling()

    if args.fields is not None:
        args.fields = list(args.fields)
        logging.info(
            "Converting only the following fields: " + f"{[x for x in args.fields]}"
        )
        field_list = args.fields
    else:
        field_list = range(1, nd2_image.sizes["v"] + 1)
    field_generator = track(field_list, description="Converting field")

    for field_id in field_generator:
        if field_id - 1 >= nd2_image.field_count():
            logging.warning(
                f"Skipped field #{field_id}(from specified "
                + "field range, not available in nd2 file)."
            )
        else:
            export_field(nd2_image, field_id - 1, args, args.channels)


def check_argument_compatibility(
    args: argparse.Namespace, nd2_image: ND2Reader2
) -> argparse.Namespace:
    assert not nd2_image.isLive(), "time-course conversion images not implemented."

    assert args.template.can_export_fields(nd2_image.field_count(), args.fields), (
        "when exporting more than 1 field, the template "
        + f"must include the {TNTFields.SERIES_ID} seed. "
        + f"Got '{args.template.template}' instead."
    )

    if args.channels is not None:
        channels = nd2_image.select_channels(args.channels)
        assert 0 != len(channels), "none of the specified channels was found."

    assert args.template.can_export_channels(
        nd2_image.channel_count(), args.channels
    ), (
        "when exporting more than 1 channel, the template "
        + f"must include either {TNTFields.CHANNEL_ID} or "
        + f"{TNTFields.CHANNEL_NAME} seeds. "
        + f"Got '{args.template.template}' instead."
    )

    if args.fields is not None:
        if np.min(args.fields) > nd2_image.field_count():
            logging.warning(
                "Skipped all available fields "
                + "(not included in specified field range."
            )

    if args.deltaZ is not None:
        logging.info(f"Enforcing a deltaZ of {args.deltaZ:.3f} um.")
    else:
        assert 1 == len(
            nd2_image.z_resolution
        ), f"Z resolution is not constant {nd2_image.z_resolution}."

    return args


@enable_rich_assert
def run(args: argparse.Namespace) -> None:
    nd2_image = ND2Reader2(args.input)
    if args.dry:
        nd2_image.log_details()
        sys.exit()

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
    add_log_file_handler(os.path.join(args.outdir, "nd2_to_tiff.log.txt"))

    nd2_image.log_details()
    args = check_argument_compatibility(args, nd2_image)

    logging.info(f"Output directory: '{args.outdir}'")

    convert_to_tiff(args, nd2_image)

    logging.info("Done. :thumbs_up: :smiley:")
