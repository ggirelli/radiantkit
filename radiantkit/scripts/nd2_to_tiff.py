"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
import logging
import numpy as np  # type: ignore
import os
import pims  # type: ignore
from radiantkit import __version__
from radiantkit.conversion import ND2Reader2
from radiantkit.exception import enable_rich_exceptions
import radiantkit.image as imt
from radiantkit.io import add_log_file_handler
import radiantkit.stat as stat
from radiantkit.string import MultiRange
from radiantkit.string import TIFFNameTemplateFields as TNTFields
from radiantkit.string import TIFFNameTemplate as TNTemplate
from rich.progress import track  # type: ignore
import sys
from typing import List, Optional, Tuple, Union


@enable_rich_exceptions
def init_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1],
        description=f"""
Convert a nd2 file into single channel tiff images.

In the case of 3+D images, the script also checks for consistent deltaZ distance across
consecutive 2D slices (i.e., dZ). If the distance is consitent, it is used to set the
tiff image dZ metadata. Otherwise, the script tries to guess the correct dZ and reports
it in the log. If the reported dZ is wrong, please enforce the correct one using the -Z
option. If a correct dZ cannot be automatically guessed, the field of view is skipped
and a warning is issued to the user. Use the --fields and -Z options to convert the
skipped field(s).

# File naming

The output tiff file names follow the specified template (-T). A template is a string
including a series of "seeds" that are replaced by the corresponding values when writing
the output file. Available seeds are:
{TNTFields.CHANNEL_NAME} : channel name, lower-cased.
{TNTFields.CHANNEL_ID} : channel ID (number).
{TNTFields.SERIES_ID} : series ID (number).
{TNTFields.DIMENSIONS} : number of dimensions, followed by "D".
{TNTFields.AXES_ORDER} : axes order (e.g., "TZYX").
Leading 0s are added up to 3 digits to any ID seed.

The default template is "{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}". Hence, when
writing the 3rd series of the "a488" channel, the output file name would be:
"a488_003.tiff".

Please, remember to escape the "$" when running from command line if using double
quotes, i.e., "\\$". Alternatively, use single quotes, i.e., '$'.""",
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
        help="""Convert only fields of view specified as when printing a set
        of pages. Omit if all fields should be converted. E.g., '1-2,5,8-9'.""",
        default=None,
    )
    parser.add_argument(
        "--channels",
        metavar="STRING",
        type=str,
        help="""Convert only specified channels. Specified as space-separated
        channel names. Omit if all channels should be converted.
        E.g., 'dapi cy5 a488'.""",
        default=None,
        nargs="+",
    )

    parser.add_argument(
        "--version", action="version", version=f"{sys.argv[0]} {__version__}"
    )

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument(
        "--deltaZ",
        "-Z",
        type=float,
        metavar="FLOAT",
        help="""If provided (in um), the script does not check delta Z
        consistency and instead uses the provided one.""",
        default=None,
    )
    advanced.add_argument(
        "--template",
        "-T",
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
        help="""Write compressed TIFF as output. Useful especially for binary or
        low-depth (e.g. labeled) images.""",
    )
    advanced.add_argument(
        "-n",
        "--dry-run",
        action="store_const",
        dest="dry",
        const=True,
        default=False,
        help="Describe input data and stop (nothing is converted).",
    )

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


@enable_rich_exceptions
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


def get_resolution_Z_mode(z_steps: List[float], field_id: int) -> float:
    z_mode = stat.get_hist_mode(stat.list_to_hist(z_steps))
    if np.isnan(z_mode):
        logging.error(
            " ".join(
                [
                    f"Z resolution is not constant in field #{field_id+1}:",
                    f"{set(z_steps)}."
                    f"Cannot automatically identify a delta Z for field #{field_id+1}.",
                    "Skipping this field.",
                    "Please enforce a delta Z manually using the --deltaZ option.",
                ]
            )
        )
        return np.nan
    logging.info(
        " ".join(
            [
                f"Z resolution is not constant in field #{field_id+1}:",
                f"{set(z_steps)}.",
                f"Using a Z resolution of {z_mode} um.",
            ]
        )
    )
    return z_mode


def get_resolution_Z(nd2_image: ND2Reader2, field_id: int, enforce: float) -> float:
    if not nd2_image.is3D():
        return 0.0

    if enforce is not None:
        return enforce

    z_steps = nd2_image.get_field_resolutionZ(field_id)
    if 1 < len(set(z_steps)):
        return get_resolution_Z_mode(z_steps, field_id)
    return z_steps[0]


def get_field(nd2_image: ND2Reader2, field_id: int, channel_id: int) -> np.ndarray:
    slicing: List[Union[slice, int]] = []
    field_of_view = nd2_image[field_id]
    for a in nd2_image.bundle_axes:
        axis_size = field_of_view.shape[nd2_image.bundle_axes.index(a)]
        if "c" == a:
            assert channel_id < axis_size
            slicing.append(channel_id)
        else:
            slicing.append(slice(0, axis_size))
    return field_of_view[tuple(slicing)]


def export_single_channel(
    field_of_view: pims.frame.Frame,
    opath: str,
    xy_resolution: Tuple[float, float] = (0.0, 0.0),
    z_resolution: float = 0.0,
    compress: bool = False,
    bundle_axes: str = "ZYX",
) -> None:
    x_pixels_per_um = 0 if 0 == xy_resolution[0] else 1 / xy_resolution[0]
    y_pixels_per_um = 0 if 0 == xy_resolution[1] else 1 / xy_resolution[1]
    imt.save_tiff(
        opath,
        field_of_view,
        compress,
        bundle_axes=bundle_axes,
        inMicrons=True,
        z_resolution=z_resolution,
        resolution=(x_pixels_per_um, y_pixels_per_um, None),
    )


def export_multiple_channels(
    nd2_image: ND2Reader2,
    field_id: int,
    args: argparse.Namespace,
    channels: Optional[List[str]] = None,
    z_resolution: float = 0.0,
) -> None:
    channels = list(nd2_image.get_channel_names()) if channels is None else channels
    channels = nd2_image.select_channels(channels)
    bundle_axes = nd2_image.bundle_axes.copy()
    if nd2_image.has_multi_channels():
        bundle_axes.pop(bundle_axes.index("c"))
    bundle_axes = "".join(bundle_axes).upper()
    for channel_id in range(nd2_image.channel_count()):
        channel_name = nd2_image.metadata["channels"][channel_id].lower()
        if channel_name in channels:
            imt.save_tiff(
                os.path.join(
                    args.outdir,
                    nd2_image.get_tiff_path(args.template, channel_id, field_id),
                ),
                get_field(nd2_image, field_id, channel_id),
                args.doCompress,
                bundle_axes=bundle_axes,
                inMicrons=True,
                z_resolution=z_resolution,
                resolution=(
                    0 if 0 == nd2_image.xy_resolution else 1 / nd2_image.xy_resolution,
                    0 if 0 == nd2_image.xy_resolution else 1 / nd2_image.xy_resolution,
                    None,
                ),
            )


def export_field(
    nd2_image: ND2Reader2,
    field_id: int,
    args: argparse.Namespace,
    channels: Optional[List[str]] = None,
) -> None:
    z_resolution = get_resolution_Z(nd2_image, field_id, args.deltaZ)
    if np.isnan(z_resolution):
        return

    try:
        export_multiple_channels(nd2_image, field_id, args, channels, z_resolution)
    except ValueError as e:
        if "could not broadcast input array from shape" in e.args[0]:
            logging.error(
                " ".join(
                    [
                        f"corrupted file raised {type(e).__name__}.",
                        "At least one frame has mismatching shape.",
                    ]
                )
            )
            logging.critical(f"{e.args[0]}")
            sys.exit()
        raise e


def convert_to_tiff(args: argparse.Namespace, nd2_image: ND2Reader2) -> None:
    if "v" in nd2_image.axes:
        nd2_image.iter_axes = "v"
    nd2_image.set_axes_for_bundling()

    if args.fields is not None:
        args.fields = list(args.fields)
        logging.info(
            "Converting only the following fields: " + f"{[x for x in args.fields]}"
        )
        field_list = args.fields
    else:
        field_list = range(1, nd2_image.field_count() + 1)
    field_generator = track(field_list, description="Converting field")

    for field_id in field_generator:
        if field_id - 1 >= nd2_image.field_count():
            logging.warning(
                "".join(
                    [
                        f"Skipped field #{field_id}(from specified ",
                        "field range, not available in nd2 file).",
                    ]
                )
            )
        else:
            export_field(nd2_image, field_id - 1, args, args.channels)


def check_channel_selection(args: argparse.Namespace, nd2_image: ND2Reader2):
    if args.channels is not None:
        channels = nd2_image.select_channels(args.channels)
        assert 0 != len(channels), "none of the specified channels was found."


def check_arguments(
    args: argparse.Namespace, nd2_image: ND2Reader2
) -> argparse.Namespace:
    assert not nd2_image.isLive(), "time-course conversion images not implemented."

    assert args.template.can_export_fields(
        nd2_image.field_count(), args.fields
    ), "".join(
        [
            "when exporting more than 1 field, the template ",
            f"must include the {TNTFields.SERIES_ID} seed. ",
            f"Got '{args.template.template}' instead.",
        ]
    )

    check_channel_selection(args, nd2_image)

    assert args.template.can_export_channels(
        nd2_image.channel_count(), args.channels
    ), "".join(
        [
            "when exporting more than 1 channel, the template ",
            f"must include either {TNTFields.CHANNEL_ID} or ",
            f"{TNTFields.CHANNEL_NAME} seeds. ",
            f"Got '{args.template.template}' instead.",
        ]
    )

    if args.fields is not None:
        if np.min(args.fields) > nd2_image.field_count():
            logging.warning(
                "".join(
                    [
                        "Skipped all available fields ",
                        "(not included in specified field range.",
                    ]
                )
            )

    if args.deltaZ is not None:
        logging.info(f"Enforcing a deltaZ of {args.deltaZ:.3f} um.")
    elif 1 < len(nd2_image.z_resolution):
        logging.warning(
            " ".join(
                [
                    "Z resolution is not constant across fields.",
                    "It will be automagically identified, field-by-field.",
                    "If the automatic Z resolution reported in the log is wrong,",
                    "please enforce the correct one using the --deltaZ option.",
                ]
            )
        )
        logging.debug(f"Z steps histogram: {nd2_image.z_resolution}.")

    return args


@enable_rich_exceptions
def run(args: argparse.Namespace) -> None:
    nd2_image = ND2Reader2(args.input)
    if args.dry:
        nd2_image.log_details()
        sys.exit()

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
    add_log_file_handler(os.path.join(args.outdir, "nd2_to_tiff.log.txt"))

    nd2_image.log_details()
    args = check_arguments(args, nd2_image)

    logging.info(f"Output directory: '{args.outdir}'")

    convert_to_tiff(args, nd2_image)

    logging.info("Done. :thumbs_up: :smiley:")
