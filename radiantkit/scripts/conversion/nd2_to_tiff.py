"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import logging
import sys
from os import mkdir
from os.path import isdir, isfile
from os.path import join as join_paths
from typing import List, Optional, Union

import click  # type: ignore
import numpy as np  # type: ignore
from rich.progress import track  # type: ignore

from radiantkit.const import CONTEXT_SETTINGS, DEFAULT_INPUT_RE, SCRITPS_INPUT_HELP
from radiantkit.conversion import ND2Reader2
from radiantkit.image import save_tiff
from radiantkit.io import add_log_file_handler
from radiantkit.scripts import options
from radiantkit.scripts.conversion.common import (
    CONVERSION_TEMPLATE_LONG_HELP_STRING,
    convert_folder,
)
from radiantkit.scripts.conversion.settings import ConversionSettings
from radiantkit.string import TIFFNameTemplateFields as TNTFields


@click.command(
    name="nd2_to_tiff",
    context_settings=CONTEXT_SETTINGS,
    help=f"""
Convert ND2 file(s) into TIFF.

{SCRITPS_INPUT_HELP}""",
)
@click.argument("input_paths", metavar="INPUT", nargs=-1, type=click.Path(exists=True))
@click.option("--info", is_flag=True, help="Show INPUT details and stop.")
@click.option("--list", is_flag=True, help="List INPUT files and stop.")
@click.option("--long-help", is_flag=True, help="Show long help page and stop.")
@click.option(
    "--output",
    "-o",
    "output_dirpath",
    type=click.Path(exists=False),
    help="Output folder path. Defaults to INPUT without extension.",
)
@click.option(
    "--fields",
    "-F",
    type=click.STRING,
    help="""\b
    Limit conversion to the specified fields of view.
    E.g., '1-3,5' converts fields: 1,2,3,5.""",
)
@click.option(
    "--channels",
    "-C",
    type=click.STRING,
    help="""\b
    Limit conversion to the specified channels.
    Separate multiple channels with a comma, e.g., 'dapi,a647'.
    """,
)
@click.option(
    "--dz",
    "-Z",
    type=click.FLOAT,
    help="Delta Z in um. Use when the script fails to recognize the correct value.",
)
@options.input_regexp(DEFAULT_INPUT_RE["nd2"])
@click.option(
    "--template",
    "-T",
    type=click.STRING,
    help=f"""\b
    Output file name template. See --long-help for more details.
    Default: '{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}'""",
    default=f"{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}",
)
@click.option(
    "--compress",
    is_flag=True,
    help="Compress output TIFF files. Useful with low bit-depth output.",
)
def run(
    input_paths: List[str],
    info: bool,
    list: bool,
    long_help: bool,
    output_dirpath: Optional[str],
    fields: Optional[str],
    channels: Optional[str],
    dz: Optional[float],
    input_re: str,
    template: str,
    compress: bool,
) -> None:
    if long_help:
        print_long_help()
        return

    settings = ConversionSettings(set(input_paths), input_re, template)
    settings.output_dirpath = output_dirpath
    settings.set_fields(fields)
    settings.set_channels(channels)
    settings.dz = dz
    settings.compress = compress
    settings.just_info = info
    settings.just_list = list

    logging.info(f"Input file(s): {settings.input_paths}")
    for path in settings.input_paths:
        if isdir(path):
            logging.info(f"Looking into folder: {path}")
            convert_folder(settings, path, convert_file)
        else:
            convert_file(settings, path, settings.output_dirpath)
        logging.info("Done. :thumbs_up: :smiley:")


def print_long_help() -> None:
    print(
        f"""
# Converting 3+D ND2 files to TIFF

In the case of 3+D images, radiant checks for a consistent dZ across consecutive 2D
slices and saves it in the output TIFF metadata. In case of inconsistent dZ, the script
tries to guess the correct value, then report it and proceed. If the reported dZ is
wrong, please enforce the correct one using the -Z option.

If a dZ cannot be automatically guessed, the affected field is skipped and the user is
warned. Use the -F and -Z options to convert the skipped field(s).

{ CONVERSION_TEMPLATE_LONG_HELP_STRING }
"""
    )


def convert_file(
    args: ConversionSettings, path: str, output_dirpath: Optional[str] = None
) -> None:
    assert isfile(path), f"input file not found: {path}"
    if args.just_list:
        return

    nd2_image = ND2Reader2(path)
    if args.just_info:
        nd2_image.log_details()
        logging.info("")
        return
    else:
        logging.info(f"Working on file '{path}'.")

    output_dirpath = args.get_output_dirpath_for_single_file(path, output_dirpath)
    if not isdir(output_dirpath):
        mkdir(output_dirpath)
    add_log_file_handler(join_paths(output_dirpath, "nd2_to_tiff.log.txt"))

    nd2_image.log_details()
    args.is_nd2_compatible(nd2_image)
    logging.info(f"Output directory: '{output_dirpath}'")

    nd2_image.set_iter_axes("v")
    nd2_image.set_axes_for_bundling()

    for field_id in track(
        args.select_fields(args.fields, nd2_image), description="Converting field"
    ):
        export_field(nd2_image, field_id, output_dirpath, args)


def export_field(
    nd2_image: ND2Reader2,
    field_id: int,
    output_dirpath: str,
    args: ConversionSettings,
) -> None:
    dz = nd2_image.get_dz(field_id, args.dz)
    if np.isnan(dz):
        return

    try:
        export_channels(nd2_image, field_id, output_dirpath, args, dz)
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


def export_channels(
    nd2_image: ND2Reader2,
    field_id: int,
    outdir: str,
    args,
    z_resolution: float = 0.0,
) -> None:
    channels = nd2_image.select_channels(
        list(nd2_image.get_channel_names()) if args.channels is None else args.channels
    )

    bundle_axes = nd2_image.bundle_axes.copy()
    if nd2_image.has_multi_channels():
        bundle_axes.pop(bundle_axes.index("c"))
    bundle_axes = "".join(bundle_axes).upper()

    for channel_id in range(nd2_image.channel_count()):
        channel_name = nd2_image.metadata["channels"][channel_id].lower()
        if channel_name in channels:
            save_tiff(
                join_paths(
                    outdir,
                    nd2_image.get_tiff_path(args.template, channel_id, field_id),
                ),
                get_field(nd2_image, field_id, channel_id),
                args.compress,
                bundle_axes=bundle_axes,
                inMicrons=True,
                z_resolution=z_resolution,
                resolution=(
                    0 if 0 == nd2_image.xy_resolution else 1 / nd2_image.xy_resolution,
                    0 if nd2_image.xy_resolution == 0 else 1 / nd2_image.xy_resolution,
                    None,
                ),
            )


def get_field(nd2_image: ND2Reader2, field_id: int, channel_id: int) -> np.ndarray:
    slicing: List[Union[slice, int]] = []
    field_of_view = nd2_image[field_id]
    for a in nd2_image.bundle_axes:
        axis_size = field_of_view.shape[nd2_image.bundle_axes.index(a)]
        if a == "c":
            assert channel_id < axis_size
            slicing.append(channel_id)
        else:
            slicing.append(slice(0, axis_size))
    return field_of_view[tuple(slicing)]
