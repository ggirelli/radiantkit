"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import logging
import sys
from os import mkdir
from os.path import isdir, isfile
from os.path import join as join_paths
from typing import List, Optional, Set

import click  # type: ignore
from rich.progress import track  # type: ignore

from radiantkit.const import CONTEXT_SETTINGS, DEFAULT_INPUT_RE, SCRITPS_INPUT_HELP
from radiantkit.conversion import CziFile2
from radiantkit.image import get_dtype, save_tiff
from radiantkit.io import add_log_file_handler
from radiantkit.scripts import options
from radiantkit.scripts.conversion.common import (
    CONVERSION_TEMPLATE_LONG_HELP_STRING,
    convert_folder,
)
from radiantkit.scripts.conversion.settings import ConversionSettings
from radiantkit.string import TIFFNameTemplateFields as TNTFields


@click.command(
    name="czi_to_tiff",
    context_settings=CONTEXT_SETTINGS,
    help=f"""
Convert CZI file(s) into TIFF.

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
@options.input_regexp(DEFAULT_INPUT_RE["czi"])
@options.filename_template(f"{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}")
@options.compress_output("Useful with low bit-depth images.")
def run(
    input_paths: List[str],
    info: bool,
    list: bool,
    long_help: bool,
    output_dirpath: Optional[str],
    fields: Optional[str],
    channels: Optional[str],
    dz: Optional[float],
    input_regexp: str,
    template: str,
    compress: bool,
) -> None:
    if long_help:
        print_long_help()
        return

    settings = ConversionSettings(set(input_paths), input_regexp, template)
    settings.output_dirpath = output_dirpath
    settings.set_fields(fields)
    settings.set_channels(channels)
    settings.dz = dz
    settings.compress = compress
    settings.just_info = info
    settings.just_list = list

    logging.info(f"Input: {settings.input_paths}")
    for path in settings.input_paths:
        if isdir(path):
            convert_folder(settings, path, convert_file)
        else:
            convert_file(settings, path, settings.output_dirpath)
        logging.info("Done. :thumbs_up: :smiley:")


def print_long_help() -> None:
    print(CONVERSION_TEMPLATE_LONG_HELP_STRING)


def convert_file(
    args: ConversionSettings, path: str, output_dirpath: Optional[str] = None
) -> None:
    logging.info(f"Working on file '{path}'.")
    assert isfile(path), f"input file not found: {path}"
    if args.just_list:
        return

    czi_image = CziFile2(path)
    if args.just_info:
        czi_image.log_details()
        logging.info("")
        sys.exit()

    output_dirpath = args.get_output_dirpath_for_single_file(path, output_dirpath)
    if not isdir(output_dirpath):
        mkdir(output_dirpath)
    add_log_file_handler(join_paths(output_dirpath, "czi_to_tiff.log.txt"))

    czi_image.log_details()
    args.is_czi_compatible(czi_image)

    logging.info(f"Output directory: '{output_dirpath}'")

    czi_image.squeeze_axes("STCZYX")
    czi_image.reorder_axes("STCZYX")

    selected_fields: Set[int] = args.select_fields(args.fields, czi_image)

    for field_id in track(selected_fields, description="Converting field"):
        export_field(czi_image, field_id, output_dirpath, args)


def export_field(
    czi_image: CziFile2,
    field_id: int,
    output_dirpath: str,
    args: ConversionSettings,
) -> None:
    channels: Set[str] = args.select_channels(args.channels, czi_image)
    field_channels: List[str] = list(czi_image.get_channel_names())
    for channel_pixels, channel_id in czi_image.get_channel_pixels(field_id):
        if field_channels[channel_id] not in channels:
            continue
        save_tiff(
            join_paths(
                output_dirpath,
                czi_image.get_tiff_path(args.template, channel_id, field_id),
            ),
            channel_pixels.astype(get_dtype(int(channel_pixels.max()))),
            args.compress,
            resolution=(
                1e-6 / czi_image.get_axis_resolution("X"),
                1e-6 / czi_image.get_axis_resolution("Y"),
            ),
            inMicrons=True,
            z_resolution=czi_image.get_axis_resolution("Z") * 1e6,
        )
