"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore
from joblib import delayed, Parallel  # type: ignore
import logging
from os.path import isdir, isfile
from radiantkit.const import CONTEXT_SETTINGS, DEFAULT_INPUT_RE, SCRITPS_INPUT_HELP
from radiantkit.image import Image
from radiantkit.scripts.tiff.settings import CompressionSettings
from typing import List


@click.command(
    name="compress",
    context_settings=CONTEXT_SETTINGS,
    help=f"""
Compress TIFF images.

{SCRITPS_INPUT_HELP}""",
)
@click.argument("input_paths", metavar="INPUT", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--suffix",
    "-S",
    type=click.STRING,
    help="Suffix for compressed output files.",
    default=".compressed",
)
@click.option(
    "--input-re",
    "-R",
    type=click.STRING,
    metavar="RE",
    help=f"""
    Regexp used to identify input TIFF files.
    Default: {DEFAULT_INPUT_RE['tiff']}""",
    default=DEFAULT_INPUT_RE["tiff"],
)
@click.option(
    "--threads",
    "-T",
    type=click.INT,
    help="Number of threads for parallelization.",
    default=1,
)
def run_compress(
    input_paths: List[str], suffix: str, input_re: str, threads: int
) -> None:
    settings = CompressionSettings(set(input_paths), suffix, input_re)
    settings.threads = threads
    logging.info(f"Input: {settings.input_paths}")

    if 1 == len(settings.input_paths):
        export_image(*next(settings.iterate_input_output()), True)
    else:
        Parallel(n_jobs=settings.threads)(
            delayed(export_image)(
                *io_paths,
                True,
            )
            for io_paths in settings.iterate_input_output()
        )


@click.command(
    name="uncompress",
    context_settings=CONTEXT_SETTINGS,
    help=f"""
Uncompress TIFF images.

{SCRITPS_INPUT_HELP}""",
)
@click.argument("input_paths", metavar="INPUT", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--suffix",
    "-S",
    type=click.STRING,
    help="Suffix for uncompressed output files.",
    default=".uncompressed",
)
@click.option(
    "--input-re",
    "-R",
    type=click.STRING,
    metavar="RE",
    help=f"""
    Regexp used to identify input TIFF files.
    Default: {DEFAULT_INPUT_RE['tiff']}""",
    default=DEFAULT_INPUT_RE["tiff"],
)
@click.option(
    "--threads",
    "-T",
    type=click.INT,
    help="Number of threads for parallelization.",
    default=1,
)
def run_uncompress(
    input_paths: List[str], suffix: str, input_re: str, threads: int
) -> None:
    settings = CompressionSettings(set(input_paths), suffix, input_re)
    settings.threads = threads
    logging.info(f"Input: {settings.input_paths}")

    if 1 == len(settings.input_paths):
        export_image(*next(settings.iterate_input_output()), False)
    else:
        Parallel(n_jobs=settings.threads)(
            delayed(export_image)(
                input_path,
                settings.output_suffix,
                False,
            )
            for input_path in settings.input_paths
        )


def export_image(input_path: str, output_path: str, compress: bool) -> str:
    assert isfile(input_path), f"input file not found: {input_path}"
    assert not isfile(output_path) and not isdir(
        output_path
    ), f"output file already exists: {output_path}"
    Image.from_tiff(input_path).to_tiff(output_path, compressed=compress)
    logging.info(f"{'Compressed' if compress else 'Uncompressed'} '{input_path}'.")
    return output_path
