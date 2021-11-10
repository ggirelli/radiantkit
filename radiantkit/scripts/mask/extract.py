"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from typing import Optional

import click  # type: ignore

from radiantkit.const import CONTEXT_SETTINGS, DEFAULT_INPUT_RE, default_subfolder
from radiantkit.scripts import options


@click.command(
    name="extract",
    context_settings=CONTEXT_SETTINGS,
    help="""Extract segmented objects.""",
)
@click.argument(
    "input_paths",
    metavar="INPUT",
    nargs=-1,
    type=click.Path(exists=True, readable=True, file_okay=True, dir_okay=True),
    required=True,
)
@click.option("--reference", "-r", type=click.STRING, help="Name of reference channel.")
@click.option(
    "--output",
    "-o",
    type=click.Path(exists=True, dir_okay=True, writable=True),
    default=default_subfolder,
    help="Output folder",
)
@options.mask_prefix()
@options.mask_suffix("mask")
@options.is_pre_labeled()
@options.do_not_rescale()
@options.compress_output()
@options.input_regexp((DEFAULT_INPUT_RE["tiff"]))
@options.threads()
@options.agree_to_all()
def run(
    input_paths: str,
    reference: Optional[str] = None,
    output: str = default_subfolder,
    mask_prefix: str = "",
    mask_suffix: str = "mask",
    labeled: bool = False,
    compress: bool = False,
    input_regexp: str = DEFAULT_INPUT_RE["tiff"],
    threads: int = 1,
    agree_to_all: bool = False,
) -> None:
    return
