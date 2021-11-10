"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from typing import Optional

import click  # type: ignore

from radiantkit.const import CONTEXT_SETTINGS, DEFAULT_INPUT_RE
from radiantkit.scripts.tiff.settings import OutOfFocusSettings


@click.command(
    name="oof",
    context_settings=CONTEXT_SETTINGS,
    help="""Find Out-Of-Focus images.

    Identifies out-of-focus REFERENCE channel fields from TIFF files in INPUT folder, by
    calculating Gradient Of Magnitude (or Sum Of Intensity, change with --mode) and
    checking if the maxima point over Z-dimension falls within the middle range of the
    stack (defined with --range).""",
)
@click.argument(
    "input_dirpath",
    metavar="INPUT",
    type=click.Path(exists=True),
)
@click.argument(
    "reference",
    type=click.STRING,
)
@click.option(
    "--output",
    "-o",
    "output_dirpath",
    type=click.Path(exists=False),
    help="Output folder path. Default: out-of-focus in INPUT folder.",
)
@click.option(
    "--mode",
    type=click.Choice(["SOI", "GOM"], case_sensitive=False),
    help="""\b
    Mode of out-of-focus field identification.
    SOI: Sum Of Intensity.
    GOM: Gradient Of Magnitude""",
    default="GOM",
)
@click.option(
    "--fraction",
    "-F",
    type=click.FLOAT,
    help="""Fraction of the stack (centered on the middle)
    for in-focus fields identification. Default: .5""",
    default=0.5,
)
@click.option(
    "--input-re",
    "-R",
    type=click.STRING,
    metavar="RE",
    help=f"""
    Regexp used to identify input TIFF files.
    Default: {DEFAULT_INPUT_RE['tiff_with_fields']}""",
    default=DEFAULT_INPUT_RE["tiff_with_fields"],
)
@click.option(
    "--rename/--dry-run",
    "-R/-N",
    default=True,
    help="""Whether to rename out-of-focus field files by adding the suffix.
    Default: rename.""",
)
@click.option(
    "--suffix",
    "-S",
    type=click.STRING,
    help="Suffix for out-of-focus fields renaming. Default: .oof.old",
    default=".oof.old",
)
def run(
    input_dirpath: str,
    reference: str,
    output_dirpath: Optional[str],
    mode: str,
    fraction: float,
    input_re: str,
    rename: bool,
    suffix: str,
) -> None:
    settings = OutOfFocusSettings(input_dirpath, reference, input_re)
    print(settings)
