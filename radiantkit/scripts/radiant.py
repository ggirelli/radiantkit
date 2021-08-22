"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore
from radiantkit import __version__
from radiantkit.const import CONTEXT_SETTINGS
from radiantkit.scripts import conversion, mask, radial, tiff
from radiantkit.scripts import report
import webbrowser
import sys


@click.group(
    name="radiant",
    context_settings=CONTEXT_SETTINGS,
    help=f"""\b
Version:    {__version__}
Author:     Gabriele Girelli
Docs:       http://ggirelli.github.io/radiantkit
Code:       http://github.com/ggirelli/radiantkit

Radial Image Analisys Toolkit (RadIAnTkit) is a Python3.6+ package containing
tools for radial analysis of microscopy image.""",
)
@click.version_option(__version__)
def main() -> None:
    pass


@click.command(
    "_docs",
    help="Open online documentation on your favorite browser.",
)
def open_documentation() -> None:
    webbrowser.open("https://ggirelli.github.io/radiantkit/")
    sys.exit()


main.add_command(open_documentation)
main.add_command(conversion.run.main)
main.add_command(mask.run.main)
main.add_command(radial.run.main)
main.add_command(tiff.run.main)
main.add_command(report.run)
