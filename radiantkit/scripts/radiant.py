"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore
from radiantkit import __version__
from radiantkit.const import CONTEXT_SETTINGS
from radiantkit.scripts import conversion


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
def main():
    pass


main.add_command(conversion.run.main)
