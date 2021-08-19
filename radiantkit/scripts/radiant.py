"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click
import radiantkit as ra


@click.group(
    name="radiant",
    context_settings=ra.const.CONTEXT_SETTINGS,
    help=f"""\b
Version:    {ra.__version__}
Author:     Gabriele Girelli
Docs:       http://ggirelli.github.io/radiantkit
Code:       http://github.com/ggirelli/radiantkit

Radial Image Analisys Toolkit (RadIAnTkit) is a Python3.6+ package containing
tools for radial analysis of microscopy image.""",
)
@click.version_option(ra.__version__)
def main():
    pass


main.add_command(ra.scripts.nd2_to_tiff.run)
