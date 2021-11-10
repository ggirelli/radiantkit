"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore

from radiantkit import __version__
from radiantkit.const import CONTEXT_SETTINGS
from radiantkit.scripts.conversion import czi_to_tiff, nd2_to_tiff


@click.group(
    name="convert",
    context_settings=CONTEXT_SETTINGS,
    help="Tools to convert proprietary formats to tiff",
)
@click.version_option(__version__)
def main() -> None:
    pass


main.add_command(czi_to_tiff.run)
main.add_command(nd2_to_tiff.run)
