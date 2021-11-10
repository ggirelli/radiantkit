"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore

from radiantkit import __version__
from radiantkit.const import CONTEXT_SETTINGS
from radiantkit.scripts.tiff import compress, out_of_focus, split


@click.group(
    name="tiff",
    context_settings=CONTEXT_SETTINGS,
    help="Tools to manipulate TIFF files",
)
@click.version_option(__version__)
def main() -> None:
    pass


main.add_command(compress.run_compress)
main.add_command(compress.run_uncompress)
main.add_command(out_of_focus.run)
main.add_command(split.run)
