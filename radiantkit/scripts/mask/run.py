"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore
from radiantkit import __version__
from radiantkit.const import CONTEXT_SETTINGS
from radiantkit.scripts.mask import extract, measure, make, select


@click.group(
    name="mask",
    context_settings=CONTEXT_SETTINGS,
    help="Tools to manipulate masks.",
)
@click.version_option(__version__)
def main() -> None:
    pass


main.add_command(extract.run)
main.add_command(measure.run)
main.add_command(make.run)
main.add_command(select.run)
