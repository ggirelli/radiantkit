"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore
from radiantkit import __version__
from radiantkit.const import CONTEXT_SETTINGS
from radiantkit.scripts.radial import population


@click.group(
    name="radial",
    context_settings=CONTEXT_SETTINGS,
    help="Tools to generate radial profiles.",
)
@click.version_option(__version__)
def main() -> None:
    pass


main.add_command(population.run)
