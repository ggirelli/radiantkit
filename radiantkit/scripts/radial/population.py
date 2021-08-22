"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore
from radiantkit.const import CONTEXT_SETTINGS


@click.command(
    name="population",
    context_settings=CONTEXT_SETTINGS,
    help="""Generate population averaged radial profiles.""",
)
def run() -> None:
    pass
