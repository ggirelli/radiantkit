"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore
from radiantkit.const import CONTEXT_SETTINGS


@click.command(
    name="select",
    context_settings=CONTEXT_SETTINGS,
    help="""Select G1 nuclei.""",
)
def run() -> None:
    pass
