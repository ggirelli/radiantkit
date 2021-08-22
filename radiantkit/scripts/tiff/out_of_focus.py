"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore
from radiantkit.const import CONTEXT_SETTINGS


@click.command(
    name="oof",
    context_settings=CONTEXT_SETTINGS,
    help="""Find Out-Of-Focus images.""",
)
def run() -> None:
    pass
