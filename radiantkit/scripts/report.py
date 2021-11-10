"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore

from radiantkit.const import CONTEXT_SETTINGS


@click.command(
    name="report",
    context_settings=CONTEXT_SETTINGS,
    help="""Generate radiant HTML report.""",
)
def run() -> None:
    pass
