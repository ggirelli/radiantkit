"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore
from radiantkit.const import CONTEXT_SETTINGS


@click.command(
    name="extract",
    context_settings=CONTEXT_SETTINGS,
    help="""Extract segmented objects.""",
)
def run() -> None:
    pass
