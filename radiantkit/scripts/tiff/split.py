"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore

from radiantkit.const import CONTEXT_SETTINGS


@click.command(
    name="split",
    context_settings=CONTEXT_SETTINGS,
    help="""Split TIFF images.""",
)
def run() -> None:
    pass
