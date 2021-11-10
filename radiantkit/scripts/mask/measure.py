"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore

from radiantkit.const import CONTEXT_SETTINGS


@click.command(
    name="measure",
    context_settings=CONTEXT_SETTINGS,
    help="""Measure segmented objects.""",
)
def run() -> None:
    pass
