"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore


def reference():
    return click.argument(
        "--reference", "-r", type=click.STRING, help="Name of reference channel."
    )
