"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import logging

from rich.logging import RichHandler  # type: ignore

from radiantkit.scripts import (
    arguments,
    conversion,
    mask,
    options,
    radial,
    report,
    tiff,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

__all__ = [
    "arguments",
    "options",
    "conversion",
    "mask",
    "radial",
    "tiff",
    "report",
]
