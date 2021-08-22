"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from radiantkit.scripts import conversion, mask, radial, tiff
from radiantkit.scripts import report

import logging
from rich.logging import RichHandler  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

__all__ = [
    "conversion",
    "mask",
    "radial",
    "tiff",
    "report",
]
