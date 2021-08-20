"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from radiantkit.scripts.conversion import common, run, settings
from radiantkit.scripts.conversion import czi_to_tiff, nd2_to_tiff

import logging
from rich.logging import RichHandler  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

__all__ = [
    "common",
    "run",
    "settings",
    "czi_to_tiff",
    "nd2_to_tiff",
]
