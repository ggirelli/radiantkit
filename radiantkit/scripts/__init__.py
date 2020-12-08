"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from radiantkit.scripts import argtools
from radiantkit.scripts import config
from radiantkit.scripts import czi_to_tiff, nd2_to_tiff
from radiantkit.scripts import select_nuclei
from radiantkit.scripts import measure_objects, export_objects
from radiantkit.scripts import radial_population, radial_object
from radiantkit.scripts import radial_trajectory, radial_points
from radiantkit.scripts import tiff_desplit, tiff_split
from radiantkit.scripts import tiff_findoof, tiff_segment, tiffcu
from radiantkit.scripts import pipeline, report

import logging
from rich.logging import RichHandler  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

__all__ = [
    "argtools",
    "config",
    "czi_to_tiff",
    "nd2_to_tiff",
    "select_nuclei",
    "measure_objects",
    "export_objects",
    "radial_population",
    "radial_object",
    "radial_trajectory",
    "radial_points",
    "tiff_desplit",
    "tiff_split",
    "tiff_findoof",
    "tiff_segment",
    "tiffcu",
    "pipeline",
    "report",
]
