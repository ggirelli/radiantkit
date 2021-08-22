"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from radiantkit.scripts import conversion, tiff
from radiantkit.scripts import select_nuclei
from radiantkit.scripts import measure_objects, export_objects
from radiantkit.scripts import radial_population, radial_object
from radiantkit.scripts import radial_trajectory, radial_points
from radiantkit.scripts import tiff_desplit, tiff_split
from radiantkit.scripts import tiff_segment
from radiantkit.scripts import pipeline, report

import logging
from rich.logging import RichHandler  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)

__all__ = [
    "conversion",
    "tiff",
    "select_nuclei",
    "measure_objects",
    "export_objects",
    "radial_population",
    "radial_object",
    "radial_trajectory",
    "radial_points",
    "tiff_desplit",
    "tiff_split",
    "tiff_segment",
    "pipeline",
    "report",
]
