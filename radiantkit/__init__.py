"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from radiantkit import const, exception, output, scripts
from radiantkit import conversion, deconvolution, segmentation
from radiantkit import image, particle, series
from radiantkit import path, plot, stat, string
from radiantkit import report, pipeline

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass

__all__ = [
    "__version__",
    "const",
    "exception",
    "output",
    "scripts",
    "conversion",
    "deconvolution",
    "segmentation",
    "image",
    "particle",
    "series",
    "path",
    "plot",
    "stat",
    "string",
    "report",
    "pipeline",
]
