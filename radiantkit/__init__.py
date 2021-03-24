"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from radiantkit import const, argtools, exception, output
from radiantkit import conversion, deconvolution, segmentation
from radiantkit import image, particle, series
from radiantkit import path, stat, string
from radiantkit import scripts
from radiantkit import report, plot, pipeline

from importlib.metadata import version

try:
    __version__ = version(__name__)
except Exception as e:
    raise e

__all__ = [
    "__version__",
    "const",
    "argtools",
    "exception",
    "output",
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
    "scripts",
    "report",
    "pipeline",
]
