"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from collections import defaultdict
from enum import Enum
from typing import DefaultDict, Dict, List, Tuple

default_inreg = "".join(
    [
        "^(?P<dw_flag>dw_)?([^\\.]*\\.)?(?P<channel_name>[^/]*)_(?P<series_id>[0-9]+)",
        "(?P<ext>(_cmle)?(\\.[^\\.]*)?\\.tiff?)$",
    ]
)
default_axes = "VTZCYX"

default_pickle = "radiant.pkl"
default_subfolder = "objects"

default_plot_npoints = 200


class ProjectionType(Enum):
    SUM = "SUM_PROJECTION"
    MAX = "MAX_PROJECTION"


class SegmentationType(Enum):
    SUM_PROJECTION = ProjectionType.SUM.value
    MAX_PROJECTION = ProjectionType.MAX.value
    THREED = "3D"

    @staticmethod
    def get_default():
        return SegmentationType.THREED


class MidsectionType(Enum):
    CENTRAL = "CENTRAL"
    LARGEST = "LARGEST"
    MAX_INTENSITY_SUM = "MAX_INTENSITY_SUM"

    @staticmethod
    def get_default():
        return MidsectionType.LARGEST


class LaminaDistanceType(Enum):
    CENTER_MAX = "CENTER_MAX"
    CENTER_TOP_QUANTILE = "CENTER_TOP_QUANTILE"
    DIFFUSION = "DIFFUSION"

    @staticmethod
    def get_default():
        return LaminaDistanceType.CENTER_TOP_QUANTILE


# Types

stub = str
required = bool
filename = str
DirectoryPathList = List[str]
OutputFileDetails = Tuple[filename, required, DirectoryPathList]
OutputFileDirpath = Dict[stub, OutputFileDetails]

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

DEFAULT_INPUT_RE: DefaultDict[str, str] = defaultdict(lambda: ".*")
DEFAULT_INPUT_RE["nd2"] = r"^.*\.nd2$"
DEFAULT_INPUT_RE["czi"] = r"^.*\.czi$"
DEFAULT_INPUT_RE["tiff"] = r"^.*\.tiff?$"

SCRITPS_INPUT_HELP = """
To apply to a single file, provide its path as INPUT. To apply to all files in a folder,
instead, specify the folder path as INPUT. To convert specific files, specify them one
after the other as INPUT."""
