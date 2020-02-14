'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

__version__ = "0.0.1"

from enum import Enum

default_inreg = ("^([^\\.]*\\.)?(?P<channel_name>[^/]*)_(?P<series_id>[0-9]+)"
                 + "(?P<ext>(_cmle)?(\\.[^\\.]*)?\\.tiff?)$")


class ProjectionType(Enum):
    SUM = 'SUM_PROJECTION'
    MAX = 'MAX_PROJECTION'


class SegmentationType(Enum):
    SUM_PROJECTION = ProjectionType.SUM.value
    MAX_PROJECTION = ProjectionType.MAX.value
    THREED = '3D'
    @staticmethod
    def get_default():
        return SegmentationType.THREED


class MidsectionType(Enum):
    CENTRAL = 'CENTRAL'
    LARGEST = 'LARGEST'
    MAX_INTENSITY_SUM = 'MAX_INTENSITY_SUM'
    @staticmethod
    def get_default():
        return MidsectionType.LARGEST


class LaminaDistanceType(Enum):
    CENTER_MAX = 'CENTER_MAX'
    CENTER_TOP_QUANTILE = 'CENTER_TOP_QUANTILE'
    DIFFUSION = 'DIFFUSION'
    @staticmethod
    def get_default():
        return LaminaDistanceType.CENTER_TOP_QUANTILE
