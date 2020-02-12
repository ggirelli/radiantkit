'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from enum import Enum
from radiantkit.image import ImageBase, ImageBinary, Image
from typing import Optional


class CenterType(Enum):
    CENTER_OF_MASS = "center of mass"
    CENTROID = "centroid"
    MAX = "max"
    QUANTILE = "quantile"


class RadialDistanceCalculator(object):
    _flatten_axes: Optional[str] = None
    _center_type: CenterType
    _quantile: Optional[float] = None

    def __init__(self, flatten: Optional[str] = None,
                 center_type: CenterType = CenterType.QUANTILE,
                 q: Optional[float] = None):
        super(RadialDistanceCalculator, self).__init__()

        if flatten is not None:
            assert all([a in ImageBase._ALLOWED_AXES for a in flatten])
            self._flatten_axes = flatten

        if center_type in CenterType:
            self._center_type = center_type
            if CenterType.QUANTILE == self._center_type:
                self.__set_quantile(q)

    def __set_quantile(self, q: Optional[float]) -> None:
        raise NotImplementedError

    def get_quantile(self, axes: Optional[str] = None) -> float:
        if self._quantile is None:
            assert axes is not None
            return 10**(-len(axes))
        return self._quantile

    def calc(self, B: ImageBinary, C: Optional[Image] = None) -> None:
        raise NotImplementedError
