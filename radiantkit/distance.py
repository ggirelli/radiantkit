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

    def __init__(self, bundle_axes: Optional[str] = None,
                 center_type: CenterType = CenterType.QUANTILE,
                 q: Optional[float] = None):
        super(RadialDistanceCalculator, self).__init__()

        if bundle_axes is not None:
            assert all([a in ImageBase._ALLOWED_AXES for a in bundle_axes])
            self._flatten_axes = bundle_axes

        if center_type in CenterType:
            self._center_type = center_type
            if CenterType.QUANTILE == self._center_type:
                self.__set_quantile(q)

    def __set_quantile(self, q: Optional[float]) -> None:
        if q is not None:
            assert q > 0 and q <= 1
            self._quantile = q

    def quantile(self, img: Optional[ImageBase] = None) -> float:
        if self._quantile is None:
            assert img is not None, (
                "either set a quantile manually or provide an image")
            return 10**(-len(img.axes))
        return self._quantile

    def __calc_contour_dist(self, B: ImageBinary) -> Image:
        raise NotImplementedError

    def __calc_center_of_mass(self, contour_dist: Image,
                              C: Optional[Image] = None, *args, **kwargs
                              ) -> Image:
        assert C is not None, (
            "'center of mass' center definition requires a grayscale image")
        raise NotImplementedError

    def __calc_centroid(self, contour_dist: Image, *args, **kwargs) -> Image:
        raise NotImplementedError

    def __calc_max(self, contour_dist: Image, *args, **kwargs) -> Image:
        raise NotImplementedError

    def __calc_quantile(self, contour_dist: Image, *args, **kwargs) -> Image:
        q = self.quantile(contour_dist)
        raise NotImplementedError

    def calc(self, B: ImageBinary, C: Optional[Image] = None) -> Image:
        calc_fun = {
            CenterType.CENTER_OF_MASS: self.__calc_center_of_mass,
            CenterType.CENTROID: self.__calc_centroid,
            CenterType.MAX: self.__calc_max,
            CenterType.QUANTILE: self.__calc_quantile
        }
        contour_dist = self.__calc_contour_dist(B)
        return calc_fun[self._center_type](contour_dist, C)
