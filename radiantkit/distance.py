'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from enum import Enum
import numpy as np  # type: ignore
from radiantkit.image import ImageBase, ImageBinary, Image
from radiantkit import stat
from scipy.ndimage.morphology import distance_transform_edt  # type: ignore
from scipy.ndimage import center_of_mass  # type: ignore
from typing import Optional, Tuple


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
            return 1-10**(-len(img.axes))
        return self._quantile

    def __calc_contour_dist(self, B: ImageBase) -> ImageBase:
        contour_dist = Image(distance_transform_edt(B.get_offset(1), B.aspect),
                             axes=B.axes)
        contour_dist.aspect = B.aspect
        return contour_dist

    def __calc_center_of_mass(self, contour_dist: ImageBase,
                              C: ImageBase) -> np.ndarray:
        center_of_mass_coords = center_of_mass(
            C.pixels[contour_dist.pixels != 0])
        center_dist = stat.array_cells_distance_to_point(
            contour_dist, center_of_mass_coords, aspect=C.aspect)
        center_dist[0 == contour_dist] = np.inf
        return center_dist

    def __calc_centroid(self, contour_dist: ImageBase) -> np.ndarray:
        centroid = np.array([c.mean() for c in np.nonzero(contour_dist)])
        center_dist = stat.array_cells_distance_to_point(
            contour_dist, centroid, aspect=contour_dist.aspect)
        center_dist[0 == contour_dist] = np.inf
        return center_dist

    def __calc_max(self, contour_dist: ImageBase) -> np.ndarray:
        center_dist = distance_transform_edt(
            contour_dist.pixels == contour_dist.pixels.max(),
            contour_dist.aspect)
        center_dist[0 == contour_dist] = np.inf
        return center_dist

    def __calc_quantile(self, contour_dist: ImageBase) -> np.ndarray:
        q = self.quantile(contour_dist)
        qvalue = np.quantile(contour_dist.pixels[contour_dist.pixels != 0], q)
        center_dist = distance_transform_edt(
            contour_dist.pixels <= qvalue, contour_dist.aspect)
        center_dist[0 == contour_dist] = np.inf
        return center_dist

    def calc(self, B: ImageBinary, C: Optional[ImageBase] = None
             ) -> Optional[Tuple[np.ndarray, np.ndarray]]:

        if self._flatten_axes is not None:
            B2 = B.flatten(self._flatten_axes)
        else:
            B2 = B.copy()

        contour_dist = self.__calc_contour_dist(B2)
        if self._center_type is CenterType.CENTER_OF_MASS:
            assert C is not None, ("'center of mass' center definition "
                                   + "requires a grayscale image")
            if self._flatten_axes is not None:
                C2 = C.flatten(self._flatten_axes)
            else:
                C2 = C.copy()
            center_dist = self.__calc_center_of_mass(contour_dist, C2)
        else:
            if self._center_type is CenterType.CENTROID:
                center_dist = self.__calc_centroid(contour_dist)
            elif self._center_type is CenterType.MAX:
                center_dist = self.__calc_max(contour_dist)
            elif self._center_type is CenterType.QUANTILE:
                center_dist = self.__calc_quantile(contour_dist)
            else:
                return None

        if self._flatten_axes is not None:
            contour_dist = contour_dist.tile_to(B.shape)
            center_dist = center_dist.tile_to(B.shape)

        return (contour_dist.pixels, center_dist.pixels)
