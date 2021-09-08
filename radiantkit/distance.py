"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from enum import Enum
import numpy as np  # type: ignore
from radiantkit.channel import ImageGrayScale
from radiantkit.image import Image, ImageBinary, offset2
from radiantkit import stat
from scipy.ndimage.morphology import distance_transform_edt  # type: ignore
from scipy.ndimage import center_of_mass  # type: ignore
from typing import Optional, Tuple


class DistanceType(Enum):
    LAMINA = "lamina_dist"
    CENTER = "center_dist"
    LAMINA_NORM = "lamina_dist_norm"

    @property
    def label(self) -> str:
        return dict(
            lamina_dist="Distance from lamina (nm)",
            center_dist="Distance from center (nm)",
            lamina_dist_norm="Normalized distance from lamina (a.u.)",
        )[self.value]


class CenterType(Enum):
    CENTER_OF_MASS = "center_of_mass"
    CENTROID = "centroid"
    MAX = "max"
    QUANTILE = "quantile"

    @staticmethod
    def get_default():
        return CenterType.QUANTILE


class RadialDistanceCalculator(object):
    _flatten_axes: Optional[str] = None
    _center_type: CenterType
    _quantile: Optional[float] = None

    def __init__(
        self,
        bundle_axes: Optional[str] = None,
        center_type: CenterType = CenterType.QUANTILE,
        q: Optional[float] = None,
    ):
        super(RadialDistanceCalculator, self).__init__()

        if bundle_axes is not None:
            assert all(a in Image._ALLOWED_AXES for a in bundle_axes)
            self._flatten_axes = bundle_axes

        if center_type in CenterType:
            self._center_type = center_type
            if CenterType.QUANTILE == self._center_type:
                self.__set_quantile(q)

    @property
    def center_type(self):
        return self._center_type

    def __set_quantile(self, q: Optional[float]) -> None:
        if q is not None:
            assert q > 0 and q <= 1
            self._quantile = q

    def quantile(self, img: Optional[Image] = None) -> float:
        if self._quantile is None:
            assert img is not None, "either set a quantile manually or provide an image"
            return 1 - 10 ** (-len(img.axes))
        return self._quantile

    def __calc_contour_dist(self, B: Image) -> Image:
        contour_dist = ImageGrayScale(
            offset2(distance_transform_edt(B.offset(1), B.aspect), -1)
        )
        contour_dist.aspect = B.aspect
        return contour_dist

    def __calc_center_of_mass(self, contour_dist: Image, C: Image) -> Optional[Image]:
        center_of_mass_coords = center_of_mass(C.pixels[contour_dist.pixels != 0])
        center_dist = stat.array_cells_distance_to_point(
            contour_dist.pixels, center_of_mass_coords, aspect=C.aspect
        )
        center_dist[contour_dist.pixels == 0] = np.inf
        return contour_dist.from_this(center_dist)

    def __calc_centroid(self, contour_dist: Image) -> Optional[Image]:
        centroid = np.array([c.mean() for c in np.nonzero(contour_dist.pixels)])
        center_dist = stat.array_cells_distance_to_point(
            contour_dist.pixels, centroid, aspect=contour_dist.aspect
        )
        center_dist[contour_dist.pixels == 0] = np.inf
        return contour_dist.from_this(center_dist)

    def __calc_max(self, contour_dist: Image) -> Optional[Image]:
        center_dist = distance_transform_edt(
            contour_dist.pixels != contour_dist.pixels.max(), contour_dist.aspect
        )
        center_dist[contour_dist.pixels == 0] = np.inf
        return contour_dist.from_this(center_dist)

    def __calc_quantile(self, contour_dist: Image) -> Optional[Image]:
        q = self.quantile(contour_dist)
        qvalue = np.quantile(contour_dist.pixels[contour_dist.pixels != 0], q)
        center = contour_dist.pixels < qvalue
        if center.sum() == 0:
            return None
        center_dist = distance_transform_edt(center, contour_dist.aspect)
        center_dist[contour_dist.pixels == 0] = np.inf
        return contour_dist.from_this(center_dist)

    def __flatten(self, img: Image) -> Image:
        if self._flatten_axes is not None:
            return img.flatten(self._flatten_axes)
        else:
            return img.copy()

    def __unflatten(self, img: Image, shape: Tuple[int, ...]) -> Image:
        if self._flatten_axes is not None:
            return img.tile_to(shape)
        else:
            return img.copy()

    def __calc_center_dist(self, contour_dist, C) -> Optional[Image]:
        if self._center_type is CenterType.CENTER_OF_MASS:
            assert C is not None, (
                "'center of mass' center definition " + "requires a grayscale image"
            )
            C2 = self.__flatten(C)
            return self.__calc_center_of_mass(contour_dist, C2)
        else:
            if self._center_type is CenterType.CENTROID:
                return self.__calc_centroid(contour_dist)
            elif self._center_type is CenterType.MAX:
                return self.__calc_max(contour_dist)
            elif self._center_type is CenterType.QUANTILE:
                return self.__calc_quantile(contour_dist)
        return None

    def calc(
        self, B: ImageBinary, C: Optional[Image] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        B2 = self.__flatten(B)

        contour_dist = self.__calc_contour_dist(B2)
        center_dist = self.__calc_center_dist(contour_dist, C)
        if center_dist is None:
            return None

        contour_dist = self.__unflatten(contour_dist, B.shape)
        center_dist = self.__unflatten(center_dist, B.shape)

        return (contour_dist.pixels, center_dist.pixels)

    def __repr__(self) -> str:
        s = f"<RDC> axes:{self._flatten_axes}; "
        s += f"center:{self._center_type}; q:{self._quantile}"
        return s
