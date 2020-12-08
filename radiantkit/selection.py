"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import logging
import numpy as np  # type: ignore
from radiantkit.image import Image, ImageBinary, ImageLabeled, pixels_are_binary
from typing import Tuple


class BoundingElement(object):
    _bounds: Tuple[slice, ...]

    def __init__(self, axes_bounds: Tuple[slice, ...]):
        super(BoundingElement, self).__init__()
        self._bounds = axes_bounds

    @property
    def bounds(self) -> Tuple[slice, ...]:
        return self._bounds

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple([int(r.stop - r.start) for r in self.bounds])

    @staticmethod
    def _from_binary_pixels(pixels: np.ndarray) -> "BoundingElement":
        assert pixels_are_binary(pixels)
        axes_bounds = []
        for axis_id in range(len(pixels.shape)):
            axes_to_sum = list(range(len(pixels.shape)))
            logging.info(axes_to_sum)
            axes_to_sum.pop(axes_to_sum.index(axis_id))
            logging.info(axes_to_sum)
            axis_projection = pixels.sum(axes_to_sum) != 0
            logging.info((axis_projection.min(), axis_projection.max()))
            axes_bounds.append(
                slice(
                    axis_projection.argmax(),
                    len(axis_projection) - axis_projection[::-1].argmax(),
                )
            )
        return BoundingElement(tuple(axes_bounds))

    @staticmethod
    def from_binary_image(B: ImageBinary) -> "BoundingElement":
        assert pixels_are_binary(B.pixels)
        return BoundingElement._from_binary_pixels(B)

    @staticmethod
    def from_labeled_image(L: ImageLabeled, key: int) -> "BoundingElement":
        assert key in L.pixels
        return BoundingElement._from_binary_pixels(L == key)

    def apply(self, img: Image) -> np.ndarray:
        assert len(self._bounds) == len(img.shape), (self._bounds, img.shape)
        return img.pixels[self._bounds].copy()

    def __repr__(self):
        return f"{len(self._bounds)}D Bounding Element: {self._bounds}"
