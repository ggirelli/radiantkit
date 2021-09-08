"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import numpy as np  # type: ignore
from radiantkit.image import Image, ImageBinary, ImageLabeled, are_pixels_binary
from typing import List, Tuple


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
        return tuple(int(r.stop - r.start) for r in self.bounds)

    @staticmethod
    def from_binary_pixels(pixels: np.ndarray) -> "BoundingElement":
        assert are_pixels_binary(pixels)
        axes_bounds = []
        for axis_id in range(len(pixels.shape)):
            axes_to_sum = list(range(len(pixels.shape)))
            axes_to_sum.pop(axes_to_sum.index(axis_id))
            axis_projection = pixels.sum(tuple(axes_to_sum)) != 0
            axes_bounds.append(
                slice(
                    axis_projection.argmax(),
                    len(axis_projection) - axis_projection[::-1].argmax(),
                )
            )
        return BoundingElement(tuple(axes_bounds))

    @staticmethod
    def from_binary_image(B: ImageBinary) -> "BoundingElement":
        assert are_pixels_binary(B.pixels)
        return BoundingElement.from_binary_pixels(B.pixels)

    @staticmethod
    def from_labeled_image(L: ImageLabeled, key: int) -> "BoundingElement":
        assert key in L.pixels
        return BoundingElement.from_binary_pixels(L.pixels == key)

    def apply_to_pixels(self, pixels: np.ndarray) -> np.ndarray:
        assert len(self._bounds) == len(pixels.shape), (self._bounds, pixels.shape)
        return pixels[self._bounds].copy()

    def apply(self, img: Image) -> np.ndarray:
        return self.apply_to_pixels(img.pixels)

    def offset(self, offset: int) -> "BoundingElement":
        offset_bounds: List[slice] = []
        for bounds in self._bounds:
            offset_bounds.append(slice(bounds.start - offset, bounds.stop + offset))
        return BoundingElement(tuple(offset_bounds))

    def __repr__(self):
        return "".join(
            [
                f"{len(self._bounds)}D Bounding Element: {self._bounds}",
                f"\nShape: {self.shape}",
            ]
        )
