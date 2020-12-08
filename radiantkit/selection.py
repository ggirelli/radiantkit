"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import numpy as np  # type: ignore
from radiantkit.image import Image, ImageBinary
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
    def from_binary_image(B: ImageBinary) -> "BoundingElement":
        assert 0 == B.pixels.min() and 1 == B.pixels.max()
        axes_bounds = []
        for axis_id in range(len(B.shape)):
            axes_to_sum = list(range(len(B.shape)))
            axes_to_sum.pop(axes_to_sum.index(axis_id))
            axis_projection = B.pixels.sum(axes_to_sum) != 0
            axes_bounds.append(
                slice(
                    axis_projection.argmax(),
                    len(axis_projection) - axis_projection[::-1].argmax(),
                )
            )
        return BoundingElement(tuple(axes_bounds))

    def apply(self, img: Image) -> np.ndarray:
        assert len(self._bounds) == len(img.shape)
        return img.pixels[self._bounds].copy()

    def __repr__(self):
        return f"{len(self._bounds)}D Bounding Element: {self._bounds}"
