'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import numpy as np  # type: ignore
from radiantkit.image import ImageBase, ImageBinary
from typing import Tuple


class BoundingElement(object):
    _bounds: Tuple[Tuple[int, int], ...]

    def __init__(self, axes_bounds: Tuple[Tuple[int, int], ...]):
        super(BoundingElement, self).__init__()
        self._bounds = axes_bounds

    @property
    def bounds(self) -> Tuple[Tuple[int, int], ...]:
        return self._bounds

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple([int(b1-b0) for (b0, b1) in self.bounds])

    @staticmethod
    def from_binary_image(B: ImageBinary) -> 'BoundingElement':
        assert 0 == B.pixels.min() and 1 == B.pixels.max()
        axes_bounds = []
        for axis_id in range(len(B.shape)):
            axis = B.pixels.sum(tuple([axis for axis in range(len(B.shape))
                                       if axis != axis_id])) != 0
            axes_bounds.append((axis.argmax(), len(axis)-axis[::-1].argmax()))
        return BoundingElement(tuple(axes_bounds))

    def apply(self, I: ImageBase) -> np.ndarray:
        assert len(self._bounds) == len(I.shape)
        for axis_id in range(len(I.shape)):
            assert self._bounds[axis_id][1] <= I.shape[axis_id]
        return I.pixels[tuple([slice(b0, b1)
                               for b0, b1 in self._bounds])].copy()

    def __repr__(self):
        return f"{len(self._bounds)}D Bounding Element: {self._bounds}"
