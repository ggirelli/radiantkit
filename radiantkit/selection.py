'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import numpy as np
from radiantkit.image import Image
from typing import Optional, Tuple, Type

class BoundingElement(object):
    _bounds: Optional[Tuple[Tuple[int]]] = None

    def __init__(self, axes_bounds: Tuple[Tuple[int]]):
        super(BoundingElement, self).__init__()
        self._bounds = axes_bounds

    @property
    def bounds(self) -> Tuple[Tuple[int]]:
        return self._bounds

    @property
    def shape(self) -> Tuple[int]:
        return tuple([int(b1-b0) for (b0, b1) in self._bounds])

    def apply(self, I: Type[Image]) -> np.ndarray:
        assert len(self._bounds) == len(I.shape)
        for axis_id in range(len(I.shape)):
            assert self._bounds[axis_id][1] <= I.shape[axis_id]
        return I.pixels[tuple([slice(b0, b1) for b0,b1 in self._bounds])]
