'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from radiantkit.image import ImageBinary, ImageLabeled
from radiantkit.selection import BoundingElement
from tqdm import tqdm
from typing import List, Optional, Type, Union

class ParticleSettings(object):
    _mask: Optional[ImageBinary] = None
    _roi: Optional[BoundingElement] = None

    def __init__(self):
        super(ParticleSettings, self).__init__()

    @property
    def mask(self):
        return self._mask

    @property
    def roi(self):
        return self._roi

    @property
    def total_size(self):
        return self.mask.sum()

    @property
    def sizeXY(self):
        return self.size("XY")

    @property
    def sizeZ(self):
        return self.size("Z")

    def size(self, axes: str) -> int:
        assert all([axis in self.mask.axes for axis in axes])
        axes_ids = tuple([self.mask.axes.index(axis)
            for axis in self.mask.axes if axis not in axes])
        return self.mask.pixels.max(axes_ids).sum()

class ParticleBase(ParticleSettings):
    def __init__(self, B: ImageBinary, roi: BoundingElement):
        super(ParticleBase, self).__init__()
        assert B.shape == roi.shape
        self._mask = B
        self._roi = roi

class Nucleus(ParticleBase):
    def __init__(self, B: ImageBinary, roi: BoundingElement):
        super(Nucleus, self).__init__(B, roi)

class ParticleFinder(object):
    def __init__(self):
        super(ParticleFinder, self).__init__()

    @staticmethod
    def get_particles_from_labeled_image(L: ImageLabeled,
            particleClass: Type[ParticleBase] = ParticleBase
        ) -> List[Type[ParticleBase]]:
        assert L.pixels.min() != L.pixels.max(), 'monochromatic image detected.'
        particle_list = []
        for lab in tqdm(range(1, L.pixels.max())):
            B = ImageBinary(L.pixels == lab)
            roi = BoundingElement.from_binary_image(B)
            B = ImageBinary(roi.apply(B))
            particle_list.append(particleClass(B, roi))
        return particle_list
