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
    _region_of_interest: Optional[BoundingElement] = None
    label: Optional[int] = None

    def __init__(self):
        super(ParticleSettings, self).__init__()

    @property
    def mask(self):
        return self._mask

    @property
    def region_of_interest(self):
        return self._region_of_interest

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
    def __init__(self, B: ImageBinary, region_of_interest: BoundingElement):
        super(ParticleBase, self).__init__()
        assert B.shape == region_of_interest.shape
        self._mask = B
        self._region_of_interest = region_of_interest

class Nucleus(ParticleBase):
    def __init__(self, B: ImageBinary, region_of_interest: BoundingElement):
        super(Nucleus, self).__init__(B, region_of_interest)

class ParticleFinder(object):
    def __init__(self):
        super(ParticleFinder, self).__init__()

    @staticmethod
    def get_particles_from_binary_image(B: ImageBinary,
            particleClass: Type[ParticleBase] = ParticleBase
        ) -> List[Type[ParticleBase]]:
        return ParticleFinder.get_particles_from_labeled_image(
            ImageLabeled(B), particleClass)

    @staticmethod
    def get_particles_from_labeled_image(L: ImageLabeled,
            particleClass: Type[ParticleBase] = ParticleBase
        ) -> List[Type[ParticleBase]]:
        assert L.pixels.min() != L.pixels.max(), 'monochromatic image detected.'
        particle_list = []
        for current_label in tqdm(range(1, L.pixels.max())):
            B = ImageBinary(L.pixels == current_label)
            region_of_interest = BoundingElement.from_binary_image(B)
            B = ImageBinary(region_of_interest.apply(B))
            particle = particleClass(B, region_of_interest)
            particle.label = current_label
            particle_list.append(particle)
        return particle_list
