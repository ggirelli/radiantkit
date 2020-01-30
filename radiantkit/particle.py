'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import logging
import numpy as  np
from radiantkit.image import ImageBase, ImageBinary, ImageLabeled
from radiantkit.selection import BoundingElement
from skimage.measure import marching_cubes_lewiner, mesh_surface_area
from tqdm import tqdm
from typing import List, Optional, Type, Union

class ParticleSettings(object):
    _mask: Optional[ImageBinary] = None
    _region_of_interest: Optional[BoundingElement] = None
    label: Optional[int] = None
    _volume: Optional[int]=None
    _surface: Optional[int]=None

    def __init__(self, B: ImageBinary, region_of_interest: BoundingElement):
        super(ParticleSettings, self).__init__()
        assert B.shape == region_of_interest.shape
        self._mask = B
        self._region_of_interest = region_of_interest

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

    @property
    def volume(self):
        if self._volume is None: self._volume = self._mask.pixels.sum()
        return self._volume
    
    @property
    def surface(self):
        if self._surface is None:
            M = self._mask.pixels.copy()
            shape = [1 for axis in M.shape]
            shape[-2:] = M.shape[-2:]
            M = np.vstack((np.zeros(shape), M, np.zeros(shape)))
            verts, faces, ns, vs = marching_cubes_lewiner(
                M, 0.0, self._mask.aspect)
            self._surface = mesh_surface_area(verts, faces)
        return self._surface

    def size(self, axes: str) -> int:
        assert all([axis in self.mask.axes for axis in axes])
        axes_ids = tuple([self.mask.axes.index(axis)
            for axis in self.mask.axes if axis not in axes])
        return self.mask.pixels.max(axes_ids).sum()

class ParticleBase(ParticleSettings):
    _intensity_sum: Optional[float]=None
    _intensity_mean: Optional[float]=None

    def __init__(self, B: ImageBinary, region_of_interest: BoundingElement):
        super(ParticleBase, self).__init__(B, region_of_interest)
    
    @property
    def intensity_sum(self):
        if self._intensity_sum is None:
            logging.warning("run init_intensity_features " +
                "to initialize this value")
        return self._intensity_sum
    
    @property
    def intensity_mean(self):
        if self._intensity_mean is None:
            logging.warning("run init_intensity_features " +
                "to initialize this value")
        return self._intensity_mean

    def init_intensity_features(self, I: Type[ImageBase]):
        pixels = self._region_of_interest.apply(I)[self._mask.pixels]
        self._intensity_mean = np.mean(pixels)
        self._intensity_sum = np.sum(pixels)

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
            ImageLabeled(B.pixels, B.axes), particleClass)

    @staticmethod
    def get_particles_from_labeled_image(L: ImageLabeled,
            particleClass: Type[ParticleBase] = ParticleBase
        ) -> List[Type[ParticleBase]]:
        assert L.pixels.min() != L.pixels.max(), 'monochromatic image detected.'

        particle_list = []
        for current_label in range(1, L.pixels.max()+1):
            B = ImageBinary(L.pixels == current_label)
            region_of_interest = BoundingElement.from_binary_image(B)

            B = ImageBinary(region_of_interest.apply(B))

            particle = particleClass(B, region_of_interest)
            particle.label = current_label

            particle_list.append(particle)

        return particle_list
