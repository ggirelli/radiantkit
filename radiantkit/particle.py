'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import itertools
from joblib import delayed, Parallel
import logging
import numpy as  np
import os
import pandas as pd
from radiantkit.image import Image, ImageBase, ImageBinary, ImageLabeled
from radiantkit import selection
from radiantkit import stat
from skimage.measure import marching_cubes_lewiner, mesh_surface_area
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Type, Union

class ParticleSettings(object):
    _mask: Optional[ImageBinary] = None
    _region_of_interest: Optional[selection.BoundingElement] = None
    label: Optional[int] = None
    _total_size: Optional[int]=None
    _surface: Optional[int]=None

    def __init__(self, B: ImageBinary,
        region_of_interest: selection.BoundingElement):
        super(ParticleSettings, self).__init__()
        assert B.shape == region_of_interest.shape
        self._mask = B
        self._region_of_interest = region_of_interest

    @property
    def mask(self) -> ImageBinary:
        return self._mask

    @property
    def region_of_interest(self) -> selection.BoundingElement:
        return self._region_of_interest

    @property
    def total_size(self) -> int:
        if self._total_size is None: self._total_size = self._mask.pixels.sum()
        return self._total_size

    @property
    def volume(self) -> int:
        return self.total_size

    @property
    def sizeXY(self) -> int:
        return self.size("XY")

    @property
    def sizeZ(self) -> int:
        return self.size("Z")
    
    @property
    def surface(self) -> float:
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

    def __init__(self, B: ImageBinary,
        region_of_interest: selection.BoundingElement):
        super(ParticleBase, self).__init__(B, region_of_interest)
    
    @property
    def intensity_sum(self) -> float:
        if self._intensity_sum is None:
            logging.warning("run init_intensity_features " +
                "to initialize this value")
        return self._intensity_sum
    
    @property
    def intensity_mean(self) -> float:
        if self._intensity_mean is None:
            logging.warning("run init_intensity_features " +
                "to initialize this value")
        return self._intensity_mean

    def init_intensity_features(self, I: Type[ImageBase]) -> None:
        pixels = self._region_of_interest.apply(I)[self._mask.pixels]
        self._intensity_mean = np.mean(pixels)
        self._intensity_sum = np.sum(pixels)

class Nucleus(ParticleBase):
    def __init__(self, B: ImageBinary,
        region_of_interest: selection.BoundingElement):
        super(Nucleus, self).__init__(B, region_of_interest)

class NucleiList(object):
    def __init__(self, nuclei: List[Nucleus]):
        super(NucleiList, self).__init__()
        self.__nuclei = nuclei

    @property
    def nuclei(self):
        return self.__nuclei.copy()
    
    @staticmethod
    def from_field_of_view(imgdir: str, maskpath: str,
        rawpath: str, loglevel: str="INFO") -> List[Nucleus]:
        I = Image.from_tiff(os.path.join(imgdir, rawpath))
        M = ImageBinary.from_tiff(os.path.join(imgdir, maskpath))
        assert I.shape == M.shape

        nuclei = ParticleFinder().get_particles_from_binary_image(M, Nucleus)
        for nucleus in nuclei:
            nucleus.init_intensity_features(I)
            nucleus.ipath = rawpath

        return NucleiList(nuclei)

    @staticmethod
    def from_multiple_fields_of_view(masklist: Tuple[str], ipath: str,
        threads: int=1) -> List['NucleiList']:
        if 1 == threads:
            nuclei = []
            for rawpath,maskpath in tqdm(masklist):
                nuclei.append(NucleiList.from_field_of_view(
                    ipath, maskpath, rawpath))
        else:
            nuclei = Parallel(n_jobs = threads, verbose = 11)(
                delayed(NucleiList.from_field_of_view)(
                    ipath, maskpath, rawpath) for rawpath,maskpath in masklist)

        return NucleiList.concat(nuclei)

    @staticmethod
    def concat(lists: List['NucleiList']) -> 'NucleiList':
        return NucleiList(list(itertools.chain(*[nl.nuclei for nl in lists])))

    def __len__(self):
        return len(self.__nuclei)

    def select_G1(self, k_sigma: float=2.5) -> Tuple[pd.DataFrame,Dict]:
        size_data = np.array([n.total_size for n in self.nuclei])
        size_fit = stat.cell_cycle_fit(size_data)
        assert size_fit[0] is not None
        size_range = stat.range_from_fit(
            size_data, *size_fit, k_sigma)

        intensity_sum_data = np.array([n.intensity_sum for n in self.nuclei])
        intensity_sum_fit = stat.cell_cycle_fit(intensity_sum_data)
        assert intensity_sum_fit[0] is not None
        intensity_sum_range = stat.range_from_fit(
            intensity_sum_data, *intensity_sum_fit, k_sigma)

        nuclei_data = pd.DataFrame.from_dict({
            'image':[n.ipath for n in self.nuclei],
            'label':[n.label for n in self.nuclei],
            'size':size_data,
            'isum':intensity_sum_data
        })

        nuclei_data['pass_size'] = np.logical_and(
            size_data >= size_range[0],
            size_data <= size_range[1])
        nuclei_data['pass_isum'] = np.logical_and(
            intensity_sum_data >= intensity_sum_range[0],
            intensity_sum_data <= intensity_sum_range[1])
        nuclei_data['pass'] = np.logical_and(
            nuclei_data['pass_size'], nuclei_data['pass_isum'])

        return (nuclei_data, {'size':{'range':size_range,'fit':size_fit},
            'isum':{'range':intensity_sum_range,'fit':intensity_sum_fit}})

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
            region_of_interest = selection.BoundingElement.from_binary_image(B)

            B = ImageBinary(region_of_interest.apply(B))

            particle = particleClass(B, region_of_interest)
            particle.label = current_label

            particle_list.append(particle)

        return particle_list
