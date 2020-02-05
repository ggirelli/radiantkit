'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import itertools
import joblib
import logging
import numpy as np
import os
import pandas as pd
from radiantkit.image import Image, ImageBase, ImageBinary, ImageLabeled
from radiantkit.selection import BoundingElement
from radiantkit.stat import cell_cycle_fit, range_from_fit
from skimage.measure import marching_cubes_lewiner, mesh_surface_area
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Type

class ParticleSettings(object):
    _mask: Optional[ImageBinary] = None
    _region_of_interest: Optional[BoundingElement] = None
    label: Optional[int] = None
    _total_size: Optional[int]=None
    _surface: Optional[int]=None

    def __init__(self, B: ImageBinary,
        region_of_interest: BoundingElement):
        super(ParticleSettings, self).__init__()
        assert B.shape == region_of_interest.shape
        self._mask = B
        self._region_of_interest = region_of_interest

    @property
    def mask(self) -> ImageBinary:
        return self._mask

    @property
    def aspect(self) -> np.ndarray:
        return self.mask.aspect
    
    @aspect.setter
    def aspect(self, spacing: np.ndarray) -> None:
        self.mask.aspect = spacing
        self._surface = None

    @property
    def region_of_interest(self) -> BoundingElement:
        return self._region_of_interest

    @property
    def total_size(self) -> int:
        if self._total_size is None: self._total_size = self._mask.pixels.sum()
        return self._total_size

    @property
    def volume(self) -> int:
        return self.total_size * np.prod(self.mask.aspect)

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
    _intensity: Dict[str, Dict[str, float]]=None

    def __init__(self, B: ImageBinary,
        region_of_interest: BoundingElement):
        super(ParticleBase, self).__init__(B, region_of_interest)
        self._intensity = {}

    @property
    def channel_names(self):
        return list(self._intensity.keys())

    def get_intensity_sum(self, channel_name: str) -> Optional[float]:
        if channel_name in self._intensity:
            return self._intensity[channel_name]['sum']
        else: return np.nan

    def get_intensity_mean(self, channel_name: str) -> Optional[float]:
        if channel_name in self._intensity:
            return self._intensity[channel_name]['mean']
        else: return np.nan

    def init_intensity_features(self, I: Image,
        channel_name: str='unknown') -> None:
        if channel_name in self._intensity: logging.warning(
            f"overwriting intensity mean of channel '{channel_name}'.")
        else: self._intensity[channel_name] = {}

        pixels = self._region_of_interest.apply(I)[self._mask.pixels]
        if I.ground[0] is not None: pixels -= I.ground[0]
        self._intensity[channel_name]['mean'] = np.mean(pixels)
        self._intensity[channel_name]['sum'] = np.sum(pixels)

class Nucleus(ParticleBase):
    def __init__(self, B: ImageBinary,
        region_of_interest: BoundingElement):
        super(Nucleus, self).__init__(B, region_of_interest)

class NucleiList(object):
    def __init__(self, nuclei: List[Nucleus]):
        super(NucleiList, self).__init__()
        self.__nuclei = nuclei

    @property
    def nuclei(self):
        return self.__nuclei.copy()
    
    @staticmethod
    def from_field_of_view(maskpath: str, rawpath: str,
        doRescale: bool=True) -> List[Nucleus]:
        I = Image.from_tiff(rawpath, doRescale=doRescale)
        M = ImageBinary.from_tiff(maskpath)
        assert I.shape == M.shape

        nuclei = ParticleFinder().get_particles_from_binary_image(M, Nucleus)
        for nucleus in nuclei:
            nucleus.init_intensity_features(I)
            nucleus.source = rawpath

        return NucleiList(nuclei)

    @staticmethod
    def from_multiple_fields_of_view(masklist: Tuple[str], ipath: str,
        doRescale: bool=True, threads: int=1) -> List['NucleiList']:
        if 1 == threads:
            nuclei = []
            for rawpath,maskpath in tqdm(masklist):
                nuclei.append(NucleiList.from_field_of_view(
                    os.path.join(ipath, maskpath),
                    os.path.join(ipath, rawpath), doRescale))
        else:
            nuclei = joblib.Parallel(n_jobs = threads, verbose = 11)(
                joblib.delayed(NucleiList.from_field_of_view)(
                    os.path.join(ipath, maskpath), os.path.join(ipath, rawpath),
                    doRescale) for rawpath,maskpath in masklist)

        return NucleiList.concat(nuclei)

    @staticmethod
    def concat(lists: List['NucleiList']) -> 'NucleiList':
        return NucleiList(list(itertools.chain(*[nl.nuclei for nl in lists])))

    def __len__(self):
        return len(self.__nuclei)

    def get_data(self):
        ndata = pd.DataFrame.from_dict({
            'image':[n.source for n in self.nuclei],
            'label':[n.label for n in self.nuclei],
            'size':[n.total_size for n in self.nuclei]
        })
        channels = list(set(itertools.chain(*[n.channel_names
            for n in self.nuclei])))
        for channel in channels:
            ndata[f'isum_{channel}'] = [n.get_intensity_sum(channel)
                for n in self.nuclei]
        return ndata

    def select_G1(self, k_sigma: float=2.5,
        channel: str='unknown') -> Tuple[pd.DataFrame,Dict]:
        ndata = self.get_data()
        isum_label = f'isum_{channel}'

        size_fit = cell_cycle_fit(ndata['size'].values)
        assert size_fit[0] is not None
        size_range = range_from_fit(
            ndata['size'].values, *size_fit, k_sigma)

        isum_fit = cell_cycle_fit(ndata[isum_label].values)
        assert isum_fit[0] is not None
        isum_range = range_from_fit(
            ndata[isum_label].values, *isum_fit, k_sigma)

        ndata['pass_size'] = np.logical_and(
            ndata['size'].values >= size_range[0],
            ndata['size'].values <= size_range[1])
        ndata['pass_isum'] = np.logical_and(
            ndata[isum_label].values >= isum_range[0],
            ndata[isum_label].values <= isum_range[1])
        ndata['pass'] = np.logical_and(ndata['pass_size'], ndata['pass_isum'])
        ndata['ref'] = channel

        return (ndata, {
            'size':{'range':size_range,'fit':size_fit},
            'isum':{'range':isum_range,'fit':isum_fit}})

class ParticleFinder(object):
    def __init__(self):
        super(ParticleFinder, self).__init__()

    @staticmethod
    def get_particles_from_binary_image(B: ImageBinary,
            particleClass: Type[ParticleBase] = ParticleBase
        ) -> List[Type[ParticleBase]]:
        return ParticleFinder.get_particles_from_labeled_image(
            B.label(), particleClass)

    @staticmethod
    def get_particles_from_labeled_image(L: ImageLabeled,
            particleClass: Type[ParticleBase] = ParticleBase
        ) -> List[Type[ParticleBase]]:
        assert L.pixels.min() != L.pixels.max(), 'monochromatic image detected.'

        particle_list = []
        for current_label in np.unique(L.pixels):
            if 0 == current_label: continue
            B = ImageBinary(L.pixels == current_label)

            region_of_interest = BoundingElement.from_binary_image(B)

            B = ImageBinary(region_of_interest.apply(B))
            B.aspect = L.aspect

            particle = particleClass(B, region_of_interest)
            particle.label = current_label

            particle_list.append(particle)

        return particle_list
