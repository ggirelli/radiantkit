'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import itertools
import logging
import os
from radiantkit import const, image, path
from radiantkit import particle
import re
import sys
from typing import Dict, List, Tuple
from typing import Callable, Iterator, Optional, Pattern, Type

class SeriesSettings(object):
    _ID: int=0
    _channel_data: Dict[str, Type[image.Image]]=None
    _mask_data: Optional[Dict]=None
    _ref: Optional[str]=None
    labeled: bool=False

    def __init__(self, ID: int, channel_paths: Dict[str,str],
        mask_path: Optional[str]=None, inreg: Optional[Pattern]=None):
        super(SeriesSettings, self).__init__()
        
        ref = path.get_image_details(mask_path, inreg)[1]
        assert ref in channel_paths

        self._ID = ID
        self._channel_data = {}
        for channel_name in channel_paths:
            self._channel_data[channel_name] = dict(
                path=channel_paths[channel_name])
        if mask_path is not None:
            self._mask_data = dict(path=mask_path)
            self._ref = ref


    @property
    def ID(self) -> int:
        return self._ID

    @property
    def channel_names(self) -> List[str]:
        return list(self._channel_data.keys())

    @property
    def channel_data(self) -> Dict[str,str]:
        return self._channel_data.copy()

    @property
    def ref(self) -> str:
        return self._ref

    @property
    def mask_path(self) -> Optional[str]:
        if self.has_mask(): return self._mask_data['path']

    @property
    def mask(self) -> Optional[Type[image.ImageBase]]:
        if self.has_mask():
            if 'I' not in self._mask_data: self.init_mask()
            return self._mask_data['I']

    def has_ref(self) -> bool:
        return self._ref is not None

    def has_mask(self) -> bool:
        return self._mask_data is not None

    def init_mask(self) -> None:
        if not self.has_mask: return None
        if not "I" in self._mask_data:
            if self.labeled: self._mask_data['I'
                ] = image.ImageLabeled.from_tiff(self.mask_path)
            else: self._mask_data['I'
                ] = image.ImageBinary.from_tiff(self.mask_path)

    def init_channel(self, channel_name: str) -> None:
        if channel_name in self._channel_data:
            if not "I" in self._channel_data[channel_name]:
                self._channel_data[channel_name]['I'] = image.Image.from_tiff(
                    self._channel_data[channel_name]['path'])

    def get_channel(self, channel_name: str) -> Optional[image.Image]:
        if channel_name in self._channel_data:
            if 'I' in self._channel_data[channel_name]:
                return self._channel_data[channel_name]['I']

    def __str__(self) -> str:
        s = f"Series #{self._ID} with {len(self.channel_names)} channels."
        if not self.has_ref(): s += " No reference."
        else:
            s += f" '{self._ref}' reference channel"
            if self.has_mask(): s += " (with mask)"
        for (name, data) in self._channel_data.items():
            s += f"\n  {name} => '{data['path']}'"
        if self.has_mask: s += f"\n  mask => '{self.mask_path}'"
        return s

class Series(SeriesSettings):
    _particles: Optional[List[Type[particle.ParticleBase]]]=None

    def __init__(self, ID: int, channel_paths: Dict[str,str],
        mask_path: Optional[str]=None, inreg: Optional[Pattern]=None):
        for (channel_name, channel_path) in channel_paths.items():
            assert os.path.isfile(channel_path)
        super(Series, self).__init__(ID, channel_paths, mask_path, inreg)

    @property
    def particles(self):
        if self._particles is None:
            logging.warning("particle attribute accessible " +
                "after running extract_particles.")
        return self._particles
    
    def extract_particles(self,
        particleClass: Type[particle.ParticleBase]) -> None:
        if not self.has_mask():
            logging.warning("mask is missing, no particles extracted.")
            return

        if self.labeled:
            fextract = particle.ParticleFinder.get_particles_from_labeled_image
        else:
            fextract = particle.ParticleFinder.get_particles_from_binary_image

        self._particles = fextract(self.mask, particleClass)
        for pbody in self._particles: pbody.source = self.mask_path

        self.mask.unload()

    @staticmethod
    def static_extract_particles(s: 'Series', channel: str) -> 'Series':
        s.extract_particles(particle.Nucleus)
        logging.info(f"{len(s.particles)} nuclei in series '{s.ID}'")

        s.init_channel(channel)
        for pbody in s.particles:
            pbody.init_intensity_features(s.get_channel(channel), channel)
        s.get_channel(channel).unload()
        return s

    def __str__(self):
        s = super(Series, self).__str__()
        if self._particles is not None:
            s += f"\n  With {len(self._particles)} particles "
            s += f"[{type(self._particles[0]).__name__}]."
        return s

class SeriesList(object):
    _series: List[Series]=None

    def __init__(self, series_list: List[Series]):
        super(SeriesList, self).__init__()
        self._series = series_list

    @property
    def channel_names(self):
        return list(set(itertools.chain(*[series.channel_names
            for series in self._series])))

    @staticmethod
    def from_directory(dpath: str, inreg: Pattern, ref: Optional[str]=None,
        maskfix: Optional[Tuple[str, str]]=None):
        channel_list = path.find_re(dpath, inreg)
        mask_list, channel_list = path.select_by_prefix_and_suffix(
            dpath, channel_list, *maskfix)
        
        mask_data = {}
        for mask_path in mask_list:
            series_id, channel_name = path.get_image_details(mask_path,inreg)
            if channel_name != ref:
                logging.warning("skipping mask for channel " +
                    f"'{channel_name}', not reference ({ref}).")
                continue
            if series_id in mask_data:
                logging.warning("found multiple masks for reference channel " +
                    f"in series {series_id}. " +
                    f"Skipping '{mask_path}'.")
                continue
            mask_data[series_id] = os.path.join(dpath, mask_path)

        channel_data = {}
        for channel_path in channel_list:
            series_id, channel_name = path.get_image_details(channel_path,inreg)
            if series_id not in channel_data: channel_data[series_id] = {}
            if channel_name in channel_data[series_id]:
                logging.warning("found multiple instances of channel " +
                    f"{channel_name} in series {series_id}. " +
                    f"Skipping '{channel_path}'.")
            channel_data[series_id][channel_name
                ] = os.path.join(dpath, channel_path)

        channel_counts = [len(x) for x in channel_data.values()]
        assert 1 == len(set(channel_counts)), "inconsistent number of channels"

        series_list = []
        for series_id in channel_data.keys():
            if series_id not in mask_data:
                logging.critical("missing mask of reference channel "
                    f"'{ref}' for series '{series_id}'")
                sys.exit()
            series_list.append(Series(series_id,
                channel_data[series_id], mask_data[series_id], inreg))

        return SeriesList(series_list)

    def __len__(self) -> int:
        return len(self._series)

    def __next__(self) -> Series:
        for series in self._series:
            yield series

    def __iter__(self) -> Iterator[Series]:
        return self.__next__()
