'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import itertools
import logging
import os
from radiantkit.image import Image, ImageBase, ImageBinary, ImageLabeled
from radiantkit.path import find_re, get_image_details
from radiantkit.path import select_by_prefix_and_suffix
from radiantkit.particle import ParticleBase, ParticleFinder
from typing import Dict, List, Tuple
from typing import Iterator, Optional, Pattern, Type, Union

class SeriesSettings(object):
    _ID: int=0
    _channels: Dict[str, Union[str, Type[Image]]]=None
    _mask: Union[str, Type[ImageBase]]=None
    _ref: Optional[str]=None
    _labeled: bool=False
    _ground_block_side: int=11

    def __init__(self, ID: int, channel_paths: Dict[str,str],
        mask_path: Optional[str]=None, inreg: Optional[Pattern]=None):
        super(SeriesSettings, self).__init__()
        self._ID = ID

        for name in channel_paths: assert os.path.isfile(channel_paths[name])
        self._channels = channel_paths.copy()

        if mask_path is not None:
            ref = get_image_details(mask_path, inreg)[1]
            assert ref in channel_paths
            self._ref = ref
            self._mask = mask_path

    @property
    def ID(self) -> int:
        return self._ID

    @property
    def labeled(self) -> bool:
        return self._labeled
    
    @labeled.setter
    def labeled(self, labeled: bool) -> None:
        if labeled != self._labeled:
            if not labeled and isinstance(self._mask, ImageLabeled):
                self._mask = self._mask.binary()
        self._labeled = labeled

    @property
    def ground_block_side(self) -> int:
        return self._ground_block_side
    
    @ground_block_side.setter
    def ground_block_side(self, bs: int) -> None:
        self._ground_block_side = bs
        if 0 != bs%2: self._ground_block_side += 1

    @property
    def channel_names(self) -> List[str]:
        return list(self._channels.keys())

    @property
    def channels(self) -> Dict[str, Union[str, Type[Image]]]:
        return self._channels.copy()

    @property
    def ref(self) -> str:
        return self._ref

    @property
    def mask_path(self) -> Optional[str]:
        if isinstance(self._mask, str): return self._mask
        else: return self._mask.path

    @property
    def mask(self) -> Optional[Type[ImageBase]]:
        if self.has_mask():
            if isinstance(self._mask, str): self.init_mask()
            return self._mask

    def has_ref(self) -> bool:
        return self._ref is not None

    def has_mask(self) -> bool:
        return self._mask is not None

    def init_mask(self) -> None:
        if not self.has_mask(): return
        if isinstance(self._mask, str):
            if self.labeled: self._mask = ImageLabeled.from_tiff(
                    self.mask_path, doRelabel=False)
            else: self._mask = ImageBinary.from_tiff(self.mask_path)

    def init_channel(self, channel_name: str) -> None:
        if channel_name in self._channels:
            if isinstance(self._channels[channel_name], str):
                self._channels[channel_name] = Image.from_tiff(
                    self._channels[channel_name])
                if self.has_mask():
                    self.init_mask()
                    if self.labeled:
                        self._channels[channel_name].update_ground(
                            self.mask.binary())
                    else: self._channels[channel_name].update_ground(self.mask)

    def get_channel(self, channel_name: str) -> Optional[Image]:
        if channel_name in self._channels:
            if isinstance(self._channels[channel_name], str):
                self.init_channel(channel_name)
            if isinstance(self._channels[channel_name], Image):
                return self._channels[channel_name]

    def unload(self) -> None:
        for channel_name in self._channels.keys():
            self.unload_channel(channel_name)
        if self.has_mask(): self.mask.unload()

    def unload_channel(self, channel_name: str) -> None:
        if channel_name in self._channels:
            if not isinstance(self._channels[channel_name], str):
                self._channels[channel_name].unload()

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
    _particles: Optional[List[Type[ParticleBase]]]=None

    def __init__(self, ID: int, channel_paths: Dict[str,str],
        mask_path: Optional[str]=None, inreg: Optional[Pattern]=None):
        super(Series, self).__init__(ID, channel_paths, mask_path, inreg)

    @property
    def particles(self):
        if self._particles is None: logging.warning(
            "particle attribute accessible after running extract_particles.")
        return self._particles
    
    def init_particles(self, channel_list: Optional[List[str]]=None,
        particleClass: Type[ParticleBase]=ParticleBase) -> None:
        if not self.has_mask():
            logging.warning("mask is missing, no particles extracted.")
            return

        if self.labeled:
            fextract = ParticleFinder.get_particles_from_labeled_image
        else: fextract = ParticleFinder.get_particles_from_binary_image
        self._particles = fextract(self.mask, particleClass)
        self.mask.unload()

        for pbody in self._particles: pbody.source = self.mask_path
        if channel_list is not None:
            for channel in channel_list:
                assert channel in self.channel_names
                for pbody in self._particles:
                    pbody.init_intensity_features(
                        self.get_channel(channel), channel)
            self.unload_channel(channel)

    @staticmethod
    def extract_particles(series: 'Series',
        channel_list: Optional[List[str]]=None,
        particleClass: Type[ParticleBase]=ParticleBase) -> 'Series':
        series.init_particles(channel_list, particleClass)
        return series

    def keep_particles(self, label_list: List[int]) -> None:
        self._particles = [p for p in self._particles if p.label in label_list]

    def __str__(self):
        s = super(Series, self).__str__()
        if self._particles is not None:
            s += f"\n  With {len(self._particles)} particles "
            s += f"[{type(self._particles[0]).__name__}]."
        return s

class SeriesList(object):
    series: List[Series]=None
    _path: Optional[str]=None
    label: Optional[str]=None

    def __init__(self, series_list: List[Series], path: Optional[str]=None):
        super(SeriesList, self).__init__()
        self.series = series_list
        if path is not None:
            assert os.path.isdir(path)
            self._path = path

    @property
    def path(self):
        return self._path

    @property
    def channel_names(self):
        return list(set(itertools.chain(*[series.channel_names
            for series in self.series])))

    @staticmethod
    def from_directory(dpath: str, inreg: Pattern, ref: Optional[str]=None,
        maskfix: Optional[Tuple[str, str]]=None):
        channel_list = find_re(dpath, inreg)
        mask_list, channel_list = select_by_prefix_and_suffix(
            dpath, channel_list, *maskfix)
        
        if ref is not None:
            mask_data = {}
            for mask_path in mask_list:
                series_id, channel_name = get_image_details(mask_path,inreg)
                if channel_name != ref:
                    logging.warning("skipping mask for channel " +
                        f"'{channel_name}', not reference ({ref}).")
                    continue
                if series_id in mask_data:
                    logging.warning("found multiple masks for reference " +
                        f"channel in series {series_id}. " +
                        f"Skipping '{mask_path}'.")
                    continue
                mask_data[series_id] = os.path.join(dpath, mask_path)

        channel_data = {}
        for channel_path in channel_list:
            series_id, channel_name = get_image_details(channel_path,inreg)
            if series_id not in channel_data: channel_data[series_id] = {}
            if channel_name in channel_data[series_id]:
                logging.warning("found multiple instances of channel " +
                    f"{channel_name} in series {series_id}. " +
                    f"Skipping '{channel_path}'.")
            channel_data[series_id][channel_name] = os.path.join(
                dpath, channel_path)

        channel_counts = [len(x) for x in channel_data.values()]
        assert 1 == len(set(channel_counts)
            ), f"inconsistent number of channels in '{dpath}' series"

        series_list = []
        for series_id in channel_data.keys():
            if ref is not None:
                assert series_id in mask_data, ("missing mask of reference " +
                    f"channel '{ref}' for series '{series_id}'")
                series_list.append(Series(series_id,
                    channel_data[series_id], mask_data[series_id], inreg))
            else:
                series_list.append(Series(series_id,
                    channel_data[series_id], inreg=inreg))

        return SeriesList(series_list, dpath)

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, i: int) -> Series:
        return self.series[i]

    def __next__(self) -> Series:
        for series in self.series:
            yield series

    def __iter__(self) -> Iterator[Series]:
        return self.__next__()
