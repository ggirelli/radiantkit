'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import itertools
import logging
import numpy as np
import os
from radiantkit.image import Image, ImageBase, ImageBinary, ImageLabeled
from radiantkit.path import find_re, get_image_details
from radiantkit.path import select_by_prefix_and_suffix
from radiantkit.particle import ParticleBase, ParticleFinder
import sys
from tqdm import tqdm
from typing import Dict, List, Tuple
from typing import Dict, Iterator, Optional, Pattern, Type, Union

class ChannelList(object):
    '''Store named Image instances (channels) with the same shape and aspect.
    A mask can be provided for background/foreground calculation.'''
    _ID: int=0
    _channels: Dict[str, Image]=None
    _ref: Optional[str]=None
    _mask: Optional[Union[ImageBinary,ImageLabeled]]=None
    _aspect: Optional[np.ndarray]=None
    _shape: Optional[Tuple[int]]=None
    _ground_block_side: int=11

    def __init__(self, ID: int):
        super(ChannelList, self).__init__()
        self._ID = ID
        self._channels = {}
    
    @property
    def ID(self) -> int:
        return self._ID

    @property
    def names(self) -> List[str]:
        return list(self._channels)

    @property
    def shape(self) -> Optional[Tuple[int]]:
        return self._shape

    @property
    def aspect(self) -> Optional[np.ndarray]:
        return self._aspect

    @aspect.setter
    def aspect(self, spacing: np.ndarray) -> None:
        if 0 != len(self):
            for name,channel in self._channels.items(): channel.aspect = spacing
            self._aspect = list(self._channels.values())[0].aspect

    @property
    def ground_block_side(self) -> int:
        return self._ground_block_side
    
    def __update_ground(self, name: Optional[str]=None):
        if name is None:
            for name,channel in self._channels.items():
                channel.update_ground(self.mask, self.ground_block_side)
        else:
            self._channels[name].update_ground(
                self.mask, self.ground_block_side)

    @ground_block_side.setter
    def ground_block_side(self, bs: int) -> None:
        if 0 != bs%2: bs += 1
        if bs != self._ground_block_side and self.is_masked():
            if 0 != len(self): self.__update_ground()
        self._ground_block_side = bs

    @property
    def reference(self) -> str:
        return self._ref

    @property
    def mask(self) -> Optional[Union[ImageBinary,ImageLabeled]]:
        return self._mask

    @staticmethod
    def from_dict(channel_paths: Dict[str, str]) -> 'ChannelList':
        CL = ChannelList()
        for name,path in channel_paths.items():
            CL.add_channel_from_tiff(name, path)
        return CL

    def is_masked(self) -> bool:
        return self._mask is not None

    def is_labeled(self) -> bool:
        return isinstance(self._mask, ImageLabeled)

    def __init_or_check_aspect(self, spacing: np.ndarray) -> None:
        if self.aspect is None:
            self._aspect = spacing
        elif any(spacing != self.aspect):
            logging.error(f"aspect mismatch. Expected {self.aspect} " +
                f"but got {spacing}")
            sys.exit()

    def __init_or_check_shape(self, shape: np.ndarray) -> None:
        if self.shape is None:
            self._shape = shape
        elif shape != self.shape:
            logging.error(f"shape mismatch. Expected {self.shape} " +
                f"but got {shape}")
            sys.exit()

    def add_mask(self, name: str, M: Union[ImageBinary,ImageLabeled],
        replace: bool=False) -> None:
        if not name in self._channels:
            logging.error(f"{name} channel unavailable. Mask not added.")
            return
        if self.mask is not None and not replace:
            logging.warning(f"mask is already present." +
                "Use replace=True to replace it.")
        self.__init_or_check_shape(M.shape)
        self.__init_or_check_aspect(M.aspect)
        self._mask = M
        self._ref = name

    def add_mask_from_tiff(self, name: str, path: str,
        labeled: bool=False, replace: bool=False) -> None:
        assert os.path.isfile(path)
        if labeled: M = ImageLabeled.from_tiff(path)
        else: M = ImageBinary.from_tiff(path)
        if self.aspect is not None: M.aspect = self.aspect
        M.unload()
        self.add_mask(name, M, replace)

    def add_channel(self, name: str, I: Image, replace: bool=False) -> None:
        if name in self._channels and not replace:
            logging.warning(f"channel {name} is already present." +
                "Use replace=True to replace it.")
        self.__init_or_check_shape(I.shape)
        self.__init_or_check_aspect(I.aspect)
        self._channels[name] = I
        if self.is_masked(): self._channels[name].update_ground(
            self.mask, self._ground_block_side)

    def add_channel_from_tiff(self, name: str, path: str,
        replace: bool=False) -> None:
        assert os.path.isfile(path)
        I = Image.from_tiff(path)
        if self.aspect is not None: I.aspect = self.aspect
        I.unload()
        self.add_channel(name, I, replace)

    def unload(self, name: Optional[str]=None) -> None:
        if name is None:
            for channel in self._channels.values():
                channel.unload()
            if self.is_masked(): self.mask.unload()
        elif name in self._channels:
            self._channels[name].unload()

    def label(self) -> None:
        if not self.is_labeled():
            self.add_mask(self.ref, self.mask.label(), True)

    def binarize(self) -> None:
        if self.is_labeled():
            self.add_mask(self.ref, self.mask.binarize(), True)

    def __len__(self) -> int:
        return len(self._channels)

    def __getitem__(self, name: str) -> Tuple[str,Image]:
        return (name, self._channels[name])

    def __next__(self) -> Tuple[str,Image]:
        for name in self._channels:
            yield (name, self._channels[name])

    def __iter__(self) -> Iterator[Tuple[str,Image]]:
        return self.__next__()

    def __contains__(self, name: str) -> bool:
        return name in self._channels

    def __str__(self) -> str:
        s = f"Series #{self._ID} with {len(self)} channels."
        for name,channel in self: s += f"\n  {name} => '{channel.path}'"
        if self.is_masked():
            s += f"\n  mask({self.reference}) => '{self.mask.path}'"
        return s

class Series(ChannelList):
    _particles: Optional[List[Type[ParticleBase]]]=None

    def __init__(self, ID: int):
        super(Series, self).__init__(ID)

    @property
    def particles(self) -> Optional[List[Type[ParticleBase]]]:
        if self._particles is None: logging.warning(
            "particle attribute accessible after running extract_particles.")
        return self._particles
    
    @property
    def aspect(self) -> np.ndarray:
        return super(Series, self).aspect

    @aspect.setter
    def aspect(self, spacing: np.ndarray) -> None:
        if 0 != len(self):
            for name,channel in self._channels.items(): channel.aspect = spacing
            self._aspect = list(self._channels.values())[0].aspect
        if self._particles is not None:
            for particle in self._particles:
                particle.aspect = spacing

    def init_particles(self, channel_names: Optional[List[str]]=None,
        particleClass: Type[ParticleBase]=ParticleBase) -> None:
        if not self.is_masked():
            logging.warning("mask is missing, no particles extracted.")
            return

        if self.is_labeled():
            fextract = ParticleFinder.get_particles_from_labeled_image
        else:
            fextract = ParticleFinder.get_particles_from_binary_image
        self._particles = fextract(self.mask, particleClass)
        self.mask.unload()

        for pbody in self._particles: pbody.source = self.mask.path
        if channel_names is not None:
            for name in channel_names:
                assert name in self
                for pbody in self._particles:
                    pbody.init_intensity_features(
                        self[name][1], name)
            self.unload(name)

    @staticmethod
    def extract_particles(series: 'Series',
        channel_list: Optional[List[str]]=None,
        particleClass: Type[ParticleBase]=ParticleBase) -> 'Series':
        series.init_particles(channel_list, particleClass)
        return series

    def keep_particles(self, label_list: List[int]) -> None:
        self._particles = [p for p in self._particles if p.label in label_list]

    def export_particles(self, path: str, compressed: bool,
        showProgress: bool=False) -> None:
        assert os.path.isdir(path)
        if showProgress: iterbar = tqdm
        else: iterbar = lambda x, *args, **kwargs: x

        for nucleus in iterbar(self.particles, desc="mask"):
            nucleus.mask.to_tiff(os.path.join(path,
                f"mask_series{self.ID:03d}_nucleus{nucleus.label:03d}"),
                compressed)

        for channel_name in iterbar(self.names, desc="channel"):
            for nucleus in iterbar(self.particles, desc="nucleus"):
                Image(nucleus.region_of_interest.apply(
                    self[channel_name][1])).to_tiff(
                    os.path.join(path, f"{channel_name}_series{self.ID:03d}_" +
                        f"nucleus{nucleus.label:03d}"), compressed)

    @staticmethod
    def static_export_particles(series: 'Series', path: str, compressed: bool,
        showProgress: bool=False) -> None:
        series.export_particles(path, compressed, showProgress)

    def __str__(self):
        s = super(Series, self).__str__()
        if self._particles is not None:
            s += f"\n  With {len(self._particles)} particles "
            s += f"[{type(self._particles[0]).__name__}]."
        return s

class SeriesList(object):
    _series: List[Series]=None
    label: Optional[str]=None

    def __init__(self, series_list: List[Series]):
        super(SeriesList, self).__init__()
        self._series = series_list

    @property
    def channel_names(self):
        return list(set(itertools.chain(*[s.names for s in self._series])))

    @staticmethod
    def from_directory(dpath: str, inreg: Pattern,
        ref: Optional[str]=None,
        maskfix: Optional[Tuple[str, str]]=None,
        aspect: Optional[np.ndarray]=None):

        masks, channels = select_by_prefix_and_suffix(
            dpath, find_re(dpath, inreg), *maskfix)
        series = {}

        for path in channels:
            sid, channel_name = get_image_details(path,inreg)
            if sid not in series:
                series[sid] = Series(sid)
                series[sid].aspect = aspect

            if channel_name in series[sid]:
                logging.warning("found multiple instances of channel " +
                    f"{channel_name} in series {sid}. Skipping '{path}'.")
                continue

            series[sid].add_channel_from_tiff(channel_name,
                os.path.join(dpath, path))
        
        if ref is not None:
            for path in masks:
                sid, channel_name = get_image_details(path,inreg)
                if sid not in series:
                    series[sid] = Series(sid)
                    series[sid].aspect = aspect

                if channel_name != ref:
                    logging.warning("skipping mask for channel " +
                        f"'{channel_name}', not reference ({ref}).")
                    continue

                series[sid].add_mask_from_tiff(channel_name,
                    os.path.join(dpath, path))

        clen = len(set([len(s) for s in series.values()]))
        assert 1 == clen, f"inconsistent number of channels in '{dpath}' series"

        return SeriesList(series.values())

    def __len__(self) -> int:
        return len(self._series)

    def __getitem__(self, i: int) -> Series:
        return self._series[i]

    def __next__(self) -> Series:
        for s in self._series: yield s

    def __iter__(self) -> Iterator[Series]:
        return self.__next__()
