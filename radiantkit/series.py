'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import ggc  # type: ignore
import itertools
import joblib  # type: ignore
import logging
import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore
from radiantkit.image import ImageBinary, ImageLabeled, Image
from radiantkit.path import find_re, get_image_details
from radiantkit.path import select_by_prefix_and_suffix
from radiantkit.particle import ParticleBase, ParticleFinder
from radiantkit.stat import quantile_from_counts
import sys
from tqdm import tqdm  # type: ignore
from typing import Dict, List, Tuple
from typing import Iterator, Optional, Pattern, Type, Union


class ChannelList(object):
    '''Store named Image instances (channels) with the same shape and aspect.
    A mask can be provided for background/foreground calculation.'''
    _ID: int = 0
    _channels: Dict[str, Image]
    _ref: Optional[str] = None
    _mask: Optional[Union[ImageBinary, ImageLabeled]] = None
    _aspect: Optional[np.ndarray] = None
    _shape: Optional[Tuple[int]] = None
    _ground_block_side: int = 11
    __current_channel: int = 0

    def __init__(self, ID: int, ground_block_side: Optional[int] = None):
        super(ChannelList, self).__init__()
        self._ID = ID
        self._channels = {}
        if ground_block_side is not None:
            self._ground_block_side = ground_block_side

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
            for name, channel in self._channels.items():
                channel.aspect = spacing
            self._aspect = list(self._channels.values())[0].aspect

    @property
    def ground_block_side(self) -> int:
        return self._ground_block_side

    @ground_block_side.setter
    def ground_block_side(self, bs: int) -> None:
        if 0 != bs % 2:
            bs += 1
        if bs != self._ground_block_side and self.is_masked():
            if 0 != len(self):
                self.__update_ground()
        self._ground_block_side = bs

    def __update_ground(self, name: Optional[str] = None):
        if self.mask is None:
            return
        if name is None:
            for name, channel in self._channels.items():
                channel.update_ground(self.mask, self.ground_block_side)
        else:
            self._channels[name].update_ground(
                self.mask, self.ground_block_side)

    @property
    def reference(self) -> Optional[str]:
        return self._ref

    @property
    def mask(self) -> Optional[Union[ImageBinary, ImageLabeled]]:
        return self._mask

    @staticmethod
    def from_dict(ID: int, channel_paths: Dict[str, str]) -> 'ChannelList':
        CL = ChannelList(ID)
        for name, path in channel_paths.items():
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
            logging.error(f"aspect mismatch. Expected {self.aspect} "
                          + f"but got {spacing}")
            sys.exit()

    def __init_or_check_shape(self, shape: np.ndarray) -> None:
        if self.shape is None:
            self._shape = shape
        elif shape != self.shape:
            logging.error(f"shape mismatch. Expected {self.shape} "
                          + f"but got {shape}")
            sys.exit()

    def add_mask(self, name: str, M: Union[ImageBinary, ImageLabeled],
                 replace: bool = False) -> None:
        if name not in self._channels:
            logging.error(f"{name} channel unavailable. Mask not added.")
            return
        if self.mask is not None and not replace:
            logging.warning(f"mask is already present."
                            + "Use replace=True to replace it.")
        self.__init_or_check_shape(M.shape)
        self.__init_or_check_aspect(M.aspect)
        self._mask = M
        self._ref = name

    def add_mask_from_tiff(
            self, name: str, path: str,
            labeled: bool = False, replace: bool = False) -> None:
        assert os.path.isfile(path)
        M: Union[ImageBinary, ImageLabeled]
        if labeled:
            M = ImageLabeled.from_tiff(path)
        else:
            M = ImageBinary.from_tiff(path)
        if self.aspect is not None:
            M.aspect = self.aspect
        M.unload()
        self.add_mask(name, M, replace)

    def add_channel(self, name: str, img: Image,
                    replace: bool = False) -> None:
        if name in self._channels and not replace:
            logging.warning(f"channel {name} is already present."
                            + "Use replace=True to replace it.")
        self.__init_or_check_shape(img.shape)
        self.__init_or_check_aspect(img.aspect)
        self._channels[name] = img
        if self.mask is not None:
            self._channels[name].update_ground(
                self.mask, self._ground_block_side)

    def add_channel_from_tiff(self, name: str, path: str,
                              replace: bool = False) -> None:
        assert os.path.isfile(path)
        img = Image.from_tiff(path)
        if self.aspect is not None:
            img.aspect = self.aspect
        img.unload()
        self.add_channel(name, img, replace)

    def unload(self, name: Optional[str] = None) -> None:
        if name is None:
            for channel in self._channels.values():
                channel.unload()
            if self.mask is not None:
                self.mask.unload()
        elif name in self._channels:
            self._channels[name].unload()

    def label(self) -> None:
        if isinstance(self.mask, ImageBinary) and self.reference is not None:
            self.add_mask(self.reference, self.mask.label(), True)

    def binarize(self) -> None:
        if isinstance(self.mask, ImageLabeled) and self.reference is not None:
            self.add_mask(self.reference, self.mask.binarize(), True)

    def __len__(self) -> int:
        return len(self._channels)

    def __getitem__(self, name: str) -> Tuple[str, Image]:
        return (name, self._channels[name])

    def __next__(self) -> Tuple[str, Image]:
        self.__current_channel += 1

        if self.__current_channel > len(self):
            raise StopIteration
        else:
            channel_names = self.names
            channel_names.sort()
            return self[channel_names[self.__current_channel-1]]

    def __iter__(self) -> Iterator[Tuple[str, Image]]:
        self.__current_channel = 0
        return self

    def __contains__(self, name: str) -> bool:
        return name in self._channels

    def __str__(self) -> str:
        s = f"Series #{self._ID} with {len(self)} channels."
        for name, channel in self:
            s += f"\n  {name} => '{channel.path}'"
        if self.mask is not None:
            s += f"\n  mask({self.reference}) => '{self.mask.path}'"
        return s


class Series(ChannelList):
    _particles: List[Type[ParticleBase]]

    def __init__(self, ID: int, ground_block_side: Optional[int] = None):
        super(Series, self).__init__(ID, ground_block_side)

    @property
    def particles(self) -> List[ParticleBase]:
        if self._particles is None:
            logging.warning("particle attribute accessible "
                            + "after running extract_particles.")
        return self._particles

    @property
    def aspect(self) -> np.ndarray:
        return super(Series, self).aspect

    @aspect.setter
    def aspect(self, spacing: np.ndarray) -> None:
        if 0 != len(self):
            for name, channel in self._channels.items():
                channel.aspect = spacing
            self._aspect = list(self._channels.values())[0].aspect
        if self._particles is not None:
            for particle in self._particles:
                particle.set_aspect(spacing)

    def init_particles(self, channel_names: Optional[List[str]] = None,
                       particleClass: Type[ParticleBase] = ParticleBase
                       ) -> None:
        if self.mask is None:
            logging.warning("mask is missing, no particles extracted.")
            return

        if isinstance(self.mask, ImageLabeled):
            self._particles = ParticleFinder.get_particles_from_labeled_image(
                self.mask, particleClass)
        elif isinstance(self.mask, ImageBinary):
            self._particles = ParticleFinder.get_particles_from_binary_image(
                self.mask, particleClass)
        self.mask.unload()

        for pbody in self._particles:
            pbody.source = self.mask.path
        if channel_names is not None:
            for name in channel_names:
                assert name in self
                for pbody in self._particles:
                    pbody.init_intensity_features(
                        self[name][1], name)
            self.unload(name)

    @staticmethod
    def extract_particles(series: 'Series',
                          channel_list: Optional[List[str]] = None,
                          particleClass: Type[ParticleBase] = ParticleBase
                          ) -> 'Series':
        series.init_particles(channel_list, particleClass)
        return series

    def keep_particles(self, label_list: List[int]) -> None:
        self._particles = [p for p in self._particles if p.label in label_list]

    def export_particles(self, path: str, compressed: bool,
                         showProgress: bool = False) -> None:
        assert os.path.isdir(path)
        if showProgress:
            iterbar = tqdm
        else:
            def iterbar(x, *args, **kwargs):
                return x

        for nucleus in iterbar(self.particles, desc="mask"):
            nucleus.mask.to_tiff(
                os.path.join(
                    path,
                    f"mask_series{self.ID:03d}_nucleus{nucleus.label:03d}"),
                compressed)

        for channel_name in iterbar(self.names, desc="channel"):
            for nucleus in iterbar(self.particles, desc="nucleus"):
                Image(nucleus.region_of_interest.apply(
                    self[channel_name][1])).to_tiff(
                    os.path.join(path, f"{channel_name}_series{self.ID:03d}_"
                                 + f"nucleus{nucleus.label:03d}"), compressed)

    @staticmethod
    def static_export_particles(series: 'Series', path: str, compressed: bool,
                                showProgress: bool = False) -> None:
        series.export_particles(path, compressed, showProgress)

    def __str__(self):
        s = super(Series, self).__str__()
        if self._particles is not None:
            s += f"\n  With {len(self._particles)} particles "
            s += f"[{type(self._particles[0]).__name__}]."
        return s


SeriesDict = Dict[int, Series]


class SeriesList(object):
    name: str
    series: List[Series]
    label: Optional[str] = None
    __current_series: int = 0

    def __init__(self, name: str = "", series_list: List[Series] = []):
        super(SeriesList, self).__init__()
        self.series = series_list
        self.name = name

    @property
    def channel_names(self):
        return list(set(itertools.chain(*[s.names for s in self.series])))

    @staticmethod
    def __initialize_channels(
            dpath: str, series: SeriesDict, channels: List[str],
            inreg: Pattern, aspect: Optional[np.ndarray] = None,
            ground_block_side: Optional[int] = None) -> SeriesDict:
        for path in tqdm(channels, desc="initializing channels"):
            image_details = get_image_details(path, inreg)
            if image_details is None:
                continue
            sid, channel_name = image_details

            if sid not in series:
                series[sid] = Series(sid, ground_block_side)
                if aspect is not None:
                    series[sid].aspect = aspect

            if channel_name in series[sid]:
                logging.warning("found multiple instances of channel "
                                + f"{channel_name} in series {sid}. "
                                + f"Skipping '{path}'.")
                continue

            series[sid].add_channel_from_tiff(
                channel_name, os.path.join(dpath, path))
        return series

    @staticmethod
    def __initialize_masks(
            ref: str, dpath: str, series: SeriesDict, masks: List[str],
            inreg: Pattern, aspect: Optional[np.ndarray] = None,
            labeled: bool = False, ground_block_side: Optional[int] = None
            ) -> SeriesDict:
        for path in tqdm(masks, desc="initializing masks"):
            image_details = get_image_details(path, inreg)
            if image_details is None:
                continue
            sid, channel_name = image_details

            if sid not in series:
                series[sid] = Series(sid, ground_block_side)
                if aspect is not None:
                    series[sid].aspect = aspect

            if channel_name != ref:
                logging.warning("skipping mask for channel "
                                + f"'{channel_name}', "
                                + f"not reference ({ref}).")
                continue

            series[sid].add_mask_from_tiff(
                channel_name, os.path.join(dpath, path), labeled)
        return series

    @staticmethod
    def from_directory(
            dpath: str, inreg: Pattern, ref: Optional[str] = None,
            maskfix: Tuple[str, str] = ("", ""),
            aspect: Optional[np.ndarray] = None, labeled: bool = False,
            ground_block_side: Optional[int] = None):

        masks, channels = select_by_prefix_and_suffix(
            dpath, find_re(dpath, inreg), *maskfix)
        series: SeriesDict = {}

        series = SeriesList.__initialize_channels(
            dpath, series, channels, inreg, aspect, ground_block_side)

        if ref is not None:
            series = SeriesList.__initialize_masks(
                ref, dpath, series, channels, inreg,
                aspect, labeled, ground_block_side)

        clen = len(set([len(s) for s in series.values()]))
        assert 1 == clen, (
            f"inconsistent number of channels in '{dpath}' series")

        return SeriesList(os.path.basename(dpath), list(series.values()))

    def extract_particles(self, particleClass: Type[ParticleBase],
                          threads: int = 1) -> None:
        threads = ggc.args.check_threads(threads)
        if 1 == threads:
            [series.init_particles(particleClass=particleClass)
                for series in tqdm(self)]
        else:
            self.series = joblib.Parallel(n_jobs=threads, verbose=11)(
                joblib.delayed(Series.extract_particles)(
                    series, series.names, particleClass)
                for series in self)

    def export_particle_features(self, path: str) -> pd.DataFrame:
        fdata = []
        for series in self:
            if series.particles is None:
                continue
            for nucleus in series.particles:
                ndata = dict(
                    root=[self.name],
                    series_id=[series.ID],
                    nucleus_id=[nucleus.label],
                    total_size=[nucleus.total_size],
                    volume=[nucleus.volume],
                    surface=[nucleus.surface],
                    shape=[nucleus.shape])

                for name in nucleus.channel_names:
                    ndata[f"{name}_isum"] = [nucleus.get_intensity_sum(name)]
                    ndata[f"{name}_imean"] = [nucleus.get_intensity_mean(name)]
                ndata = pd.DataFrame.from_dict(ndata)
                fdata.append(ndata)

        df = pd.concat(fdata, sort=False)
        df.to_csv(path, index=False, sep="\t")
        return df

    def particle_feature_labels(self) -> Dict[str, str]:
        dfu = dict(total_size='Size (vx)', volume='Volume (nm^3)',
                   shape='Shape', surface='Surface (nm^2)',
                   sizeXY='XY size (px)', sizeZ='Z size (px)')
        for channel in self.channel_names:
            dfu[f'{channel}_isum'] = f'"{channel}" intensity sum (a.u.)'
            dfu[f'{channel}_imean'] = f'"{channel}" intensity mean (a.u.)'
        return dfu

    def get_particles(self) -> Iterator[Type[ParticleBase]]:
        for s in self:
            if s.particles is None:
                continue
            for p in s.particles:
                yield p

    def get_particle_single_px_stats(self) -> pd.DataFrame:
        box_stats = []
        for channel_name in tqdm(self.channel_names,
                                 desc='calculating channel box stats'):
            odata = pd.DataFrame.from_dict(dict(value=[0], count=[0]))
            odata.set_index('value')
            for series in self:
                if series.particles is None:
                    continue
                channel = series[channel_name][1]
                for nucleus in series.particles:
                    odata = odata.add(
                        fill_value=0,
                        other=nucleus.get_intensity_value_counts(channel))
            odata.sort_index(inplace=True)
            odata['cumsum'] = np.cumsum(odata['count'])

            q1 = quantile_from_counts(odata['value'].values,
                                      odata['cumsum'].values, .25, True)
            median = quantile_from_counts(odata['value'].values,
                                          odata['cumsum'].values, .5, True)
            q3 = quantile_from_counts(odata['value'].values,
                                      odata['cumsum'].values, .75, True)
            iqr = q3-q1
            whisk_low = max(q1-iqr, odata['value'].min())
            whisk_high = min(q3+iqr, odata['value'].max())
            outliers = np.append(
                odata['value'].values[odata['value'].values < whisk_low],
                odata['value'].values[odata['value'].values > whisk_low])

            box_stats.append(pd.DataFrame.from_dict(dict(
                root=[self.name], channel=[channel_name],
                vmin=[odata['value'].min()], vmax=[odata['value'].max()],
                whisk_low=[whisk_low], whisk_high=[whisk_high],
                q1=[q1], median=[median], q3=[q3],
                n_outliers=[len(outliers)], outliers=[outliers]
            )))
        return pd.concat(box_stats)

    def export_particle_tiffs(self, path: str, threads: int = 1,
                              compressed: bool = False) -> None:
        threads = ggc.args.check_threads(threads)
        assert os.path.isdir(path)
        if 1 == threads:
            for series in tqdm(self, desc="series"):
                series.export_particles(path, compressed)
        else:
            joblib.Parallel(n_jobs=threads, verbose=11)(
                joblib.delayed(Series.static_export_particles)(
                    series, path, compressed) for series in self)

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, i: int) -> Series:
        return self.series[i]

    def __next__(self) -> Series:
        self.__current_series += 1
        if self.__current_series > len(self):
            raise StopIteration
        else:
            return self[self.__current_series-1]

    def __iter__(self) -> Iterator[Series]:
        self.__current_series = 0
        return self
