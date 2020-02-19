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
import pickle
from radiantkit.distance import CenterType, RadialDistanceCalculator
from radiantkit.image import ImageBinary, ImageLabeled, Image
from radiantkit.path import find_re, get_image_details
from radiantkit.path import select_by_prefix_and_suffix
from radiantkit.particle import Nucleus, Particle, ParticleFinder
from radiantkit import stat
import sys
from tqdm import tqdm  # type: ignore
from typing import Dict, List, Tuple
from typing import Iterator, Optional, Pattern, Type, Union

ChannelName = str
DistanceType = str
ChannelRadialProfileData = Dict[
    DistanceType, Tuple[stat.PolyFitResult, pd.DataFrame]]
RadialProfileData = Dict[ChannelName, ChannelRadialProfileData]


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

    def __init__(self, ID: int, ground_block_side: Optional[int] = None,
                 aspect: Optional[np.ndarray] = None):
        super(ChannelList, self).__init__()
        self._ID = ID
        self._channels = {}
        if ground_block_side is not None:
            self._ground_block_side = ground_block_side
        self._aspect = aspect

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

    def __repr__(self) -> str:
        s = f"Series #{self._ID} with {len(self)} channels."
        for name, channel in self:
            s += f"\n  {name} => '{channel.path}'"
            s += f" [loaded:{channel.loaded}]"
        if self.mask is not None:
            s += f"\n  mask({self.reference}) => '{self.mask.path}'"
            s += f" [loaded:{self.mask.loaded}]"
        return s


class Series(ChannelList):
    _particles: List[Nucleus]

    def __init__(self, ID: int, ground_block_side: Optional[int] = None,
                 aspect: Optional[np.ndarray] = None):
        super(Series, self).__init__(ID, ground_block_side)
        self._particles = []

    @property
    def particles(self) -> List[Nucleus]:
        if 0 == len(self._particles):
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

    def __init_particles_intensity_features(
            self, channel_names: Optional[List[str]] = None):
        if channel_names is not None:
            for name in channel_names:
                assert name in self
                for pbody in self._particles:
                    pbody.init_intensity_features(
                        self[name][1], name)
            self.unload(name)

    def __run_particle_finder(
            self, particleClass: Type[Particle] = Particle) -> None:
        if isinstance(self.mask, ImageLabeled):
            self._particles = ParticleFinder.get_particles_from_labeled_image(
                self.mask, particleClass)
        elif isinstance(self.mask, ImageBinary):
            self._particles = ParticleFinder.get_particles_from_binary_image(
                self.mask, particleClass)

    def init_particles(self,
                       particleClass: Type[Particle] = Particle,
                       channel_list: Optional[List[str]] = None,
                       reInit: bool = False) -> None:
        if 0 != len(self._particles) and not reInit:
            return

        if self.mask is None:
            logging.warning("mask is missing, no particles extracted.")
            return

        self.__run_particle_finder(particleClass)
        self.mask.unload()

        for pbody in self._particles:
            pbody.source = self.mask.path
        self.__init_particles_intensity_features(channel_list)

    @staticmethod
    def extract_particles(series: 'Series',
                          particleClass: Type[Particle] = Particle,
                          channel_list: Optional[List[str]] = None
                          ) -> 'Series':
        series.init_particles(particleClass, channel_list)
        return series

    def keep_particles(self, label_list: List[int]) -> None:
        self._particles = [p for p in self._particles if p.label in label_list]

    def export_particles(self, path: str, compressed: bool) -> None:
        assert os.path.isdir(path)

        for channel_name in self.names:
            for nucleus in self.particles:
                basename = f"series{self.ID:03d}_nucleus{nucleus.label:03d}"

                nucleus.mask.to_tiff(os.path.join(path,
                                     f"mask_{basename}.tif"), compressed)

                if nucleus.has_distances():
                    lamina_dist, center_dist = nucleus.distances

                    Image(center_dist).to_tiff(
                        os.path.join(path, f"centerDist_{basename}.tif"),
                        compressed)
                    Image(lamina_dist).to_tiff(
                        os.path.join(path, f"laminaDist_{basename}.tif"),
                        compressed)

                Image(nucleus.region_of_interest.apply(
                    self[channel_name][1])).to_tiff(
                    os.path.join(path, f"{channel_name}_{basename}.tif"),
                    compressed)
            self.unload(channel_name)

    @staticmethod
    def static_export_particles(series: 'Series', path: str,
                                compressed: bool) -> None:
        series.export_particles(path, compressed)

    def init_particles_distances(
            self, rdc: RadialDistanceCalculator, reInit: bool = False) -> None:
        C = None
        if self.reference is not None and (
                rdc.center_type is CenterType.CENTER_OF_MASS):
            C = self[self.reference][1]
        for particle in self._particles:
            if not particle.has_distances() or reInit:
                particle.init_distances(rdc, C)
        if C is not None:
            C.unload()

    def get_particles_intensity_at_distance(
            self, channel_name: str) -> pd.DataFrame:
        assert channel_name in self.names
        assert all([p.has_distances for p in self._particles])
        df = pd.concat([p.get_intensity_at_distance(self[channel_name][1])
                        for p in self._particles])
        self.unload(channel_name)
        df['channel'] = channel_name
        df['series_label'] = self.ID
        return df

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
    _reference: Optional[str] = None

    def __init__(self, name: str = "", series_list: List[Series] = [],
                 ref: Optional[str] = None):
        super(SeriesList, self).__init__()
        self.series = series_list
        self.name = name
        self._reference = ref

    @property
    def channel_names(self) -> List[str]:
        return list(set(itertools.chain(*[s.names for s in self.series])))

    @property
    def reference(self) -> Optional[str]:
        return self._reference

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
                series[sid] = Series(sid, ground_block_side, aspect)

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
            inreg: Pattern, labeled: bool = False,
            aspect: Optional[np.ndarray] = None,
            ground_block_side: Optional[int] = None) -> SeriesDict:
        for path in tqdm(masks, desc="initializing masks"):
            image_details = get_image_details(path, inreg)
            if image_details is None:
                continue
            sid, channel_name = image_details

            if sid not in series:
                series[sid] = Series(sid, ground_block_side, aspect)

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
                ref, dpath, series, masks, inreg,
                labeled, aspect, ground_block_side)

        clen = len(set([len(s) for s in series.values()]))
        assert 1 == clen, (
            f"inconsistent number of channels in '{dpath}' series")

        return SeriesList(os.path.basename(dpath), list(series.values()), ref)

    def extract_particles(self, particleClass: Type[Particle],
                          channel_list: Optional[List[str]] = None,
                          threads: int = 1) -> None:
        threads = ggc.args.check_threads(threads)
        if 1 == threads:
            [series.init_particles(particleClass, channel_list)
                for series in tqdm(self)]
        else:
            self.series = joblib.Parallel(n_jobs=threads, verbose=11)(
                joblib.delayed(Series.extract_particles)(
                    series, particleClass, channel_list)
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
                    shape=[nucleus.shape()])

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

    def particles(self) -> Iterator[Particle]:
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

            q1 = stat.quantile_from_counts(
                odata['value'].values, odata['cumsum'].values, .25, True)
            median = stat.quantile_from_counts(
                odata['value'].values, odata['cumsum'].values, .5, True)
            q3 = stat.quantile_from_counts(
                odata['value'].values, odata['cumsum'].values, .75, True)
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

    def __prep_single_channel_profile(
            self, channel_name: ChannelName, rdc: RadialDistanceCalculator,
            nbins: int = 200, deg: int = 5, reInit: bool = False
            ) -> ChannelRadialProfileData:
        channel_idata_dflist = []
        for s in self.series:
            s.init_particles_distances(rdc, reInit)
            channel_idata_dflist.append(
                s.get_particles_intensity_at_distance(channel_name))
        channel_intensity_data = pd.concat(channel_idata_dflist)

        logging.info("fitting polynomial curve")
        return dict(
            lamina_dist=stat.radial_fit(
                channel_intensity_data['lamina_dist'],
                channel_intensity_data['ivalue'],
                nbins, deg),
            center_dist=stat.radial_fit(
                channel_intensity_data['center_dist'],
                channel_intensity_data['ivalue'],
                nbins, deg),
            lamina_dist_norm=stat.radial_fit(
                channel_intensity_data['lamina_dist_norm'],
                channel_intensity_data['ivalue'],
                nbins, deg))

    def get_radial_profiles(
            self, rdc: RadialDistanceCalculator,
            nbins: int = 200, deg: int = 5,
            reInit: bool = False
            ) -> RadialProfileData:
        profiles: RadialProfileData = {}
        for channel_name in tqdm(self.channel_names, desc="channel"):
            logging.info(f"extracting vx values for channel '{channel_name}'")
            profiles[channel_name] = self.__prep_single_channel_profile(
                channel_name, rdc, nbins, deg, reInit)
        return profiles

    def to_pickle(self, dpath: str, pickle_name: str = "radiant.pkl") -> None:
        assert os.path.isdir(dpath)
        pickle_path = os.path.join(dpath, pickle_name)
        with open(pickle_path, "wb") as PO:
            pickle.dump(self, PO)

    def unload(self, name: Optional[str] = None) -> None:
        for series in self.series:
            series.unload(name)

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
