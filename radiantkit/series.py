"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
import itertools
from joblib import cpu_count, delayed, Parallel  # type: ignore
import logging
import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore
import pickle
from radiantkit.distance import CenterType, RadialDistanceCalculator
from radiantkit.channel import ImageGrayScale, ChannelList
from radiantkit.image import ImageBinary, ImageLabeled
from radiantkit.path import find_re, get_image_details
from radiantkit.path import select_by_prefix_and_suffix
from radiantkit.particle import Nucleus, Particle, ParticleFinder
from radiantkit.scripts import argtools
from radiantkit import stat
from rich.progress import track  # type: ignore
from typing import Dict, List, Tuple
from typing import Iterator, Optional, Pattern, Type

ChannelName = str
DistanceType = str
ChannelRadialProfileData = Dict[DistanceType, Tuple[stat.PolyFitResult, pd.DataFrame]]
RadialProfileData = Dict[ChannelName, ChannelRadialProfileData]


class Series(ChannelList):
    _particles: List[Nucleus]

    def __init__(
        self,
        ID: int,
        ground_block_side: Optional[int] = None,
        aspect: Optional[np.ndarray] = None,
    ):
        super(Series, self).__init__(ID, ground_block_side, aspect)
        self._particles = []

    @property
    def particles(self) -> List[Nucleus]:
        if 0 == len(self._particles):
            logging.warning(
                "particle attribute accessible after running '.extract_particles()'."
            )
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
                particle.aspect(spacing)

    def __init_particles_intensity_features(
        self, channel_names: Optional[List[str]] = None
    ):
        if channel_names is not None:
            for name in channel_names:
                assert name in self
                for pbody in self._particles:
                    pbody.init_intensity_features(self[name][1], name)
            self.unload(name)

    def __run_particle_finder(self, particleClass: Type[Particle] = Particle) -> None:
        if self.mask is None or 0 == self.mask.pixels.max():
            self._particles = []
            return
        if isinstance(self.mask, ImageLabeled):
            self._particles = ParticleFinder.get_particles_from_labeled_image(
                self.mask, particleClass
            )
        elif isinstance(self.mask, ImageBinary):
            self._particles = ParticleFinder.get_particles_from_binary_image(
                self.mask, particleClass
            )

    def init_particles(
        self,
        particleClass: Type[Particle] = Particle,
        channel_list: Optional[List[str]] = None,
        reInit: bool = False,
    ) -> None:
        logging.info(f"initializing particles in series {self.ID}")
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
    def extract_particles(
        series: "Series",
        particleClass: Type[Particle] = Particle,
        channel_list: Optional[List[str]] = None,
    ) -> "Series":
        series.init_particles(particleClass, channel_list)
        return series

    def keep_particles(self, label_list: List[int]) -> None:
        self._particles = [p for p in self._particles if p.idx in label_list]

    def export_particles(self, path: str, compressed: bool) -> None:
        assert os.path.isdir(path)

        for channel_name in self.names:
            for nucleus in self.particles:
                basename = f"series{self.ID:03d}_nucleus{nucleus.idx:03d}"

                nucleus.to_tiff(os.path.join(path, f"mask_{basename}.tif"), compressed)

                if nucleus.has_distances():
                    center_dist, lamina_dist = nucleus.distances

                    ImageGrayScale(center_dist).to_tiff(
                        os.path.join(path, f"centerDist_{basename}.tif"), compressed
                    )
                    ImageGrayScale(lamina_dist).to_tiff(
                        os.path.join(path, f"laminaDist_{basename}.tif"), compressed
                    )

                ImageGrayScale(nucleus.roi.apply(self[channel_name][1])).to_tiff(
                    os.path.join(path, f"{channel_name}_{basename}.tif"), compressed
                )
            self.unload(channel_name)

    @staticmethod
    def static_export_particles(series: "Series", path: str, compressed: bool) -> None:
        series.export_particles(path, compressed)

    def init_particles_distances(
        self, rdc: RadialDistanceCalculator, reInit: bool = False
    ) -> "Series":
        C = None
        if self.reference is not None and (
            rdc.center_type is CenterType.CENTER_OF_MASS
        ):
            C = self[self.reference][1]
        for particle in self._particles:
            if not particle.has_distances() or reInit:
                particle.init_distances(rdc, C)
        if C is not None:
            C.unload()
        return self

    @staticmethod
    def static_init_particles_distances(
        series: "Series", rdc: RadialDistanceCalculator, reInit: bool = False
    ) -> "Series":
        return series.init_particles_distances(rdc, reInit)

    def get_particles_intensity_at_distance(self, channel_name: str) -> pd.DataFrame:
        assert channel_name in self.names
        assert all([p.has_distances for p in self._particles])

        if self.reference is not None and self.reference != channel_name:
            df = pd.concat(
                [
                    p.get_intensity_at_distance(
                        self[channel_name][1], self[self.reference][1]
                    )
                    for p in self._particles
                ]
            )
            self.unload(self.reference)
        else:
            df = pd.concat(
                [
                    p.get_intensity_at_distance(self[channel_name][1])
                    for p in self._particles
                ]
            )
        self.unload(channel_name)

        df["reference"] = self.reference
        df["channel"] = channel_name
        df["series_label"] = self.ID
        return df

    @staticmethod
    def static_get_particles_intensity_at_distance(
        series: "Series", channel_name: str
    ) -> pd.DataFrame:
        return series.get_particles_intensity_at_distance(channel_name)

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
    def channel_names(self) -> List[str]:
        return list(set(itertools.chain(*[s.names for s in self.series])))

    @staticmethod
    def __initialize_channels(
        dpath: str,
        series: SeriesDict,
        channels: List[str],
        inreg: Pattern,
        aspect: Optional[np.ndarray] = None,
        ground_block_side: Optional[int] = None,
        do_rescale: bool = False,
    ) -> SeriesDict:
        for path in track(channels, description="initializing channels"):
            image_details = get_image_details(path, inreg)
            if image_details is None:
                continue
            sid, channel_name = image_details

            if sid not in series:
                series[sid] = Series(sid, ground_block_side, aspect)
            series[sid].do_rescale = do_rescale

            if channel_name in series[sid]:
                logging.warning(
                    "found multiple instances of channel "
                    + f"{channel_name} in series {sid}. "
                    + f"Skipping '{path}'."
                )
                continue

            series[sid].add_channel_from_tiff(channel_name, os.path.join(dpath, path))
        return series

    @staticmethod
    def __initialize_masks(
        ref: str,
        dpath: str,
        series: SeriesDict,
        masks: List[str],
        inreg: Pattern,
        labeled: bool = False,
        aspect: Optional[np.ndarray] = None,
        ground_block_side: Optional[int] = None,
    ) -> SeriesDict:
        for path in track(masks, description="initializing masks"):
            image_details = get_image_details(path, inreg)
            if image_details is None:
                continue
            sid, channel_name = image_details

            if sid not in series:
                series[sid] = Series(sid, ground_block_side, aspect)

            if channel_name != ref:
                logging.warning(
                    "skipping mask for channel "
                    + f"'{channel_name}', "
                    + f"not reference ({ref})."
                )
                continue

            series[sid].add_mask_from_tiff(
                channel_name, os.path.join(dpath, path), labeled
            )
        return series

    @staticmethod
    def from_directory(
        dpath: str,
        inreg: Pattern,
        ref: Optional[str] = None,
        maskfix: Tuple[str, str] = ("", ""),
        aspect: Optional[np.ndarray] = None,
        labeled: bool = False,
        ground_block_side: Optional[int] = None,
        do_rescale: bool = False,
    ):

        masks, channels = select_by_prefix_and_suffix(
            dpath, find_re(dpath, inreg), *maskfix
        )
        series: SeriesDict = {}

        series = SeriesList.__initialize_channels(
            dpath, series, channels, inreg, aspect, ground_block_side, do_rescale
        )

        if ref is not None:
            series = SeriesList.__initialize_masks(
                ref, dpath, series, masks, inreg, labeled, aspect, ground_block_side
            )

        clen = len(set([len(s) for s in series.values()]))
        assert 1 == clen, f"inconsistent number of channels in '{dpath}' series"

        return SeriesList(os.path.basename(dpath), list(series.values()))

    def extract_particles(
        self,
        particleClass: Type[Particle],
        channel_list: Optional[List[str]] = None,
        threads: int = 1,
    ) -> None:
        threads = cpu_count() if threads > cpu_count() else threads
        if 1 == threads:
            for series in track(self):
                series.init_particles(particleClass, channel_list)
        else:
            self.series = Parallel(n_jobs=threads, verbose=11)(
                delayed(Series.extract_particles)(series, particleClass, channel_list)
                for series in self
            )

    def export_particle_features(self, path: str) -> pd.DataFrame:
        fdata = []
        for series in self:
            if series.particles is None:
                continue
            for nucleus in series.particles:
                ndata = dict(
                    root=[self.name],
                    series_id=[series.ID],
                    nucleus_id=[nucleus.idx],
                    total_size=[nucleus.total_size],
                    volume=[nucleus.volume],
                    surface=[nucleus.surface],
                    shape=[nucleus.shape_descriptor()],
                )

                for name in nucleus.channel_names:
                    ndata[f"{name}_isum"] = [nucleus.get_intensity_sum(name)]
                    ndata[f"{name}_imean"] = [nucleus.get_intensity_mean(name)]
                ndata = pd.DataFrame.from_dict(ndata)
                fdata.append(ndata)

        df = pd.concat(fdata, sort=False)
        df.to_csv(path, index=False, sep="\t")
        return df

    def particle_feature_labels(self) -> Dict[str, str]:
        dfu = dict(
            total_size="Size (vx)",
            volume="Volume (nm^3)",
            shape="Shape",
            surface="Surface (nm^2)",
            sizeXY="XY size (px)",
            sizeZ="Z size (px)",
        )
        for channel in self.channel_names:
            dfu[f"{channel}_isum"] = f'"{channel}" intensity sum (a.u.)'
            dfu[f"{channel}_imean"] = f'"{channel}" intensity mean (a.u.)'
        return dfu

    def particles(self) -> Iterator[Particle]:
        for s in self:
            if s.particles is None:
                continue
            for p in s.particles:
                yield p

    def get_particle_single_px_stats(self) -> pd.DataFrame:
        box_stats = []
        for channel_name in track(
            self.channel_names, description="calculating channel box stats"
        ):
            odata = pd.DataFrame.from_dict(dict(value=[0], count=[0]))
            odata.set_index("value")
            for series in self:
                if series.particles is None:
                    continue
                channel = series[channel_name][1]
                for nucleus in series.particles:
                    odata = odata.add(
                        fill_value=0, other=nucleus.get_intensity_value_counts(channel)
                    )
            odata.sort_index(inplace=True)
            odata["cumsum"] = np.cumsum(odata["count"])

            q1 = stat.quantile_from_counts(
                odata["value"].values, odata["cumsum"].values, 0.25, True
            )
            median = stat.quantile_from_counts(
                odata["value"].values, odata["cumsum"].values, 0.5, True
            )
            q3 = stat.quantile_from_counts(
                odata["value"].values, odata["cumsum"].values, 0.75, True
            )
            iqr = q3 - q1
            whisk_low = max(q1 - iqr, odata["value"].min())
            whisk_high = min(q3 + iqr, odata["value"].max())
            outliers = np.append(
                odata["value"].values[odata["value"].values < whisk_low],
                odata["value"].values[odata["value"].values > whisk_low],
            )

            box_stats.append(
                pd.DataFrame.from_dict(
                    dict(
                        root=[self.name],
                        channel=[channel_name],
                        vmin=[odata["value"].min()],
                        vmax=[odata["value"].max()],
                        whisk_low=[whisk_low],
                        whisk_high=[whisk_high],
                        q1=[q1],
                        median=[median],
                        q3=[q3],
                        n_outliers=[len(outliers)],
                        outliers=[outliers],
                    )
                )
            )
        return pd.concat(box_stats)

    def export_particle_tiffs(
        self, path: str, threads: int = 1, compressed: bool = False
    ) -> None:
        threads = cpu_count() if threads > cpu_count() else threads
        assert os.path.isdir(path)
        if 1 == threads:
            for series in track(self, description="series"):
                series.export_particles(path, compressed)
        else:
            Parallel(n_jobs=threads, verbose=11)(
                delayed(Series.static_export_particles)(series, path, compressed)
                for series in self
            )

    def __retrieve_channel_intensity_at_distance(
        self,
        channel_name: str,
        rdc: RadialDistanceCalculator,
        threads: int = 1,
        reInit: bool = False,
    ) -> pd.DataFrame:
        assert threads > 0
        if 1 == threads:
            channel_idata_dflist = []
            for s in self.series:
                s.init_particles_distances(rdc, reInit)
                channel_idata_dflist.append(
                    s.get_particles_intensity_at_distance(channel_name)
                )
            return pd.concat(channel_idata_dflist)
        else:
            self.series = Parallel(n_jobs=threads, verbose=0)(
                delayed(Series.static_init_particles_distances)(s, rdc, reInit)
                for s in self.series
            )
            return pd.concat(
                Parallel(n_jobs=threads, verbose=0)(
                    delayed(Series.static_get_particles_intensity_at_distance)(
                        s, channel_name
                    )
                    for s in self.series
                )
            )

    def __prep_single_channel_profile(
        self,
        channel_name: ChannelName,
        rdc: RadialDistanceCalculator,
        nbins: int = 200,
        deg: int = 5,
        threads: int = 1,
        reInit: bool = False,
    ) -> List[Tuple[ChannelName, ChannelRadialProfileData]]:
        logging.info(
            f"extracting vx values for channel '{channel_name}'"
            + f" [threads:{threads}]"
        )

        channel_intensity_data = self.__retrieve_channel_intensity_at_distance(
            channel_name, rdc, threads, reInit
        )

        logging.info("fitting polynomial curve")
        profiles = [
            (
                channel_name,
                dict(
                    lamina_dist=stat.radial_fit(
                        channel_intensity_data["lamina_dist"],
                        channel_intensity_data["ivalue"],
                        nbins,
                        deg,
                    ),
                    center_dist=stat.radial_fit(
                        channel_intensity_data["center_dist"],
                        channel_intensity_data["ivalue"],
                        nbins,
                        deg,
                    ),
                    lamina_dist_norm=stat.radial_fit(
                        channel_intensity_data["lamina_dist_norm"],
                        channel_intensity_data["ivalue"],
                        nbins,
                        deg,
                    ),
                ),
            )
        ]

        if "ivalue_norm" in channel_intensity_data.columns:
            logging.info("fitting normalized polynomial curve")
            profiles.append(
                (
                    f"{channel_name}_over_ref",
                    dict(
                        lamina_dist=stat.radial_fit(
                            channel_intensity_data["lamina_dist"],
                            channel_intensity_data["ivalue_norm"],
                            nbins,
                            deg,
                        ),
                        center_dist=stat.radial_fit(
                            channel_intensity_data["center_dist"],
                            channel_intensity_data["ivalue_norm"],
                            nbins,
                            deg,
                        ),
                        lamina_dist_norm=stat.radial_fit(
                            channel_intensity_data["lamina_dist_norm"],
                            channel_intensity_data["ivalue_norm"],
                            nbins,
                            deg,
                        ),
                    ),
                )
            )

        return profiles

    def get_radial_profiles(
        self,
        rdc: RadialDistanceCalculator,
        nbins: int = 200,
        deg: int = 5,
        reInit: bool = False,
        threads: int = 1,
    ) -> RadialProfileData:
        profiles: RadialProfileData = {}
        for channel_name in track(self.channel_names, description="channel"):
            profiles.update(
                self.__prep_single_channel_profile(
                    channel_name, rdc, nbins, deg, threads, reInit
                )
            )
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
        return sorted(self.series, key=lambda s: s.ID)[i]

    def __next__(self) -> Series:
        self.__current_series += 1
        if self.__current_series > len(self):
            raise StopIteration
        else:
            return self[self.__current_series - 1]

    def __iter__(self) -> Iterator[Series]:
        self.__current_series = 0
        return self


def init_series_list(
    args: argparse.Namespace,
) -> Tuple[argparse.Namespace, SeriesList]:
    pickled = False
    series_list = None
    pickle_path = os.path.join(args.input, args.pickle_name)
    args = argtools.set_default_args_for_series_init(args)

    if os.path.exists(pickle_path):
        if not args.import_instance:
            logging.info(f"found '{args.pickle_name}' file in input folder.")
            logging.info("use --import-instance flag to unpickle it.")
        if args.import_instance:
            with open(pickle_path, "rb") as PI:
                series_list = pickle.load(PI)
                pickled = True

    if series_list is None:
        logging.info("parsing series folder")
        series_list = SeriesList.from_directory(
            args.input,
            args.inreg,
            args.ref_channel,
            (args.mask_prefix, args.mask_suffix),
            args.aspect,
            args.labeled,
            args.block_side,
            args.do_rescaling,
        )

    logging.info(
        f"parsed {len(series_list)} series with "
        + f"{len(series_list.channel_names)} channels each"
        + f": {series_list.channel_names}"
    )

    args = argtools.check_parallelization_and_pickling(args, pickled)

    return args, series_list


def pickle_series_list(args: argparse.Namespace, series_list: SeriesList) -> None:
    if args.export_instance:
        logging.info("Pickling instance")
        series_list.unload()
        series_list.to_pickle(args.input, args.pickle_name)
