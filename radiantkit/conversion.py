'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
from czifile import CziFile
from nd2reader import ND2Reader
from nd2reader.parser import Parser as ND2Parser
import numpy as np
from typing import Iterable, Optional, Set, Tuple
import warnings
import xml.etree.ElementTree as ET

class ND2Reader2(ND2Reader):
    def __init__(self, filename):
        super(ND2Reader2, self).__init__(filename)

    def field_count(self) -> int:
        if not 'v' in self.axes: return 1
        return self.sizes['v']

    def isLive(self) -> bool:
        if "t" in self.axes: return 1 < self.sizes["t"]
        return False

    def is3D(self) -> bool:
        return 'z' in self.axes

    def hasMultiChannels(self) -> bool:
        if "c" in self.axes: return 1 < self.channel_count()
        return False

    def get_channel_names(self) -> Iterable[str]:
        for channel in self.metadata['channels']:
            yield channel.lower()

    def channel_count(self) -> int:
        if not 'c' in self.sizes: n = 1
        else: n = self.sizes["c"]
        assert (len(list(self.get_channel_names())) == n
            ), "channel count mismatch."
        return n

    def set_axes_for_bundling(self):
        if self.is3D():
            self.bundle_axes = "zyxc" if self.hasMultiChannels() else "zyx"
        else:
            self.bundle_axes = "yxc" if "c" in self.axes else "yx"

    @staticmethod
    def get_resolutionZ(nd2path: str, field_id: int) -> Set[int]:
        with open(nd2path, "rb") as ND2H:
            parser = ND2Parser(ND2H)
            Zdata = np.array(parser._raw_metadata.z_data)
            Zlevels = np.array(parser.metadata['z_levels']).astype('int')
            Zlevels = Zlevels + len(Zlevels) * field_id
            Zdata = Zdata[Zlevels]
            return set(np.round(np.diff(Zdata), 3))

class CziFile2(CziFile):
    __pixels: Optional[np.ndarray] = None

    def __init__(self, filename):
        super(CziFile2, self).__init__(filename)
    
    @property
    def pixels(self) -> np.ndarray:
        if self.__pixels is None:
            with warnings.catch_warnings(record = True) as w:
                self.__pixels = self.asarray()
        return self.__pixels

    def field_count(self) -> int:
        if not "S" in self.axes: return 1
        return self.pixels.shape[self.axes.index("S")]

    def isLive(self) -> bool:
        if "T" in self.axes: return 1 < self.shape[self.axes.index("T")]
        return False

    def is3D(self) -> bool:
        return 'Z' in self.axes

    def hasMultiChannels(self) -> bool:
        if "C" in self.axes: return 1 < self.channel_count()
        return False

    def get_channel_names(self) -> Iterable[str]:
        channel_path = "Metadata/DisplaySetting/Channels/Channel/DyeName"
        for x in ET.fromstring(self.metadata()).findall(channel_path):
            yield x.text.replace(" ", "").lower()

    def channel_count(self) -> int:
        if not 'C' in self.axes: n = 1
        else: n = self.pixels.shape[self.axes.index("C")]
        assert (len(list(self.get_channel_names())) == n
            ), "channel count mismatch."
        return n

    def get_resolution(self) -> dict:
        resolution_path = "Metadata/Scaling/Items/Distance"
        resolution = []
        for x in ET.fromstring(self.metadata()).findall(resolution_path):
            if x.attrib['Id'] in ["X", "Y", "Z"]:
                resolution.append((x.attrib['Id'], float(x[0].text)))
        return dict(resolution)

    def squeeze_axes(self, skip: str) -> None:
        axes = list(self.axes)
        for axis in axes:
            axis_id = axes.index(axis)
            if axis in skip: continue
            self.__pixels = np.squeeze(self.pixels, axis_id)
            self.shape = self.pixels.shape
            axes.pop(axis_id)
        self.axes = "".join(axes)

    def reorder_axes(self, bundle_axes: str) -> None:
        if self.axes == bundle_axes: return
        bundle_axes = list(bundle_axes)
        assert len(bundle_axes) == len(self.axes)
        assert all([axis in self.axes for axis in bundle_axes])
        self.__pixels = np.moveaxis(self.pixels, range(len(axes)),
            [bundle_axes.index(axis) for axis in self.axes])
        self.shape = self.pixels.shape
        self.axes =  bundle_axes

    def get_channel_pixels(self, args: argparse.Namespace,
        field_id: Optional[int] = None) -> Iterable[Tuple[np.ndarray, int]]:
        if field_id is not None: field = self.pixels[field_id, :]
        else:
            assert "C" == self.axes[0]
            field = self.pixels
        for channel_id  in range(self.channel_count()):
            yield (field[channel_id], channel_id)
