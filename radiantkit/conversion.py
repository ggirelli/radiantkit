'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from nd2reader import ND2Reader
from nd2reader.parser import Parser as ND2Parser
import numpy as np
from typing import Set

class ND2Reader2(ND2Reader):
    def __init__(self, filename):
        super().__init__(filename)

    def field_count(self) -> int:
        if not 'v' in self.axes: return 1
        return self.sizes['v']

    def isLive(self) -> bool:
        if "t" in self.axes: return 1 < self.sizes["t"]
        return False

    def is3D(self) -> bool:
        return 'z' in self.axes

    def hasMultiChannels(self) -> bool:
        if "c" in self.axes: return 1 < self.sizes['c']
        return False

    def channel_count(self) -> int:
        if not 'c' in self.axes: return 1
        return self.sizes['c']

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

