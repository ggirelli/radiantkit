'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import logging
import os
from radiantkit import image
from radiantkit import path
import re
import sys
from typing import Dict, List, Optional, Pattern, Tuple, Type

class SeriesSettings(object):
    _ID: int=0
    _channel_data: Dict[str, Type[image.Image]]={}
    _mask_data: Optional[Dict]=None
    _ref: Optional[str]=None

    def __init__(self, ID: int, channel_paths: Dict[str,str],
        mask_path: Optional[str]=None, inreg: Optional[Pattern]=None):
        super(SeriesSettings, self).__init__()
        
        ref = path.get_image_details(mask_path, inreg)[1]
        assert ref in channel_paths

        self._ID = ID
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
    def mask_data(self):
        if self.has_mask: return self._mask_data.copy()

    def has_ref(self):
        return self._ref is not None

    def has_mask(self):
        return self._mask_data is not None

    def __repr__(self):
        s = f"Series #{self._ID} with {len(self._channel_data)} channels."
        if not self.has_ref():
            s += " No reference."
        else:
            s += f" '{self._ref}' reference channel"
            if self.has_mask():
                s += " (with mask)"
        for (name, data) in self._channel_data.items():
            s += f"\n  {name} => '{data['path']}'"
        if self.has_mask:
            s += f"\n  mask => '{self.mask_data['path']}'"
        return s

class Series(SeriesSettings):
    def __init__(self, ID: int, channel_paths: Dict[str,str],
        mask_path: Optional[str]=None, inreg: Optional[Pattern]=None):
        for (channel_name, channel_path) in channel_paths.items():
            assert os.path.isfile(channel_path)
        super(Series, self).__init__(ID, channel_paths, mask_path, inreg)

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

        return series_list
