"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import logging
import numpy as np  # type: ignore
import os
from radiantkit.image import ImageGrayScale, ImageBinary, ImageLabeled
import sys
from typing import Dict, Iterator, List, Optional, Tuple, Union


class ChannelList(object):
    """Store named Image instances (channels) with the same shape and aspect.
    A mask can be provided for background/foreground calculation."""

    _ID: int = 0
    _channels: Dict[str, ImageGrayScale]
    _ref: Optional[str] = None
    _mask: Optional[Union[ImageBinary, ImageLabeled]] = None
    _aspect: Optional[np.ndarray] = None
    _shape: Optional[Tuple[int, ...]] = None
    _ground_block_side: int = 11
    __current_channel: int = 0
    _do_rescale: bool = False

    def __init__(
        self,
        ID: int,
        ground_block_side: Optional[int] = None,
        aspect: Optional[np.ndarray] = None,
    ):
        super(ChannelList, self).__init__()
        self._ID = ID
        self._channels = {}
        if ground_block_side is not None:
            self._ground_block_side = ground_block_side
        if aspect is not None:
            self._aspect = np.array(aspect)

    @property
    def ID(self) -> int:
        return self._ID

    @property
    def names(self) -> List[str]:
        return list(self._channels)

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        return self._shape

    @property
    def aspect(self) -> Optional[np.ndarray]:
        return self._aspect

    @aspect.setter
    def aspect(self, spacing: np.ndarray) -> None:
        if spacing is None:
            return
        spacing = np.array(spacing)
        if 0 != len(self):
            for name, channel in self._channels.items():
                channel.aspect = spacing
            self._aspect = list(self._channels.values())[0].aspect
        else:
            self._aspect = spacing

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
            self._channels[name].update_ground(self.mask, self.ground_block_side)

    @property
    def reference(self) -> Optional[str]:
        return self._ref

    @property
    def mask(self) -> Optional[Union[ImageBinary, ImageLabeled]]:
        return self._mask

    @property
    def do_rescale(self) -> bool:
        return self._do_rescale

    @do_rescale.setter
    def do_rescale(self, do_rescale: bool):
        self._do_rescale = do_rescale

    @staticmethod
    def from_dict(ID: int, channel_paths: Dict[str, str]) -> "ChannelList":
        CL = ChannelList(ID)
        for name, path in channel_paths.items():
            CL.add_channel_from_tiff(name, path)
        return CL

    def is_masked(self) -> bool:
        return self._mask is not None

    def is_labeled(self) -> bool:
        return isinstance(self._mask, ImageLabeled)

    def mask_is_not_empty(self) -> bool:
        if self._mask is None:
            return False
        elif 0 == self._mask.pixels.shape[0]:
            return False
        elif 0 == self._mask.pixels.max():
            return False
        return True

    def __init_or_check_aspect(self, spacing: np.ndarray) -> None:
        if self.aspect is None:
            self._aspect = spacing
        elif any(spacing != self.aspect):
            logging.error(
                f"aspect mismatch. Expected {self.aspect} " + f"but got {spacing}"
            )
            sys.exit()

    def __init_or_check_shape(self, shape: Tuple[int, ...]) -> None:
        if self.shape is None:
            self._shape = shape
        elif shape != self.shape:
            logging.error(
                f"shape mismatch. Expected {self.shape} " + f"but got {shape}"
            )
            sys.exit()

    def add_mask(
        self, name: str, mask: Union[ImageBinary, ImageLabeled], replace: bool = False
    ) -> None:
        if name not in self._channels:
            logging.error(f"{name} channel unavailable. Mask not added.")
            return
        if self.mask is not None and not replace:
            logging.warning("mask is already present. Use replace=True to replace it.")
        self.__init_or_check_shape(mask.shape)
        self.__init_or_check_aspect(mask.aspect)
        self._mask = mask
        self._ref = name

    def add_mask_from_tiff(
        self, name: str, path: str, labeled: bool = False, replace: bool = False
    ) -> None:
        assert os.path.isfile(path)
        mask: Union[ImageBinary, ImageLabeled]
        if labeled:
            mask = ImageLabeled.from_tiff(path)
        else:
            mask = ImageBinary.from_tiff(path)
        if self.aspect is not None:
            mask.aspect = self.aspect
        mask.unload()
        self.add_mask(name, mask, replace)

    def add_channel(
        self, name: str, img: ImageGrayScale, replace: bool = False
    ) -> None:
        if name in self._channels and not replace:
            logging.warning(
                "".join(
                    [
                        f"channel {name} is already present.",
                        "Use replace=True to replace it.",
                    ]
                )
            )
        self.__init_or_check_shape(img.shape)
        self.__init_or_check_aspect(img.aspect)
        self._channels[name] = img
        if self.mask is not None:
            self._channels[name].update_ground(self.mask, self._ground_block_side)

    def add_channel_from_tiff(
        self, name: str, path: str, replace: bool = False
    ) -> None:
        assert os.path.isfile(path)
        img = ImageGrayScale.from_tiff(path, do_rescale=self.do_rescale)
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

    def __getitem__(self, name: str) -> Tuple[str, ImageGrayScale]:
        return (name, self._channels[name])

    def __next__(self) -> Tuple[str, ImageGrayScale]:
        self.__current_channel += 1

        if self.__current_channel > len(self):
            raise StopIteration
        else:
            channel_names = self.names
            channel_names.sort()
            return self[channel_names[self.__current_channel - 1]]

    def __iter__(self) -> Iterator[Tuple[str, ImageGrayScale]]:
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
