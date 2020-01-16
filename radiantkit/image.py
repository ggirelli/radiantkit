'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import numpy as np
import os
from skimage.io import imread
from typing import List, Tuple
import warnings

class ImageSettings(object):
    """docstring for ImageSettings"""

    _axes_order: str = "TZYX"

    def __init__(self):
        super(ImageSettings, self).__init__()

    @property
    def nd(self) -> int:
        return len(self._axes_order)

class Image(ImageSettings):
    """docstring for Image"""

    __pixels: np.ndarray = None

    def __init__(self, *args, **kwargs):
        super(Image, self).__init__(*args, **kwargs)

    @property
    def shape(self) -> Tuple[int]:
        return self.__pixels.shape
    
    @property
    def pixels(self) -> np.ndarray:
        return self.__pixels

    def from_tiff(self, path: str) -> None:
        self.__pixels = self.read_tiff(path)
        self.__update_axes_order_according_to_input()

    def from_matrix(self, m: np.ndarray) -> None:
        self.__pixels = m.copy()
        self.__update_axes_order_according_to_input()
    
    def __update_axes_order_according_to_input(self) -> None:
        self.__pixels = self.extract_nd(self.__pixels, self.nd)
        if len(self.__pixels.shape) != self.nd:
            self._axes_order = self._axes_order[-self.nd:]

    @staticmethod
    def read_tiff(path: str) -> np.ndarray:
        assert os.path.isfile(path), f"file not found: '{path}'"
        try:
            with warnings.catch_warnings(record=True) as warning_list:
                I = imread(impath)
                warning_list = [str(e) for e in warning_list]
                if any(["axes do not match shape" in e for e in warning_list]):
                    print(f"WARNING: "+
                        "image axes do not match metadata in '{path}'")
                    print("Using the image axes.")
        except (ValueError, TypeError) as e:
            print(f"ERROR: "+
                "cannot read image from '{path}', file seems corrupted.")
            raise
        return I

    @staticmethod
    def extract_nd(I: np.ndarray, nd: int) -> np.ndarray:
        if len(I.shape) <= nd: return I
        while len(I.shape) > nd: I = I[0]
        if 0 in I.shape: print("WARNING: the image contains empty dimensions.")
        return I

    def clear_borders(self, dimensions: List[int]) -> None:
        pass

class Image3D(Image):
    """docstring for Image3D"""

    _axes_order: str = "ZYX"

    def __init__(self, arg):
        super(Image3D, self).__init__()
        self.arg = arg

class MaskedImage(ImageSettings):
    """docstring for MaskedImage"""
    def __init__(self, arg):
        super(MaskedImage, self).__init__()
        self.arg = arg

def get_huygens_rescaling_factor(path: str) -> float:
    basename,ext = tuple(os.path.splitext(os.path.basename(path)))
    path = os.path.join(os.path.dirname(path), f"{basename}_history.txt")
    if not os.path.exists(path): return 1
    needle = 'Stretched to Integer type'
    with open(path, 'r') as fhistory:
        factor = fhistory.readlines()
        factor = [x for x in factor if needle in x]
    if 0 == len(factor): return 1
    elif 1 == len(factor): return float(factor[0].strip().split(' ')[-1])
    else: return np.prod([float(f.strip().split(' ')[-1]) for f in factor])
