'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import logging
import numpy as np
import os
from radiantkit import const
from skimage import filters
from skimage.io import imread
from skimage.measure import label
from skimage.morphology import closing
from skimage.morphology import square, cube
from skimage.segmentation import clear_border
import tifffile
from typing import List, Tuple
import warnings

class ImageSettings(object):
    _axes_order: str = "TZYX"

    def __init__(self):
        super(ImageSettings, self).__init__()

    @property
    def nd(self) -> int:
        return len(self._axes_order)

class Image(ImageSettings):
    __path_to_local: str = None
    __pixels: np.ndarray = None

    def __init__(self, *args, **kwargs):
        super(Image, self).__init__(*args, **kwargs)

    @property
    def shape(self) -> Tuple[int]:
        return self.__pixels.shape
    
    @property
    def pixels(self) -> np.ndarray:
        if self.__pixels is None and self.__path_to_local is not None:
            self.load_from_local()
        return self.__pixels

    def from_tiff(self, path: str) -> None:
        self.__pixels = self.read_tiff(path)
        self.__path_to_local = path
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
                I = imread(path)
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

    @staticmethod
    def save_tiff(path: str, I: np.ndarray, dtype: str, compressed: bool,
        bundled_axes: str="CZYX", inMicrons: bool=False,
        ResolutionZ: float=None, forImageJ: bool=False, **kwargs):
        
        if not "C" in bundled_axes: bundled_axes = f"C{bundled_axes}"
        while len(bundled_axes) > len(I.shape):
            new_shape = [1]
            [new_shape.append(n) for n in I.shape]
            I.shape = new_shape

        assert_msg = "shape mismatch between bundled axes and image."
        assert len(bundled_axes) == len(I.shape), assert_msg

        metadata = {'axes' : bundled_axes}
        if inMicrons: metadata['unit'] = "um"
        if not type(None) == ResolutionZ: metadata['spacing'] = ResolutionZ

        if compressed:
            tifffile.imsave(path, I.astype(dtype),
                shape = I.shape, compress = 9,
                dtype = dtype, imagej = forImageJ,
                metadata = metadata, **kwargs)
        else:
            tifffile.imsave(path, I.astype(dtype),
                shape = I.shape, compress = 0,
                dtype = dtype, imagej = forImageJ,
                metadata = metadata, **kwargs)

    @staticmethod
    def z_project(I: np.ndarray, projection_type: const.ProjectionType):
        if projection_type == const.ProjectionType.SUM_PROJECTION:
            I = I.sum(0).astype(I.dtype)
        elif projection_type == const.ProjectionType.MAX_PROJECTION:
            I = I.max(0).astype(I.dtype)
        return I

    @staticmethod
    def close(I: np.ndarray) -> np.ndarray:
        assert 1 == I.max()
        if 2 == len(I.shape):
            I = closing(I, square(3))
        elif 3 == len(i.shape):
            I = closing(I, cube(3))
        else:
            logging.info("Close operation not implemented for images with " +
                f"{len(I.shape)} dimensions.")
        return I

    @staticmethod
    def threshold_adaptive(I: np.ndarray, block_size: int,
        method: str, mode: str, *args, **kwargs) -> np.ndarray:
        assert 0 == block_size % 2

        def threshold_adaptive_slice(I: np.ndarray, block_size: int,
            method: str, mode: str, *args, **kwargs) -> np.ndarray:
            threshold = filters.threshold_local(I, block_size, *args,
                method = method, mode = mode, **kwargs)
            return I >= threshold

        if 2 == len(i.shape):
            mask = threshold_adaptive_slice(I, block_size,
                method, mode, *args, **kwargs)
        elif 3 == len(i.shape):
            mask = I.copy()
            for slice_id in range(mask.shape[0]):
                mask[slice_id, :, :] = threshold_adaptive_slice(
                    mask[slice_id, :, :], block_size, method, mode,
                    *args, **kwargs)
        else:
            logging.info("Local threshold not implemented for images with " +
                f"{len(I.shape)} dimensions.")
        return mask

    @staticmethod
    def clear_XY_borders(self, I: np.ndarray) -> np.ndarray:
        if 2 == len(I.shape):
            return clear_border(I)
        elif 3 == len(I.shape):
            border_labels = []
            border_labels.extend(np.unique(I[:, 0, :]).tolist())
            border_labels.extend(np.unique(I[:, -1, :]).tolist())
            border_labels.extend(np.unique(I[:, :, 0]).tolist())
            border_labels.extend(np.unique(I[:, :, -1]).tolist())
            border_labels = set(border_labels)
            for lab in border_labels: I[I == lab] = 0
            if 1 != I.max(): I = label(I)
            return I
        else:
            logging.warning("XY border clearing not implemented for images " +
                f"with {len(I.shape)} dimensions.")
            return I

    @staticmethod
    def clear_Z_borders(self, I: np.ndarray) -> np.ndarray:
        if 2 == len(I.shape): return I
        elif 3 == len(I.shape):
            border_labels = []
            border_labels.extend(np.unique(I[0, :, :]).tolist())
            border_labels.extend(np.unique(I[-1, :, :]).tolist())
            border_labels = set(border_labels)
            for lab in border_labels: I[I == lab] = 0
            if 1 != I.max(): I = label(I)
            return I
        else:
            logging.warning("Z border clearing not implemented for images " +
                f"with {len(I.shape)} dimensions.")
            return I

    def load_from_local(self) -> None:
        assert self.__path_to_local is not None
        assert os.path.isfile(self.__path_to_local)
        self.from_tiff(self.__path_to_local)

    def unload(self) -> None:
        if self.__path_to_local is None:
            logging.error("cannot unload Image without path_to_local.")
            return
        if not os.path.isfile(self.__path_to_local):
            logging.error("path_to_local not found, cannot unload: " +
                self.__path_to_local)
            return
        self.__pixels = None

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

def get_dtype(imax):
    depths = [8, 16]
    for depth in depths:
        if imax <= 2**depth-1:
            return("uint%d" % (depth,))
    return("uint")
