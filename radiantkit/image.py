'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from enum import auto, Enum
import logging
import numpy as np
import os
from radiantkit import const
from scipy import ndimage as ndi
import skimage as ski
from skimage.morphology import square, cube
from skimage.morphology import closing, opening
from skimage.morphology import dilation, erosion
from skimage.segmentation import clear_border
import tifffile
from typing import List, Optional, Tuple, Union
import warnings

class ImageSettings(object):
    _ALLOWED_AXES: str="VCTZYX"
    _axes_order: str="VCTZYX"
    _aspect: np.ndarray=np.ones(6)

    def __init__(self):
        super(ImageSettings, self).__init__()
    
    @property
    def nd(self) -> int:
        return len(self._axes_order)

    @property
    def aspect(self) -> np.ndarray:
        return self._aspect

    @aspect.setter
    def aspect(self, spacing: np.ndarray) -> None:
        if len(self.aspect) == len(spacing): self._aspect = spacing
        elif len(spacing) < len(self._axes_order):
            self.aspect[-len(spacing):] = spacing
            logging.warning(f"aspect changed to {self.aspect} " +
                f"(used only last {len(self.aspect)} values)")
        else:
            self.aspect = spacing[-len(self.aspect):]
            logging.warning(f"aspect changed to {self.aspect} " +
                f"(used only last {len(self.aspect)} values)")

class ImageBase(ImageSettings):
    _path_to_local: Optional[str] = None
    _pixels: Optional[np.ndarray] = None
    _shape: Tuple[int]=None

    def __init__(self, pixels: np.ndarray, path: Optional[str]=None,
        axes: Optional[str]=None):
        super(ImageSettings, self).__init__()
        assert len(pixels.shape) <= len(self._ALLOWED_AXES)
        self._pixels = pixels.copy()
        self._remove_empty_axes()
        self._shape = self._pixels.shape
        if axes is not None:
            assert len(axes) == len(self.shape)
            assert all([c in self._ALLOWED_AXES for c in axes])
            assert all([1 == axes.count(c) for c in set(axes)])
            self._axes_order = axes
        else:
            self._axes_order = self._ALLOWED_AXES[-len(self.shape):]
        self._aspect = self._aspect[-len(self.shape):]
        if path is not None:
            if os.path.isfile(path): self._path_to_local = path

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    @property
    def pixels(self) -> np.ndarray:
        if self._pixels is None and self._path_to_local is not None:
            self.load_from_local()
        return self._pixels

    @property
    def dtype(self) -> str:
        return get_dtype(self.pixels.max())

    @property
    def path(self):
        return self._path_to_local

    @property
    def axes(self) -> str:
        return self._axes_order

    @axes.setter
    def axes(self, new_axes: str) -> None:
        assert all([c in self.__ALLOWED_AXES for c in new_axes])
        assert all([1 == c.count(new_axes) for c in set(new_axes)])
        if len(new_axes) < len(self.axes):
            for c in self.axes:
                if not c in new_axes:
                    axes_id = self.axes.index(c)
                    self._pixels = np.delete(self.pixels,
                        slice(1, self.pixels.shape[axes_id]), axes_id)
                    self._pixels = np.squeeze(self.pixels, axes_id)
                    self._aspect.pop(axes_id)
        
        while len(new_axes) > len(self.axes):
            for c in new_axes:
                if not c in self.axes:
                    self._axes_order = f"{c}{self.axes}"
                    new_shape = [1]
                    new_shape.extend(self._pixels.shape)
                    self._pixels.shape = new_shape
                    self._aspect = np.append(1, self._aspect)

        if new_axes != self.axes:
            assert len(new_axes) == len(self.axes)
            assert all([c in new_axes for c in self.axes])
            new_axes_order = [new_axes.index(c) for c in self.axes]
            self._pixels = np.moveaxis(self.pixels,
                range(len(self.axes)), new_axes_order)
            self._shape = self._pixels.shape
            self._aspect = self._aspect[new_axes_order]
            self._axes_order = new_axes

    @property
    def loaded(self):
        if not self.is_loadable(): return True
        else: return self._pixels is not None

    @staticmethod
    def from_tiff(path: str) -> 'ImageBase':
        return ImageBase(read_tiff(path), path)
    
    def _extract_nd(self) -> None:
        self._pixels = extract_nd(self._pixels, self.nd)
        assert len(self._pixels.shape) <= self.nd
        if len(self._pixels.shape) != self.nd:
            self._axes_order = self._axes_order[-self.nd:]

    def _remove_empty_axes(self) -> None:
        if len(self.pixels.shape) != self.nd: self._extract_nd()
        while 1 == self.pixels.shape[0]:
            new_shape = list(self.pixels.shape)
            new_shape.pop(0)
            self.pixels.shape = new_shape
            self._axes_order = self._axes_order[1:]

    def axis_shape(self, axis: str) -> int:
        if not axis in self._axes_order: return None
        return self.shape[self._axes_order.index(axis)]

    def z_project(self, projection_type: const.ProjectionType) -> np.ndarray:
        return z_project(self.pixels, projection_type)

    def is_loadable(self) -> bool:
        return self.path is not None and os.path.isfile(self.path)

    def load_from_local(self) -> None:
        assert self._path_to_local is not None
        assert os.path.isfile(self._path_to_local)
        self._pixels = self.from_tiff(self._path_to_local).pixels

    def unload(self) -> None:
        if self._path_to_local is None:
            logging.error("cannot unload ImageBase without path_to_local.")
            return
        if not os.path.isfile(self._path_to_local):
            logging.error("path_to_local not found, cannot unload: " +
                self._path_to_local)
            return
        self._pixels = None

    def to_tiff(self, path: str, compressed: bool,
        bundle_axes: Optional[str]=None, inMicrons: bool=False,
        ResolutionZ: Optional[float]=None, forImageJ:
        bool=False, **kwargs) -> None:
        if bundle_axes is None: bundle_axes = self._axes_order
        save_tiff(path, self.pixels, self.dtype, compressed, bundle_axes,
            inMicrons, ResolutionZ, forImageJ, **kwargs)

    def __repr__(self) -> str:
        s = f"{self.nd}D {self.__class__.__name__}: "
        s += f"{'x'.join([str(d) for d in self.shape])} [{self.axes}]"
        if self.loaded: s += ' [loaded]'
        else: s += ' [unloaded]'
        if self.is_loadable(): s += f"; From '{self._path_to_local}'"
        return s

class ImageLabeled(ImageBase):
    def __init__(self, pixels: np.ndarray, path: Optional[str]=None,
        axes: Optional[str]=None, doRelabel: bool=True):
        super(ImageLabeled, self).__init__(pixels, path, axes)
        if doRelabel: self._relabel()

    @property
    def max(self):
        return self.pixels.max()

    @staticmethod
    def from_tiff(path: str, axes: Optional[str]=None,
        doRelabel: bool=True) -> 'ImageLabeled':
        return ImageLabeled(read_tiff(path), path, axes, doRelabel)

    def _relabel(self) -> None:
        self._pixels = self._pixels.copy() > self._pixels.min()
        self._pixels = ski.measure.label(self._pixels)

    def clear_XY_borders(self) -> None:
        self._pixels = clear_XY_borders(self._pixels)

    def clear_Z_borders(self) -> None:
        self._pixels = clear_Z_borders(self._pixels)

    def sizeXY(self, lab: int) -> int:
        return self.size(lab, "XY")

    def sizeZ(self, lab: int) -> int:
        return self.size(lab, "Z")

    def size(self, lab: int, axes: str) -> int:
        assert all([axis in self.axes for axis in axes])
        axes_ids = tuple([self.axes.index(axis)
            for axis in self.axes if axis not in axes])
        return (self.pixels == lab).max(axes_ids).sum()

    def __remove_labels_by_size(self, labels: List[int],
        sizes: List[int], pass_range: Tuple[Union[int,float]],
        axes: str="total") -> None:
        assert 2 == len(pass_range)
        assert pass_range[0] <= pass_range[1]

        sizes = np.array(sizes)
        labels = np.array(labels)
        filtered = np.logical_or(sizes < pass_range[0], sizes > pass_range[1])

        logging.info(f"removing {filtered.sum()}/{self.max} labels " +
            f"outside of {axes} size range {pass_range}")
        logging.debug(np.array((labels, sizes)))
        self._pixels[np.isin(self.pixels, labels[filtered])] = 0
        self._pixels = ski.measure.label(self.pixels)
        logging.info(f"retained {self.max} labels")

    def filter_size(self, axes: str,
        pass_range: Tuple[Union[int,float]]) -> None:
        labels = np.unique(self.pixels)
        labels = labels[0 != labels]
        sizes = []
        for current_label in labels:
            logging.debug(f"Calculating {axes} size for label {current_label}")
            sizes.append(self.size(current_label, axes))
        self.__remove_labels_by_size(labels, sizes, pass_range, axes)

    def filter_total_size(self, pass_range: Tuple[Union[int,float]]) -> None:
        labels, sizes = np.unique(self.pixels, return_counts=True)
        self.__remove_labels_by_size(labels, sizes, pass_range)

    def inherit_labels(self, mask2d: 'ImageLabeled') -> None:
        self._pixels = inherit_labels(self, mask2d)

    def binary(self) -> 'ImageBinary':
        return ImageBinary(self.pixels, self._path_to_local, self._axes_order)

    def __repr__(self) -> str:
        s = super(ImageLabeled, self).__repr__()
        s += f"; Max label: {self.pixels.max}"
        return s

class ImageBinary(ImageBase):
    def __init__(self, pixels: np.ndarray, path: Optional[str]=None,
        axes: Optional[str]=None, doRebinarize: bool=True):
        super(ImageBinary, self).__init__(pixels, path, axes)
        if doRebinarize: self._rebinarize()
        assert 1 == self.pixels.max()

    @staticmethod
    def from_tiff(path: str, axes: Optional[str]=None,
        doRebinarize: bool=True) -> 'ImageBinary':
        return ImageBinary(read_tiff(path), path, axes, doRebinarize)

    def _rebinarize(self) -> None:
        self._pixels = self.pixels > self.pixels.min()

    def fill_holes(self) -> None:
        self._pixels = fill_holes(self.pixels)

    def close(self, block_side: int=3) -> None:
        self._pixels = closing2(self.pixels, block_side)

    def open(self, block_side: int=3) -> None:
        self._pixels = opening2(self.pixels, block_side)

    def dilate(self, block_side: int=3) -> None:
        self._pixels = dilate(self.pixels, block_side)

    def erode(self, block_side: int=3) -> None:
        self._pixels = erode(self.pixels, block_side)

    def logical_and(self, B: 'ImageBinary') -> None:
        self._pixels = np.logical_and(self.pixels, B.pixels)

    def logical_or(self, B: 'ImageBinary') -> None:
        self._pixels = np.logical_or(self.pixels, B.pixels)

    def logical_xor(self, B: 'ImageBinary') -> None:
        self._pixels = np.logical_xor(self.pixels, B.pixels)

    def invert(self) -> None:
        self._pixels = np.logical_not(self.pixels)

    def label(self) -> ImageLabeled:
        return ImageLabeled(self.pixels)

    def to_tiff(self, path: str, compressed: bool,
        bundle_axes: Optional[str]=None, inMicrons: bool=False,
        ResolutionZ: Optional[float]=None, forImageJ:
        bool=False, **kwargs) -> None:
        if bundle_axes is None: bundle_axes = self._axes_order
        save_tiff(path, self.pixels*np.iinfo(self.dtype).max, self.dtype,
            compressed, bundle_axes, inMicrons, ResolutionZ, forImageJ,
            **kwargs)

    def __repr__(self) -> str:
        s = super(ImageBinary, self).__repr__()
        s += f"; Foreground voxels: {self.pixels.sum()}"
        s += f"; Background voxels: {(self.pixels*(-1)+1).sum()}"
        return s

class Image(ImageBase):
    _rescale_factor: float = 1.
    _background: float=None
    _foreground: float=None

    def __init__(self, pixels: np.ndarray, path: Optional[str]=None,
        axes: Optional[str]=None):
        super(Image, self).__init__(pixels, path, axes)
    
    @property
    def ground(self):
        return (self._background, self._foreground)

    @property
    def rescale_factor(self) -> float:
        return self._rescale_factor
    
    @rescale_factor.setter
    def rescale_factor(self, new_factor: float) -> None:
        self._pixels = self.pixels*self.rescale_factor
        self.__rescale_factor = new_factor
        self._pixels = self.pixels/self.rescale_factor

    @staticmethod
    def from_tiff(path: str, axes: Optional[str]=None,
        doRescale: bool=True) -> 'Image':
        I = Image(read_tiff(path), path, axes)
        if doRescale: I.rescale_factor = I.get_huygens_rescaling_factor()
        return I

    def get_huygens_rescaling_factor(self) -> float:
        if self._path_to_local is None: return 1.
        return get_huygens_rescaling_factor(self._path_to_local)

    def threshold_global(self, thr: Union[int,float]) -> ImageBinary:
        return ImageBinary(self.pixels>thr, doRebinarize=False)

    def threshold_adaptive(self, block_size: int,
        method: str, mode: str, *args, **kwargs) -> ImageBinary:
        return ImageBinary(threshold_adaptive(self.pixels, block_size,
            method, mode, *args, **kwargs), doRebinarize=False)

    def update_ground(self, M: ImageBinary, block_side: int=11) -> None:
        M = dilate(M.pixels, block_side)

        self._foreground = np.median(self.pixels[M])
        self._background = np.median(self.pixels[np.logical_not(M)])

    def __repr__(self) -> str:
        s = super(Image, self).__repr__()
        if self.ground[0] is not None:
            s += f"\nBack/foreground: {self.ground}"
        return s

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

def get_dtype(imax: Union[int,float]) -> str:
    depths = [8, 16]
    for depth in depths:
        if imax <= 2**depth-1:
            return("uint%d" % (depth,))
    return("uint")

def read_tiff(path: str) -> np.ndarray:
    assert os.path.isfile(path), f"file not found: '{path}'"
    try:
        with warnings.catch_warnings(record=True) as warning_list:
            I = ski.io.imread(path)
            warning_list = [str(e) for e in warning_list]
            if any(["axes do not match shape" in e for e in warning_list]):
                logging.warning(f"image axes do not match metadata in '{path}'")
                logging.warning("using the image axes.")
    except (ValueError, TypeError) as e:
        logging.critical(f"cannot read image '{path}', file seems corrupted.")
        raise
    return I

def extract_nd(I: np.ndarray, nd: int) -> np.ndarray:
    if len(I.shape) <= nd: return I
    while len(I.shape) > nd: I = I[0]
    if 0 in I.shape: logging.warning("the image contains empty dimensions.")
    return I

def save_tiff(path: str, I: np.ndarray, dtype: str, compressed: bool,
    bundle_axes: str="CTZYX", inMicrons: bool=False,
    ResolutionZ: Optional[float]=None, forImageJ: bool=False, **kwargs) -> None:
    
    while len(bundle_axes) > len(I.shape):
        new_shape = [1]
        [new_shape.append(n) for n in I.shape]
        I.shape = new_shape

    assert_msg = "shape mismatch between bundled axes and image."
    assert len(bundle_axes) == len(I.shape), assert_msg

    metadata = {'axes' : bundle_axes}
    if inMicrons: metadata['unit'] = "um"
    if ResolutionZ is not None: metadata['spacing'] = ResolutionZ

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

def z_project(I: np.ndarray,
    projection_type: const.ProjectionType) -> np.ndarray:
    if projection_type == const.ProjectionType.SUM_PROJECTION:
        I = I.sum(0).astype(I.dtype)
    elif projection_type == const.ProjectionType.MAX_PROJECTION:
        I = I.max(0).astype(I.dtype)
    return I

def threshold_adaptive(I: np.ndarray, block_size: int,
    method: str, mode: str, *args, **kwargs) -> np.ndarray:
    assert 1 == block_size % 2

    def threshold_adaptive_slice(I: np.ndarray, block_size: int,
        method: str, mode: str, *args, **kwargs) -> np.ndarray:
        threshold = ski.filters.threshold_local(I, block_size, *args,
            method = method, mode = mode, **kwargs)
        return I >= threshold

    if 2 == len(I.shape):
        mask = threshold_adaptive_slice(I, block_size,
            method, mode, *args, **kwargs)
    elif 3 == len(I.shape):
        mask = I.copy()
        for slice_id in range(mask.shape[0]):
            logging.debug(f"ADAPT_THR SLICE#({slice_id})")
            mask[slice_id, :, :] = threshold_adaptive_slice(
                mask[slice_id, :, :], block_size, method, mode,
                *args, **kwargs)
    else:
        logging.info("Local threshold not implemented for images with " +
            f"{len(I.shape)} dimensions.")
    return mask

def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = ndi.binary_fill_holes(mask)
    if 3 == len(mask.shape):
        for slice_id in range(mask.shape[0]):
            logging.debug(f"FILL_HOLES SLICE#({slice_id})")
            mask[slice_id, :, :] = ndi.binary_fill_holes(mask[slice_id, :, :])
    elif 2 != len(mask.shape):
        logging.warning("3D hole filling not performed on images with " +
            f"{len(mask.shape)} dimensions.")
    return mask

def closing2(mask: np.ndarray, block_side: int=3) -> np.ndarray:
    assert 1 == mask.max()
    if 2 == len(mask.shape):
        mask = closing(mask, square(block_side))
    elif 3 == len(mask.shape):
        mask = closing(mask, cube(block_side))
    else:
        logging.info("Close operation not implemented for images with " +
            f"{len(mask.shape)} dimensions.")
    return mask

def opening2(mask: np.ndarray, block_side: int=3) -> np.ndarray:
    assert 1 == mask.max()
    if 2 == len(mask.shape):
        mask = opening(mask, square(block_side))
    elif 3 == len(mask.shape):
        mask = opening(mask, cube(block_side))
    else:
        logging.info("Open operation not implemented for images with " +
            f"{len(mask.shape)} dimensions.")
    return mask

def dilate(mask: np.ndarray, block_side: int=3) -> np.ndarray:
    assert 1 == mask.max()
    if 2 == len(mask.shape):
        mask = dilation(mask, square(block_side))
    elif 3 == len(mask.shape):
        mask = dilation(mask, cube(block_side))
    else:
        logging.info("Dilate operation not implemented for images with " +
            f"{len(mask.shape)} dimensions.")
    return mask

def erode(mask: np.ndarray, block_side: int=3) -> np.ndarray:
    assert 1 == mask.max()
    if 2 == len(mask.shape):
        mask = erosion(mask, square(block_side))
    elif 3 == len(mask.shape):
        mask = erosion(mask, cube(block_side))
    else:
        logging.info("Erode operation not implemented for images with " +
            f"{len(mask.shape)} dimensions.")
    return mask

def clear_XY_borders(L: np.ndarray) -> np.ndarray:
    if 2 == len(L.shape):
        return clear_border(L)
    elif 3 == len(L.shape):
        border_labels = []
        border_labels.extend(np.unique(L[:, 0, :]).tolist())
        border_labels.extend(np.unique(L[:, -1, :]).tolist())
        border_labels.extend(np.unique(L[:, :, 0]).tolist())
        border_labels.extend(np.unique(L[:, :, -1]).tolist())
        border_labels = set(border_labels)
        for lab in border_labels: L[L == lab] = 0
        return ski.measure.label(L)
    else:
        logging.warning("XY border clearing not implemented for images " +
            f"with {len(L.shape)} dimensions.")
        return L

def clear_Z_borders(L: np.ndarray) -> np.ndarray:
    if 2 == len(L.shape): return L
    elif 3 == len(L.shape):
        border_labels = []
        border_labels.extend(np.unique(L[0, :, :]).tolist())
        border_labels.extend(np.unique(L[-1, :, :]).tolist())
        border_labels = set(border_labels)
        for lab in border_labels: L[L == lab] = 0
        return ski.measure.label(L)
    else:
        logging.warning("Z border clearing not implemented for images " +
            f"with {len(L.shape)} dimensions.")
        return L

def inherit_labels(mask: Union[ImageBinary, ImageLabeled],
    mask2d: Union[ImageBinary, ImageLabeled]) -> ImageLabeled:
    assert 2 == len(mask2d.shape)
    if 2 == len(mask.shape):
        assert mask2d.shape == mask.shape
        return mask2d.pixels[np.logical_and(mask.pixels>0, mask2d.pixels>0)]
    elif 3 == len(mask.shape):
        assert mask2d.shape == mask[-2:].shape
        new_mask = mask.pixels.copy()
        for slice_id in range(mask.shape[0]):
            new_mask[slice_id,:,:] = mask2d.pixels[np.logical_and(
                mask.pixels[slice_id,:,:]>0, mask2d.pixels>0)]
        return ImageLabeled(new_mask, doRelabel=False)
    else:
        self.logger.warning("mask combination not allowed for images " +
            f"with {len(mask.shape)} dimensions.")
        return mask
