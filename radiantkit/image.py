"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from enum import Enum
import logging
import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore
from radiantkit import const, stat
from radiantkit.deconvolution import get_deconvolution_rescaling_factor
from scipy import ndimage as ndi  # type: ignore
import skimage as ski  # type: ignore
from skimage.morphology import square, cube  # type: ignore
from skimage.morphology import closing, opening
from skimage.morphology import dilation, erosion
from skimage.segmentation import clear_border  # type: ignore
import tifffile as tf  # type: ignore
from typing import Any, Dict, List, Optional, Tuple, Union


class ImageBase(object):
    _ALLOWED_AXES: str = const.default_axes
    _axes_order: str = const.default_axes
    _aspect: np.ndarray = np.ones(6)

    def __init__(self):
        super(ImageBase, self).__init__()

    @property
    def nd(self) -> int:
        return len(self._axes_order)

    @property
    def aspect(self) -> np.ndarray:
        return self._aspect

    @aspect.setter
    def aspect(self, spacing: np.ndarray) -> None:
        if spacing is None:
            return
        spacing = np.array(spacing)
        aspect_slice = slice(len(self.aspect) - len(spacing), len(self.aspect))
        if len(self.aspect) == len(spacing):
            self._aspect = spacing
        elif len(spacing) < len(self._axes_order):
            self.aspect[aspect_slice] = spacing
            logging.warning(
                f"aspect changed to {self.aspect} "
                + f"(used only last {len(self.aspect)} values)"
            )
        else:
            self.aspect = spacing[aspect_slice]
            logging.warning(
                f"aspect changed to {self.aspect} "
                + f"(used only last {len(self.aspect)} values)"
            )


class Image(ImageBase):
    _path_to_local: Optional[str] = None
    _pixels: np.ndarray
    _shape: Tuple[int]

    def __init__(
        self, pixels: np.ndarray, path: Optional[str] = None, axes: Optional[str] = None
    ):
        super(ImageBase, self).__init__()
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
            self._axes_order = self._ALLOWED_AXES[
                slice(
                    len(self._ALLOWED_AXES) - len(self.shape), len(self._ALLOWED_AXES)
                )
            ]
        self._aspect = self._aspect[
            slice(len(self.aspect) - len(self.shape), len(self.aspect))
        ]
        if path is not None:
            if os.path.isfile(path):
                self._path_to_local = path

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

    def __remove_axes(self, new_axes: str) -> None:
        while any([a not in new_axes for a in self.axes]):
            for c in self.axes:
                if c not in new_axes:
                    axes_id = self.axes.index(c)
                    self._pixels = np.delete(
                        self.pixels, slice(1, self.pixels.shape[axes_id]), axes_id
                    )
                    self._pixels = np.squeeze(self.pixels, axes_id)
                    self._aspect.pop(axes_id)
                    break

    def __add_axes(self, new_axes: str) -> None:
        while any([a not in self.axes for a in new_axes]):
            for c in new_axes:
                if c not in self.axes:
                    self._axes_order = f"{c}{self.axes}"
                    new_shape = [1]
                    new_shape.extend(self.pixels.shape)
                    self.pixels.shape = new_shape
                    self._aspect = np.append(1, self._aspect)
                    break

    def __reorder_axes(self, new_axes: str) -> None:
        assert len(self.axes) == len(new_axes)
        assert all([a in self.axes for a in new_axes])
        if new_axes != self.axes:
            assert len(new_axes) == len(self.axes)
            assert all([c in new_axes for c in self.axes])
            new_axes_order = [new_axes.index(c) for c in self.axes]
            self._pixels = np.moveaxis(
                self.pixels, range(len(self.axes)), new_axes_order
            )
            self._shape = self._pixels.shape
            self._aspect = self._aspect[new_axes_order]
            self._axes_order = new_axes

    @property
    def axes(self) -> str:
        return self._axes_order

    @axes.setter
    def axes(self, new_axes: str) -> None:
        assert all([c in self._ALLOWED_AXES for c in new_axes])
        assert all([1 == c.count(new_axes) for c in set(new_axes)])
        self.__remove_axes(new_axes)
        self.__add_axes(new_axes)
        self.__reorder_axes(new_axes)

    @property
    def loaded(self):
        if not self.is_loadable():
            return True
        else:
            return self._pixels is not None

    @staticmethod
    def from_tiff(path: str) -> "Image":
        return Image(read_tiff(path), path)

    def _extract_nd(self) -> None:
        self._pixels = extract_nd(self._pixels, self.nd)
        assert len(self._pixels.shape) <= self.nd
        if len(self._pixels.shape) != self.nd:
            self._axes_order = self._axes_order[
                slice(len(self._axes_order) - self.nd, len(self._axes_order))
            ]

    def _remove_empty_axes(self) -> None:
        if len(self.pixels.shape) != self.nd:
            self._extract_nd()
        while 1 == self.pixels.shape[0]:
            new_shape = list(self.pixels.shape)
            new_shape.pop(0)
            self.pixels.shape = new_shape
            self._axes_order = self._axes_order[1:]

    def axis_shape(self, axis: str) -> Optional[int]:
        if axis not in self._axes_order:
            return None
        return self.shape[self._axes_order.index(axis)]

    def flatten(
        self,
        axes_to_keep: str,
        projection_type: const.ProjectionType = const.ProjectionType.SUM,
    ) -> "Image":
        axes_to_flatten = tuple(
            [ai for ai in range(len(self.axes)) if self.axes[ai] not in axes_to_keep]
        )
        if projection_type is const.ProjectionType.SUM:
            pixels = self.pixels.sum(axes_to_flatten, keepdims=True)
        elif projection_type is const.ProjectionType.MAX:
            pixels = self.pixels.max(axes_to_flatten, keepdims=True)
        else:
            raise ValueError
        return self.from_this(pixels)

    def z_project(self, projection_type: const.ProjectionType) -> np.ndarray:
        return z_project(self.pixels, projection_type)

    def tile_to(self, shape: Tuple[int, ...]) -> "Image":
        return self.from_this(tile_to(self.pixels, shape))

    def is_loadable(self) -> bool:
        return self.path is not None and os.path.isfile(self.path)

    def load_from_local(self) -> None:
        assert self._path_to_local is not None
        assert os.path.isfile(self._path_to_local), self._path_to_local
        self._pixels = self.from_tiff(self._path_to_local).pixels

    def unload(self) -> None:
        if self._path_to_local is None:
            logging.error("cannot unload Image without path_to_local.")
            return
        if not os.path.isfile(self._path_to_local):
            logging.error(
                "path_to_local not found, cannot unload: " + self._path_to_local
            )
            return
        self._pixels = None

    def to_tiff(
        self,
        path: str,
        compressed: bool,
        bundle_axes: Optional[str] = None,
        inMicrons: bool = False,
        z_resolution: Optional[float] = None,
        forImageJ: bool = False,
        **kwargs,
    ) -> None:
        if bundle_axes is None:
            bundle_axes = self._axes_order
        save_tiff(
            path,
            self.pixels.astype(self.dtype),
            compressed,
            bundle_axes,
            inMicrons,
            z_resolution,
            forImageJ,
            **kwargs,
        )

    def offset(self, offset: int) -> np.ndarray:
        return self.from_this(offset2(self.pixels, offset))

    def copy(self) -> "Image":
        return self.from_this(self.pixels, True)

    def from_this(self, pixels: np.ndarray, keepPath: bool = False) -> "Image":
        if keepPath:
            I2 = type(self)(pixels, self._path_to_local, self.axes)
        else:
            I2 = type(self)(pixels, axes=self.axes)
        I2.aspect = self.aspect
        return I2

    def __repr__(self) -> str:
        s = f"{self.nd}D {self.__class__.__name__}: "
        s += f"{'x'.join([str(d) for d in self.shape])} [{self.axes}, "
        s += f"aspect: {'x'.join([str(d) for d in self.aspect])} nm]"
        if self.loaded:
            s += " [loaded]"
        else:
            s += " [unloaded]"
        if self.is_loadable():
            s += f"; From '{self._path_to_local}'"
        return s


class ImageLabeled(Image):
    def __init__(
        self,
        pixels: np.ndarray,
        path: Optional[str] = None,
        axes: Optional[str] = None,
        doRelabel: bool = True,
    ):
        super(ImageLabeled, self).__init__(pixels, path, axes)
        if doRelabel:
            self._relabel()

    @property
    def max(self):
        return self.pixels.max()

    @staticmethod
    def from_tiff(
        path: str, axes: Optional[str] = None, doRelabel: bool = True
    ) -> "ImageLabeled":
        return ImageLabeled(read_tiff(path), path, axes, doRelabel)

    def _relabel(self) -> None:
        self._pixels = self.pixels.copy() > self.pixels.min()
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
        axes_ids = tuple(
            [self.axes.index(axis) for axis in self.axes if axis not in axes]
        )
        return (self.pixels == lab).max(axes_ids).sum()

    def __remove_labels_by_size(
        self,
        labels: List[int],
        sizes: np.ndarray,
        pass_range: stat.Interval,
        axes: str = "total",
    ) -> None:
        assert 2 == len(pass_range)
        assert pass_range[0] <= pass_range[1]

        labels = np.array(labels)
        filtered = np.logical_or(sizes < pass_range[0], sizes > pass_range[1])

        logging.info(
            f"removing {filtered.sum()}/{self.max} labels "
            + f"outside of {axes} size range {pass_range}"
        )
        logging.debug(np.array((labels, sizes)))
        self.pixels[np.isin(self.pixels, labels[filtered])] = 0
        self._pixels = ski.measure.label(self.pixels)
        logging.info(f"retained {self.max} labels")

    def filter_size(
        self, axes: str, pass_range: Tuple[Union[int, float], Union[int, float]]
    ) -> None:
        labels = np.unique(self.pixels)
        labels = labels[0 != labels]
        sizes = []
        for current_label in labels:
            logging.debug(f"Calculating {axes} size for label {current_label}")
            sizes.append(self.size(current_label, axes))
        self.__remove_labels_by_size(labels, np.array(sizes), pass_range, axes)

    def filter_total_size(self, pass_range: Tuple[float, float]) -> None:
        labels, sizes = np.unique(self.pixels, return_counts=True)
        self.__remove_labels_by_size(labels, sizes, pass_range)

    def inherit_labels(self, mask2d: "ImageLabeled") -> None:
        self._pixels = inherit_labels(self, mask2d)

    def binarize(self) -> "ImageBinary":
        B = ImageBinary(self.pixels, None, self._axes_order)
        B.aspect = self.aspect
        return B

    def __repr__(self) -> str:
        s = super(ImageLabeled, self).__repr__()
        s += f"; Max label: {self.pixels.max}"
        return s


class ImageBinary(Image):
    _background: float = 0
    _foreground: float = 0

    def __init__(
        self,
        pixels: np.ndarray,
        path: Optional[str] = None,
        axes: Optional[str] = None,
        doRebinarize: bool = True,
    ):
        super(ImageBinary, self).__init__(pixels, path, axes)
        if doRebinarize:
            self._rebinarize()
        assert 1 == self.pixels.max()
        self._foreground = self.pixels.sum()
        self._background = np.prod(self.pixels.shape) - self._foreground

    @property
    def background(self):
        return self._background

    @property
    def foreground(self):
        return self._foreground

    @staticmethod
    def from_tiff(
        path: str, axes: Optional[str] = None, doRebinarize: bool = True
    ) -> "ImageBinary":
        return ImageBinary(read_tiff(path), path, axes, doRebinarize)

    def _rebinarize(self) -> None:
        self._pixels = self.pixels > self.pixels.min()

    def fill_holes(self) -> None:
        self._pixels = fill_holes(self.pixels)

    def close(self, block_side: int = 3) -> None:
        self._pixels = closing2(self.pixels, block_side)

    def open(self, block_side: int = 3) -> None:
        self._pixels = opening2(self.pixels, block_side)

    def dilate(self, block_side: int = 3) -> None:
        self._pixels = dilate(self.pixels, block_side)

    def erode(self, block_side: int = 3) -> None:
        self._pixels = erode(self.pixels, block_side)

    def dilate_fill_erode(self, block_side: int = 3) -> None:
        self.dilate(block_side)
        self.fill_holes()
        self.erode(block_side)

    def logical_and(self, B: "ImageBinary") -> None:
        self._pixels = np.logical_and(self.pixels, B.pixels)

    def logical_or(self, B: "ImageBinary") -> None:
        self._pixels = np.logical_or(self.pixels, B.pixels)

    def logical_xor(self, B: "ImageBinary") -> None:
        self._pixels = np.logical_xor(self.pixels, B.pixels)

    def invert(self) -> None:
        self._pixels = np.logical_not(self.pixels)

    def label(self) -> ImageLabeled:
        L = ImageLabeled(self.pixels, None, self._axes_order)
        L.aspect = self.aspect
        return L

    def to_tiff(
        self,
        path: str,
        compressed: bool,
        bundle_axes: Optional[str] = None,
        inMicrons: bool = False,
        z_resolution: Optional[float] = None,
        forImageJ: bool = False,
        **kwargs,
    ) -> None:
        if bundle_axes is None:
            bundle_axes = self._axes_order
        save_tiff(
            path,
            (self.pixels * np.iinfo(self.dtype).max).astype(self.dtype),
            compressed,
            bundle_axes,
            inMicrons,
            z_resolution,
            forImageJ,
            **kwargs,
        )

    def __repr__(self) -> str:
        s = super(ImageBinary, self).__repr__()
        s += f"; Foreground voxels: {self.foreground}"
        s += f"; Background voxels: {self.background}"
        return s


class SliceDescriptorMode(Enum):
    GRADIENT_OF_MAGNITUDE = "Gradient of magnitude"
    INTENSITY_SUM = "Intensity integral over slice"


class ImageGrayScale(Image):
    _rescale_factor: float = 1.0
    _background: Optional[float] = None
    _foreground: Optional[float] = None

    def __init__(
        self,
        pixels: np.ndarray,
        path: Optional[str] = None,
        axes: Optional[str] = None,
        do_rescale: bool = False,
    ):
        super(ImageGrayScale, self).__init__(pixels, path, axes)
        if do_rescale:
            self._rescale_factor = self.get_deconvolution_rescaling_factor()

    @property
    def background(self):
        return self._background

    @property
    def foreground(self):
        return self._foreground

    @property
    def rescale_factor(self) -> float:
        return self._rescale_factor

    @rescale_factor.setter
    def rescale_factor(self, new_factor: float) -> None:
        self._pixels = self.pixels / self.rescale_factor
        self._rescale_factor = new_factor
        self._pixels = self.pixels * self.rescale_factor

    @property
    def pixels(self) -> np.ndarray:
        if self._pixels is None and self._path_to_local is not None:
            self.load_from_local()
        return self._pixels / self._rescale_factor

    @staticmethod
    def from_tiff(
        path: str,
        axes: Optional[str] = None,
        do_rescale: bool = False,
        default_axes: str = const.default_axes[1:],
    ) -> "ImageGrayScale":
        img = ImageGrayScale(
            read_tiff(path, default_axes=default_axes), path, axes, do_rescale
        )
        return img

    def get_deconvolution_rescaling_factor(self) -> float:
        if self._path_to_local is None:
            return 1.0
        return get_deconvolution_rescaling_factor(self._path_to_local)

    def threshold_global(self, thr: Union[int, float]) -> ImageBinary:
        return ImageBinary(self.pixels > thr, doRebinarize=False)

    def threshold_adaptive(
        self, block_size: int, method: str, mode: str, *args, **kwargs
    ) -> ImageBinary:
        return ImageBinary(
            threshold_adaptive(self.pixels, block_size, method, mode, *args, **kwargs),
            doRebinarize=False,
        )

    def update_ground(
        self, M: Union[ImageBinary, ImageLabeled], block_side: int = 11
    ) -> None:
        if isinstance(M, ImageLabeled):
            M = M.binarize()
        M = dilate(M.pixels, block_side)
        self._foreground = np.median(self.pixels[M])
        self._background = np.median(self.pixels[np.logical_not(M)])

    def __repr__(self) -> str:
        s = super(ImageGrayScale, self).__repr__()
        if self.background is not None:
            s += f"; Back/foreground: {(self.background, self.foreground)}"
        return s

    def describe_slices(
        self, mode: SliceDescriptorMode = SliceDescriptorMode.GRADIENT_OF_MAGNITUDE
    ) -> List[float]:
        slice_descriptors = []
        if SliceDescriptorMode.GRADIENT_OF_MAGNITUDE == mode:
            for zi in range(self.shape[0]):
                slice_descriptors.append(self.pixels[zi].sum())
        elif SliceDescriptorMode.GRADIENT_OF_MAGNITUDE == mode:
            for zi in range(self.shape[0]):
                dx = stat.gpartial(self.pixels[zi, :, :], 1, 1)
                dy = stat.gpartial(self.pixels[zi, :, :], 2, 1)
                slice_descriptors.append(
                    np.mean(np.mean((dx ** 2 + dy ** 2) ** (1 / 2)))
                )
        else:
            slice_descriptors = []
        return slice_descriptors

    def is_in_focus(
        self,
        mode: SliceDescriptorMode = SliceDescriptorMode.GRADIENT_OF_MAGNITUDE,
        fraction: float = 0.5,
    ) -> Tuple[bool, pd.DataFrame]:
        slice_descriptors = self.describe_slices()
        profile = pd.DataFrame.from_dict(
            {
                "Z-slice index": np.array(range(self.shape[0])) + 1,
                mode.value: slice_descriptors,
            }
        )
        max_slice_id = slice_descriptors.index(max(slice_descriptors))
        halfrange = self.shape[0] * fraction / 2.0
        halfstack = self.shape[0] / 2.0
        return (
            max_slice_id >= (halfstack - halfrange)
            and max_slice_id <= (halfstack + halfrange),
            profile,
        )


def get_dtype(imax: Union[int, float]) -> str:
    depths = [8, 16]
    for depth in depths:
        if imax <= 2 ** depth - 1:
            return "uint%d" % (depth,)
    return "uint"


def get_bundle_axes_from_metadata(
    t: tf.TiffFile, default_axes: str = const.default_axes[1:]
) -> str:
    bundle_axes = default_axes
    metadata_field_list = [x for x in dir(t) if "metadata" in x]
    for metadata_field in metadata_field_list:
        metadata = getattr(t, metadata_field)
        if metadata is not None:
            if "axes" in metadata[0]:
                logging.debug(
                    f"read axes field from {metadata_field}: {metadata[0]['axes']}"
                )
                return metadata[0]["axes"]
    return bundle_axes[-len(t.asarray().shape):]


def read_tiff(
    path: str,
    expected_axes: Optional[str] = "ZYX",
    default_axes: str = const.default_axes[1:],
) -> np.ndarray:
    assert os.path.isfile(path), f"file not found: '{path}'"
    try:
        t = tf.TiffFile(path)
        bundle_axes = get_bundle_axes_from_metadata(t, default_axes)
        img = t.asarray()
    except (ValueError, TypeError) as e:
        logging.critical(f"cannot read image '{path}', file seems corrupted.\n{e}")
        raise
    img, bundle_axes = enforce_default_axis_bundle(img, bundle_axes, const.default_axes[1:])
    return img


def extract_nd(img: np.ndarray, nd: int) -> np.ndarray:
    if len(img.shape) <= nd:
        return img
    while len(img.shape) > nd:
        img = img[0]
    if 0 in img.shape:
        logging.warning("the image contains empty dimensions.")
    return img


def get_sampleformat_tag(dtype):
    if np.issubdtype(dtype, np.integer):
        return 1
    elif np.issubdtype(dtype, np.signedinteger):
        return 2
    elif np.issubdtype(dtype, np.float):
        return 3
    else:
        return 4


def add_missing_axes(
    img: np.ndarray, bundle_axes: str, expected_axes: str = const.default_axes[1:]
) -> Tuple[np.ndarray, str]:
    new_shape = []
    for a in expected_axes:
        if a not in bundle_axes.upper():
            bundle_axes = f"{a}{bundle_axes.upper()}"
            new_shape.append(1)
    new_shape.extend(img.shape)
    img = img.reshape(new_shape)
    return (img, bundle_axes)


def remove_unexpected_axes(
    img: np.ndarray,
    bundle_axes: str,
    expected_axes: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    if (
        expected_axes is None
        or expected_axes == bundle_axes
        or all([c in expected_axes for c in bundle_axes])
    ):
        return (img, bundle_axes)
    bundle_axes_list = list(bundle_axes.upper())
    slicing: List[Union[slice, int]] = []
    for aidx in range(len(bundle_axes_list)):
        a = bundle_axes_list[aidx]
        if a in expected_axes:
            slicing.append(slice(0, img.shape[aidx]))
        else:
            logging.debug(f"dropped axis {a} (i:{aidx}).")
            slicing.append(0)
            new_aidx = bundle_axes.index(a)
            bundle_axes = "".join(
                list(bundle_axes)[:new_aidx] + list(bundle_axes)[(new_aidx + 1) :]
            )
    return (img[tuple(slicing)].squeeze(), bundle_axes)


def reorder_axes(
    img: np.ndarray, bundle_axes: str, expected_axes: str = const.default_axes[1:]
) -> Tuple[np.ndarray, str]:
    if bundle_axes != expected_axes:
        bundle_axes_list = list(bundle_axes.upper())
        while bundle_axes != expected_axes:
            for i in range(len(expected_axes)):
                if bundle_axes[i] != expected_axes[i]:
                    a1 = (bundle_axes[i], i)
                    a2 = (expected_axes[i], bundle_axes.index(expected_axes[i]))
                    bundle_axes_list[a1[1]] = a2[0]
                    bundle_axes_list[a2[1]] = a1[0]
                    bundle_axes = "".join(bundle_axes_list)
                    img = np.swapaxes(img, a1[1], a2[1])
    return (img, bundle_axes)


def enforce_default_axis_bundle(
    img: np.ndarray, bundle_axes: str, expected_axes: str = const.default_axes[1:]
) -> Tuple[np.ndarray, str]:
    logging.info(bundle_axes)
    img, bundle_axes = remove_unexpected_axes(img, bundle_axes, expected_axes)
    logging.info(bundle_axes)
    img, bundle_axes = add_missing_axes(img, bundle_axes, expected_axes)
    logging.info(bundle_axes)
    img, bundle_axes = reorder_axes(img, bundle_axes, expected_axes)
    logging.info((img.shape, bundle_axes, "final"))
    return (img, bundle_axes)


def save_tiff(
    path: str,
    img: np.ndarray,
    compressed: bool,
    bundle_axes: str = "ZYX",
    inMicrons: bool = False,
    z_resolution: Optional[float] = None,
    forImageJ: bool = True,
    forceTZCYX: bool = True,
    **kwargs,
) -> None:
    assert len(bundle_axes) == len(
        img.shape
    ), f"shape mismatch between bundled axes ({bundle_axes}) and image ({img.shape})."

    if forceTZCYX:
        img, bundle_axes = enforce_default_axis_bundle(
            img, bundle_axes, const.default_axes[1:]
        )

    metadata: Dict[str, Any] = dict(axes=bundle_axes)
    if inMicrons:
        metadata["unit"] = "um"
    if z_resolution is not None:
        metadata["spacing"] = z_resolution
    compressionLevel = 0 if not compressed else 9

    tf.imwrite(
        path,
        img,
        compress=compressionLevel,
        imagej=forImageJ,
        metadata=metadata,
        software=f"radiant v{const.__version__}",
        extratags=[(339, "i", 1, get_sampleformat_tag(img.dtype), False)],
        **kwargs,
    )


def z_project(img: np.ndarray, projection_type: const.ProjectionType) -> np.ndarray:
    if projection_type == const.ProjectionType.SUM:
        img = img.sum(0).astype(img.dtype)
    elif projection_type == const.ProjectionType.MAX:
        img = img.max(0).astype(img.dtype)
    return img


def offset2(img: np.ndarray, offset: int) -> int:
    if 0 == offset:
        return img
    if offset < 0:
        offset *= -1
        return img[tuple([slice(offset, -offset) for a in range(len(img.shape))])]
    else:
        canvas = np.zeros(np.array(img.shape) + 2 * offset)
        canvas[
            tuple([slice(offset, img.shape[a] + offset) for a in range(len(img.shape))])
        ] = img
        return canvas


def threshold_adaptive(
    img: np.ndarray, block_size: int, method: str, mode: str, *args, **kwargs
) -> np.ndarray:
    assert 1 == block_size % 2

    def threshold_adaptive_slice(
        img: np.ndarray, block_size: int, method: str, mode: str, *args, **kwargs
    ) -> np.ndarray:
        threshold = ski.filters.threshold_local(
            img, block_size, *args, method=method, mode=mode, **kwargs
        )
        return img >= threshold

    if 2 == len(img.shape):
        mask = threshold_adaptive_slice(img, block_size, method, mode, *args, **kwargs)
    elif 3 == len(img.shape):
        mask = img.copy()
        for slice_id in range(mask.shape[0]):
            logging.debug(f"ADAPT_THR SLICE#({slice_id})")
            mask[slice_id, :, :] = threshold_adaptive_slice(
                mask[slice_id, :, :], block_size, method, mode, *args, **kwargs
            )
    else:
        logging.info(
            "Local threshold not implemented for images with "
            + f"{len(img.shape)} dimensions."
        )
        raise ValueError
    return mask


def fill_holes(mask: np.ndarray) -> np.ndarray:
    mask = ndi.binary_fill_holes(mask)
    if 3 == len(mask.shape):
        for slice_id in range(mask.shape[0]):
            logging.debug(f"FILL_HOLES SLICE#({slice_id})")
            mask[slice_id, :, :] = ndi.binary_fill_holes(mask[slice_id, :, :])
    elif 2 != len(mask.shape):
        logging.warning(
            "3D hole filling not performed on images with "
            + f"{len(mask.shape)} dimensions."
        )
        raise ValueError
    return mask


def closing2(mask: np.ndarray, block_side: int = 3) -> np.ndarray:
    assert 1 == mask.max()
    if block_side > 1:
        if 2 == len(mask.shape):
            mask = closing(mask, square(block_side))
        elif 3 == len(mask.shape):
            mask = closing(mask, cube(block_side))
        else:
            logging.info(
                "Close operation not implemented for images with "
                + f"{len(mask.shape)} dimensions."
            )
            raise ValueError
    return mask


def opening2(mask: np.ndarray, block_side: int = 3) -> np.ndarray:
    assert 1 == mask.max()
    if block_side > 1:
        if 2 == len(mask.shape):
            mask = opening(mask, square(block_side))
        elif 3 == len(mask.shape):
            mask = opening(mask, cube(block_side))
        else:
            logging.info(
                "Open operation not implemented for images with "
                + f"{len(mask.shape)} dimensions."
            )
            raise ValueError
    return mask


def dilate(mask: np.ndarray, block_side: int = 3) -> np.ndarray:
    assert 1 == mask.max()
    if block_side > 1:
        if 2 == len(mask.shape):
            mask = dilation(mask, square(block_side))
        elif 3 == len(mask.shape):
            mask = dilation(mask, cube(block_side))
        else:
            logging.info(
                "Dilate operation not implemented for images with "
                + f"{len(mask.shape)} dimensions."
            )
            raise ValueError
    return mask


def erode(mask: np.ndarray, block_side: int = 3) -> np.ndarray:
    assert 1 == mask.max()
    if block_side > 1:
        if 2 == len(mask.shape):
            mask = erosion(mask, square(block_side))
        elif 3 == len(mask.shape):
            mask = erosion(mask, cube(block_side))
        else:
            logging.info(
                "Erode operation not implemented for images with "
                + f"{len(mask.shape)} dimensions."
            )
            raise ValueError
    return mask


def clear_XY_borders(L: np.ndarray) -> np.ndarray:
    if 2 == len(L.shape):
        return clear_border(L)
    elif 3 == len(L.shape):
        border_labels: List[int] = []
        border_labels.extend(np.unique(L[:, 0, :]).tolist())
        border_labels.extend(np.unique(L[:, -1, :]).tolist())
        border_labels.extend(np.unique(L[:, :, 0]).tolist())
        border_labels.extend(np.unique(L[:, :, -1]).tolist())
        for lab in set(border_labels):
            L[L == lab] = 0
        return ski.measure.label(L)
    else:
        logging.warning(
            "XY border clearing not implemented for images "
            + f"with {len(L.shape)} dimensions."
        )
        raise ValueError


def clear_Z_borders(L: np.ndarray) -> np.ndarray:
    if 2 == len(L.shape):
        return L
    elif 3 == len(L.shape):
        border_labels: List[int] = []
        border_labels.extend(np.unique(L[0, :, :]).tolist())
        border_labels.extend(np.unique(L[-1, :, :]).tolist())
        for lab in set(border_labels):
            L[L == lab] = 0
        return ski.measure.label(L)
    else:
        logging.warning(
            "Z border clearing not implemented for images "
            + f"with {len(L.shape)} dimensions."
        )
        raise ValueError


def inherit_labels(
    mask: Union[ImageBinary, ImageLabeled], mask2d: Union[ImageBinary, ImageLabeled]
) -> ImageLabeled:
    assert 2 == len(mask2d.shape)
    if 2 == len(mask.shape):
        assert mask2d.shape == mask.shape
        return mask2d.pixels[np.logical_and(mask.pixels > 0, mask2d.pixels > 0)]
    elif 3 == len(mask.shape):
        assert mask2d.shape == mask.shape[-2:]
        new_mask = mask.pixels.copy()
        for slice_id in range(mask.shape[0]):
            new_mask[slice_id, :, :] = mask2d.pixels[
                np.logical_and(mask.pixels[slice_id, :, :] > 0, mask2d.pixels > 0)
            ]
        return ImageLabeled(new_mask, doRelabel=False)
    else:
        logging.warning(
            "mask combination not allowed for images "
            + f"with {len(mask.shape)} dimensions."
        )
        raise ValueError


def tile_to(img: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    assert len(shape) == len(img.shape)
    new_shape = list(shape)
    for ai in range(len(img.shape)):
        if 1 != img.shape[ai]:
            new_shape[ai] = 1
    return np.tile(img, new_shape)
