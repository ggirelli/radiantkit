'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import logging
import numpy as np
from radiantkit import const
from radiantkit.image import ImageBase, ImageBinary, ImageLabeled
from skimage.filters import threshold_otsu
from typing import Optional, Tuple, Type, Union

class BinarizerSettings(object):
	segmentation_type: const.SegmentationType = (
		const.SegmentationType.get_default())
	do_global: bool = True
	global_closing: bool = True
	do_local: bool = True
	_local_side: int = 101
	local_method: str = 'gaussian'
	local_mode: str = 'constant'
	local_closing: bool = True
	do_clear_XY_borders: bool = True
	do_clear_Z_borders: bool = False
	do_fill_holes: bool = True
	logger: Optional[logging.Logger] = None

	def __init__(self,
		logger: logging.Logger = logging.getLogger("radiantkit")):
		super(BinarizerSettings, self).__init__()
		self.logger = logger

	@property
	def local_side(self) -> int:
		return self._local_side

	@local_side.setter
	def local_side(self, value: int) -> None:
		if 0 == value % 2: value += 1
		self._local_side = value

class Binarizer(BinarizerSettings):
	def __init__(self,
		logger: logging.Logger = logging.getLogger("radiantkit")):
		super(Binarizer, self).__init__(logger)

	@staticmethod
	def inherit_labels(mask: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
		assert 2 == len(mask2d.shape)
		if 2 == len(mask.shape):
			assert mask2d.shape == mask.shape
			return mask2d[np.logical_and(mask>0, mask2d>0)]
		elif 3 == len(mask.shape):
			assert mask2d.shape == mask[-2:].shape
			new_mask = mask.copy()
			for slice_id in range(mask.shape[0]):
				new_mask[slice_id,:,:] = mask2d[np.logical_and(
					mask[slice_id,:,:]>0, mask2d>0)]
			return new_mask
		else:
			self.logger.warning("mask combination not allowed for images " +
				f"with {len(mask.shape)} dimensions.")
			return mask

	def run(self, I: Type[ImageBase],
		mask2d: Optional[ImageBinary]=None) -> ImageBinary:
		if not self.do_global and not self.do_local:
			self.logger.warning("no threshold applied.")
			return I

		if self.segmentation_type in (const.SegmentationType.SUM_PROJECTION,
			const.SegmentationType.MAX_PROJECTION):
			self.logger.info(f"projecting over Z [{self.segmentation_type}].")
			I.z_project(const.ProjectionType(self.segmentation_type))

		mask = []
		global_threshold = 0
		if self.do_global:
			global_threshold = threshold_otsu(I.pixels)
			self.logger.info(f"applying global threshold of {global_threshold}")
			gmask = I.threshold_global(global_threshold)
			if self.global_closing: gmask.close()
			mask.append(gmask)
		if self.do_local and 1 < self.local_side:
			self.logger.info("applying adaptive threshold to neighbourhood " +
				f"with side of {self.local_side} px. " +
				f"({self.local_method}, {self.local_mode})")
			local_mask = I.threshold_adaptive(
				self.local_side, self.local_method, self.local_mode)
			if self.local_closing: local_mask.close()
			mask.append(local_mask)

		while 1 < len(mask):
			mask[0].logical_and(mask[1])
			mask.pop(1)
		mask = mask[0]

		if mask2d is not None:
			logging.info("combining with 2D mask")
			mask.logical_and(ImageBinary(mask2d))

		mask = ImageLabeled(mask.pixels)
		if self.do_clear_XY_borders:
			logging.info("clearing XY borders")
			mask.clear_XY_borders()
		if self.do_clear_Z_borders:
			logging.info("clearing Z borders")
			mask.clear_Z_borders()

		mask = ImageBinary(mask.pixels)
		if self.do_fill_holes:
			logging.info("filling holes")
			mask.fill_holes()

		if mask2d is not None:
			logging.info("recovering labels from 2D mask")
			mask = self.inherit_labels(mask.pixels, mask2d.pixels)
		else: mask = mask.pixels

		return ImageBinary(mask)

