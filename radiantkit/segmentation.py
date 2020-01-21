'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import logging as log
import numpy as np
from radiantkit import const
from radiantkit import image as imt
from skimage.filters import threshold_otsu
from typing import Tuple, Union

class BinarizerSettings(object):
	segmentation_type: const.SegmentationType = None
	analysis_type: const.AnalysisType = None
	do_global: bool = True
	do_local: bool = True
	_local_side: int = 101
	local_method: str = 'median'
	local_mode: str = 'constant'
	local_closing: bool = True
	do_clear_XY_borders: bool = True
	do_clear_Z_borders: bool = False
	do_fill_holes: bool = True
	radius_interval: Tuple[float] = (10., float('inf'))
	min_z_size: Union[float,int] = .25
	logger: log.Logger = None

	def __init__(self, logger: log.Logger = log.getLogger("radiantkit")):
		super(BinarizerSettings, self).__init__()
		self.logger = logger

	@property
	def local_side(self) -> int:
		return self._local_side

	@local_side.setter
	def local_side(self, value: int) -> None:
		if 0 != value % 2: value += 1
		self._local_side = value

class Binarizer(BinarizerSettings):
	def __init__(self, logger: log.Logger = log.getLogger("radiantkit")):
		super(Binarizer, self).__init__(logger)

	def run(self, I: np.ndarray, mask2: np.ndarray = None) -> np.ndarray:
		if not self.do_global and not self.do_local:
			self.logger.warning("no threshold applied.")
			return I

		if self.segmentation_type in (const.SegmentationType.SUM_PROJECTION,
			const.SegmentationType.MAX_PROJECTION):
			self.logger.info(f"projecting over Z [{self.segmentation_type}].")
			I = imt.Image.z_project(I,
				const.ProjectionType(self.segmentation_type))

		masks = []
		global_threshold = 0
		if self.do_global:
			global_threshold = threshold_otsu(I)
			self.logger.info(f"applying global threshold of {global_threshold}")
			mask.append(imt.Image.close(I > thr))
		if self.do_local and 1 < self.local_side:
			self.logger.info("applying adaptive threshold to neighbourhood " +
				f"with side of {self.local_side} px.")
			local_mask = imt.Image.threshold_adaptive(I,
					self.local_side, self.local_method, self.local_mode)
			if self.local_closing: local_mask = imt.Image.close(local_mask)
			mask.append(local_mask)

		while 1 < len(mask):
			mask[0] = np.logical_and(mask[0], mask[1])
			mask.pop(1)
		mask = mask[0]

		if mask2 is not None: mask = np.logical_and(mask, mask2>0)

		if self.do_clear_XY_borders:
			pass

		if self.do_clear_Z_borders:
			pass

		if self.do_fill_holes:
			pass

		return I

