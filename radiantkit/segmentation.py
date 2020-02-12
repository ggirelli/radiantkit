'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import logging
from radiantkit import const
from radiantkit.image import Image, ImageBinary, ImageLabeled
from skimage.filters import threshold_otsu  # type: ignore
from typing import Optional, Union


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
    logger: logging.Logger

    def __init__(self,
                 logger: logging.Logger = logging.getLogger("radiantkit")):
        super(BinarizerSettings, self).__init__()
        self.logger = logger

    @property
    def local_side(self) -> int:
        return self._local_side

    @local_side.setter
    def local_side(self, value: int) -> None:
        if 0 == value % 2:
            value += 1
        self._local_side = value


class Binarizer(BinarizerSettings):
    def __init__(self,
                 logger: logging.Logger = logging.getLogger("radiantkit")):
        super(Binarizer, self).__init__(logger)

    def run(self, img: Image,
            mask2d: Optional[Union[ImageBinary, ImageLabeled]] = None
            ) -> Union[Image, ImageBinary]:
        if not self.do_global and not self.do_local:
            self.logger.warning("no threshold applied.")
            return img

        if self.segmentation_type in (const.SegmentationType.SUM_PROJECTION,
                                      const.SegmentationType.MAX_PROJECTION):
            self.logger.info(f"projecting over Z [{self.segmentation_type}].")
            img.z_project(const.ProjectionType(self.segmentation_type))

        mask_list = []
        global_threshold = 0
        if self.do_global:
            global_threshold = threshold_otsu(img.pixels)
            self.logger.info(
                f"applying global threshold of {global_threshold}")
            gmask = img.threshold_global(global_threshold)
            if self.global_closing:
                gmask.close()
            mask_list.append(gmask)
        if self.do_local and 1 < self.local_side:
            self.logger.info("applying adaptive threshold to neighbourhood "
                             + f"with side of {self.local_side} px. "
                             + f"({self.local_method}, {self.local_mode})")
            local_mask = img.threshold_adaptive(
                self.local_side, self.local_method, self.local_mode)
            if self.local_closing:
                local_mask.close()
            mask_list.append(local_mask)

        while 1 < len(mask_list):
            mask_list[0].logical_and(mask_list[1])
            mask_list.pop(1)
        M = mask_list[0]

        if mask2d is not None:
            logging.info("combining with 2D mask")
            M.logical_and(ImageBinary(mask2d.pixels))

        L = ImageLabeled(M.pixels)
        if self.do_clear_XY_borders:
            logging.info("clearing XY borders")
            L.clear_XY_borders()
        if self.do_clear_Z_borders:
            logging.info("clearing Z borders")
            L.clear_Z_borders()

        M = ImageBinary(L.pixels)
        if self.do_fill_holes:
            logging.info("filling holes")
            M.fill_holes()

        return M
