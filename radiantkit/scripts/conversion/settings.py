"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import logging
import sys
from os.path import basename, dirname, isdir, isfile
from os.path import join as join_paths
from os.path import splitext
from typing import Optional, Set, Union

import numpy as np  # type: ignore

from radiantkit.const import DEFAULT_INPUT_RE
from radiantkit.conversion import CziFile2, ND2Reader2
from radiantkit.string import MultiRange
from radiantkit.string import TIFFNameTemplate as TNTemplate
from radiantkit.string import TIFFNameTemplateFields as TNTFields


class ConversionSettings(object):
    _input_paths: Set[str] = set()
    _output_dirpath: Optional[str] = None
    _fields: Optional[Set[int]] = None
    _channels: Optional[Set[str]] = None
    dz: Optional[float] = None
    input_re: str = DEFAULT_INPUT_RE[""]
    template: TNTemplate
    compress: bool = False
    just_info: bool = False
    just_list: bool = False

    def __init__(self, input_paths: Set[str], input_re: str, template: str):
        super(ConversionSettings, self).__init__()
        self.input_paths = input_paths
        self.input_re = input_re
        self.template = TNTemplate(template)

    @property
    def input_paths(self) -> Set[str]:
        return self._input_paths

    @input_paths.setter
    def input_paths(self, path_list: Set[str]) -> None:
        file_count = 0
        dir_count = 0
        for path in set(path_list):
            if isfile(path):
                file_count += 1
            elif isdir(path):
                dir_count += 1
            else:
                assert False, f"input path not found: {path}"
        if dir_count > 0:
            assert 1 == dir_count, "only one directory is allowed per run"
            assert (
                0 == file_count
            ), "please provide either files or a directory, not both."
        self._input_paths = set(path_list)

    @property
    def output_dirpath(self) -> Optional[str]:
        return self._output_dirpath

    @output_dirpath.setter
    def output_dirpath(self, path: Optional[str]) -> None:
        if path is not None:
            assert not isfile(path), f"output path exists as a file: {path}"
            assert not isdir(path), f"output path exists as a folder: {path}"
        self._output_dirpath = path

    @property
    def fields(self) -> Optional[Set[int]]:
        return self._fields

    def set_fields(self, fields_str: Optional[str]) -> None:
        self._fields = (
            {x - 1 for x in MultiRange(fields_str)} if fields_str is not None else None
        )

    @property
    def channels(self) -> Optional[Set[str]]:
        return self._channels

    def set_channels(self, channel_str: Optional[str]) -> None:
        if channel_str is not None:
            self._channels = set(channel_str.split(","))
        else:
            self._channels = None

    @staticmethod
    def select_fields(
        fields: Optional[Set[int]],
        image: Union[ND2Reader2, CziFile2],
        verbose: bool = True,
    ) -> Set[int]:
        if fields is None:
            return set(range(image.field_count()))
        if np.array(list(fields)).min() > image.field_count():
            if verbose:
                logging.warning(
                    "".join(
                        [
                            "Skipped all available fields ",
                            "(not included in specified field range.",
                        ]
                    )
                )
            return set()
        else:
            if verbose:
                logging.info(
                    f"Converting only the following fields: {[x+1 for x in fields]}"
                )
            return {x for x in fields if x <= image.field_count()}

    def check_fields(self, image: Union[ND2Reader2, CziFile2]) -> bool:
        self._fields = self.select_fields(self.fields, image, False)
        assert self.template.can_export_fields(
            image.field_count(), self.fields
        ), "".join(
            [
                "when exporting more than 1 field, the template ",
                f"must include the {TNTFields.SERIES_ID} seed. ",
                f"Got '{self.template.template}' instead.",
            ]
        )
        return True

    @staticmethod
    def select_channels(
        channels: Optional[Set[str]],
        image: Union[ND2Reader2, CziFile2],
        verbose: bool = True,
    ) -> Set[str]:
        if channels is None:
            channels = set(image.get_channel_names())
        else:
            channels = set(image.select_channels(channels))
            if not channels:
                logging.error("None of the specified channels was found.")
                sys.exit()
            if verbose:
                logging.info(f"Converting only the following channels: {channels}")
        return channels

    def check_channels(self, image: Union[ND2Reader2, CziFile2]) -> bool:
        self._channels = self.select_channels(self.channels, image, False)
        assert self.template.can_export_channels(
            image.channel_count(), self.channels
        ), "".join(
            [
                "when exporting more than 1 channel, the template ",
                f"must include either {TNTFields.CHANNEL_ID} or ",
                f"{TNTFields.CHANNEL_NAME} seeds. ",
                f"Got '{self.template.template}' instead.",
            ]
        )
        return True

    def check_nd2_dz(self, nd2_image: ND2Reader2) -> bool:
        if self.dz is not None:
            logging.info(f"Enforcing a deltaZ of {self.dz:.3f} um.")
        elif len(nd2_image.z_resolution) > 1:
            logging.warning(
                " ".join(
                    [
                        "Z resolution is not constant across fields.",
                        "It will be automagically identified, field-by-field.",
                        "If the automatic Z resolution reported in the log is wrong,",
                        "please enforce the correct one using the --dz option.",
                    ]
                )
            )
            logging.debug(f"Z steps histogram: {nd2_image.z_resolution}.")
        return True

    def is_nd2_compatible(self, nd2_image: ND2Reader2) -> bool:
        assert not nd2_image.isLive(), "time-course conversion images not implemented."
        self.check_fields(nd2_image)
        self.check_channels(nd2_image)
        self.check_nd2_dz(nd2_image)
        return True

    def is_czi_compatible(self, czi_image: CziFile2) -> bool:
        assert not czi_image.isLive(), "time-course conversion images not implemented."
        self.check_fields(czi_image)
        self.check_channels(czi_image)
        return True

    def get_output_dirpath_for_single_file(
        self, file_path: str, output_dirpath: Optional[str] = None
    ) -> str:
        if output_dirpath is None:
            output_dirpath = join_paths(
                dirname(file_path), splitext(basename(file_path))[0]
            )
        assert not isfile(output_dirpath), f"output path is a file: {output_dirpath}"
        return output_dirpath
