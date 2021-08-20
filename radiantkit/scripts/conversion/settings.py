"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import logging
import numpy as np  # type: ignore
from os.path import basename, dirname, isdir, isfile, join as join_paths, splitext
import radiantkit as ra
from radiantkit.string import TIFFNameTemplateFields as TNTFields
from radiantkit.string import TIFFNameTemplate as TNTemplate
import sys
from typing import List, Optional, Set, Union


class ConversionSettings(object):
    """docstring for ND2toTIFFsettings"""

    _input_paths: Set[str] = set()
    _output_dirpath: Optional[str] = None
    _fields: Optional[Set[int]] = None
    _channels: Optional[Set[str]] = None
    dz: Optional[float] = None
    input_re: str = ra.const.DEFAULT_INPUT_RE["nd2"]
    template: TNTemplate
    compress: bool = False
    just_info: bool = False
    just_list: bool = False

    def __init__(self, **args):
        super(ConversionSettings, self).__init__()
        for field in [
            "input",
            "output",
            "fields",
            "channels",
            "dz",
            "input_re",
            "template",
            "compress",
            "info",
            "list",
        ]:
            assert field in args, f"missing field '{field}'"
        self.input_paths = args["input"]
        self.output_dirpath = args["output"]
        self.fields = args["fields"]
        self.channels = args["channels"]
        self.dz = args["dz"]
        self.input_re = args["input_re"]
        self.template = TNTemplate(args["template"])
        self.compress = args["compress"]
        self.just_info = args["info"]
        self.just_list = args["list"]

    @property
    def input_paths(self) -> Set[str]:
        return self._input_paths

    @input_paths.setter
    def input_paths(self, path_list: List[str]) -> None:
        file_count = 0
        dir_count = 0
        for path in set(path_list):
            if isfile(path):
                file_count += 1
            elif isdir(path):
                dir_count += 1
            else:
                assert False, f"input path not found: {path}"
        if 0 < dir_count:
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

    @fields.setter
    def fields(self, fields_str: Optional[str]) -> None:
        if fields_str is not None:
            self._fields = set(ra.string.MultiRange(fields_str))
        else:
            self._fields = None

    @property
    def channels(self) -> Optional[Set[str]]:
        return self._channels

    @channels.setter
    def channels(self, channels_str: Optional[str]) -> None:
        if channels_str is not None:
            self._channels = set(channels_str.split(","))
        else:
            self._channels = None

    @staticmethod
    def select_fields(
        fields: Optional[Set[int]],
        image: Union[ra.conversion.ND2Reader2, ra.conversion.CziFile2],
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
                    "Converting only the following fields: " + f"{[x for x in fields]}"
                )
            return set([x - 1 for x in fields if x <= image.field_count()])

    def check_fields(
        self, image: Union[ra.conversion.ND2Reader2, ra.conversion.CziFile2]
    ) -> bool:
        self.fields = self.select_fields(self.fields, image, False)
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
        image: Union[ra.conversion.ND2Reader2, ra.conversion.CziFile2],
        verbose: bool = True,
    ) -> Set[str]:
        if channels is None:
            channels = set(image.get_channel_names())
        else:
            channels = set(image.select_channels(channels))
            if 0 == len(channels):
                logging.error("None of the specified channels was found.")
                sys.exit()
            if verbose:
                logging.info(f"Converting only the following channels: {channels}")
        return channels

    def check_channels(
        self, image: Union[ra.conversion.ND2Reader2, ra.conversion.CziFile2]
    ) -> bool:
        self.channels = self.select_channels(self.channels, image, False)
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

    def check_nd2_dz(self, nd2_image: ra.conversion.ND2Reader2) -> bool:
        if self.dz is not None:
            logging.info(f"Enforcing a deltaZ of {self.dz:.3f} um.")
        elif 1 < len(nd2_image.z_resolution):
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

    def is_nd2_compatible(self, nd2_image: ra.conversion.ND2Reader2) -> bool:
        assert not nd2_image.isLive(), "time-course conversion images not implemented."
        self.check_fields(nd2_image)
        self.check_channels(nd2_image)
        self.check_nd2_dz(nd2_image)
        return True

    def is_czi_compatible(self, czi_image: ra.conversion.CziFile2) -> bool:
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
