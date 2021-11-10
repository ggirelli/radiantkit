"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import re
from os.path import isdir, isfile
from typing import Iterator, Optional, Set, Tuple

from joblib import cpu_count  # type: ignore

from radiantkit.const import DEFAULT_INPUT_RE
from radiantkit.path import add_suffix, find_re


class CompressionSettings(object):
    _input_paths: Set[str]
    _output_suffix: str
    input_re: str = DEFAULT_INPUT_RE[""]
    _threads: int = 1

    def __init__(self, input: Set[str], suffix: str, input_re: str):
        super(CompressionSettings, self).__init__()
        self.input_paths = input
        self.output_suffix = suffix
        self.input_re = input_re

    @property
    def input_paths(self) -> Set[str]:
        return self._input_paths

    @input_paths.setter
    def input_paths(self, input_paths: Set[str]) -> None:
        file_count = 0
        dir_count = 0
        for path in set(input_paths):
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
            self._input_paths = set(
                find_re(next(iter(input_paths)), re.compile(self.input_re))
            )
        else:
            self._input_paths = set(input_paths)

    @property
    def output_suffix(self) -> str:
        return self._output_suffix

    @output_suffix.setter
    def output_suffix(self, suffix: str) -> None:
        pass

    @property
    def threads(self) -> int:
        return self._threads

    @threads.setter
    def threads(self, threads: int) -> None:
        self._threads = max(1, min(cpu_count(), threads))

    def iterate_input_output(self) -> Iterator[Tuple[str, str]]:
        for input_path in self.input_paths:
            yield (input_path, add_suffix(input_path, self.output_suffix))


class OutOfFocusSettings(object):
    _input_dirpath: str
    _reference: str
    _input_re: str
    _output_dirpath: Optional[str]
    _mode: str = "GOM"
    _fraction: float = 0.5
    rename: bool = True
    suffix: str = ".oof.old"

    def __init__(self, input_dirpath: str, reference: str, input_re: str):
        super(OutOfFocusSettings, self).__init__()
        self._input_dirpath = input_dirpath
        self._reference = reference
        self._input_re = input_re
