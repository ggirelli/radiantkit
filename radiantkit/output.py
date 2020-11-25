"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import os
import pandas as pd  # type: ignore
import pickle
from radiantkit.const import DirectoryPathList
from typing import Optional, Pattern

DEFAULT_SUBDIRS: DirectoryPathList = ["objects"]


class OutputDirectories(object):
    _dirpath: str
    _subdirs: DirectoryPathList
    _is_root: bool

    def __init__(
        self,
        dirpath: str,
        subdirs: Optional[DirectoryPathList] = None,
        group_by: Optional[Pattern] = None,
    ):
        super(OutputDirectories, self).__init__()
        assert os.path.isdir(dirpath)
        self._dirpath = dirpath
        if subdirs is None:
            self._subdirs = DEFAULT_SUBDIRS
        else:
            self._subdirs = subdirs
        self._is_root = False

    @property
    def is_root(self) -> bool:
        return self._is_root

    @is_root.setter
    def is_root(self, is_root: bool):
        self._is_root = is_root


class Output(OutputDirectories):
    def __init__(self, *args, **kwargs):
        super(Output, self).__init__(*args, **kwargs)

    @staticmethod
    def search_directory(
        filename: str, dirpath: str, subdirs: DirectoryPathList
    ) -> DirectoryPathList:
        if os.path.isfile(os.path.join(dirpath, filename)):
            return [dirpath]
        for subdirpath in subdirs:
            if os.path.isfile(os.path.join(dirpath, subdirpath, filename)):
                return [os.path.join(dirpath, subdirpath)]
        return []

    def __search_root(self, filename: str) -> DirectoryPathList:
        found_in: DirectoryPathList = []
        for fname in os.listdir(self._dirpath):
            if os.path.isdir(fname):
                found_in.extend(
                    self.search_directory(
                        filename, os.path.join(self._dirpath, fname), self._subdirs
                    )
                )
        return found_in

    def search(self, filename: str) -> DirectoryPathList:
        if self._is_root:
            return self.__search_root(filename)
        else:
            return self.search_directory(filename, self._dirpath, self._subdirs)


class OutputReader(object):
    def __init__(self):
        super(OutputReader, self).__init__()

    @staticmethod
    def read_csv(
        path: str,
        sep: str = ",",
    ) -> pd.DataFrame:
        return pd.read_csv(path, sep=sep)

    @staticmethod
    def read_tsv(path: str) -> pd.DataFrame:
        return OutputReader.read_csv(path, "\t")

    @staticmethod
    def read_pkl(path: str) -> pd.DataFrame:
        with open(path, "rb") as PIH:
            return pd.DataFrame(pickle.load(PIH))

    @staticmethod
    def read_single_file(path: str) -> pd.DataFrame:
        if path.endswith(".csv"):
            return OutputReader.read_csv(path)
        if path.endswith(".tsv"):
            return OutputReader.read_tsv(path)
        elif path.endswith(".pkl"):
            return OutputReader.read_pkl(path)
        else:
            raise NotImplementedError(
                f"cannot read file of type '{os.path.splitext(path)[-1]}'"
            )
