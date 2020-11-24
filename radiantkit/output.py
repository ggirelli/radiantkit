"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import os
from typing import List, Optional

DEFAULT_SUBDIRS: List[str] = ["objects"]


class OutputDirectories(object):
    _dirpath: str
    _subdirs: List[str]
    _is_root: bool

    def __init__(self, dirpath: str, subdirs: Optional[List[str]] = None):
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
    def search_directory(filename: str, dirpath: str, subdirs: List[str]) -> List[str]:
        if os.path.isfile(os.path.join(dirpath, filename)):
            return [dirpath]
        for subdirpath in subdirs:
            if os.path.isfile(os.path.join(dirpath, subdirpath, filename)):
                return [os.path.join(dirpath, subdirpath)]
        return []

    def __search_root(self, filename: str) -> List[str]:
        found_in: List[str] = []
        for fname in os.listdir(self._dirpath):
            if os.path.isdir(fname):
                found_in.extend(self.search_directory(
                    filename, os.path.join(self._dirpath, fname), self._subdirs
                ))
        return found_in

    def search(self, filename: str) -> List[str]:
        if self._is_root:
            return self.__search_root(filename)
        else:
            return self.search_directory(filename, self._dirpath, self._subdirs)
