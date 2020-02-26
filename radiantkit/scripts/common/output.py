'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from enum import Enum
import logging
import os
from radiantkit import path, scripts
from typing import Dict, List, Optional, Pattern


class OutputType(Enum):
    SELECT_NUCLEI = "select_nuclei"
    MEASURE_OBJECTS = "measure_objects"
    RADIAL_POPULATION = "radial_population"


class OutputChecker(object):
    __type: OutputType

    def __init__(self, ot: OutputType):
        super(OutputChecker, self).__init__()
        assert ot in OutputType
        self.__type = ot

    @property
    def type(self):
        return self.__type

    def __output_files_in_folder(self, dpath: str) -> bool:
        if not os.path.isdir(dpath):
            return False
        flist = os.listdir(dpath)
        return all([oname in flist for oname in getattr(
                scripts, self.__type.value).__OUTPUT__])

    def is_in_folder(
            self, dpath: str, subname: Optional[str] = None) -> bool:
        if self.__output_files_in_folder(dpath):
            return True
        if subname is not None:
            dpath = os.path.join(dpath, subname)
            if self.__output_files_in_folder(dpath):
                return True
        return False


class OutputFinder(object):
    def __init__(self):
        super(OutputFinder, self).__init__()

    def search_output_types(
            self, dpath: str,
            output_list: List[Dict[str, List[OutputType]]],
            subname: str = "objects"
            ) -> List[Dict[str, List[OutputType]]]:
        current_output_list = []
        for otype in OutputType:
            if OutputChecker(otype).is_in_folder(dpath, subname):
                current_output_list.append(otype)

        if 0 == len(current_output_list):
            output_list.append({dpath: []})
        else:
            output_list.append({dpath: current_output_list})

        return output_list

    def nested_search_output_types(
            self, dpath: str, inreg: Pattern
            ) -> List[Dict[str, List[OutputType]]]:
        output_list: List[Dict[str, List[OutputType]]] = []
        if 0 == len(path.find_re(dpath, inreg)):
            subfolder_list = [f for f in os.scandir(dpath) if os.path.isdir(f)]
            for f in subfolder_list:
                fpath = f.path
                logging.info(f"looking into subfolder '{f.name}'")
                output_list = self.search_output_types(fpath, output_list)
        else:
            output_list = self.search_output_types(dpath, output_list)

        return output_list
