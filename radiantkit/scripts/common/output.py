'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from enum import Enum
import logging
import os
import pandas as pd  # type: ignore
from radiantkit import path, scripts
from typing import Dict, List, Optional, Pattern


class OutputType(Enum):
    """Enum for registered radiant script stubs.

    Each radiant script that with recognizable output must have a stub in this
    Enum class. Easily extract script output filenames and reporting condition
    (all or any) with the corresponding functions.

    Extends:
        Enum

    Variables:
        SELECT_NUCLEI {str} -- stub for select_nuclei
        MEASURE_OBJECTS {str} -- stub for measure_objects
        RADIAL_POPULATION {str} -- stub for radial_population
    """
    SELECT_NUCLEI = "select_nuclei"
    MEASURE_OBJECTS = "measure_objects"
    RADIAL_POPULATION = "radial_population"

    def filenames(self):
        return list(getattr(scripts, self.value).__OUTPUT__.keys())

    def file_labels(self):
        return list(getattr(scripts, self.value).__OUTPUT__.items())

    def report_condition(self):
        return getattr(scripts, self.value).__OUTPUT_CONDITION__


DirectoryPath = str
OutputTypeList = List[OutputType]


class OutputChecker(object):
    """Check class for OutputType.

    Check if a specific OutputType is present in a folder, or specified sub-
    -folder, based on its report condition.

    Variables:
        __type: {OutputType}
    """
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
        return self.__type.report_condition()([
            oname in flist for oname in self.__type.filenames()])

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

    @staticmethod
    def search(
            dpath: str, output_list: Dict[DirectoryPath, OutputTypeList],
            subname: str = "objects") -> Dict[DirectoryPath, OutputTypeList]:
        current_output_list = []
        for otype in OutputType:
            if OutputChecker(otype).is_in_folder(dpath, subname):
                current_output_list.append(otype)

        if dpath not in output_list:
            output_list[dpath] = []
        if 0 == len(current_output_list):
            current_output_list = []
        output_list[dpath].extend(current_output_list)

        return output_list

    @staticmethod
    def search_recursive(
            dpath: str, inreg: Pattern
            ) -> Dict[DirectoryPath, OutputTypeList]:
        output_list: Dict[DirectoryPath, OutputTypeList] = {}

        if 0 != len(path.find_re(dpath, inreg)):
            return OutputFinder.search(dpath, output_list)

        subfolder_list = [f for f in os.scandir(dpath) if os.path.isdir(f)]
        for f in subfolder_list:
            logging.info(f"looking into subfolder '{f.name}'")
            output_list.update(
                OutputFinder.search_recursive(f.path, inreg))

        return output_list


class OutputReader(object):
    def __init__(self):
        super(OutputReader, self).__init__()

    @staticmethod
    def read(otype: OutputType, path_list: List[str],
             subname: str = "objects") -> Dict[str, pd.DataFrame]:
        output_data: Dict[str, pd.DataFrame] = {}
        for ofname, oflab in otype.file_labels():
            merged_output_df = pd.DataFrame()
            for root_path in path_list:
                opath = os.path.join(root_path, ofname)
                if not os.path.isfile(opath):
                    opath = os.path.join(root_path, subname, ofname)
                if not os.path.isfile(opath):
                    continue
                output_df = pd.read_csv(opath, sep="\t")
                output_df['root'] = os.path.dirname(root_path)
                output_df['base'] = os.path.basename(root_path)
                merged_output_df = pd.concat([merged_output_df, output_df])
            output_data[oflab] = merged_output_df
        return output_data

    @staticmethod
    def read_recursive(dpath: str, inreg: Pattern, subname: str = "objects"
                       ) -> Dict[str, pd.DataFrame]:
        output_list = OutputFinder.search_recursive(dpath, inreg)

        output_locations: Dict[OutputType, List[DirectoryPath]] = {}
        for tpath, output_type_list in output_list.items():
            for otype in output_type_list:
                if otype not in output_locations:
                    output_locations[otype] = []
                output_locations[otype].append(tpath)

        output: Dict[str, Dict[str, pd.DataFrame]] = {}
        for otype, locations in output_locations.items():
            output[otype.value] = OutputReader.read(otype, locations, subname)

        return output
