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


def output_in_folder(otype: OutputType, ipath: str) -> bool:
    if not os.path.isdir(ipath):
        return False
    flist = os.listdir(ipath)
    return all([oname in flist for oname in getattr(
            scripts, otype.value).__OUTPUT__])


def output_type_in_folder(
        otype: OutputType, ipath: str, subname: Optional[str] = None) -> bool:
    if output_in_folder(otype, ipath):
        return True

    if subname is not None:
        ipath = os.path.join(ipath, subname)
        if output_in_folder(otype, ipath):
            return True

    return False


def search_radiant_output_types(
        ipath: str, subname: str = "objects") -> List[OutputType]:
    return [otype for otype in OutputType
            if output_type_in_folder(otype, ipath, subname)]


def get_output_list(ipath: str) -> Optional[List[OutputType]]:
    output_list = search_radiant_output_types(ipath)
    if 0 == len(output_list):
        return None
    else:
        return output_list


def get_output_list_per_folder(
        ipath: str, inreg: Pattern) -> List[Dict[str, List[OutputType]]]:
    output_list = []
    if 0 == len(path.find_re(ipath, inreg)):
        subfolder_list = [f for f in os.scandir(ipath) if os.path.isdir(f)]
        for f in subfolder_list:
            fpath = f.path
            logging.info(f"looking into subfolder '{f.name}'")
            output = get_output_list(fpath)
            if output is not None:
                output_list.append({fpath: output})
    else:
        output = get_output_list(ipath)
        if output is not None:
            output_list.append({ipath: output})

    return output_list
