'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from enum import Enum
import logging
import os
import pandas as pd  # type: ignore
import pickle
import plotly.graph_objects as go  # type: ignore
from radiantkit import const, path, plot, scripts
from typing import Any, Callable, Dict, List, Optional, Pattern


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

    @property
    def filenames(self):
        return list(getattr(scripts, self.value).__OUTPUT__.values())

    @property
    def file_labels(self):
        return list(getattr(scripts, self.value).__OUTPUT__.items())

    @property
    def report_condition(self):
        return getattr(scripts, self.value).__OUTPUT_CONDITION__

    @property
    def label(self):
        return getattr(scripts, self.value).__LABEL__

    def plot(self, **data) -> go.Figure:
        plot_fun: Dict[str, Callable] = dict(
            select_nuclei=plot.plot_nuclear_selection,
            measure_objects=plot.plot_nuclear_features,
            radial_population=plot.plot_profiles
        )
        return plot_fun[self.value](**data)

    @staticmethod
    def to_dict():
        return dict([(x.value, x.label) for x in OutputType])


DirectoryPath = str
OutputTypeList = List[OutputType]
ScriptStub = str
ScriptLabel = str
OutputFileLabel = str
OutputData = Dict[OutputFileLabel, Any]
BaseName = str


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
        return self.__type.report_condition([
            oname in flist for oname in self.__type.filenames])

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
            subname: str = const.default_subfolder
            ) -> Dict[DirectoryPath, OutputTypeList]:
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
            dpath: str, inreg: Pattern, skipHidden: bool = True
            ) -> Dict[DirectoryPath, OutputTypeList]:
        output_list: Dict[DirectoryPath, OutputTypeList] = {}

        if 0 != len(path.find_re(dpath, inreg)):
            return OutputFinder.search(dpath, output_list)

        subfolder_list = [f for f in os.scandir(dpath) if os.path.isdir(f)]
        if skipHidden:
            subfolder_list = [
                f for f in subfolder_list if not f.name.startswith(".")]

        for f in subfolder_list:
            logging.info(f"looking into subfolder '{f.name}'")
            current_output = OutputFinder.search_recursive(f.path, inreg)
            path_list = [x.value for x in current_output[f.path]]
            logging.info(f"found output for: {path_list}")
            output_list.update(current_output)

        return output_list


class OutputReader(object):
    def __init__(self):
        super(OutputReader, self).__init__()

    @staticmethod
    def read_csv(ofname: str, root_path: DirectoryPath,
                 subname: str = const.default_subfolder, sep=","
                 ) -> Optional[pd.DataFrame]:
        opath = os.path.join(root_path, ofname)
        if not os.path.isfile(opath):
            opath = os.path.join(root_path, subname, ofname)
        if not os.path.isfile(opath):
            return None
        output_df = pd.read_csv(opath, sep=sep)
        output_df['root'] = os.path.dirname(root_path)
        output_df['base'] = os.path.basename(root_path)

        return output_df

    @staticmethod
    def read_tsv(ofname: str, dpath: DirectoryPath,
                 subname: str = const.default_subfolder) -> pd.DataFrame:
        return OutputReader.read_csv(ofname, dpath, subname, "\t")

    @staticmethod
    def read_pkl(ofname: str, root_path: DirectoryPath,
                 subname: str = const.default_subfolder) -> Dict:
        opath = os.path.join(root_path, ofname)
        if not os.path.isfile(opath):
            opath = os.path.join(root_path, subname, ofname)
        if not os.path.isfile(opath):
            raise FileNotFoundError
        with open(opath, "rb") as PIH:
            output_data = {'data': pickle.load(PIH)}
        output_data['root'] = os.path.dirname(root_path)
        output_data['base'] = os.path.basename(root_path)

        return output_data

    @staticmethod
    def read_single_file(
            oflab: OutputFileLabel, ofname: str,
            dpath: DirectoryPath, subname: str = const.default_subfolder
            ) -> OutputData:
        if ofname.endswith(".csv"):
            return {oflab: OutputReader.read_csv(
                ofname, dpath, subname)}
        if ofname.endswith(".tsv"):
            return {oflab: OutputReader.read_tsv(
                ofname, dpath, subname)}
        elif ofname.endswith(".pkl"):
            return {oflab: OutputReader.read_pkl(
                ofname, dpath, subname)}
        else:
            raise NotImplementedError

    @staticmethod
    def read(otype: OutputType, path_list: List[DirectoryPath],
             subname: str = const.default_subfolder
             ) -> Dict[DirectoryPath, OutputData]:
        output_data: Dict[DirectoryPath, OutputData] = {}
        for root_path in path_list:
            root_data: OutputData = {}
            for oflab, ofname in otype.file_labels:
                root_data.update(OutputReader.read_single_file(
                    oflab, ofname, root_path, subname))
            output_data.update({root_path: root_data})
        return output_data

    @staticmethod
    def read_recursive(dpath: str, inreg: Pattern,
                       subname: str = const.default_subfolder
                       ) -> Dict[ScriptStub, OutputData]:
        output_list = OutputFinder.search_recursive(dpath, inreg)

        output_locations: Dict[OutputType, List[DirectoryPath]] = {}
        for tpath, output_type_list in output_list.items():
            for otype in output_type_list:
                if otype not in output_locations:
                    output_locations[otype] = []
                output_locations[otype].append(tpath)

        output: Dict[ScriptStub, Dict[DirectoryPath, OutputData]] = {}
        for otype, locations in output_locations.items():
            output[otype.value] = OutputReader.read(otype, locations, subname)

        return output


class OutputPlotter(object):
    def __init__(self):
        super(OutputPlotter, self).__init__()

    @staticmethod
    def plot(output_list: Dict[ScriptStub, Dict[DirectoryPath, OutputData]]
             ) -> Dict[ScriptStub, Dict[BaseName, go.Figure]]:
        plot_dict: Dict[ScriptStub, Dict[BaseName, go.Figure]] = {}
        for script_stub, script_data in output_list.items():
            plot_dict[script_stub] = {}
            logging.info(f"preparing plot for '{script_stub}' data")
            plot_dict[script_stub] = OutputType(
                script_stub).plot(**script_data)
        return plot_dict
