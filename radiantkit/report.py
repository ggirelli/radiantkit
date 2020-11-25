"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from abc import abstractmethod
from collections import defaultdict
from importlib import import_module
import logging
import os
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from radiantkit.const import DirectoryPathList, OutputFileDirpath
from radiantkit import scripts
from radiantkit.output import Output, OutputDirectories, OutputReader
from types import ModuleType
from typing import DefaultDict, Dict, List, Optional


class ReportBase(OutputDirectories):
    _stub: str
    _files: OutputFileDirpath
    __located_files: OutputFileDirpath

    def __init__(self, *args, **kwargs):
        super(ReportBase, self).__init__(*args, **kwargs)

    def _search(self) -> None:
        output = Output(self._dirpath, self._subdirs)
        output.is_root = self.is_root
        self.__located_files = {}
        for stub, (filename, required, _) in self._files.items():
            found_in: DirectoryPathList = output.search(filename)
            if not found_in and required:
                logging.warning(
                    f"missing required output file '{filename}' "
                    + f"for script '{self._stub}' report."
                )
                continue
            else:
                self.__located_files[stub] = (filename, required, found_in)

    def _read(self) -> DefaultDict[str, Dict[str, pd.DataFrame]]:
        data: DefaultDict[str, Dict[str, pd.DataFrame]] = defaultdict(lambda: {})
        for stub, (filename, _, dirpathlist) in self.__located_files.items():
            for dirpath in dirpathlist:
                filepath = os.path.join(dirpath, filename)
                logging.info(f"reading [{stub}] output file '{filepath}'.")
                assert os.path.isfile(filepath)
                data[stub][dirpath] = OutputReader.read_single_file(filepath)
        return data

    @staticmethod
    def figure_to_html(fig: go.Figure, include_plotlyjs: bool = False) -> str:
        return (
            fig.to_html(include_plotlyjs=False).split("<body>")[1].split("</body>")[0]
        )

    @abstractmethod
    def plot(
        self, data: DefaultDict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, go.Figure]]:
        ...

    @abstractmethod
    def make_page(self, fig_data: Dict[str, Dict[str, go.Figure]]) -> None:
        ...

    def make(self) -> None:
        self._search()
        return self.make_page(self.plot(self._read()))


class ReportMaker(OutputDirectories):
    __reportable: List[ReportBase]

    def __init__(self, *args, **kwargs):
        super(ReportMaker, self).__init__(*args, **kwargs)

    def __get_report_instance(self, module_name: str) -> Optional[ReportBase]:
        """ReportMaker supports script modules with a defined 'Report' class that is a
        subclass of ReportBase"""
        try:
            script_module: ModuleType = import_module(
                f"radiantkit.scripts.{module_name}"
            )
            if "Report" in dir(script_module):
                if issubclass(script_module.Report, ReportBase):  # type: ignore
                    report = script_module.Report(self._dirpath)  # type: ignore
                    report.is_root = self.is_root
                    return report
        except ModuleNotFoundError:
            pass
        return None

    def __find_reportable(self) -> None:
        self.__reportable = []
        assert os.path.isdir(self._dirpath)
        for module_name in dir(scripts):
            report_instance = self.__get_report_instance(module_name)
            if report_instance is not None:
                self.__reportable.append(report_instance)

    def make(self) -> None:
        self.__find_reportable()
        for report in self.__reportable:
            report.make()
