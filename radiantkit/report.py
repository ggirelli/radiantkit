"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from abc import abstractmethod
from importlib import import_module
import logging
import os
from radiantkit import scripts
from radiantkit.output import Output, OutputDirectories
from types import ModuleType
from typing import Dict, List, Optional, Tuple


class ReportBase(OutputDirectories):
    _stub: str
    _files: Dict[str, Tuple[str, bool]]

    def __init__(self, *args, **kwargs):
        super(ReportBase, self).__init__(*args, **kwargs)

    def __found_output(self) -> bool:
        output = Output(self._dirpath, self._subdirs)
        output.is_root = self.is_root
        for stub, (filename, required) in self._files.items():
            found_in: List[str] = output.search(filename)
            if not found_in and required:
                logging.warning(
                    f"missing required output file '{filename}' "
                    + f"for script '{self._stub}' report."
                )
                return False
        return True

    def make(self) -> None:
        if not self.__found_output():
            return
        raise NotImplementedError("here report creation should proceed")

    @abstractmethod
    def plot(self, *args, **kwargs) -> None:
        pass


class ReportMaker(OutputDirectories):
    __reportable: List[ReportBase]

    def __init__(self, *args, **kwargs):
        super(ReportMaker, self).__init__(*args, **kwargs)

    def __get_report_instance(self, module_name: str) -> Optional[ReportBase]:
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
