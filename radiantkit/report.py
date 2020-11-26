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
from radiantkit import const, scripts
from radiantkit.output import Output, OutputDirectories, OutputReader
from types import ModuleType
from typing import Any, DefaultDict, Dict, List, Optional


class ReportBase(OutputDirectories):
    _idx: float
    _stub: str
    _title: str
    _files: const.OutputFileDirpath
    _log: const.OutputFileDirpath
    _args: const.OutputFileDirpath
    __located_files: const.OutputFileDirpath

    def __init__(self, *args, **kwargs):
        super(ReportBase, self).__init__(*args, **kwargs)

    @property
    def idx(self) -> float:
        return self._idx

    @property
    def stub(self) -> str:
        return self._stub

    @property
    def title(self) -> str:
        return self._title

    @property
    def files(self) -> const.OutputFileDirpath:
        return self._files

    def _search(self, files: const.OutputFileDirpath) -> const.OutputFileDirpath:
        output = Output(self._dirpath, self._subdirs)
        output.is_root = self.is_root
        located_files: const.OutputFileDirpath = {}
        for stub, (filename, required, _) in self._files.items():
            found_in: const.DirectoryPathList = output.search(filename)
            if not found_in and required:
                logging.warning(
                    f"missing required output file '{filename}' "
                    + f"for script '{self._stub}' report."
                )
                continue
            else:
                located_files[stub] = (filename, required, found_in)
        return located_files

    def _read(
        self, located_files: const.OutputFileDirpath
    ) -> DefaultDict[str, Dict[str, Any]]:
        logging.info(f"reading output files of '{self._stub}'.")
        data: DefaultDict[str, Dict[str, pd.DataFrame]] = defaultdict(lambda: {})
        for stub, (filename, _, dirpathlist) in located_files.items():
            for dirpath in dirpathlist:
                filepath = os.path.join(dirpath, filename)
                assert os.path.isfile(filepath)
                data[stub][dirpath] = OutputReader.read_single_file(filepath)
        return data

    @staticmethod
    def figure_to_html(
        fig: go.Figure,
        include_plotlyjs: bool = False,
        classes: List[str] = [],
        data: Dict[str, str] = {},
    ) -> str:
        html = (
            fig.to_html(include_plotlyjs=False)
            .split("<body>")[1]
            .split("</body>")[0]
            .strip()
        )[5:-6]

        div_opening = "<div"
        if classes:
            div_opening += f" class='{' '.join(classes)}'"
        for k, v in data.items():
            div_opening += f" data-{k}='{v}'"
        div_opening += ">"

        return f"{div_opening}{html}</div>"

    @abstractmethod
    def _plot(
        self, data: DefaultDict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, go.Figure]]:
        ...

    def _make_plot_panels(self, fig_data: Dict[str, Dict[str, go.Figure]]) -> str:
        assert self._stub in fig_data

        figure_panels: str = "\n\t".join(
            [
                self.figure_to_html(
                    fig,
                    classes=[self._stub, "panel", "hidden"],
                    data=dict(condition=os.path.basename(dpath)),
                )
                for dpath, fig in sorted(
                    fig_data[self._stub].items(), key=lambda x: x[0]
                )
            ]
        )

        page = f"""
<!--Report for {self._stub}-->
<div class='report hidden' data-report='{self._stub}'>
    <small>Select a condition to update the plot below.</small>
    <select class='{self._stub} u-full-width'>"""
        for d in sorted(fig_data[self._stub].keys()):
            page += f"\n\t\t\t<option>{os.path.basename(d)}</option>"
        page += f"""
    </select>
    {figure_panels}
    <script type='text/javascript'>
        // Condition selection
        $('.{self.stub}.panel[data-condition='+$('select.{self.stub}').val()+']').removeClass('hidden');
        $('select.{self.stub}').change(function(e) {{
            $('.{self.stub}.panel:not(.hidden)').addClass('hidden');
            $('.{self.stub}.panel[data-condition='+$(this).val()+']').removeClass('hidden');
        }})
    </script>
</div>"""

        return page

    @abstractmethod
    def _make_html(self, fig_data: Dict[str, Dict[str, go.Figure]]) -> str:
        ...

    def make(self) -> str:
        return self._make_html(self._plot(self._read(self._search(self._files))))


class ReportMaker(OutputDirectories):
    jquery_src: str = "https://code.jquery.com/jquery-3.5.1.min.js"
    plotly_src: str = "https://cdn.plot.ly/plotly-latest.min.js"
    skeleton_href: str = (
        "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css"
    )
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
        self.__reportable = sorted(self.__reportable, key=lambda r: r.idx)

    def __make_head(self) -> str:
        return f"""<head>
    <script src="{self.plotly_src}"></script>
    <link rel="stylesheet" href="{self.skeleton_href}" />
    <script src="{self.jquery_src}"></script>
    <style type="text/css">
        .hidden {{
            display: none;
        }}
    </style>
</head>"""

    def __make_body(self) -> str:
        body = """<body>
    <h2>Radiant report</h2>
    <div class='row'>
        <!--Headers-->
        <div class='three columns'>
            <h5>Index</h5>
        </div>
        <div class='nine columns'>
            <h5>Report</h5>
        </div>
    </div>
    <div class='row'>
        <!--Report index-->
        <div id='index-list' class='three columns'>"""
        for report in self.__reportable:
            body += f"""
            <a class='button u-full-width'
                href='#' data-report='{report.stub}'>{report.title}</a>"""
        body += """
        </div>
        <!--Report results-->
        <div id='report-list' class='nine columns'>"""
        for report in self.__reportable:
            body += report.make().replace("\n", "\n\t\t\t") + "\n"
        body += """
        </div>
    </div>
    <script type='text/javascript'>
        // Report selection
        show_report = function(selected_report) {{
            $('#report-list div.report:not(.hidden)')
                .addClass('hidden');
            $('#report-list div.report[data-report="'+selected_report+'"]')
                .removeClass('hidden');
        }}

        // Show the first report from the index
        $('#index-list a.button').first()
            .removeClass('hidden').addClass('button-primary');
        show_report($('#index-list a.button.button-primary').data('report'));

        // Show the selected report
        $('#index-list a.button').click(function(e) {{
            e.preventDefault();
            selected_report = $(this).data('report');
            $('#index-list a.button-primary').removeClass('button-primary');
            $(this).addClass('button-primary');
            show_report(selected_report);
        }});
    </script>
</body>"""
        return body

    def make(self) -> None:
        self.__find_reportable()
        with open("test.html", "w+") as OH:
            OH.write("<!DOCTYPE html>\n<html lang='en'>\n")
            OH.write(self.__make_head())
            OH.write(self.__make_body())
            OH.write("\n</html>")
