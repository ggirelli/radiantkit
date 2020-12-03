"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from abc import abstractmethod
from collections import defaultdict
from importlib import import_module
import inspect
import logging
import os
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
from radiantkit import const, scripts
from radiantkit.output import Output, OutputDirectories, OutputReader
from typing import Any, DefaultDict, Dict, List, Optional, Tuple


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
        for stub, (filename, required, _) in files.items():
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
        self, data: DefaultDict[str, Dict[str, Any]], *args, **kwargs
    ) -> Dict[str, Dict[str, go.Figure]]:
        ...

    def _make_panel_page(
        self, panel_type: str, panels: str, keys: List[str], msg: str = ""
    ) -> str:
        page = f"""
        <small>{msg}</small>
        <select class='{self._stub} {panel_type}-panel u-full-width'>""".replace(
            f"\n{' '*4}", "\n"
        )
        for d in keys:
            page += f"""
        <option>{os.path.basename(d)}</option>"""
        page += f"""
        </select>
        {panels}
        <script type='text/javascript'>
            // Condition selection
            $('div.{self.stub}.{panel_type}-panel[data-condition='+
                $('select.{panel_type}-panel.{self.stub}').val()+']').removeClass('hidden');
            $('select.{panel_type}-panel.{self.stub}').change(function(e) {{
                $('div.{self.stub}.{panel_type}-panel:not(.hidden)').addClass('hidden');
                $('div.{self.stub}.{panel_type}-panel[data-condition='+$(this).val()+']')
                    .removeClass('hidden');
            }})
        </script>""".replace(
            f"\n{' '*4}", "\n"
        )
        return page

    def _make_log_panels(self, log_data: DefaultDict[str, Dict[str, Any]]) -> str:
        panels: str = "\n\t".join(
            [
                f"""<div class='{self._stub} log-panel hidden'
                    data-condition='{os.path.basename(dpath)}'>
                    <pre style='overflow: auto;'><code>
                {log.strip()}
                    </code></pre>
                </div>""".replace(
                    f"\n{' '*4*3}", "\n"
                )
                for dpath, log in sorted(log_data["log"].items(), key=lambda x: x[0])
            ]
        )

        return self._make_panel_page(
            "log",
            panels,
            sorted(log_data["log"].keys()),
            "Select a condition to show its log below.",
        )

    def _make_arg_panels(self, arg_data: DefaultDict[str, Dict[str, Any]]) -> str:
        panels: str = "\n\t".join(
            [
                f"""<div class='{self._stub} args-panel hidden'
                    data-condition='{os.path.basename(dpath)}'>
                    <pre style='overflow: auto;'><code>
                {args}
                    </code></pre>
                </div>""".replace(
                    f"\n{' '*4*3}", "\n"
                )
                for dpath, args in sorted(arg_data["args"].items(), key=lambda x: x[0])
            ]
        )

        return self._make_panel_page(
            "args",
            panels,
            sorted(arg_data["args"].keys()),
            "Select a condition to show its arguments below.",
        )

    def _make_plot_panels(self, fig_data: Dict[str, Dict[str, go.Figure]]) -> str:
        assert self._stub in fig_data

        panels: str = "\n\t".join(
            [
                self.figure_to_html(
                    fig,
                    classes=[self._stub, "plot-panel", "hidden"],
                    data=dict(condition=os.path.basename(dpath)),
                )
                for dpath, fig in sorted(
                    fig_data[self._stub].items(), key=lambda x: x[0]
                )
            ]
        )

        return self._make_panel_page(
            "plot",
            panels,
            sorted(fig_data[self._stub].keys()),
            "Select a condition to update the plot below.",
        )

    def _make_html(
        self,
        fig_data: Optional[Dict[str, Dict[str, go.Figure]]] = None,
        log_data: Optional[DefaultDict[str, Dict[str, Any]]] = None,
        arg_data: Optional[DefaultDict[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        page = ReportPage(self._stub, 0)
        if fig_data is not None:
            page.add_panel("plot", "Plots", self._make_plot_panels(fig_data))
        if log_data is not None:
            page.add_panel("log", "Log", self._make_log_panels(log_data))
        if arg_data is not None:
            page.add_panel("arg", "Args", self._make_arg_panels(arg_data))
        return page.make()

    def make(self) -> str:
        logging.info(f"reading output files of '{self._stub}'.")
        output_data = self._read(self._search(self._files))
        logging.info(f"reading logs and args of '{self._stub}'.")
        log_data = self._read(self._search(self._log))
        arg_data = self._read(self._search(self._args))
        return self._make_html(
            fig_data=self._plot(output_data, arg_data=arg_data),
            log_data=log_data,
            arg_data=arg_data,
        )


class ReportPage(object):
    _page_id: str
    _nesting_level: int = 0
    _panels: Dict[
        str, Tuple[str, str]
    ]  # Dict[panel_idx, Tuple[panel_label, panel_html]]

    def __init__(self, pid: str, nesting_level: int):
        super(ReportPage, self).__init__()
        self._page_id = pid
        self._nesting_level = max(0, nesting_level)
        self._panels = {}

    @property
    def id(self) -> str:
        return self._page_id

    @property
    def html_id(self):
        return (
            self._page_id
            if 0 == self._nesting_level
            else f"{self._page_id}-{self._nesting_level}"
        )

    @property
    def html_class(self):
        return "page" if 0 == self._nesting_level else f"page-{self._nesting_level}"

    def add_panel(self, idx: str, label: str, content: str) -> None:
        if idx in self._panels:
            logging.warning(f"replacing panel '{idx}'.")
        self._panels[idx] = (label, content)

    def __build_panel_index(self) -> str:
        buttons = [
            f"""
            <div class='four columns'>
                <a class='button u-full-width' data-target='{pidx}-panel' href='#'>
                    {plab}</a>
            </div>""".replace(
                "\n", f"\n{' '*4*2}"
            )
            for pidx, (plab, _) in self._panels.items()
        ]
        index = f"""\n    <div id='{self.html_id}-page-index-wrapper'
        class='panel-index'>"""
        for i in range(0, len(buttons), 3):
            buttons_subset = "".join(buttons[slice(i, i + 3)])
            index += f"""
                <div class='row'>{buttons_subset}
                </div>""".replace(
                f"\n{' '*4*2}", "\n"
            )
        index += """\n    </div>"""
        return index

    def __wrap_panels(self) -> str:
        panels = ""
        for pidx, (plab, phtml) in self._panels.items():
            phtml = phtml.replace("\n", "\n\t")
            panels += f"""
            <!--{plab} panel-->
            <div class='{pidx}-panel page-wrapper hidden'>{phtml}
            </div>""".replace(
                f"\n{' '*4*2}", "\n"
            )
        return panels

    def make(self) -> str:
        subclass = " subpage" if self._nesting_level > 0 else ""
        page = f"""
        <!--Report for {self.html_id}-->
        <div class='{self.html_class}{subclass} hidden'
            data-page='{self.html_id}'>""".replace(
            f"\n{' '*4*2}", "\n"
        )
        page += self.__build_panel_index()
        page += self.__wrap_panels()
        page += f"""
            <!--Closure-->
            <script type='text/javascript'>
                // Show first panel
                $('div.{self.html_class}[data-page="{self.html_id}"] .page-wrapper')
                    .first().removeClass('hidden');
                $('div.{self.html_class}[data-page="{self.html_id}"] '+
                    '.page-wrapper .subpage').first().removeClass('hidden');
                $('#{self.html_id}-page-index-wrapper a.button')
                    .first().addClass('button-primary');

                // Select panel type
                $('#{self.html_id}-page-index-wrapper a.button').click(function(e) {{
                    e.preventDefault();
                    $('div.{self.html_class}[data-page="{self.html_id}"] '
                        +'.page-wrapper:not(.hidden)').addClass('hidden');
                    $('#{self.html_id}-page-index-wrapper a.button-primary')
                        .removeClass('button-primary');
                    $(this).addClass('button-primary');

                    selected_page = $(
                        'div.{self.html_class}[data-page="{self.html_id}"]'
                        +' .page-wrapper.'+$(this).data('target')
                    );
                    selected_page.removeClass('hidden');

                    first_subpage = selected_page.children('.subpage').first();
                    first_subpage.removeClass('hidden');
                    first_subpage.children('.panel-index').find('a.button')
                        .first().click();
                }});
            </script>
        </div>""".replace(
            f"\n{' '*4*2}", "\n"
        )
        return page


class ReportMaker(OutputDirectories):
    jquery_src: str = "https://code.jquery.com/jquery-3.5.1.min.js"
    plotly_src: str = "https://cdn.plot.ly/plotly-latest.min.js"
    skeleton_href: str = (
        "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css"
    )
    __reportable: List[ReportBase]
    title: str
    footer: str

    def __init__(self, *args, **kwargs):
        super(ReportMaker, self).__init__(*args, **kwargs)
        self.title = ""
        self.footer = ""

    def __get_report_instance(self, module_name: str) -> Optional[ReportBase]:
        """ReportMaker supports script modules with a defined 'Report' class that is a
        subclass of ReportBase"""
        try:
            script_module = import_module(f"radiantkit.scripts.{module_name}")
            for name, obj in inspect.getmembers(
                script_module,
                lambda x: inspect.isclass(x) and issubclass(x, ReportBase),
            ):
                report = obj(self._dirpath)  # type: ignore
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
            <meta charset="utf-8">
            <title>{self.title}</title>
            <script src="{self.plotly_src}"></script>
            <link rel="stylesheet" href="{self.skeleton_href}" />
            <script src="{self.jquery_src}"></script>
            <style type="text/css">
                footer {{
                    display: float;
                    padding: .5em;
                    border-top: 1px solid #dadada;
                }}
                .hidden {{
                    display: none;
                }}
            </style>
        </head>""".replace(
            f"\n{' '*4*2}", "\n"
        )

    def __make_body(self) -> str:
        body = """
        <body>
            <header>
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
            </header>
            <div class='row'>
                <!--Report index-->
                <div id='index-list' class='three columns'>""".replace(
            f"\n{' '*4*2}", "\n"
        )
        for report in self.__reportable:
            body += f"""
            <a class='button u-full-width'
                href='#' data-page='{report.stub}'>{report.title}</a>"""
        body += """
        </div>
        <!--Report results-->
        <div id='report-list' class='nine columns'>"""
        for report in self.__reportable:
            body += report.make().replace("\n", f"\n{' '*4*3}") + "\n"
        body += f"""
                </div>
            </div>
            <script type='text/javascript'>
                // Report selection
                show_report = function(selected_report) {{
                    $('#report-list div.page:not(.hidden)')
                        .addClass('hidden');
                    $('#report-list div.page[data-page="'+selected_report+'"]')
                        .removeClass('hidden');
                }}

                // Show the first report from the index
                $('#index-list a.button').first()
                    .removeClass('hidden').addClass('button-primary');
                show_report($('#index-list a.button.button-primary').data('page'));

                // Show the selected report
                $('#index-list a.button').click(function(e) {{
                    e.preventDefault();
                    selected_report = $(this).data('page');
                    $('#index-list a.button-primary').removeClass('button-primary');
                    $(this).addClass('button-primary');
                    show_report(selected_report);
                }});
            </script>
            <footer class="u-full-width">
                <small>{self.footer}</small>
            </footer>
        </body>""".replace(
            f"\n{' '*4*2}", "\n"
        )
        return body

    def make(self) -> None:
        self.__find_reportable()
        with open("test.html", "w+") as OH:
            OH.write("<!DOCTYPE html>\n<html lang='en'>\n")
            OH.write(self.__make_head())
            OH.write(self.__make_body())
            OH.write("\n</html>")
