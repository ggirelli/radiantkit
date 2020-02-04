'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
from datetime import datetime
from jinja2 import PackageLoader
from jinja2 import Environment, Template, select_autoescape
from os.path import basename, dirname
from radiantkit import plot
import tempfile
from typing import Optional

class Report(object):
    _env: Optional[Environment]=None
    _template: Optional[Template]=None

    def __init__(self, template: str):
        super(Report, self).__init__()
        self._env = Environment(
                loader=PackageLoader('radiantkit', 'templates'),
                autoescape=select_autoescape(['html', 'xml'])
            )
        self._env.filters['basename'] = basename
        self._env.filters['dirname'] = dirname
        self._template = self._env.get_template(template)
        print((type(self._env),type(self._template)))

    def render(self, path: str, **kwargs) -> None:
        with open(path, "w+") as OH:
            OH.write(self._template.render(**kwargs))

def report_select_nuclei(args: argparse.Namespace,
    opath: str, **kwargs) -> None:

    report = Report('select_nuclei_report_template.html')

    figure = plot.plot_nuclear_selection(kwargs['data'], kwargs['ref'],
        kwargs['size_range'], kwargs['intensity_sum_range'])

    report.render(opath, args=args,
        data=kwargs['data'], series_list=kwargs['series_list'],
        plot_json=figure.to_json(), now=str(datetime.now()))
