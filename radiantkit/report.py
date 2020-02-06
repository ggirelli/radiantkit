'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
from datetime import datetime
import jinja2 as jj2
import os
import radiantkit as ra
import tempfile
from typing import Optional

class Report(object):
    _env: Optional[jj2.Environment]=None
    _template: Optional[jj2.Template]=None

    def __init__(self, template: str):
        super(Report, self).__init__()
        self._env = jj2.Environment(
                loader=jj2.PackageLoader('radiantkit', 'templates'),
                autoescape=jj2.select_autoescape(['html', 'xml'])
            )
        self._env.filters['basename'] = os.path.basename
        self._env.filters['dirname'] = os.path.dirname
        self._template = self._env.get_template(template)

    def render(self, path: str, **kwargs) -> None:
        with open(path, "w+") as OH:
            OH.write(self._template.render(**kwargs))

def report_select_nuclei(args: argparse.Namespace,
    opath: str, online: bool=False, **kwargs) -> None:

    report = Report('select_nuclei_report_template.html')
    details = kwargs['details']

    figure = ra.plot.plot_nuclear_selection(kwargs['data'], kwargs['ref'],
        details['size']['range'], details['isum']['range'],
        details['size']['fit'], details['isum']['fit'],)

    report.render(opath, online=online, args=args, details=details,
        data=kwargs['data'], series_list=kwargs['series_list'],
        plot_json=figure.to_json(), now=str(datetime.now()))
