'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
from datetime import datetime
import jinja2 as jj2
import os
from typing import Any, Dict


class JinjaTemplate(object):
    _env: jj2.Environment
    _template: jj2.Template

    def __init__(self, template: str):
        super(JinjaTemplate, self).__init__()
        self._env = jj2.Environment(
                loader=jj2.PackageLoader('radiantkit', 'templates'),
                autoescape=jj2.select_autoescape(['html', 'xml'])
            )
        self._template = self._env.get_template(template)

    def render(self, path: str, **kwargs) -> None:
        with open(path, "w+") as OH:
            OH.write(self._template.render(**kwargs))


class Report(JinjaTemplate):
    _env: jj2.Environment
    _template: jj2.Template

    def __init__(self, template: str):
        super(Report, self).__init__(template)
        self._env.filters['basename'] = os.path.basename
        self._env.filters['dirname'] = os.path.dirname
        self._template = self._env.get_template(template)

    def render(self, path: str, **kwargs) -> None:
        assert "title" in kwargs
        super(Report, self).render(path, **kwargs)


def general_report(
        args: argparse.Namespace, output_list: Dict[str, Any]
        ) -> None:
    repi = Report("reports/main.tpl.html")
    repi.render(
        "radiant.html",
        args=args, odata=output_list,
        title="RadIAntKit",
        now=str(datetime.now()), online=args.online)
