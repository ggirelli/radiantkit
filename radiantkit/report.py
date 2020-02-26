'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import jinja2 as jj2
import os


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
