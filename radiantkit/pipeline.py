'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from enum import Enum
import os
from radiantkit.report import JinjaTemplate
import re
import stat
from typing import Optional


class ShellType(Enum):
    BASH = 'bash'
    ZSH = 'zsh'


class PipelineTemplate(object):
    _name: str
    _config: JinjaTemplate
    _snakef: JinjaTemplate

    def __init__(self, name: str):
        super(PipelineTemplate, self).__init__()
        self._name = name
        self._config = JinjaTemplate(f'snakemake/{name}.config.yaml')
        self._snakef = JinjaTemplate(f'snakemake/{name}.snakefile')

    def render(self, root: str, shell_type: Optional[str] = None,
               conda_env: Optional[str] = None, **kwargs) -> None:
        assert os.path.isdir(root)
        if shell_type is not None:
            assert any([shell_type == x.value
                        for x in ShellType.__members__.values()])
        if conda_env is not None:
            assert re.search(r'[a-zA-Z0-9_]+', conda_env) is not None

        self._config.render(os.path.join(
            root, f"{self._name}.config.yaml"), **kwargs)
        self._snakef.render(os.path.join(
            root, f"{self._name}.snakefile"), **kwargs)

        run_path = os.path.join(root, "run.sh")
        JinjaTemplate('snakemake/run.sh').render(
            run_path, name=self._name,
            shell_type=shell_type, conda_env=conda_env, **kwargs)
        os.chmod(run_path, os.stat(run_path).st_mode | stat.S_IEXEC)


def setup_workflow(root: str, name: str, **kwargs) -> None:
    ppt = PipelineTemplate(name)
    ppt.render(root, root_folder=root)
