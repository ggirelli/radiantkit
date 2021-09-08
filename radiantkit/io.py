# PYTHON_ARGCOMPLETE_OK

"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import os
import logging
from rich.console import Console  # type: ignore
from rich.logging import RichHandler  # type: ignore
from typing import Optional


def add_log_file_handler(path: str, logger_name: Optional[str] = None) -> None:
    """Adds log file handler to logger.

    By defaults, adds the handler to the root logger.

    Arguments:
        path {str} -- path to output log file

    Keyword Arguments:
        logger_name {str} -- logger name (default: {""})
    """
    assert not os.path.isdir(path)
    log_dir = os.path.dirname(path)
    assert os.path.isdir(log_dir) or log_dir == ""
    fh = RichHandler(console=Console(file=open(path, mode="w+")), markup=True)
    fh.setLevel(logging.INFO)
    logging.getLogger(logger_name).addHandler(fh)
    logging.info(f"[green]Log to[/]: '{path}'")
