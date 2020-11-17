"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import logging
import sys
from typing import Callable


def enable_rich_exceptions(fun: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except Exception as e:
            logging.exception(e)
            sys.exit()

    return wrapper
