"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from importlib.metadata import version
from typing import List

from radiantkit import const, conversion

try:
    __version__ = version(__name__)
except Exception as e:
    raise e

__all__ = ["__version__", "const", "conversion"]
__path__: List[str]
