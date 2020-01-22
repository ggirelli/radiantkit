'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import re
import _sre
from typing import Iterator

class MultiRange(object):
    __string_range: str = None
    __multirange: list = None
    __reg: _sre.SRE_PATTERN = re.compile(r'^[0-9-,]+$')

    def __init__(self, s: str):
        super(MultiRange, self).__init__()
        assert self.__reg.search(s) is not None
        self.__string_range = s

        string_range_list = [b.strip() for b in self.__string_range.split(",")]
        
        self.__multirange = []
        for string_range in string_range_list:
            extremities = [int(x) for x in string_range.split("-")]
            assert len(extremities) in (1, 2)
            if 2 == len(extremities):
                assert extremities[1] > extremities[2]
            multirange.append(tuple(extremities))
        multirange = sorted(multirange, key = lambda x: x[0])

    def __get__(self) -> Iterator[int]:
        for extremities in multirange:
            if 1 == len(extremities):
                yield extremities[0]
            else:
                for i in range(extremities[0] to extremities[1]):
                    yield i

    def __init__(self, arg):
        
        self.arg = arg
        


