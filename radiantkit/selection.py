'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from typing import Tuple

class RegionOfInterest(object):
    """docstring for RegionOfInterest"""
    def __init__(self):
        super(RegionOfInterest, self).__init__()

class SquareOfInterest(RegionOfInterest):
    """docstring for SquareOfInterest"""

    __P1: Tuple[int] = None
    __P2: Tuple[int] = None

    def __init__(self, P1:(int,int), P2:(int,int), *args, **kwargs):
        super(SquareOfInterest, self).__init__(*args, **kwargs)
        assert all([c>=0 for c in P1]) and all([c>=0 for c in P2])
        self.__P1=P1
        self.__P2=P2

class BoxOfInterest(RegionOfInterest):
    """docstring for BoxOfInterest"""

    __P1: Tuple[int] = None
    __P2: Tuple[int] = None

    def __init__(self, P1:(int,int,int), P2:(int,int,int), *args, **kwargs):
        super(BoxOfInterest, self).__init__(*args, **kwargs)
        assert all([c>=0 for c in P1]) and all([c>=0 for c in P2])
        self.__P1=P1
        self.__P2=P2
