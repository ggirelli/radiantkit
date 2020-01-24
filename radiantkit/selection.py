'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from typing import Tuple

class RegionOfInterest(object):
    def __init__(self):
        super(RegionOfInterest, self).__init__()

class BoundingElement(RegionOfInterest):
    def __init__(self, axes_bounds: Tuple[Tuple[int]]):
        super(BoundingElement, self).__init__()
        
