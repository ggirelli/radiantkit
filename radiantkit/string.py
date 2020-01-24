'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import re
from typing import Iterator, Optional

class MultiRange(object):
    __string_range: Optional[str] = None
    __extremes_list: Optional[list] = None
    __reg: re.Pattern = re.compile(r'^[0-9-, ]+$')
    zero_indexed = False

    def __init__(self, s: str):
        super(MultiRange, self).__init__()
        assert (self.__reg.search(s) is not None
            ), ("cannot parse range string. " +
            "It should only contains numbers, commas, dashes, and spaces.")[0]
        self.__string_range = s

        string_range_list = [b.strip() for b in self.__string_range.split(",")]
        
        self.__extremes_list = []
        for string_range in string_range_list:
            extremes = [int(x) for x in string_range.split("-")]
            assert len(extremes) in (1, 2), "a range should be specified as A-B"
            extremes = list(set(extremes))
            if 2 == len(extremes):
                assert extremes[1] >= extremes[0]
            self.__extremes_list.append(tuple(extremes))

        self.__extremes_list = sorted(
            self.__extremes_list, key = lambda x: x[0])
        self.__clean_extremes_list()

    def __clean_extremes_list(self):
        is_clean = False
        while not is_clean:
            popped = 0
            i = 0
            while i < len(self.__extremes_list)-1:
                A = self.__extremes_list[i]
                B = self.__extremes_list[i+1]
                if 1 == len(A):
                    if 1 == len(B):
                        if A == B:
                            self.__extremes_list.pop(i)
                            popped = 1
                            break
                        elif A[0] == B[0]-1:
                            self.__extremes_list[i] = (A[0], B[0])
                            self.__extremes_list.pop(i+1)
                            popped = 1
                            break
                    elif 2 == len(B):
                        if A[0] >= B[0]-1:
                            self.__extremes_list[i+1] = (A[0], B[1])
                            self.__extremes_list.pop(i)
                            popped = 1
                            break
                elif 2 == len(A):
                    if 1 == len(B):
                        if B[0] in (A[1], A[1]+1):
                            self.__extremes_list[i] = (A[0], B[0])
                            self.__extremes_list.pop(i+1)
                            popped = 1
                            break
                        elif A[1] > B[0]:
                            self.__extremes_list.pop(i+1)
                            popped = 1
                            break
                    elif 2 == len(B):
                        if A[1] >= B[0] and A[1] < B[1]:
                            self.__extremes_list[i] = (A[0], B[1])
                            self.__extremes_list.pop(i+1)
                            popped = 1
                            break
                        elif A[1] >= B[1]:
                            self.__extremes_list.pop(i+1)
                            popped = 1
                            break
                i += 1
            if i >= len(self.__extremes_list)-2+popped: is_clean = True

    def __next__(self) -> int:
        for extremes in self.__extremes_list:
            if 1 == len(extremes):
                yield extremes[0] - self.zero_indexed
            else:
                for i in range(extremes[0], extremes[1]+1):
                    yield i - self.zero_indexed

    def __iter__(self) -> Iterator[int]:
        return self.__next__()
