"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import re
from string import Template
from typing import Iterator, List, Optional, Pattern, Set, Tuple


class MultiRange(object):
    __current_item: Tuple[int, int] = (0, 0)
    __string_range: str
    __extremes_list: List[Tuple[int, int]]
    __reg: Pattern = re.compile(r"^[0-9-, ]+$")
    __length: Optional[int] = None
    __ready: bool = False

    def __init__(self, s: str):
        super(MultiRange, self).__init__()

        assert self.__reg.search(s) is not None, " ".join(
            [
                "cannot parse range string. It should only contains numbers,",
                "commas, dashes, and spaces.",
            ]
        )
        self.__string_range = s

        string_range_list = [b.strip() for b in self.__string_range.split(",")]

        self.__extremes_list = []
        for string_range in string_range_list:
            extremes = [int(x) for x in string_range.split("-")]
            if len(extremes) == 1:
                extremes = [extremes[0], extremes[0]]
            assert 2 == len(extremes), "a range should be specified as A-B"
            assert extremes[1] >= extremes[0]
            self.__extremes_list.append((extremes[0], extremes[1]))

        self.__extremes_list = sorted(self.__extremes_list, key=lambda x: x[0])
        self.__clean_extremes_list()

        assert 0 < self.__extremes_list[0][0], "'page' count starts from 1."
        self.__ready = True

    @property
    def length(self):
        if self.__length is None and self.__ready:
            self.__length = 0
            for a, b in self.__extremes_list:
                self.__length += b - a + 1
        return self.__length

    def __check_overlap(self, i: int) -> bool:
        A = self.__extremes_list[i]
        B = self.__extremes_list[i + 1]
        if A[1] >= B[0] and A[1] < B[1]:
            self.__extremes_list[i] = (A[0], B[1])
            self.__extremes_list.pop(i + 1)
            return True
        elif A[1] >= B[1]:
            self.__extremes_list.pop(i + 1)
            return True
        return False

    def __clean_extremes_list(self) -> None:
        is_clean = False
        while not is_clean:
            popped = 0
            i = 0
            while i < len(self.__extremes_list) - 1:
                if self.__check_overlap(i):
                    popped = 1
                    break
                i += 1
            if i >= len(self.__extremes_list) - 2 + popped:
                is_clean = True

    def __next__(self) -> int:
        current_range_id, current_item = self.__current_item
        if current_range_id >= len(self.__extremes_list):
            raise StopIteration
        current_range = self.__extremes_list[current_range_id]
        if current_item >= current_range[1] - current_range[0]:
            self.__current_item = (current_range_id + 1, 0)
        else:
            self.__current_item = (current_range_id, current_item + 1)
        return current_range[0] + current_item

    def __iter__(self) -> Iterator[int]:
        self.__current_item = (0, 0)
        return self

    def __len__(self) -> Optional[int]:
        return self.length


class TIFFNameTemplateFields(object):
    CHANNEL_NAME = "${channel_name}"
    CHANNEL_ID = "${channel_id}"
    SERIES_ID = "${series_id}"
    DIMENSIONS = "${dimensions}"
    AXES_ORDER = "${axes_order}"


class TIFFNameTemplate(Template):
    def __init__(self, s: str):
        super(TIFFNameTemplate, self).__init__(s)

    def can_export_fields(
        self, n_fields: int, selected_fields: Optional[Set[int]] = None
    ) -> bool:
        return (
            n_fields <= 1
            or TIFFNameTemplateFields.SERIES_ID in self.template
            or (selected_fields is None or len(selected_fields) <= 1)
            and selected_fields is not None
        )

    def can_export_channels(
        self, n_channels: int, selected_channels: Optional[Set[str]]
    ) -> bool:
        seeds_missing = all(
            x not in self.template
            for x in [
                TIFFNameTemplateFields.CHANNEL_ID,
                TIFFNameTemplateFields.CHANNEL_NAME,
            ]
        )
        return (
            n_channels <= 1
            or not seeds_missing
            or (selected_channels is None or len(selected_channels) <= 1)
            and selected_channels is not None
        )


def add_leading_delim(suffix: str, delim: str = ".") -> str:
    if len(suffix) != 0 and not suffix.startswith(delim):
        suffix = f"{delim}{suffix}"
    return suffix


def add_leading_dot(suffix: str) -> str:
    return add_leading_delim(suffix)


def add_trailing_delim(prefix: str, delim: str = ".") -> str:
    if len(prefix) != 0 and not prefix.endswith(delim):
        prefix = f"{prefix}{delim}"
    return prefix


def add_trailing_dot(prefix: str) -> str:
    return add_trailing_delim(prefix)
