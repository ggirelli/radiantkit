"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import logging
import os
from radiantkit import string
import re
from typing import List, Optional, Pattern, Tuple


FileList = List[str]
RawMaskPair = Tuple[str, str]


def add_suffix(path: str, suffix: str, delim: str = ".") -> str:
    fname, fext = os.path.splitext(path)
    if not fname.endswith(suffix):
        fname += string.add_leading_delim(suffix, delim)
    return f"{fname}{fext}"


def add_extension(path: str, ext: str, delim: str = ".") -> str:
    ext = string.add_leading_delim(ext, delim)
    if not path.endswith(ext):
        path += ext
    return path


def find_re(ipath: str, ireg: Pattern) -> FileList:
    flist = [
        f
        for f in os.listdir(ipath)
        if (os.path.isfile(os.path.join(ipath, f)) and re.match(ireg, f) is not None)
    ]
    return flist


def select_by_prefix_and_suffix(
    dpath: str, ilist: FileList, prefix: str = "", suffix: str = ""
) -> Tuple[FileList, FileList]:
    olist = ilist.copy()
    if 0 != len(suffix):
        olist = [f for f in olist if os.path.splitext(f)[0].endswith(suffix)]
    if 0 != len(prefix):
        olist = [f for f in olist if os.path.splitext(f)[0].startswith(prefix)]
    return (olist, [x for x in ilist if x not in olist])


def pair_raw_mask_images(
    dpath: str, flist: List[str], prefix: str = "", suffix: str = ""
) -> List[RawMaskPair]:
    olist: List[RawMaskPair] = []
    for fpath in flist:
        fbase, fext = os.path.splitext(fpath)
        fbase = fbase[slice(len(prefix), len(fbase) - len(suffix) + 1)]
        raw_image = f"{fbase}{fext}"
        if not os.path.isfile(os.path.join(dpath, raw_image)):
            logging.warning(f"missing raw image for mask '{fpath}', skipped.")
            flist.pop(flist.index(fpath))
        else:
            olist.append((raw_image, fpath))
    return olist


def get_image_details(path: str, inreg: Pattern) -> Optional[Tuple[int, str]]:
    fmatch = re.match(inreg, os.path.basename(path))
    if fmatch is not None:
        finfo = fmatch.groupdict()
        return (int(finfo["series_id"]), finfo["channel_name"])
    else:
        return None
