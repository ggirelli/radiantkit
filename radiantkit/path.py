'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import logging
import os
import re
from typing import List, Optional, Pattern, Tuple


def add_leading_dot(s: str) -> str:
    if '.' != s[0]:
        s = '.' + s
    return s


def add_extension(path: str, ext: str) -> str:
    ext = add_leading_dot(ext)
    if not path.endswith(ext):
        path += ext
    return path


def find_re(ipath: str, ireg: Pattern) -> List[str]:
    flist = [f for f in os.listdir(ipath)
             if (os.path.isfile(os.path.join(ipath, f))
                 and re.match(ireg, f) is not None)]
    return flist


def select_by_prefix_and_suffix(dpath: str, ilist: List[str],
                                prefix: str = "", suffix: str = ""
                                ) -> Tuple[List[str], List[str]]:
    olist = ilist.copy()
    if 0 != len(suffix):
        olist = [f for f in olist if os.path.splitext(f)[0].endswith(suffix)]
    if 0 != len(prefix):
        olist = [f for f in olist if os.path.splitext(f)[0].startswith(prefix)]
    return (olist, [x for x in ilist if x not in olist])


def pair_raw_mask_images(
        dpath: str, flist: List[str], prefix: str = "", suffix: str = ""
        ) -> List[Tuple[str, str]]:
    olist: List[Tuple[str, str]] = []
    for fpath in flist:
        fbase, fext = os.path.splitext(fpath)
        fbase = fbase[len(prefix):-len(suffix)]
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
        return (int(finfo['series_id']), finfo['channel_name'])
    else:
        return None
