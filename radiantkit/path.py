'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import os
import re
from typing import List, Pattern, Tuple

def add_leading_dot(s: str) -> str:
    if '.' != s[0]: s = '.' + s
    return s

def add_extension(path: str, ext: str) -> str:
    ext = add_leading_dot(ext)
    if not path.endswith(ext): path += ext
    return path

def find_re(ipath: str, ireg: Pattern) -> List[str]:
    flist = [f for f in os.listdir(ipath) 
        if os.path.isfile(os.path.join(ipath, f))
        and not type(None) == type(re.match(ireg, f))]
    return flist

def select_by_prefix_and_suffix(dpath: str, flist: List[str],
    prefix: str="", suffix: str= "") -> List[str]:
    if 0 != len(suffix): flist = [f for f in flist
        if os.path.splitext(f)[0].endswith(suffix)]
    if 0 != len(prefix): flist = [f for f in flist
        if os.path.splitext(f)[0].startswith(prefix)]
    return flist

def pair_raw_mask_images(dpath: str, flist: List[str],
    prefix: str="", suffix: str="") -> List[Tuple[str]]:
    for fpath in flist:
        fbase, fext = os.path.splitext(fpath)
        fbase = fbase[len(prefix):-len(suffix)]
        raw_image = f"{fbase}{fext}"
        if not os.path.isfile(os.path.join(dpath, raw_image)):
            logging.warning(f"missing raw image for mask '{fpath}', skipped.")
            flist.pop(flist.index(fpath))
        else:
            flist[flist.index(fpath)] = (raw_image,fpath)
    return flist
