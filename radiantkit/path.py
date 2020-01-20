'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

def add_leading_dot(s: str) -> str:
    if '.' != s[0]: s = '.' + s
    return s

def add_extension(path: str, ext: str) -> str:
    ext = add_leading_dot(ext)
    if not path.endswith(ext): path += ext
    return path
