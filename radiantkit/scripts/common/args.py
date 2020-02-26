'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import logging
import os
from radiantkit import const


def check_axes(axes: str) -> None:
    if axes is not None:
        assert all([a in const.default_axes for a in axes])


def check_output_folder_path(opath: str) -> None:
    assert not os.path.isfile(opath)
    if not os.path.isdir(opath):
        os.mkdir(opath)


def set_default_args_for_series_init(
        args: argparse.Namespace) -> argparse.Namespace:
    if "aspect" not in args:
        args.aspect = None
    if "labeled" not in args:
        args.labeled = None
    if "block_side" not in args:
        args.block_side = None
    return args


def check_parallelization_and_pickling(
        args: argparse.Namespace, pickled: bool) -> argparse.Namespace:
    args.pre_threads = args.threads
    if pickled:
        args.threads = 1
        logging.warning(
            "deactivated parallelization when loading pickled instance.")
    return args
