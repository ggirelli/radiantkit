'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import logging
import os
import pickle
from radiantkit import const
from radiantkit import series
from typing import Tuple


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
    if pickled:
        args.threads = 1
        logging.warning(
            "deactivated parallelization when loading pickled instance.")
    return args


def init_series_list(args: argparse.Namespace
                     ) -> Tuple[argparse.Namespace, series.SeriesList]:
    pickled = False
    series_list = None
    pickle_path = os.path.join(args.input, args.pickle_name)
    args = set_default_args_for_series_init(args)

    if os.path.exists(pickle_path):
        if not args.import_instance:
            logging.info(f"found '{args.pickle_name}' file in input folder.")
            logging.info("use --import-instance flag to unpickle it.")
        if args.import_instance:
            with open(pickle_path, "rb") as PI:
                series_list = pickle.load(PI)
                pickled = True

    if series_list is None:
        logging.info(f"parsing series folder")
        series_list = series.SeriesList.from_directory(
            args.input, args.inreg, args.ref_channel,
            (args.mask_prefix, args.mask_suffix),
            args.aspect, args.labeled, args.block_side)

    logging.info(f"parsed {len(series_list)} series with "
                 + f"{len(series_list.channel_names)} channels each"
                 + f": {series_list.channel_names}")

    args = check_parallelization_and_pickling(args, pickled)

    return args, series_list


def pickle_series_list(args: argparse.Namespace,
                       series_list: series.SeriesList) -> None:
    if args.export_instance:
        logging.info("Pickling instance")
        series_list.unload()
        series_list.to_pickle(args.input, args.pickle_name)
