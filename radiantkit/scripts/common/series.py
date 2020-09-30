"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
import logging
import os
import pickle
from radiantkit import series
from radiantkit.scripts.common import args as ra_args
from typing import Tuple


def init_series_list(
    args: argparse.Namespace,
) -> Tuple[argparse.Namespace, series.SeriesList]:
    pickled = False
    series_list = None
    pickle_path = os.path.join(args.input, args.pickle_name)
    args = ra_args.set_default_args_for_series_init(args)

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
            args.input,
            args.inreg,
            args.ref_channel,
            (args.mask_prefix, args.mask_suffix),
            args.aspect,
            args.labeled,
            args.block_side,
        )

    logging.info(
        f"parsed {len(series_list)} series with "
        + f"{len(series_list.channel_names)} channels each"
        + f": {series_list.channel_names}"
    )

    args = ra_args.check_parallelization_and_pickling(args, pickled)

    return args, series_list


def pickle_series_list(
    args: argparse.Namespace, series_list: series.SeriesList
) -> None:
    if args.export_instance:
        logging.info("Pickling instance")
        series_list.unload()
        series_list.to_pickle(args.input, args.pickle_name)
