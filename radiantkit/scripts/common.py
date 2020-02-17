'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import logging
import os
import pickle
from radiantkit import series
from typing import List


def set_default_args(args: argparse.Namespace,
                     label_list: List[str]) -> argparse.Namespace:
    for label in label_list:
        if label not in args:
            args[label] = None
    return args


def init_series_list(args: argparse.Namespace) -> series.SeriesList:
    series_list = None
    pickle_path = os.path.join(args.input, args.pickle_name)
    args = set_default_args(args, ['aspect', 'labeled', 'block_side'])

    if os.path.exists(pickle_path):
        if not args.import_architecture:
            logging.info(f"Found '{args.pickle_name}' file in input folder."
                         + "Use --import-architecture flag to unpickle it.")
        if args.import_architecture:
            with open(pickle_path, "rb") as PI:
                series_list = pickle.load(PI)

    if series_list is None:
        logging.info(f"parsing series folder")
        series_list = series.SeriesList.from_directory(
            args.input, args.inreg, args.dna_channel,
            (args.mask_prefix, args.mask_suffix),
            args.aspect, args.labeled, args.block_side)

    logging.info(f"parsed {len(series_list)} series with "
                 + f"{len(series_list.channel_names)} channels each"
                 + f": {series_list.channel_names}")

    return series_list


def pickle_series_list(args: argparse.Namespace,
                       series_list: series.SeriesList) -> None:
    if args.export_architecture:
        logging.info("Pickling architecture")
        series_list.to_pickle(args.input, args.pickle_name)
