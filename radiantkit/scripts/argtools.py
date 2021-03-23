"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
from joblib import cpu_count  # type: ignore
import logging
import os
import pickle as pk
from radiantkit import const
import sys


def check_axes(axes: str) -> None:
    if axes is not None:
        assert all([a in const.default_axes for a in axes])


def check_output_folder_path(opath: str) -> None:
    assert not os.path.isfile(opath)
    if not os.path.isdir(opath):
        os.mkdir(opath)


def set_default_args_for_series_init(args: argparse.Namespace) -> argparse.Namespace:
    if "aspect" not in args:
        args.aspect = None
    if "labeled" not in args:
        args.labeled = None
    if "block_side" not in args:
        args.block_side = None
    return args


def check_parallelization_and_pickling(
    args: argparse.Namespace, pickled: bool
) -> argparse.Namespace:
    args.pre_threads = args.threads
    if pickled:
        args.threads = 1
        logging.warning("deactivated parallelization when loading pickled instance.")
    return args


def add_version_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--version",
        action="version",
        version="%s %s"
        % (
            sys.argv[0],
            const.__version__,
        ),
    )
    return parser


def add_threads_argument(parser: argparse._ArgumentGroup) -> argparse._ArgumentGroup:
    parser.add_argument(
        "--threads",
        metavar="NUMBER",
        type=int,
        default=1,
        help="""Number of threads for parallelization. Default: 1""",
    )
    return parser


def check_threads(threads: int) -> int:
    return max(1, min(cpu_count(), threads))


def add_pattern_argument(parser: argparse._ArgumentGroup) -> argparse._ArgumentGroup:
    parser.add_argument(
        "--inreg",
        type=str,
        metavar="REGEXP",
        help=f"""Regular expression to identify input TIFF images.
        Must contain 'channel_name' and 'series_id' fields.
        Default: '{const.default_inreg}'""",
        default=const.default_inreg,
    )
    return parser


def dump_args(args: argparse.Namespace, path) -> None:
    with open(os.path.join(args.input, path), "wb") as AH:
        args.run = None
        args.parse = None
        pk.dump(args, AH)
