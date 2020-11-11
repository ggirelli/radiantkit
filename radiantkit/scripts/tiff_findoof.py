"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""


import argparse
from joblib import cpu_count, delayed, Parallel  # type: ignore
import logging
import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore
from radiantkit.const import __version__
from radiantkit import channel, path, stat
from radiantkit.exception import enable_rich_assert
from radiantkit.io import add_log_file_handler
from rich.logging import RichHandler  # type: ignore
import sys
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)


@enable_rich_assert
def init_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1],
        description="""
Calculate gradient magnitude over Z for every image in the input folder with a
filename matching the --inreg. Use --range to change the in-focus
definition.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Find out of focus fields of view.",
    )

    parser.add_argument("input", type=str, help="Path to folder with tiff images.")

    parser.add_argument(
        "--output",
        type=str,
        help="Path to output tsv file. Default: oof.tsv in input folder.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        help="Fraction of stack (middle-centered) for in-focus fields. Default: .5",
        default=0.5,
    )

    # parser.add_argument(
    #     "--plot",
    #     action="store_const",
    #     const=True,
    #     default=False,
    #     help="""Generate pdf plot of intensity sum per Z-slice.""",
    # )

    parser.add_argument(
        "--version",
        action="version",
        version="%s %s"
        % (
            sys.argv[0],
            __version__,
        ),
    )

    advanced = parser.add_argument_group("advanced arguments")
    default_inreg = "^.*\\.tiff?$"
    advanced.add_argument(
        "--inreg",
        type=str,
        metavar="REGEXP",
        help="""Regular expression to identify input TIFF images.
        Default: '%s'"""
        % (default_inreg,),
        default=default_inreg,
    )
    advanced.add_argument(
        "--threads",
        metavar="NUMBER",
        type=int,
        default=1,
        help="""Number of threads for parallelization. Default: 1""",
    )
    advanced.add_argument(
        "--intensity-sum",
        action="store_const",
        const=True,
        default=False,
        help="""Use intensity sum instead of gradient magnitude.""",
    )
    advanced.add_argument(
        "--rename",
        action="store_const",
        const=True,
        default=False,
        help="""Rename out-of-focus images by adding the '.old' suffix.""",
    )

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


@enable_rich_assert
def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    if args.output is None:
        args.output = os.path.join(args.input, "oof.tsv")
    args.threads = cpu_count() if args.threads > cpu_count() else args.threads
    return args


def describe_slices(
    args: argparse.Namespace, img: channel.ImageGrayScale
) -> List[float]:
    slice_descriptors = []
    for zi in range(img.shape[0]):
        if args.intensity_sum:
            slice_descriptors.append(img.pixels[zi].sum())
        else:
            dx = stat.gpartial(img.pixels[zi, :, :], 1, 1)
            dy = stat.gpartial(img.pixels[zi, :, :], 2, 1)
            slice_descriptors.append(np.mean(np.mean((dx ** 2 + dy ** 2) ** (1 / 2))))
    return slice_descriptors


def is_OOF(args: argparse.Namespace, ipath: str) -> pd.DataFrame:
    img = channel.ImageGrayScale.from_tiff(os.path.join(args.input, ipath))

    slice_descriptors = describe_slices(args, img)

    profile_data = pd.DataFrame.from_dict(
        dict(
            path=np.repeat(ipath, img.shape[0]),
            x=np.array(range(img.shape[0])) + 1,
            y=slice_descriptors,
        )
    )

    max_slice_id = slice_descriptors.index(max(slice_descriptors))
    halfrange = img.shape[0] * args.fraction / 2.0
    halfstack = img.shape[0] / 2.0

    response = "out-of-focus"
    if max_slice_id >= (halfstack - halfrange):
        if max_slice_id <= (halfstack + halfrange):
            response = "in-focus"
    logging.info(f"{ipath} is {response}.")
    profile_data["response"] = response

    if "out-of-focus" == response and args.rename:
        os.rename(os.path.join(args.input, ipath), os.path.join(args.input, ipath))

    return profile_data


@enable_rich_assert
def run(args: argparse.Namespace) -> None:
    assert os.path.isdir(args.input), f"image directory not found: '{args.input}'"
    add_log_file_handler(os.path.join(args.input, "oof.log.txt"))
    logging.info(f"Input:\t\t{args.input}")
    logging.info(f"Output:\t\t{args.output}")
    logging.info(f"Fraction:\t{args.fraction}")
    logging.info(f"Rename:\t\t{args.rename}")
    if args.intensity_sum:
        logging.info("Mode:\t\tintensity_sum")
    else:
        logging.info("Mode:\t\tgradient_of_magnitude")
    logging.info(f"Regexp:\t\t{args.inreg}")
    logging.info(f"Threads:\t{args.threads}")

    series_data = Parallel(n_jobs=args.threads, verbose=11)(
        delayed(is_OOF)(args, impath) for impath in path.find_re(args.input, args.inreg)
    )

    pd.concat(series_data).to_csv(args.output, "\t", index=False)
    # if args.plot:
    #     plot_profile(args, series_data, f"{os.path.splitext(args.output)[0]}.pdf")

    logging.info("Done. :thumbs_up: :smiley:")
