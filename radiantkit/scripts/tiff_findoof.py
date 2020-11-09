"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""


import argparse
from ggc.args import check_threads  # type: ignore
from joblib import delayed, Parallel  # type: ignore
import logging
import matplotlib as mplt  # type: ignore
import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore
from radiantkit.const import __version__
from radiantkit import channel, path, plot, stat
from rich.logging import RichHandler  # type: ignore
import sys
from tqdm import tqdm  # type: ignore
from typing import List

mplt.use("ps")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)


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
    parser.add_argument("output", type=str, help="Path to output tsv file.")

    parser.add_argument(
        "--range",
        type=float,
        metavar="NUMBER",
        help="""Fraction of stack (middle-centered) for an in-focus field of
        view. Default: .5""",
        default=0.5,
    )

    parser.add_argument(
        "--plot",
        action="store_const",
        const=True,
        default=False,
        help="""Generate pdf plot of intensity sum per Z-slice.""",
    )

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
    advanced.add_argument(
        "--silent",
        action="store_const",
        const=True,
        default=False,
        help="""Silent run.""",
    )

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    args.threads = check_threads(args.threads)
    return args


def plot_profile(
    args: argparse.Namespace, series_data: pd.DataFrame, path: str
) -> None:
    mplt.pyplot.figure(figsize=[12, 8])

    xmax = []
    ymax = []
    image_names = []
    for profile_data in series_data:
        image_names.append(os.path.basename(profile_data["path"].values[0]))
        xmax.append(max(profile_data["x"]))
        ymax.append(max(profile_data["y"]))
        mplt.pyplot.plot(profile_data["x"], profile_data["y"], linewidth=0.5)
    xmax = max(xmax)
    ymax = max(ymax)

    mplt.pyplot.xlabel("Z-slice index")
    if args.intensity_sum:
        mplt.pyplot.ylabel("Intensity sum [a.u.]")
    else:
        mplt.pyplot.ylabel("Gradient magnitude [a.u.]")
    mplt.pyplot.title("Focus analysis")

    mplt.pyplot.legend(
        image_names, bbox_to_anchor=(1.04, 1), loc="upper left", prop={"size": 6}
    )
    mplt.pyplot.subplots_adjust(right=0.75)

    mplt.pyplot.gca().axvline(
        x=xmax * args.range / 2, ymax=ymax, linestyle="--", color="k"
    )
    mplt.pyplot.gca().axvline(
        x=xmax - xmax * args.range / 2, ymax=ymax, linestyle="--", color="k"
    )

    plot.export(path)


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


def is_OOF(
    args: argparse.Namespace, ipath: str, logger: logging.Logger
) -> pd.DataFrame:
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
    halfrange = img.shape[0] * args.range / 2.0
    halfstack = img.shape[0] / 2.0

    response = "out-of-focus"
    if max_slice_id >= (halfstack - halfrange):
        if max_slice_id <= (halfstack + halfrange):
            response = "in-focus"
    logger.info(f"{ipath} is {response}.")
    profile_data["response"] = response

    if "out-of-focus" == response and args.rename:
        os.rename(os.path.join(args.input, ipath), os.path.join(args.input, ipath))

    return profile_data


def run(args: argparse.Namespace) -> None:
    logger = logging.getLogger()
    if 1 == args.threads:
        FH = logging.FileHandler(
            filename=f"{os.path.splitext(args.output)[0]}.log", mode="w+"
        )
        FH.setLevel(logging.INFO)
        logger.addHandler(FH)

    if not os.path.isdir(args.input):
        logger.error(f"image directory not found: '{args.input}'")
        sys.exit()

    imlist = path.find_re(args.input, args.inreg)

    if 1 == args.threads:
        t = imlist if args.silent else tqdm(imlist, desc=os.path.dirname(args.input))
        series_data = [is_OOF(args, impath, logger) for impath in t]
    else:
        verbosity = 11 if not args.silent else 0
        series_data = Parallel(n_jobs=args.threads, verbose=verbosity)(
            delayed(is_OOF)(args, impath, logger) for impath in imlist
        )

    pd.concat(series_data).to_csv(args.output, "\t", index=False)
    if args.plot:
        plot_profile(args, series_data, f"{os.path.splitext(args.output)[0]}.pdf")
