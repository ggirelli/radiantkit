"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
from joblib import cpu_count, delayed, Parallel  # type: ignore
import logging
import os
from radiantkit.const import __version__
from radiantkit import image, path
import re
import sys


def init_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1],
        description="""
(Un)compress TIFF images.
Provide either a single input/output image paths, or input/output folder paths.
In case of folder input/output, all tiff files in the input folder with file
name matching the specified pattern are (un)compressed and saved to the output
folder. When (un)compressing multiple files, the --threads option allows to
parallelize on multiple threads. Disk read/write operations become the
bottleneck when parallelizing.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="(Un)compress TIFF images.",
    )

    parser.add_argument(
        "input",
        type=str,
        help="""Path to the TIFF image to (un)compress, or
        to a folder containing multiple TIFF images. In the latter case, the
        --inreg pattern is used to identify the image file.""",
    )
    parser.add_argument(
        "output",
        type=str,
        help="""Path to output TIFF image, or output folder
        if the input is a folder.""",
    )

    parser.add_argument(
        "-u",
        const=True,
        default=False,
        action="store_const",
        dest="doUncompress",
        help="Uncompress TIFF files.",
    )
    parser.add_argument(
        "-c",
        const=True,
        default=False,
        action="store_const",
        dest="doCompress",
        help="Compress TIFF files.",
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
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        metavar="NUMBER",
        help="""Number of threads for parallelization. Used only to
        (un)compress multiple images (i.e., input is a folder). Default: 1""",
    )

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    args.inreg = re.compile(args.inreg)

    if not args.doCompress and not args.doUncompress:
        logging.error("please, use either -c (compress) or -u (uncompress).")
        sys.exit()
    if args.doCompress and args.doUncompress:
        logging.error("please, use either -c (compress) or -u (uncompress).")
        sys.exit()
    args.threads = cpu_count() if args.threads > cpu_count() else args.threads
    args.process_multiple_files = False
    if os.path.isdir(args.input):
        args.process_multiple_files = True
        assert not os.path.isfile(args.output), "in/output should be folders."
    else:
        assert not os.path.isdir(args.output), "in/output should be files."
        assert os.path.isfile(args.input), f"input not found: '{args.input}'"

    return args


def export_image(ipath: str, opath: str, compress: bool = None) -> str:
    idir = os.path.dirname(ipath)
    ipath = os.path.basename(ipath)
    odir = os.path.dirname(opath)
    opath = os.path.basename(opath)

    if compress is None:
        compress = False
    img = image.Image.from_tiff(os.path.join(idir, ipath))
    if opath is None:
        opath = ipath

    if not compress:
        img.to_tiff(os.path.join(odir, opath), compressed=False)
        label = "Uncompressed"
    else:
        img.to_tiff(os.path.join(odir, opath), compressed=True)
        label = "Compressed"

    logging.info(f"{label} '{os.path.join(idir, ipath)}'")
    return os.path.join(odir, opath)


def run(args: argparse.Namespace) -> None:
    if args.process_multiple_files:
        if not os.path.isdir(args.input):
            logging.error(f"image folder not found: '{args.input}'")
            sys.exit()

        if not os.path.isdir(args.output):
            os.mkdir(args.output)

        imglist = path.find_re(args.input, args.inreg)

        Parallel(n_jobs=args.threads)(
            delayed(export_image)(
                os.path.join(args.input, ipath),
                os.path.join(args.output, ipath),
                compress=args.doCompress,
            )
            for ipath in imglist
        )
    else:
        export_image(args.input, args.output, args.doCompress)
