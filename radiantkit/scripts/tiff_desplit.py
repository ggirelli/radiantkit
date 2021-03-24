"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
from radiantkit import argtools as ap


def init_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1],
        description="Long description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Stitch together images split with tiff_split. *NOT IMPLEMENTED*",
    )

    parser.add_argument("input", type=str, help="""Path to the czi file to convert.""")

    parser.add_argument(
        "-o",
        "--outdir",
        metavar="outdir",
        type=str,
        default=None,
        help="""Path to output TIFF folder, created if missing. Default to a
        folder with the input file basename.""",
    )

    parser.add_argument(
        "-C",
        "--compressed",
        action="store_const",
        dest="doCompress",
        const=True,
        default=False,
        help="Force compressed TIFF as output.",
    )

    parser = ap.add_version_argument(parser)
    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    return args


def run(args: argparse.Namespace) -> None:
    raise NotImplementedError
