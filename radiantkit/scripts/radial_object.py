"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
from radiantkit.const import __version__
import sys


def init_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1],
        description=f"""Long description""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Generate single-cell radial profiles. *NOT IMPLEMENTED*",
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to folder containing deconvolved tiff images and masks.",
    )
    parser.add_argument(
        "ref_channel", type=str, help="Name of channel with DNA staining intensity."
    )

    parser.add_argument(
        "--version", action="version", version=f"{sys.argv[0]} {__version__}"
    )

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    return args


def run(args: argparse.Namespace) -> None:
    raise NotImplementedError
