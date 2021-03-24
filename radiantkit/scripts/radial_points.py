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
        help="Measure the radial position of a set of points." + " *NOT IMPLEMENTED*",
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to folder containing deconvolved tiff images and masks.",
    )
    parser.add_argument(
        "ref_channel", type=str, help="Name of channel with DNA staining intensity."
    )

    parser = ap.add_version_argument(parser)
    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    return args


def run(args: argparse.Namespace) -> None:
    raise NotImplementedError
