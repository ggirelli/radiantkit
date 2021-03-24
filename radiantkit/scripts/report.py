"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
import logging
import os
import radiantkit as ra
from radiantkit import const, report
from radiantkit import argtools as ap
from radiantkit.exception import enable_rich_exceptions
import sys


@enable_rich_exceptions
def init_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1],
        description="Long description",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Generate radiant report(s).",
    )

    parser.add_argument("input", type=str, help="Path to folder with radiant output.")

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument(
        "--subdir",
        type=str,
        metavar="STRING",
        default="objects",
        help="Name of subfolder for nested search. Default: 'objects'",
    )
    advanced.add_argument(
        "--not-root",
        action="store_const",
        dest="is_root",
        const=False,
        default=True,
        help="Input folder is single-condition (not root folder).",
    )
    advanced.add_argument(
        "--inreg",
        type=str,
        metavar="REGEXP",
        help=f"""Regular expression to identify input TIFF images.
        Must contain 'channel_name' and 'series_id' fields.
        Default: '{const.default_inreg}'""",
        default=const.default_inreg,
    )

    advanced.add_argument(
        "--offline",
        action="store_const",
        dest="online",
        const=False,
        default=True,
        help="""Generate report that does not
        require a live internet connection to be visualized.""",
    )

    parser = ap.add_version_argument(parser)
    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


@enable_rich_exceptions
def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    args.input = os.path.abspath(args.input)
    assert os.path.isdir(args.input)
    args.version = ra.__version__
    return args


@enable_rich_exceptions
def run(args: argparse.Namespace) -> None:
    logging.info(f"looking at '{args.input}'")

    repmaker = report.ReportMaker(args.input)
    repmaker.is_root = args.is_root
    repmaker.title = f"Radiant report - {args.input}"
    repmaker.footer = "".join(
        [
            f"Generated with <code>{' '.join(sys.argv)}</code> ",
            f"(<code>v{ra.__version__}</code>).",
        ]
    )
    repmaker.make()

    logging.info("Done. :thumbs_up: :smiley:")
