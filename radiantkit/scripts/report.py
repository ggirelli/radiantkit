"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
import logging
import os
from radiantkit import const, output, report
from rich.logging import RichHandler  # type: ignore
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)


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

    parser.add_argument(
        "--version", action="version", version=f"{sys.argv[0]} {const.__version__}"
    )

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    assert os.path.isdir(args.input)
    args.version = const.__version__
    return args


def run(args: argparse.Namespace) -> None:
    logging.info(f"looking at '{args.input}'")
    output_list = output.OutputReader.read_recursive(args.input, args.inreg)

    logging.info("preparing plots")
    plot_data = output.OutputPlotter.plot(output_list)

    logging.info("generating HTML report")
    report.general_report(args.input, args, output_list, plot_data)

    raise NotImplementedError
