'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import logging
import os
from radiantkit import const
from radiantkit.scripts.common import output
import sys

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s '
    + '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')


def init_parser(subparsers: argparse._SubParsersAction
                ) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split('.')[-1], description=f'''Long description''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Generate radiant report(s).")

    parser.add_argument(
        'input', type=str,
        help='''Path to folder with radiant output.''')

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument(
        '--subdir', type=str, metavar="STRING", default='objects',
        help=f"""Name of subfolder for nested search. Default: 'objects'""")
    advanced.add_argument(
        '--inreg', type=str, metavar="REGEXP",
        help=f"""Regular expression to identify input TIFF images.
        Must contain 'channel_name' and 'series_id' fields.
        Default: '{const.default_inreg}'""", default=const.default_inreg)

    parser.add_argument('--version', action='version',
                        version=f'{sys.argv[0]} {const.__version__}')

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    assert os.path.isdir(args.input)
    return args


def run(args: argparse.Namespace) -> None:
    logging.info(f"looking at '{args.input}'")

    ofinder = output.OutputFinder()
    output_list = ofinder.nested_search_output_types(args.input, args.inreg)

    if 1 == len(output_list):
        # single condition report
        pass
    else:
        # multi-condition report
        pass

    raise NotImplementedError
