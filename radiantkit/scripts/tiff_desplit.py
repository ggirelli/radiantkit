'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import logging
from radiantkit.const import __version__
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
        help="Short description")

    parser.add_argument(
        'input', type=str,
        help='''Path to the czi file to convert.''')

    parser.add_argument(
        '-o', '--outdir', metavar="outdir", type=str, default=None,
        help="""Path to output TIFF folder, created if missing. Default to a
        folder with the input file basename.""")

    parser.add_argument(
        '-C', '--compressed', action='store_const', dest='doCompress',
        const=True, default=False, help='Force compressed TIFF as output.')

    parser.add_argument('--version', action='version',
                        version=f'{sys.argv[0]} {__version__}')

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    return args


def run(args: argparse.Namespace) -> None:
    pass
