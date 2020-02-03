'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import logging
from radiantkit.const import __version__
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s ' +
    '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

def init_parser(subparsers: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(__name__.split('.')[-1], description = f'''
Long description''',
        formatter_class = argparse.RawDescriptionHelpFormatter,
        help="Analyze YFISH experiments.")

    parser.add_argument('input', type = str,
        help = '''Path to root folder (see description for details).''')
    parser.add_argument('input', type = str,
        help = '''Path to output folder.''')

    parser.add_argument('--version', action = 'version',
        version = f'{sys.argv[0]} {__version__}')

    critical = parser.add_argument_group("Critical arguments")
    critical.add_argument('--aspect', type=float, nargs=3, help="""Physical size
    of Z, Y and X voxel sides. Default: 300.0 216.6 216.6""",
    metavar=('Z','Y','X'), default=[300., 216.6, 216.6])
    critical.add_argument('--ref', metavar = "CHANNEL_NAME",
    	type = str, help = """Name of reference channel. Must have been
    	previously segmented. Default: 'dapi'""", default = "dapi")
    critical.add_argument('--mask-prefix', type=str, metavar="TEXT",
        help="""Prefix for output binarized images name.
        Default: ''.""", default='')
    critical.add_argument('--mask-suffix', type=str, metavar="TEXT",
        help="""Suffix for output binarized images name.
        Default: 'mask'.""", default='mask')
    critical.add_argument('--method', type=str, metavar="TEXT",
        help="""""", default=None)
    critical.add_argument('--midsection', type=str, metavar="TEXT",
        help="""""", default=None)
    critical.add_argument('--distance', type=str, metavar="TEXT",
        help="""""", default=None)

    nuclear_selection = parser.add_argument_group("Nuclei selection")
    nuclear_selection.add_argument('--k-sigma', type=float, metavar="NUMBER",
        help="""Suffix for output binarized images name.
        Default: 2.5""", default=2.5)
    nuclear_selection.add_argument('--use-all-nuclei', action='store_const',
        dest='skip_nuclear_selection', const=True, default=False,
        help='Skip selection of G1 nuclei.')

    minor = parser.add_argument_group("Minor arguments")
    minor.add_argument('--unit', type = str, metavar="STRING", default="nm",
        help="""Unit of measure for the aspect. Default: nm""")
    minor.add_argument('--description', type=str, nargs='*', metavar="STRING",
        help = """Space separated 'condition:description' couples.
        'condition' is the name of a condition folder. 'description' is a
        descriptive label that replace folder names in the report.
        Use '--' after the last one.""")
    minor.add_argument('--note', type=str, help="""A short description of the
        dataset. Included in the final report. Use double quotes.""")

    advanced = parser.add_argument_group("Advanced arguments")
    advanced.add_argument('--use-labels',
        action='store_const', dest='labeled',
        const=True, default=False,
        help='Use labels from masks instead of relabeling.')
    advanced.add_argument('--no-rescaling',
        action='store_const', dest='do_rescaling',
        const=False, default=True,
        help='Do not rescale image even if deconvolved.')
    advanced.add_argument('--debug',
        action='store_const', dest='debug_mode',
        const=True, default=False, help='Log also debugging messages.')
    default_inreg='^.*\.tiff?$'
    advanced.add_argument('--inreg', type=str, metavar="REGEXP",
        help="""Regular expression to identify input TIFF images.
        Default: '%s'""" % (default_inreg,), default=default_inreg)
    advanced.add_argument('-t', type=int, metavar="NUMBER", dest="threads",
        help="""Number of threads for parallelization. Default: 1""",
        default=1)
    
    parser.set_defaults(parse=parse_arguments, run=run)

    return parser

def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    return args

def run(args: argparse.Namespace) -> None:
	pass
