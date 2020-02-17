'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import ggc  # type: ignore
import logging as log
import os
from radiantkit import const
from radiantkit import particle, series
from radiantkit import io, string
from radiantkit.scripts import common
import re
import sys

log.basicConfig(
    level=log.INFO, format='%(asctime)s '
    + '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')


def init_parser(subparsers: argparse._SubParsersAction
                ) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1],
        description='''Export objects from masks as TIFF images.''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Export objects from masks as TIFF images.")

    parser.add_argument(
        'input', type=str,
        help='Path to folder containing deconvolved tiff images and masks.')
    parser.add_argument(
        'ref_channel', type=str,
        help='Name of channel with masks to be used.')

    parser.add_argument(
        '--output', type=str,
        help='''Path to folder where output should be written to.
        Defaults to "objects" subfolder in the input directory.''')
    parser.add_argument(
        '--version', action='version',
        version='%s %s' % (sys.argv[0], const.__version__,))

    critical = parser.add_argument_group("critical arguments")
    critical.add_argument(
        '--mask-prefix', type=str, metavar="TEXT",
        help="""Prefix for output binarized images name.
        Default: ''.""", default='')
    critical.add_argument(
        '--mask-suffix', type=str, metavar="TEXT",
        help="""Suffix for output binarized images name.
        Default: 'mask'.""", default='mask')

    pickler = parser.add_argument_group("pickle arguments")
    pickler.add_argument(
        '--pickle-name', type=str, metavar="STRING",
        help=f"""Filename for input/output pickle file.
        Default: '{const.default_pickle}'""", default=const.default_pickle)
    pickler.add_argument(
        '--export-architecture', action='store_const',
        dest='export_architecture', const=True, default=False,
        help='Export pickled series architecture.')
    pickler.add_argument(
        '--import-architecture', action='store_const',
        dest='import_architecture', const=True, default=False,
        help='Unpickle architecture if pickle file is found.')

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument(
        '--use-labels', action='store_const', dest='labeled',
        const=True, default=False,
        help='Use labels from masks instead of relabeling.')
    advanced.add_argument(
        '--uncompressed', action='store_const', dest='compressed',
        const=False, default=True,
        help='Generate uncompressed TIFF binary masks.')
    advanced.add_argument(
        '--inreg', type=str, metavar="REGEXP",
        help=f"""Regular expression to identify input TIFF images.
        Must contain 'channel_name' and 'series_id' fields.
        Default: '{const.default_inreg}'""", default=const.default_inreg)
    advanced.add_argument(
        '--threads', type=int, metavar="NUMBER", dest="threads", default=1,
        help="""Number of threads for parallelization. Default: 1""")
    advanced.add_argument(
        '-y', '--do-all', action='store_const', const=True, default=False,
        help="""Do not ask for settings confirmation and proceed.""")

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    args.version = const.__version__

    if args.output is None:
        args.output = os.path.join(args.input, 'objects')
    assert not os.path.isfile(args.output)
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    assert '(?P<channel_name>' in args.inreg
    assert '(?P<series_id>' in args.inreg
    args.inreg = re.compile(args.inreg)

    args.mask_prefix = string.add_trailing_dot(args.mask_prefix)
    args.mask_suffix = string.add_leading_dot(args.mask_suffix)

    args.threads = ggc.args.check_threads(args.threads)

    return args


def print_settings(args: argparse.Namespace, clear: bool = True) -> str:
    s = f"""# Object extraction v{args.version}

    ---------- SETTING : VALUE ----------

       Input directory : '{args.input}'
      Output directory : '{args.output}'
Reference channel name : '{args.ref_channel}'

           Mask prefix : '{args.mask_prefix}'
           Mask suffix : '{args.mask_suffix}'

            Use labels : {args.labeled}
            Compressed : {args.compressed}

           Pickle name : {args.pickle_name}
         Import pickle : {args.import_architecture}
         Export pickle : {args.export_architecture}

               Threads : {args.threads}
                Regexp : {args.inreg.pattern}
    """
    if clear:
        print("\033[H\033[J")
    print(s)
    return(s)


def confirm_arguments(args: argparse.Namespace) -> None:
    settings_string = print_settings(args)
    if not args.do_all:
        io.ask("Confirm settings and proceed?")

    assert os.path.isdir(args.input), f"image folder not found: {args.input}"

    settings_path = os.path.join(args.output, "extract_objects.config.txt")
    with open(settings_path, "w+") as OH:
        ggc.args.export_settings(OH, settings_string)


def export_tiffs(args: argparse.Namespace,
                 series_list: series.SeriesList) -> None:
    tiff_path = os.path.join(args.output, "tiff")
    assert not os.path.isfile(tiff_path)
    if not os.path.isdir(tiff_path):
        os.mkdir(tiff_path)

    log.info(f"exporting nuclei images to '{tiff_path}'")
    series_list.export_particle_tiffs(
        tiff_path, args.threads, args.compressed)


def run(args: argparse.Namespace) -> None:
    confirm_arguments(args)
    args, series_list = common.init_series_list(args)

    log.info(f"extracting nuclei")
    series_list.extract_particles(particle.Nucleus, threads=args.threads)

    export_tiffs(args, series_list)

    common.pickle_series_list(args, series_list)
