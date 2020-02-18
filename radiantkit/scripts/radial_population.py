'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import ggc  # type: ignore
import logging
import os
from radiantkit import const
from radiantkit import distance, io, report, string
from radiantkit import particle, series
from radiantkit.scripts import common
import re
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
        help="Generate average radial profiles for a cell population.")

    parser.add_argument(
        'input', type=str,
        help='Path to folder containing deconvolved tiff images and masks.')
    parser.add_argument(
        'ref_channel', type=str,
        help='Name of channel with DNA staining intensity.')

    parser.add_argument(
        '--output', type=str,
        help='''Path to folder where output should be written to.
        Defaults to "objects" subfolder in the input directory.''')
    parser.add_argument('--version', action='version',
                        version=f'{sys.argv[0]} {const.__version__}')

    critical = parser.add_argument_group("critical arguments")
    critical.add_argument(
        '--aspect', type=float, nargs=3, help="""Physical size
        of Z, Y and X voxel sides in nm. Default: 300.0 216.6 216.6""",
        metavar=('Z', 'Y', 'X'), default=[300., 216.6, 216.6])
    critical.add_argument(
        '--axes', type=str, metavar="STRING",
        help="""Axes to be used for distance calculation.""")
    critical.add_argument(
        '--center-type', type=str,
        default=distance.CenterType.get_default().name,
        choices=[t.name for t in distance.CenterType],
        help=f"""Type of center for distance normalization.
        Default: {distance.CenterType.get_default().name}""")
    critical.add_argument(
        '--quantile', type=float, metavar="NUMBER", help=f"""Quantile used to
        identify the center when '--center-type
        {distance.CenterType.QUANTILE.name}' is used.
        A number from 0 to 1 is expected. Defaults to 1e-N where N is
        the number of axes in an image.""")
    critical.add_argument(
        '--mask-prefix', type=str, metavar="TEXT",
        help="""Prefix for output binarized images name.
        Default: ''.""", default='')
    critical.add_argument(
        '--mask-suffix', type=str, metavar="TEXT",
        help="""Suffix for output binarized images name.
        Default: 'mask'.""", default='mask')
    critical.add_argument(
        '--bins', type=int, metavar="NUMBER", default=200,
        help=f"""Number of bins for polynomial fitting. Default: 200.""")
    critical.add_argument(
        '--degree', type=int, metavar="NUMBER", default=5,
        help=f"""Degree of polynomial fitting. Default: 5.""")

    report = parser.add_argument_group("report arguments")
    report.add_argument(
        '--no-report', action='store_const',
        help="""Do not generate an HTML report.""",
        dest="mk_report", const=False, default=True)
    report.add_argument(
        '--online-report', action='store_const',
        help="""Make a smaller HTML report by linking remote JS libraries.""",
        dest="online_report", const=True, default=False)

    pickler = parser.add_argument_group("pickle arguments")
    pickler.add_argument(
        '--pickle-name', type=str, metavar="STRING",
        help=f"""Filename for input/output pickle file.
        Default: '{const.default_pickle}'""", default=const.default_pickle)
    pickler.add_argument(
        '--export-instance', action='store_const',
        dest='export_instance', const=True, default=False,
        help='Export pickled series instance.')
    pickler.add_argument(
        '--import-instance', action='store_const',
        dest='import_instance', const=True, default=False,
        help='Unpickle instance if pickle file is found.')

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument(
        '--block-side', type=int, metavar="NUMBER",
        help="""Structural element side for dilation-based background/
        foreground measurement. Should be odd. Default: 11.""", default=11)
    advanced.add_argument(
        '--use-labels', action='store_const', dest='labeled',
        const=True, default=False,
        help='Use labels from masks instead of relabeling.')
    advanced.add_argument(
        '--no-rescaling', action='store_const', dest='do_rescaling',
        const=False, default=True,
        help='Do not rescale image even if deconvolved.')
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

    if args.axes is not None:
        assert all([a in const.default_axes for a in args.axes])
    if args.center_type is distance.CenterType.QUANTILE:
        if args.quantile is not None:
            assert args.quantile > 0 and args.quantile <= 1
    args.center_type = distance.CenterType[args.center_type]

    if not 0 != args.block_side % 2:
        logging.warning("changed ground block side from "
                        + f"{args.block_side} to {args.block_side+1}")
        args.block_side += 1

    args.threads = ggc.args.check_threads(args.threads)

    return args


def print_settings(args: argparse.Namespace, clear: bool = True) -> str:
    s = f"""# Object extraction v{args.version}

    ---------- SETTING : VALUE ----------

       Input directory : '{args.input}'
      Output directory : '{args.output}'
Reference channel name : '{args.ref_channel}'

    Voxel aspect (ZYX) : {args.aspect}
                  Axes : {args.axes}
           Center type : {args.center_type}
              Quantile : {args.quantile}
                  Bins : {args.bins}
                Degree : {args.degree}

           Mask prefix : '{args.mask_prefix}'
           Mask suffix : '{args.mask_suffix}'

     Ground block side : {args.block_side}
            Use labels : {args.labeled}
               Rescale : {args.do_rescaling}
            Compressed : {args.compressed}

           Pickle name : {args.pickle_name}
         Import pickle : {args.import_instance}
         Export pickle : {args.export_instance}

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

    assert os.path.isdir(args.input), f"input folder not found: {args.input}"

    settings_path = os.path.join(args.output, "radial_population.config.txt")
    with open(settings_path, "w+") as OH:
        ggc.args.export_settings(OH, settings_string)


def mk_report(args: argparse.Namespace, profiles: series.RadialProfileData,
              series_list: series.SeriesList) -> None:
    if args.mk_report:
        report_path = os.path.join(args.input, "radial_population.report.html")
        logging.info(f"writing report to\n{report_path}")
        report.report_select_nuclei(
            args, report_path, args.online_report,
            profiles=profiles, series_list=series_list)


def run(args: argparse.Namespace) -> None:
    confirm_arguments(args)
    args, series_list = common.init_series_list(args)

    logging.info(f"extracting nuclei")
    series_list.extract_particles(particle.Nucleus, threads=args.threads)
    logging.info(f"extracted {len(list(series_list.particles()))} nuclei")

    logging.info(f"generating radial profiles")
    rdc = distance.RadialDistanceCalculator(
        args.axes, args.center_type, args.quantile)
    profiles = series_list.get_radial_profiles(rdc, args.bins, args.degree)

    mk_report(args, profiles, series_list)
