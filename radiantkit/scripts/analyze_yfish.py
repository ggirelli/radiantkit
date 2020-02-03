'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
from ggc.prompt import ask
from ggc.args import check_threads, export_settings
import itertools
from joblib import delayed, Parallel
import logging
import os
from radiantkit import const, path
from radiantkit import particle, series
import re
import sys
from tqdm import tqdm
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s ' +
    '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

def init_parser(subparsers: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(__name__.split('.')[-1],
        description = f'''Analyze a multi-condition YFISH experiment.''',
        formatter_class = argparse.RawDescriptionHelpFormatter,
        help="Analyze a multi-condition YFISH experiment.")

    parser.add_argument('input', type = str,
        help = '''Path to root folder (see description for details).''')
    parser.add_argument('output', type = str,
        help = '''Path to output folder.''')

    parser.add_argument('--version', action = 'version',
        version = f'{sys.argv[0]} {const.__version__}')

    critical = parser.add_argument_group("Critical arguments")
    critical.add_argument('--aspect', type=float, nargs=3, help="""Physical size
    of Z, Y and X voxel sides in nm. Default: 300.0 216.6 216.6""",
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
    critical.add_argument('--method', type=str,
        help=f"""Analysis method. One of:
        Default: '{const.AnalysisType.get_default().value}'""",
        default=const.AnalysisType.get_default().value,
        choices=[e.value for e in const.AnalysisType])
    critical.add_argument('--midsection', type=str,
        help=f"""Midsection type. One of:
        Default: '{const.MidsectionType.get_default().value}'""",
        default=const.MidsectionType.get_default().value,
        choices=[e.value for e in const.MidsectionType])
    critical.add_argument('--distance', type=str,
        help=f"""Radial distance type. One of:
        Default: '{const.LaminaDistanceType.get_default().value}'""",
        default=const.LaminaDistanceType.get_default().value,
        choices=[e.value for e in const.LaminaDistanceType])
    critical.add_argument('--dist-quant',
        type=float, default=None, metavar="NUMBER",
        help=f"""Quantile (fraction) for CENTER_TOP_QUANTILE distance type.
        Defaults to 0.01 for 2D, and 0.001 for 3D analysis types.""")

    nuclear_selection = parser.add_argument_group("Nuclei selection")
    nuclear_selection.add_argument('--k-sigma', type=float, metavar="NUMBER",
        help="""Suffix for output binarized images name.
        Default: 2.5""", default=2.5)
    nuclear_selection.add_argument('--use-all-nuclei', action='store_const',
        dest='skip_nuclear_selection', const=True, default=False,
        help='Skip selection of G1 nuclei.')

    minor = parser.add_argument_group("Minor arguments")
    minor.add_argument('--description', type=str, nargs='*', metavar="STRING",
        help = """Space separated 'condition:description' couples.
        'condition' is the name of a condition folder. 'description' is a
        descriptive label that replaces the corresponding folder name in the
        final report. Use '--' after the last one.""")
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
    default_inreg ="^([^\\.]*\\.)?(?P<channel_name>[^/]*)_(?P<series_id>[0-9]+)"
    default_inreg+="(?P<ext>(_cmle)?(\\.[^\\.]*)?\\.tiff?)$"
    advanced.add_argument('--inreg', type=str, metavar="REGEXP",
        help=f"""Regular expression to identify input TIFF images.
        Must contain 'channel_name' and 'series_id' fields.
        Default: '{default_inreg}'""", default=default_inreg)
    advanced.add_argument('-t', type=int, metavar="NUMBER", dest="threads",
        help="""Number of threads for parallelization. Default: 1""",
        default=1)
    advanced.add_argument('-y', '--do-all', action='store_const',
        help="""Do not ask for settings confirmation and proceed.""",
        const=True, default=False)
    
    parser.set_defaults(parse=parse_arguments, run=run)

    return parser

def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    args.version = const.__version__

    assert not os.path.isfile(args.output)
    assert all([v >= 0 for v in args.aspect])
    assert args.k_sigma >= 0

    if args.dist_quant is None:
        if args.method is const.AnalysisType.THREED:
            args.dist_quant = 1e-3
        else:
            args.dist_quant = 1e-2
    assert args.dist_quant >= 0 and args.dist_quant <= 1

    assert '(?P<channel_name>' in args.inreg
    assert '(?P<series_id>' in args.inreg
    args.inreg = re.compile(args.inreg)

    if 0 != len(args.mask_prefix):
        if '.' != args.mask_prefix[-1]:
            args.mask_prefix = f"{args.mask_prefix}."
    if 0 != len(args.mask_suffix):
        if '.' != args.mask_suffix[0]:
            args.mask_suffix = f".{args.mask_suffix}"

    args.threads = check_threads(args.threads)

    if args.debug_mode: logging.getLogger().level = logging.DEBUG

    if args.description is not None:
        assert all([1 == s.count(":") for s in args.description])
        args.description = dict([s.split(":") for s in args.description])
        args.readable_description = ("\n"+" "*21).join([f"{c} => {v}"
            for (c,v) in args.description.items()])
    else:
        args.readable_description = "*NONE*"

    return args

def print_settings(args: argparse.Namespace, clear: bool = True) -> str:

    s = f"""
    # YFISH analysis v{args.version}
    
    ---------- SETTING : VALUE ----------
    
       Input directory : '{args.input}'
      Output directory : '{args.output}'
    
    Voxel aspect (ZYX) : {args.aspect}
     Reference channel : {args.ref}
    
           Mask prefix : '{args.mask_prefix}'
           Mask suffix : '{args.mask_suffix}'
    
                Method : {args.method}
            Midsection : {args.midsection}
              Distance : {args.distance}
              Quantile : {args.dist_quant}
    
               K sigma : {args.k_sigma}
         Select nuclei : {not args.skip_nuclear_selection}
    
                  Note : {args.note}
           Description : {args.readable_description}
    
            Use labels : {args.labeled}
               Rescale : {args.do_rescaling}
                 Debug : {args.debug_mode}
                Regexp : {args.inreg.pattern}
               Threads : {args.threads}
        """
    if clear: print("\033[H\033[J")
    print(s)
    return(s)

def confirm_arguments(args: argparse.Namespace) -> None:
    settings_string = print_settings(args)
    if not args.do_all: ask("Confirm settings and proceed?")

    assert os.path.isdir(args.input), f"input folder not found: {args.input}"
    if not os.path.isdir(args.output): os.mkdir(args.output)

    with open(os.path.join(args.output,
        "analyze_yfish.config.txt"), "w+") as OH:
        export_settings(OH, settings_string)

def build_conditions(args: argparse.Namespace) -> Dict:
    conditions = {}
    for condition_name in os.listdir(args.input):
        condition_folder = os.path.join(args.input, condition_name)
        conditions[condition_name] = dict(
            series=series.SeriesList.from_directory(condition_folder,
                args.inreg, args.ref, (args.mask_prefix, args.mask_suffix)))
        logging.info(f"parsed {len(conditions[condition_name]['series'])} " +
            f"series from condition '{condition_name}'")

    if args.description is not None:
        for condition_name in conditions:
            if condition_name in args.description:
                conditions[condition_name]['label'
                    ] = args.description[condition_name]
            else: conditions[condition_name]['label'] = condition_name
    else:
        for condition_name in conditions:
            conditions[condition_name]['label'] = condition_name

    return conditions
    
def run_series(series: series.Series) -> series.Series:
    series.extract_particles(particle.Nucleus)
    logging.info(f"Extracted {len(series.particles)} nuclei " +
        f"from series '{series.ID}'")
    return series

def run(args: argparse.Namespace) -> None:
    confirm_arguments(args)

    for name,condition in build_conditions(args).items():
        if 1 == args.threads:
            condition['series'] = [run_series(series)
                for series in tqdm(condition['series'])]
        else:
            condition['series'] = Parallel(n_jobs=args.threads, verbose=11)(
                delayed(run_series)(series) for series in condition['series'])

        ndata, details = particle.NucleiList(list(itertools.chain(
            *[s.particles for s in condition['series']]))
            ).select_G1(args.k_sigma)

        print(ndata)
        print(details)
