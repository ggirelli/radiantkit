'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from argparse import _SubParsersAction, RawDescriptionHelpFormatter
from argparse import ArgumentParser, Namespace
from ggc.prompt import ask
from ggc.args import check_threads, export_settings
from joblib import delayed, Parallel
from itertools import chain
import logging
from numpy import set_printoptions
from os.path import isdir, join as path_join
from radiantkit.const import __version__, default_inreg
from radiantkit.particle import NucleiList
from radiantkit.series import Series, SeriesList
from radiantkit.report import report_select_nuclei
from re import compile as re_compile
from sys import argv as sys_argv
from tqdm import tqdm
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s ' +
    '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

def init_parser(subparsers: _SubParsersAction
    ) -> ArgumentParser:
    parser = subparsers.add_parser(__name__.split(".")[-1], description = '''
Select nuclei (objects) from segmented images based on their size (volume in 3D,
area in 2D) and integral of intensity from raw image.

To achieve this, the script looks for mask/raw image pairs in the input folder.
Mask images are identified by the specified prefix/suffix. For example, a pair
with suffix "mask" would be:
    [RAW] "dapi_001.tiff" and [MASK] "dapi_001.mask.tiff".

Nuclei are extracted and size and integral of intensity are calculated. Then,
their density profile is calculated across all images. A sum of Gaussian is fit
to the profiles and a range of +-k_sigma around the peak of the first Gaussian
is selected. If the fit fails, a single Gaussian is fitted and the range is 
selected in the same manner around its peak. If this fit fails, the selected
range corresponds to the FWHM range around the first peak of the profiles. In
the last scenario, k_sigma is ignored.

A tabulation-separated table is generated with the nuclear features and whether
they pass the filter(s). Alongside it, an html report is generated with
interactive data visualization.
''', formatter_class = RawDescriptionHelpFormatter,
        help = "Select G1 nuclei.")

    parser.add_argument('input', type=str,
        help='Path to folder containing deconvolved tiff images.')
    parser.add_argument('dna_channel', type=str,
        help='Name of channel with DNA staining intensity.')

    parser.add_argument('--k-sigma', type=float, metavar="NUMBER",
        help="""Suffix for output binarized images name.
        Default: 2.5""", default=2.5)
    parser.add_argument('--mask-prefix', type=str, metavar="TEXT",
        help="""Prefix for output binarized images name.
        Default: ''.""", default='')
    parser.add_argument('--mask-suffix', type=str, metavar="TEXT",
        help="""Suffix for output binarized images name.
        Default: 'mask'.""", default='mask')

    parser.add_argument('--version', action='version',
        version='%s %s' % (sys_argv[0], __version__,))

    advanced = parser.add_argument_group("Advanced")
    advanced.add_argument('--uncompressed',
        action='store_const', dest='compressed',
        const=False, default=True,
        help='Generate uncompressed TIFF binary masks.')
    advanced.add_argument('--no-rescaling',
        action='store_const', dest='do_rescaling',
        const=False, default=True,
        help='Do not rescale image even if deconvolved.')
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

def parse_arguments(args: Namespace) -> Namespace:
    args.version = __version__

    assert '(?P<channel_name>' in args.inreg
    assert '(?P<series_id>' in args.inreg
    args.inreg = re_compile(args.inreg)

    if 0 != len(args.mask_prefix):
        if '.' != args.mask_prefix[-1]:
            args.mask_prefix = f"{args.mask_prefix}."
    if 0 != len(args.mask_suffix):
        if '.' != args.mask_suffix[0]:
            args.mask_suffix = f".{args.mask_suffix}"

    args.threads = check_threads(args.threads)

    return args

def print_settings(args: Namespace, clear: bool = True) -> str:
    s = f"""# Nuclei selection v{args.version}

---------- SETTING : VALUE ----------

   Input directory : '{args.input}'
  DNA channel name : '{args.dna_channel}'

       Mask prefix : '{args.mask_prefix}'
       Mask suffix : '{args.mask_suffix}'
        Compressed : {args.compressed}

           Rescale : {args.do_rescaling}
           Threads : {args.threads}
            Regexp : {args.inreg.pattern}
    """
    if clear: print("\033[H\033[J")
    print(s)
    return(s)

def confirm_arguments(args: Namespace) -> None:
    settings_string = print_settings(args)
    if not args.do_all: ask("Confirm settings and proceed?")

    assert isdir(args.input
        ), f"image folder not found: {args.input}"

    with open(path_join(args.input, "select_nuclei.config.txt"), "w+") as OH:
        export_settings(OH, settings_string)

def run(args: Namespace) -> None:
    confirm_arguments(args)

    series_list = SeriesList.from_directory(args.input, args.inreg,
        args.dna_channel, (args.mask_prefix, args.mask_suffix))
    logging.info(f"parsed {len(series_list)} series with " +
        f"{len(series_list.channel_names)} channels each" +
        f": {series_list.channel_names}")

    if 1 == args.threads:
        series_list = [Series.static_extract_particles(s, args.dna_channel)
            for s in tqdm(series_list)]
    else:
        series_list = Parallel(n_jobs=args.threads, verbose=11)(
            delayed(Series.static_extract_particles)(s, args.dna_channel)
            for s in series_list)

    nuclei = NucleiList(list(chain(*[s.particles for s in series_list])))
    logging.info(f"extracted {len(nuclei)} nuclei.")

    nuclei_data, details = nuclei.select_G1(args.k_sigma, args.dna_channel)

    set_printoptions(formatter={'float_kind':'{:.2E}'.format})
    logging.info(f"size fit:\n{details['size']['fit']}")
    set_printoptions(formatter={'float_kind':'{:.2E}'.format})
    logging.info(f"size range: {details['size']['range']}")
    set_printoptions(formatter={'float_kind':'{:.2E}'.format})
    logging.info(f"intensity sum fit:\n{details['isum']['fit']}")
    set_printoptions(formatter={'float_kind':'{:.2E}'.format})
    logging.info(f"intensity sum range: {details['isum']['range']}")

    ndpath = path_join(args.input, "select_nuclei.data.tsv")
    logging.info(f"writing nuclear data to:\n{ndpath}")
    nuclei_data.to_csv(ndpath, sep="\t", index=False)

    report_path = path_join(args.input, "select_nuclei.report.html")
    logging.info(f"writing report to\n{report_path}")
    report_select_nuclei(args, report_path, data=nuclei_data,
        size_range=details['size']['range'],
        intensity_sum_range=details['isum']['range'],
        series_list=series_list, ref=args.dna_channel)
