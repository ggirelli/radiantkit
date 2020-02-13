'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse as argp
import ggc  # type: ignore
import joblib  # type: ignore
import itertools
import logging as log
import numpy as np  # type: ignore
import os
import pandas as pd  # type: ignore
from radiantkit import const
from radiantkit.image import ImageBinary, ImageLabeled
from radiantkit.particle import NucleiList, Nucleus
from radiantkit.series import Series, SeriesList
from radiantkit import path, report, string
import re
import sys
from tqdm import tqdm  # type: ignore
from typing import Dict, List, Pattern

log.basicConfig(
    level=log.INFO, format='%(asctime)s '
    + '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')


def init_parser(subparsers: argp._SubParsersAction) -> argp.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1], description='''
Select nuclei (objects) from segmented images based on their size (volume in
3D, area in 2D) and integral of intensity from raw image.

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
''', formatter_class=argp.RawDescriptionHelpFormatter,
        help="Select G1 nuclei.")

    parser.add_argument(
        'input', type=str,
        help='Path to folder containing deconvolved tiff images and masks.')
    parser.add_argument(
        'dna_channel', type=str,
        help='Name of channel with DNA staining intensity.')

    parser.add_argument(
        '--k-sigma', type=float, metavar="NUMBER",
        help="""Suffix for output binarized images name.
        Default: 2.5""", default=2.5)
    parser.add_argument(
        '--mask-prefix', type=str, metavar="TEXT",
        help="""Prefix for output binarized images name.
        Default: ''.""", default='')
    parser.add_argument(
        '--mask-suffix', type=str, metavar="TEXT",
        help="""Suffix for output binarized images name.
        Default: 'mask'.""", default='mask')

    parser.add_argument('--version', action='version',
                        version='%s %s' % (sys.argv[0], const.__version__,))

    report = parser.add_argument_group("report arguments")
    report.add_argument(
        '--no-report', action='store_const',
        help="""Do not generate an HTML report.""",
        dest="mk_report", const=False, default=True)
    report.add_argument(
        '--online-report', action='store_const',
        help="""Make a smaller HTML report by linking remote JS libraries.""",
        dest="online_report", const=True, default=False)

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument(
        '--block-side', type=int, metavar="NUMBER",
        help="""Structural element side for dilation-based background/foreground
        measurement. Should be odd. Default: 11.""", default=11)
    advanced.add_argument(
        '--use-labels', action='store_const', dest='labeled',
        const=True, default=False,
        help='Use labels from masks instead of relabeling.')
    advanced.add_argument(
        '--no-rescaling', action='store_const', dest='do_rescaling',
        const=False, default=True,
        help='Do not rescale image even if deconvolved.')
    advanced.add_argument(
        '--no-remove', action='store_const', dest='remove_labels',
        const=False, default=True,
        help='Do not regenerate masks after removing discarded nuclei labels.')
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


def parse_arguments(args: argp.Namespace) -> argp.Namespace:
    args.version = const.__version__

    assert '(?P<channel_name>' in args.inreg
    assert '(?P<series_id>' in args.inreg
    args.inreg = re.compile(args.inreg)

    args.mask_prefix = string.add_leading_dot(args.mask_prefix)
    args.mask_suffix = string.add_trailing_dot(args.mask_suffix)

    if not 0 != args.block_side % 2:
        log.warning("changed ground block side from "
                    + f"{args.block_side} to {args.block_side+1}")
        args.block_side += 1

    args.threads = ggc.args.check_threads(args.threads)

    return args


def print_settings(args: argp.Namespace, clear: bool = True) -> str:
    s = f"""# Nuclei selection v{args.version}

    ---------- SETTING : VALUE ----------

       Input directory : '{args.input}'
      DNA channel name : '{args.dna_channel}'
               K sigma : {args.k_sigma}

           Mask prefix : '{args.mask_prefix}'
           Mask suffix : '{args.mask_suffix}'

     Ground block side : {args.block_side}
            Use labels : {args.labeled}
               Rescale : {args.do_rescaling}
         Remove labels : {args.remove_labels}
            Compressed : {args.compressed}

               Threads : {args.threads}
                Regexp : {args.inreg.pattern}
    """
    if clear:
        print("\033[H\033[J")
    print(s)
    return(s)


def confirm_arguments(args: argp.Namespace) -> None:
    settings_string = print_settings(args)
    if not args.do_all:
        ggc.prompt.ask("Confirm settings and proceed?")

    assert os.path.isdir(args.input), (
        f"image folder not found: {args.input}")

    with open(os.path.join(
            args.input, "select_nuclei.config.txt"), "w+") as OH:
        ggc.args.export_settings(OH, settings_string)


def extract_passing_nuclei_per_series(
        ndata: pd.DataFrame, inreg: Pattern) -> Dict[int, List[int]]:
    passed = ndata.loc[ndata['pass'], ['image', 'label']]
    passed['series_id'] = []
    for p in passed['image'].values:
        image_details = path.get_image_details(p, inreg)
        assert image_details is not None
        passed['series_id'].append(image_details[0])
    passed.drop('image', 1, inplace=True)
    passed = dict([
        (sid, passed.loc[passed['series_id'] == sid, 'label'].values)
        for sid in set(passed['series_id'].values)])
    return passed


def remove_labels_from_series_mask(
        series: Series, labels: List[int],
        labeled: bool, compressed: bool) -> None:
    if series.mask is None:
        return None
    series.mask.load_from_local()
    os.rename(series.mask.path, f"{series.mask.path}.old")
    if labeled:
        L = series.mask.pixels
        L[np.logical_not(np.isin(L, labels))] = 0
        L = ImageLabeled(L)
        L.to_tiff(series.mask.path, compressed)
    else:
        if isinstance(series.mask, ImageBinary):
            L = series.mask.label().pixels
        else:
            L = series.mask.pixels
        L[np.logical_not(np.isin(L, labels))] = 0
        M = ImageBinary(L)
        M.to_tiff(series.mask.path, compressed)


def run(args: argp.Namespace) -> None:
    confirm_arguments(args)

    series_list = SeriesList.from_directory(
        args.input, args.inreg, args.dna_channel,
        (args.mask_prefix, args.mask_suffix),
        None, args.labeled, args.block_side)
    log.info(f"parsed {len(series_list)} series with "
             + f"{len(series_list.channel_names)} channels each"
             + f": {series_list.channel_names}")

    log.info(f"extracting nuclei")
    series_list.extract_particles(Nucleus, [args.dna_channel], args.threads)

    nuclei = NucleiList(list(itertools.chain(
        *[s.particles for s in series_list])))
    log.info(f"extracted {len(nuclei)} nuclei.")

    nuclei_data, details = nuclei.select_G1(args.k_sigma, args.dna_channel)

    passed = extract_passing_nuclei_per_series(nuclei_data, args.inreg)
    if args.remove_labels:
        log.info("removing discarded nuclei labeles from masks")
        if 1 == args.threads:
            for series in tqdm(series_list):
                remove_labels_from_series_mask(
                    series, passed[series.ID], args.labeled, args.compressed)
        else:
            joblib.Parallel(n_jobs=args.threads, verbose=11)(
                joblib.delayed(remove_labels_from_series_mask)(
                    series, passed[series.ID], args.labeled, args.compressed)
                for series in series_list)
        n_removed = len(nuclei)-len(list(itertools.chain(*passed.values())))
        log.info(f"removed {n_removed} nuclei labels")

    np.set_printoptions(formatter={'float_kind': '{:.2E}'.format})
    log.info(f"size fit:\n{details['size']['fit']}")
    np.set_printoptions(formatter={'float_kind': '{:.2E}'.format})
    log.info(f"size range: {details['size']['range']}")
    np.set_printoptions(formatter={'float_kind': '{:.2E}'.format})
    log.info(f"intensity sum fit:\n{details['isum']['fit']}")
    np.set_printoptions(formatter={'float_kind': '{:.2E}'.format})
    log.info(f"intensity sum range: {details['isum']['range']}")

    ndpath = os.path.join(args.input, "select_nuclei.data.tsv")
    log.info(f"writing nuclear data to:\n{ndpath}")
    nuclei_data.to_csv(ndpath, sep="\t", index=False)

    if args.mk_report:
        report_path = os.path.join(args.input, "select_nuclei.report.html")
        log.info(f"writing report to\n{report_path}")
        report.report_select_nuclei(
            args, report_path, args.online_report,
            data=nuclei_data, details=details, series_list=series_list)
