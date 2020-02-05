'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse as argp
import ggc
import joblib
import itertools
import logging as log
import numpy as np
import os
import pandas as pd
from radiantkit.const import __version__, default_inreg
from radiantkit.image import ImageBinary, ImageLabeled
from radiantkit.particle import NucleiList, Nucleus
from radiantkit.path import get_image_details
from radiantkit.series import Series, SeriesList
from radiantkit.report import report_select_nuclei
import re
import sys
from tqdm import tqdm
from typing import Dict, List, Pattern

log.basicConfig(level=log.INFO, format='%(asctime)s ' +
    '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

def init_parser(subparsers: argp._SubParsersAction
    ) -> argp.ArgumentParser:
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
''', formatter_class = argp.RawDescriptionHelpFormatter,
        help = "Select G1 nuclei.")

    parser.add_argument('input', type=str,
        help='Path to folder containing deconvolved tiff images and masks.')
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
        version='%s %s' % (sys.argv[0], __version__,))

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument('--block-side', type=int, metavar="NUMBER",
        help="""Structural element side for dilation-based background/foreground
        measurement. Should be odd. Default: 11.""", default=11)
    advanced.add_argument('--use-labels',
        action='store_const', dest='labeled',
        const=True, default=False,
        help='Use labels from masks instead of relabeling.')
    advanced.add_argument('--no-rescaling',
        action='store_const', dest='do_rescaling',
        const=False, default=True,
        help='Do not rescale image even if deconvolved.')
    advanced.add_argument('--no-remove',
        action='store_const', dest='remove_labels',
        const=False, default=True,
        help='Do not remove labels of discarded nuclei.')
    advanced.add_argument('--uncompressed',
        action='store_const', dest='compressed',
        const=False, default=True,
        help='Generate uncompressed TIFF binary masks.')
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

def parse_arguments(args: argp.Namespace) -> argp.Namespace:
    args.version = __version__

    assert '(?P<channel_name>' in args.inreg
    assert '(?P<series_id>' in args.inreg
    args.inreg = re.compile(args.inreg)

    if 0 != len(args.mask_prefix):
        if '.' != args.mask_prefix[-1]:
            args.mask_prefix = f"{args.mask_prefix}."
    if 0 != len(args.mask_suffix):
        if '.' != args.mask_suffix[0]:
            args.mask_suffix = f".{args.mask_suffix}"

    if not 0 != args.block_side%2:
        log.warning("changed ground block side from " +
            f"{args.block_side} to {args.block_side+1}")
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
    if clear: print("\033[H\033[J")
    print(s)
    return(s)

def confirm_arguments(args: argp.Namespace) -> None:
    settings_string = print_settings(args)
    if not args.do_all: ggc.prompt.ask("Confirm settings and proceed?")

    assert os.path.isdir(args.input
        ), f"image folder not found: {args.input}"

    with open(os.path.join(args.input, "select_nuclei.config.txt"), "w+") as OH:
        ggc.args.export_settings(OH, settings_string)

def extract_passing_nuclei_per_series(
    ndata: pd.DataFrame, inreg: Pattern) -> Dict[int,List[int]]:
    passed = ndata.loc[ndata['pass'], ['image', 'label']]
    passed['series_id'] = [get_image_details(p, inreg)[0]
        for p in passed['image'].values]
    passed.drop('image', 1, inplace=True)
    passed = dict([
        (sid, passed.loc[passed['series_id']==sid, 'label'].values)
        for sid in set(passed['series_id'].values)])
    return passed

def remove_labels_from_series_mask(series: Series, labels: List[int],
    labeled: bool, compressed: bool) -> None:
    os.rename(series.mask.path, f"{series.mask.path}.old")
    if labeled:
        L = series.mask.pixels
        L[np.logical_not(np.isin(L, labels))] = 0
        L = ImageLabeled(L)
        L.to_tiff(series.mask.path, compressed)
    else:
        L = series.mask.label().pixels
        L[np.logical_not(np.isin(L, labels))] = 0
        M = ImageBinary(L)
        M.to_tiff(series.mask.path, compressed)

def run(args: argp.Namespace) -> None:
    confirm_arguments(args)

    series_list = SeriesList.from_directory(args.input, args.inreg,
        args.dna_channel, (args.mask_prefix, args.mask_suffix))
    log.info(f"parsed {len(series_list)} series with " +
        f"{len(series_list.channel_names)} channels each" +
        f": {series_list.channel_names}")
    for series in series_list: series.labeled = args.labeled
    for series in series_list: series.ground_bloc_side = args.block_side

    log.info(f"extracting nuclei")
    if 1 == args.threads:
        series_list = [Series.extract_particles(s, [args.dna_channel], Nucleus)
            for s in tqdm(series_list)]
    else:
        series_list = joblib.Parallel(n_jobs=args.threads, verbose=11)(
            joblib.delayed(Series.extract_particles
                )(s, [args.dna_channel], Nucleus) for s in series_list)

    nuclei = NucleiList(list(itertools.chain(
        *[s.particles for s in series_list])))
    log.info(f"extracted {len(nuclei)} nuclei.")

    nuclei_data, details = nuclei.select_G1(args.k_sigma, args.dna_channel)

    passed = extract_passing_nuclei_per_series(nuclei_data, args.inreg)
    if args.remove_labels:
        log.info("removing discarded nuclei labeles from masks")
        if 1 == args.threads:
            for series in tqdm(series_list):
                remove_labels_from_series_mask(series, passed[series.ID],
                    args.labeled, args.compressed)
        else:
            joblib.Parallel(n_jobs=args.threads, verbose=11)(
                joblib.delayed(remove_labels_from_series_mask
                    )(series, passed[series.ID], args.labeled, args.compressed)
                    for series in series_list)
        n_removed = len(nuclei)-len(list(itertools.chain(*passed.values())))
        log.info(f"removed {n_removed} nuclei labels")

    np.set_printoptions(formatter={'float_kind':'{:.2E}'.format})
    log.info(f"size fit:\n{details['size']['fit']}")
    np.set_printoptions(formatter={'float_kind':'{:.2E}'.format})
    log.info(f"size range: {details['size']['range']}")
    np.set_printoptions(formatter={'float_kind':'{:.2E}'.format})
    log.info(f"intensity sum fit:\n{details['isum']['fit']}")
    np.set_printoptions(formatter={'float_kind':'{:.2E}'.format})
    log.info(f"intensity sum range: {details['isum']['range']}")

    ndpath = os.path.join(args.input, "select_nuclei.data.tsv")
    log.info(f"writing nuclear data to:\n{ndpath}")
    nuclei_data.to_csv(ndpath, sep="\t", index=False)

    report_path = os.path.join(args.input, "select_nuclei.report.html")
    log.info(f"writing report to\n{report_path}")
    report_select_nuclei(args, report_path, data=nuclei_data,
        details=details, series_list=series_list, ref=args.dna_channel)
