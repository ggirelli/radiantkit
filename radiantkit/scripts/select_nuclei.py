'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
from distutils.util import convert_path
from ggc.prompt import ask
from ggc.args import check_threads, export_settings
import itertools
from joblib import delayed, Parallel
import logging
import numpy as np
import os
import pandas as pd
from radiantkit.const import __version__
from radiantkit import image, particle
from radiantkit import stat
from radiantkit.report import report_select_nuclei
import re
import sys
import tempfile
from tqdm import tqdm
from typing import List, Pattern, Type

logging.basicConfig(level=logging.INFO, format='%(asctime)s ' +
    '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

def init_parser(subparsers: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(__name__.split(".")[-1], description = '''
Lorem ipsum dolor sit amet, consectetur adipisicing elit. Commodi blanditiis
totam delectus provident non ipsa maxime reprehenderit soluta assumenda
accusantium. Iure eaque suscipit voluptatibus expedita adipisci, doloremque ab,
ea magnam.''', formatter_class = argparse.RawDescriptionHelpFormatter,
        help = f"{__name__.split('.')[-1]} -h")

    parser.add_argument('input', type=str,
        help='Path to folder containing deconvolved tiff images.')

    parser.add_argument('--k-sigma', type=float, metavar="NUMBER",
        help="""Suffix for output binarized images name.
        Default: 2.5""", default=2.5)
    parser.add_argument('--outprefix', type=str, metavar="TEXT",
        help="""Prefix for output binarized images name.
        Default: ''.""", default='')
    parser.add_argument('--outsuffix', type=str, metavar="TEXT",
        help="""Suffix for output binarized images name.
        Default: 'mask'.""", default='mask')

    parser.add_argument('--uncompressed',
        action='store_const', dest='compressed',
        const=False, default=True,
        help='Generate uncompressed TIFF binary masks.')

    default_inreg='^.*\.tiff?$'
    parser.add_argument('--inreg', type=str, metavar="REGEXP",
        help="""Regular expression to identify input TIFF images.
        Default: '%s'""" % (default_inreg,), default=default_inreg)
    parser.add_argument('-t', type=int, metavar="NUMBER", dest="threads",
        help="""Number of threads for parallelization. Default: 1""",
        default=1)
    parser.add_argument('-y', '--do-all', action='store_const',
        help="""Do not ask for settings confirmation and proceed.""",
        const=True, default=False)

    parser.add_argument('--version', action='version',
        version='%s %s' % (sys.argv[0], __version__,))

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser

def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    args.version = __version__

    args.inreg = re.compile(args.inreg)
    if 0 != len(args.outprefix):
        if '.' != args.outprefix[-1]:
            args.outprefix = f"{args.outprefix}."
    if 0 != len(args.outsuffix):
        if '.' != args.outsuffix[0]:
            args.outsuffix = f".{args.outsuffix}"

    args.threads = check_threads(args.threads)

    return args

def print_settings(args: argparse.Namespace, clear: bool = True) -> str:
    s = f"""
# Nuclei selection v{args.version}

---------- SETTING :  VALUE ----------

   Input directory :  '{args.input}'

       Mask prefix :  '{args.outprefix}'
       Mask suffix :  '{args.outsuffix}'
        Compressed :  {args.compressed}

           Threads :  {args.threads}
            Regexp :  {args.inreg.pattern}
    """
    if clear: print("\033[H\033[J")
    print(s)
    return(s)

def confirm_arguments(args: argparse.Namespace) -> None:
    settings_string = print_settings(args)
    if not args.do_all: ask("Confirm settings and proceed?")

    assert os.path.isdir(args.input
        ), f"image folder not found: {args.input}"

    with open(os.path.join(args.input, "select_nuclei.config.txt"), "w+") as OH:
        export_settings(OH, settings_string)

def find_images(ipath: str, inreg: Pattern) -> List[str]:
    imglist = [f for f in os.listdir(ipath) 
        if os.path.isfile(os.path.join(ipath, f))
        and not type(None) == type(re.match(inreg, f))]
    return imglist

def select_masks(ipath: str, imglist: List[str],
    prefix: str="", suffix: str= "") -> List[str]:
    if 0 != len(suffix): imglist = [f for f in imglist
        if os.path.splitext(f)[0].endswith(suffix)]
    if 0 != len(prefix): imglist = [f for f in imglist
        if os.path.splitext(f)[0].startswith(prefix)]

    for imgpath in imglist:
        imgbase, imgext = os.path.splitext(imgpath)
        imgbase = imgbase[len(prefix):-len(suffix)]
        raw_image = f"{imgbase}{imgext}"
        if not os.path.isfile(os.path.join(ipath, raw_image)):
            logging.warning(f"missing raw image for mask '{imgpath}', skipped.")
            imglist.pop(imglist.index(imgpath))
        else:
            imglist[imglist.index(imgpath)] = (raw_image,imgpath)

    return imglist

def run(args: argparse.Namespace) -> None:
    confirm_arguments(args)
    
    imglist = find_images(args.input, args.inreg)
    masklist = select_masks(args.input, imglist, args.outprefix, args.outsuffix)
    logging.info(f"working on {len(masklist)}/{len(imglist)} images.")
    assert 0 != len(masklist)

    nuclei = extract_nuclei_from_masks(masklist, args.input, args.threads)
    logging.info(f"extracted {len(nuclei)} nuclei.")

    size_data = np.array([n.total_size for n in nuclei])
    size_fit = stat.cell_cycle_fit(size_data)
    assert size_fit[0] is not None
    np.set_printoptions(formatter={'float_kind':'{:.2E}'.format})
    logging.info(f"size fit:\n{size_fit}")
    size_range = stat.range_from_fit(
        size_data, *size_fit, args.k_sigma)
    np.set_printoptions(formatter={'float_kind':'{:.2E}'.format})
    logging.info(f"size range: {size_range}")

    intensity_sum_data = np.array([n.intensity_sum for n in nuclei])
    intensity_sum_fit = stat.cell_cycle_fit(intensity_sum_data)
    assert intensity_sum_fit[0] is not None
    np.set_printoptions(formatter={'float_kind':'{:.2E}'.format})
    logging.info(f"intensity sum fit:\n{intensity_sum_fit}")
    intensity_sum_range = stat.range_from_fit(
        intensity_sum_data, *intensity_sum_fit, args.k_sigma)
    np.set_printoptions(formatter={'float_kind':'{:.2E}'.format})
    logging.info(f"size range: {intensity_sum_range}")

    nuclei_data = pd.DataFrame.from_dict({
        'image':[n.ipath for n in nuclei],
        'label':[n.label for n in nuclei],
        'size':size_data,
        'isum':intensity_sum_data
    })

    nuclei_data['pass_size'] = np.logical_and(
        size_data >= size_range[0],
        size_data <= size_range[1])
    nuclei_data['pass_isum'] = np.logical_and(
        intensity_sum_data >= intensity_sum_range[0],
        intensity_sum_data <= intensity_sum_range[1])
    nuclei_data['pass'] = np.logical_and(
        nuclei_data['pass_size'], nuclei_data['pass_isum'])

    ndpath = os.path.join(args.input, "select_nuclei.data.tsv")
    logging.info(f"writing nuclear data to:\n{ndpath}")
    nuclei_data.to_csv(ndpath, sep="\t", index=False)

    report_path = os.path.join(args.input, "select_nuclei.report.html")
    logging.info(f"writing report to\n{report_path}")
    report_select_nuclei(args, report_path, data=nuclei_data,
        size_range=size_range, intensity_sum_range=intensity_sum_range,
        masklist=sorted(masklist))
