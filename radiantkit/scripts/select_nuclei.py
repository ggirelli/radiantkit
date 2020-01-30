'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
from ggc.prompt import ask
from ggc.args import check_threads, export_settings
from joblib import delayed, Parallel
import logging
import os
from radiantkit.const import __version__
from radiantkit import image, particle
import re
import sys
from tqdm import tqdm
from typing import List, Type

logging.basicConfig(level=logging.INFO, format='%(asctime)s ' +
    '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

def parse_arguments() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description='''
...
    ''', formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input', type=str,
        help='Path to folder containing deconvolved tiff images.')

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

    args = parser.parse_args()
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

def retrieve_nuclei(imgdir: str, maskpath: str, rawpath: str,
    loglevel: str="INFO") -> List[Type[particle.ParticleBase]]:
    logging.getLogger().setLevel(loglevel)
    I = image.Image.from_tiff(os.path.join(imgdir, rawpath))
    M = image.ImageBinary.from_tiff(os.path.join(imgdir, maskpath))
    assert I.shape == M.shape
    nuclei = particle.ParticleFinder(
        ).get_particles_from_binary_image(M, particle.Nucleus)

    # cycle through nuclei and calculate features passing I

    return nuclei

def run(args: argparse.Namespace) -> None:
    imglist = [f for f in os.listdir(args.input) 
        if os.path.isfile(os.path.join(args.input, f))
        and not type(None) == type(re.match(args.inreg, f))]
    
    if 0 != len(args.outsuffix): imglist = [f for f in imglist
        if os.path.splitext(f)[0].endswith(args.outsuffix)]
    if 0 != len(args.outprefix): imglist = [f for f in imglist
        if os.path.splitext(f)[0].startswith(args.outprefix)]

    n_masks = len(imglist)
    for imgpath in imglist:
        imgbase, imgext = os.path.splitext(imgpath)
        imgbase = imgbase[len(args.outprefix):-len(args.outsuffix)]
        raw_image = f"{imgbase}{imgext}"
        if not os.path.isfile(os.path.join(args.input, raw_image)):
            logging.warning(f"missing raw image for mask '{imgpath}', skipped.")
            imglist.pop(imglist.index(imgpath))
        else:
            imglist[imglist.index(imgpath)] = (raw_image,imgpath)

    logging.info(f"working on {len(imglist)}/{n_masks} images.")

    if 1 == args.threads:
        nuclei = []
        for rawpath,maskpath in tqdm(imglist):
            nuclei.extend(retrieve_nuclei(args.input, rawpath, maskpath))
    else:
        nuclei = Parallel(n_jobs = args.threads, verbose = 11)(
            delayed(retrieve_nuclei)(args.input, rawpath, maskpath)
            for rawpath,maskpath in imglist)
    logging.info(f"extracted {len(nuclei)} nuclei.")

def main():
    args = parse_arguments()
    confirm_arguments(args)
    run(args)
