'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
from ggc.prompt import ask
from ggc.args import check_threads, export_settings
from joblib import delayed, Parallel
import logging
import numpy as np
import os
from radiantkit import const, segmentation
from radiantkit.image import Image, ImageBinary, ImageLabeled
import re
import sys
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = '''
Perform automatic 3D segmentation of DNA staining. Images are first identified
based on a regular expression matching the file name. Then, they are re-scaled
(if deconvolved with Huygens software). Afterwards, a global (Otsu) and local
(gaussian) thresholds are applied to binarize the image in 3D. Finally, holes
are filled in 3D and a closing operation is performed to remove small objects.
Objects are filtered based on volume and Z size. Moreover, objects touching the
XY image borders are discarded. Use the --labeled flag to label identified
objects with different intensity levels.
    ''', formatter_class = argparse.RawDescriptionHelpFormatter)

    parser.add_argument('imgFolder', type = str,
        help = 'Path to folder containing deconvolved tiff images.')
    parser.add_argument('outFolder', type = str,
        help = '''Path to output folder where binarized images will be
        stored (created if does not exist).''')

    default_inreg = '^.*\.tiff?$'
    parser.add_argument('--inreg', type = str,
        help = """regular expression to identify images from imgFolder.
        Default: '%s'""" % (default_inreg,), default = default_inreg)
    parser.add_argument('--outprefix', type = str,
        help = """prefix to add to the name of output binarized images.
        Default: 'mask_', 'cmask_' if --compressed is used.""",
        default = 'mask_')
    parser.add_argument('--neighbour', type = int,
        help = """Side of neighbourhood square/cube. Must be odd.
        Default: 101""", default = 101)
    parser.add_argument('--radius', type = float, nargs = 2,
        help = """Range of object radii [vx] to be considered a nucleus.
        Default: [10, Inf]""", default = [10., float('Inf')])
    parser.add_argument('--min-Z', type = float, help = """Minimum fraction of
        stack occupied by an object to be considered a nucleus. Default: .25""",
        default = .25)
    parser.add_argument('-t', '--threads', type = int,
        help = """Number of threads for parallelization. Default: 1""",
        default = 1)
    parser.add_argument('-2', '--manual-2d-masks', type = str,
        help = """Path to folder with 2D masks with matching name,
        to combine with 3D masks.""",  metavar = "MAN2DDIR")
    parser.add_argument('-F', '--dilate-fill-erode', type = int,
        metavar = "DFE", help = """Number of pixels for dilation/erosion in a
        dilate-fill-erode operation. Default: 10. Set to 0 to skip.""",
        default = 10)

    parser.add_argument('--clear-Z',
        action = 'store_const', dest = 'do_clear_Z',
        const = True, default = False,
        help = """Remove objects touching the bottom/top of the stack.""",)
    parser.add_argument('--labeled',
        action = 'store_const', dest = 'labeled',
        const = True, default = False,
        help = 'Import/Export masks as labeled instead of binary.')
    parser.add_argument('--compressed',
        action = 'store_const', dest = 'compressed',
        const = True, default = False,
        help = 'Generate compressed TIF binary masks.')
    parser.add_argument('-y', '--do-all', action = 'store_const',
        help = """Do not ask for settings confirmation and proceed.""",
        const = True, default = False)
    parser.add_argument('--debug', '-u',
        action = 'store_const', dest = 'debug_mode',
        const = True, default = False,
        help = 'Log also debugging messages.')

    version = "3.1.1"
    parser.add_argument('--version', action = 'version',
        version = '%s %s' % (sys.argv[0], version,))

    args = parser.parse_args()
    args.version = version

    args.inreg = re.compile(args.inreg)
    if args.compressed and "mask_" == args.outprefix:
        args.outprefix = 'cmask_'

    assert 1 == args.neighbour % 2
    assert args.min_Z >= 0 and args.min_Z <= 1

    args.combineWith2D = args.manual_2d_masks is not None
    if args.combineWith2D:
        assert os.path.isdir(args.manual_2d_masks
            ), f"2D mask folder not found, '{args.manual_2d_masks}'"

    args.threads = check_threads(args.threads)

    return args

def print_settings(args: argparse.Namespace, clear: bool = True) -> str:
    s = f"""
# Automatic 3D segmentation v{args.version}

---------- SETTING :  VALUE ----------

   Input directory :  '{args.imgFolder}'
  Output directory :  '{args.outFolder}'

       Mask prefix :  '{args.outprefix}'
     Neighbourhood :  {args.neighbour}
          2D masks : '{args.manual_2d_masks}'
           Labeled :  {args.labeled}
        Compressed :  {args.compressed}

 Dilate-fill-erode :  {args.dilate_fill_erode}
 Minimum Z portion :  {args.min_Z:.2f}
    Minimum radius :  [{args.radius[0]:.2f}, {args.radius[1]:.2f}] vx
           Clear Z :  {args.do_clear_Z}

           Threads :  {args.threads}
            Regexp :  '{args.inreg}'
             Debug :  '{args.debug_mode}'

    """
    if clear: print("\033[H\033[J")
    print(s)
    return(s)

def confirm_arguments(args: argparse.Namespace) -> None:
    settings_string = print_settings(args)
    if not args.do_all: ask("Confirm settings and proceed?")

    assert os.path.isdir(args.imgFolder
        ), f"image folder not found: {args.imgFolder}"
    if not os.path.isdir(args.outFolder): os.mkdir(args.outFolder)

    with open(os.path.join(args.outFolder, "settings.txt"), "w+") as OH:
        export_settings(OH, settings_string)

def run_segmentation(args: argparse.Namespace,
    imgpath: str, imgdir: str) -> None:
    I = Image.from_tiff(os.path.join(imgdir, imgpath))
    I.rescale_factor = I.get_huygens_rescaling_factor()

    binarizer = segmentation.Binarizer()
    binarizer.segmentation_type = const.SegmentationType.THREED
    binarizer.local_side = args.neighbour
    binarizer.min_z_size = args.min_Z
    binarizer.do_clear_Z_borders = args.do_clear_Z

    mask2d = None
    if args.combineWith2D:
        if os.path.isdir(args.manual_2d_masks):
            mask2_path = os.path.join(
                args.manual_2d_masks, os.path.basename(imgpath))
            if os.path.isfile(mask2d_path):
                mask2d = ImageBinary.from_tiff(mask2_path, axes="YX").pixels

    L = binarizer.run(I, mask2d).label()

    xy_size_range = (np.pi*args.radius[0]**2, np.pi*args.radius[1]**2)
    logging.info(f"Filtering XY size: {xy_size_range}")
    L.filter_size("XY", xy_size_range)
    z_size_range = (args.min_Z*I.axis_shape("Z"), np.inf)
    logging.info(f"Filtering Z size: {z_size_range}")
    L.filter_size("Z", z_size_range)

    print(L.pixels.max())

def run(args: argparse.Namespace) -> None:
    imglist = [f for f in os.listdir(args.imgFolder) 
        if os.path.isfile(os.path.join(args.imgFolder, f))
        and not type(None) == type(re.match(args.inreg, f))]
    logging.info(f"found {len(imglist)} image(s) to segment.")
    if 1 == args.threads:
        for imgpath in tqdm(imglist): run_segmentation(
            args, imgpath, args.imgFolder)
    else:
        Parallel(n_jobs = args.threads, verbose = 11)(
            delayed(run_segmentation)(
                args, imgpath, args.imgFolder)
            for imgpath in imglist)

def main():
    args = parse_arguments()
    if args.debug_mode: logging.getLogger().level = logging.DEBUG
    confirm_arguments(args)
    run(args)
