"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import logging
import argparse
from ggc.args import check_threads, export_settings  # type: ignore
from joblib import delayed, Parallel  # type: ignore
import numpy as np  # type: ignore
import os
from radiantkit.const import __version__
from radiantkit import const, path, stat, string
from radiantkit import channel, image, io, segmentation
import re
import sys
from tqdm import tqdm  # type: ignore
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s "
    + "[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
)


def init_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1],
        description="""
Perform automatic 3D segmentation of TIFF images. The default parameters are
optimized for nuclear DNA staining and voxel size of 0.13x0.13x0.3 uM.

The input images are first identified based on a regular expression matched to
the file name. Then, they are re-scaled (if deconvolved with Huygens software).
Afterwards, a global (Otsu) and local (gaussian) thresholds are applied to
binarize the image in 3D. Finally, holes are filled in 3D and closed to remove
small objects. Finally. objects are filtered based on volume and Z size.
Moreover, objects touching the XY image borders are discarded.

If a folder path is provided with the -2 option, any binary file with name
matching the one of an input image will be combined to the binarized image.

Use the --labeled flag to label identified objects with different intensity
levels. By default, the script generates compressed binary tiff images; use the
--uncompressed flag to generate normal tiff images instead.

Input images that have the specified prefix and suffix are not segmented.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Segment tiff images (default optimized for DAPI staining).",
    )

    parser.add_argument(
        "input",
        type=str,
        help="""Path single tiff image,
        or to folder containing deconvolved tiff images.""",
    )

    parser.add_argument(
        "-o",
        metavar="DIRPATH",
        type=str,
        default=None,
        help="""Path to output TIFF folder. Defaults to the input folder""",
        dest="output",
    )
    parser.add_argument(
        "--outprefix",
        type=str,
        metavar="TEXT",
        default="",
        help="""Prefix for output binarized images name. Default: ''.""",
    )
    parser.add_argument(
        "--outsuffix",
        type=str,
        metavar="TEXT",
        default="mask",
        help="""Suffix for output binarized images name. Default: 'mask'.""",
    )
    parser.add_argument(
        "--neighbour",
        type=int,
        metavar="NUMBER",
        help="""Side of neighbourhood region for adaptig thresholding.
        Must be odd. Default: 101""",
        default=101,
    )
    parser.add_argument(
        "--radius",
        type=float,
        nargs=2,
        help="""Filter range of object radii [px]. Default: [10, Inf]""",
        default=[10.0, float("Inf")],
        metavar=("MIN_RADIUS", "MAX_RADIUS"),
    )
    parser.add_argument(
        "--min-Z",
        type=float,
        metavar="FRACTION",
        default=0.25,
        help="""Minimum stack fraction occupied by an object. Default: .25""",
    )
    parser.add_argument(
        "--mask-2d",
        type=str,
        metavar="DIRPATH",
        help="""Path to folder with 2D masks with matching name,
        to combine with 3D masks.""",
    )

    parser.add_argument(
        "--clear-Z",
        action="store_const",
        dest="do_clear_Z",
        const=True,
        default=False,
        help="""Remove objects touching the bottom/top of the stack.""",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%s %s"
        % (
            sys.argv[0],
            __version__,
        ),
    )

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument(
        "--dilate-fill-erode",
        type=int,
        metavar="NUMBER",
        default=0,
        help="""Number of pixels for dilation/erosion steps
        in a dilate-fill-erode operation. Default: 0. Set to 0 to skip.""",
    )
    advanced.add_argument(
        "--labeled",
        action="store_const",
        dest="labeled",
        const=True,
        default=False,
        help="Export masks as labeled instead of binary.",
    )
    advanced.add_argument(
        "--uncompressed",
        action="store_const",
        dest="compressed",
        const=False,
        default=True,
        help="Generate uncompressed TIFF binary masks.",
    )
    advanced.add_argument(
        "--no-rescaling",
        action="store_const",
        dest="do_rescaling",
        const=False,
        default=True,
        help="Do not rescale image even if deconvolved.",
    )
    advanced.add_argument(
        "--debug",
        action="store_const",
        dest="debug_mode",
        const=True,
        default=False,
        help="Log also debugging messages. Silenced by --silent.",
    )
    advanced.add_argument(
        "--silent",
        action="store_const",
        dest="silent",
        const=True,
        default=False,
        help="Limits logs to critical events only.",
    )
    default_inreg = "^.*\\.tiff?$"
    advanced.add_argument(
        "--inreg",
        type=str,
        metavar="REGEXP",
        help="""Regular expression to identify input TIFF images.
        Default: '%s'"""
        % (default_inreg,),
        default=default_inreg,
    )
    advanced.add_argument(
        "--threads",
        type=int,
        metavar="NUMBER",
        dest="threads",
        default=1,
        help="""Number of threads for parallelization. Default: 1""",
    )
    advanced.add_argument(
        "-y",
        "--do-all",
        action="store_const",
        const=True,
        default=False,
        help="""Do not ask for settings confirmation and proceed.""",
    )

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    args.version = __version__

    if args.output is None:
        if os.path.isfile(args.input):
            args.output = os.path.dirname(args.input)
        else:
            args.output = args.input

    args.inreg = re.compile(args.inreg)
    args.outprefix = string.add_trailing_dot(args.outprefix)
    args.outsuffix = string.add_leading_dot(args.outsuffix)

    assert 1 == args.neighbour % 2
    assert args.min_Z >= 0 and args.min_Z <= 1

    if args.mask_2d is not None:
        assert os.path.isdir(
            args.mask_2d
        ), f"2D mask folder not found, '{args.mask_2d}'"

    args.threads = check_threads(args.threads)

    loglvl = 20
    loglvl = logging.DEBUG if args.debug_mode else 20
    logging.getLogger().level = logging.CRITICAL if args.silent else loglvl

    return args


def print_settings(args: argparse.Namespace, clear: bool = True) -> str:
    s = f"""# Automatic 3D segmentation v{args.version}

    ---------- SETTING : VALUE ----------

       Input directory : '{args.input}'
      Output directory : '{args.output}'

           Mask prefix : '{args.outprefix}'
           Mask suffix : '{args.outsuffix}'
         Neighbourhood : {args.neighbour}
              2D masks : '{args.mask_2d}'
               Labeled : {args.labeled}
            Compressed : {args.compressed}

     Dilate-fill-erode : {args.dilate_fill_erode}
     Minimum Z portion : {args.min_Z:.2f}
        Minimum radius : [{args.radius[0]:.2f}, {args.radius[1]:.2f}] vx
               Clear Z : {args.do_clear_Z}

               Rescale : {args.do_rescaling}
               Threads : {args.threads}
                Regexp : {args.inreg.pattern}
                 Debug : {args.debug_mode}
                Silent : {args.silent}
    """
    if clear:
        print("\033[H\033[J")
    print(s)
    return s


def confirm_arguments(args: argparse.Namespace) -> None:
    settings_string = print_settings(args)
    if not args.do_all:
        io.ask("Confirm settings and proceed?")

    if not os.path.isfile(args.input):
        assert os.path.isdir(args.input), f"image folder not found: {args.input}"
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    with open(os.path.join(args.output, "tiff_segment.config.txt"), "w+") as OH:
        export_settings(OH, settings_string)


def read_mask_2d(
    args: argparse.Namespace, imgpath: str
) -> Optional[image.ImageLabeled]:
    mask2d = None
    if args.mask_2d is not None:
        if os.path.isdir(args.manual_2d_masks):
            mask2d_path = os.path.join(args.manual_2d_masks, os.path.basename(imgpath))
            if os.path.isfile(mask2d_path):
                mask2d = image.ImageLabeled.from_tiff(
                    mask2d_path, axes="YX", doRelabel=False
                )
    return mask2d


def run_segmentation(
    args: argparse.Namespace, imgpath: str, imgdir: str, loglevel: str = "INFO"
) -> None:
    logging.getLogger().setLevel(loglevel)
    logging.info(f"Segmenting image '{imgpath}'")

    img = channel.ImageGrayScale.from_tiff(
        os.path.join(imgdir, imgpath), doRescale=args.do_rescaling
    )
    logging.info(f"image axes: {img.axes}")
    logging.info(f"image shape: {img.shape}")
    if args.do_rescaling:
        logging.info(f"rescaling factor: {img.rescale_factor}")

    binarizer = segmentation.Binarizer()
    binarizer.segmentation_type = const.SegmentationType.THREED
    binarizer.local_side = args.neighbour
    binarizer.do_clear_Z_borders = args.do_clear_Z

    M2D = read_mask_2d(args, imgpath)
    M = binarizer.run(img, M2D)
    assert isinstance(M, image.ImageBinary)

    logging.info(f"dilate-fill-erode with side {args.dilate_fill_erode}")
    M.dilate_fill_erode(args.dilate_fill_erode)
    logging.info("labeling")
    L = M.label()

    size_range = stat.radius_interval_to_size(args.radius, len(L.axes))
    logging.info(f"filtering total size: {size_range}")
    L.filter_total_size(size_range)
    z_size_range = (args.min_Z * img.axis_shape("Z"), np.inf)
    logging.info(f"filtering Z size: {z_size_range}")
    L.filter_size("Z", z_size_range)

    if M2D is not None:
        logging.info("recovering labels from 2D mask")
        L.inherit_labels(M2D)

    if 0 == L.pixels.max():
        logging.warning(f"skipped image '{imgpath}' (only background)")
        return

    imgbase, imgext = os.path.splitext(imgpath)
    if not args.labeled:
        logging.info("writing output")
        M = image.ImageBinary(L.pixels)
        M.to_tiff(
            os.path.join(
                args.output, f"{args.outprefix}{imgbase}{args.outsuffix}{imgext}"
            ),
            args.compressed,
        )
    else:
        logging.info("writing labeled output")
        L.to_tiff(
            os.path.join(
                args.output, f"{args.outprefix}{imgbase}{args.outsuffix}{imgext}"
            ),
            args.compressed,
        )


def run(args: argparse.Namespace) -> None:
    confirm_arguments(args)
    if os.path.isfile(args.input):
        assert re.match(
            args.inreg, os.path.basename(args.input)
        ), "the provided image name does not match the pattern"
        imglist = [os.path.basename(args.input)]
        args.input = os.path.dirname(args.input)
    else:
        imglist = path.find_re(args.input, args.inreg)

    _, imglist = path.select_by_prefix_and_suffix(
        args.input, imglist, args.outprefix, args.outsuffix
    )

    logLevel = logging.getLogger().level
    logLevel = 10 if args.debug_mode else 20
    logLevel = 50 if args.silent else logLevel

    logging.info(f"found {len(imglist)} image(s) to segment.")
    if 1 == args.threads:
        for imgpath in tqdm(imglist):
            run_segmentation(args, imgpath, args.input)
    else:
        Parallel(n_jobs=args.threads, verbose=11)(
            delayed(run_segmentation)(args, imgpath, args.input, logLevel)
            for imgpath in imglist
        )
