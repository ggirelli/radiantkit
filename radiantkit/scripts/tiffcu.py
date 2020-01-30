'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
from ggc.args import check_threads
from joblib import Parallel, delayed
import logging
import multiprocessing
import os
from radiantkit.const import __version__
import radiantkit.image as imt
import re
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s ' +
    '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = '''
(Un)compress TIFF images. Provide either a single input/output image paths, or
input/output folder paths. In case of folder input/output, all tiff files in the
input folder with file name matching the specified pattern are (un)compressed
and saved to the output folder. When (un)compressing multiple files, the
--threads option allows to parallelize on multiple threads. Disk read/write
operations become the bottleneck when parallelizing.
    ''', formatter_class = argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input', type = str,
        help = '''Path to the TIFF image to (un)compress, or to a folder
        containing multiple TIFF images. In the latter case, the --inreg pattern
        is used to identify the image file.''')
    parser.add_argument('output', type = str,
        help = '''Path to output TIFF image, or output folder if the input is a
        folder.''')

    default_inreg = '^.*\.tiff?$'
    parser.add_argument('--inreg', type = str, default = default_inreg,
        help = """Regular expression to identify image files, when the input is
        a folder. Default: '%s'""" % (default_inreg,))
    parser.add_argument('--threads', '-t', type = int, default = 1,
        help = """Number of threads for parallelization. Used only to
        (un)compress multiple images (i.e., input is a folder). Default: 1""")

    parser.add_argument('-u', const = True, default = False,
        action = 'store_const', dest = 'doUncompress',
        help = 'Uncompress TIFF files.')
    parser.add_argument('-c', const = True, default = False,
        action = 'store_const', dest = 'doCompress',
        help = 'Compress TIFF files.')

    parser.add_argument('--version', action = 'version',
        version = '%s %s' % (sys.argv[0], __version__,))

    args = parser.parse_args()
    args.inreg = re.compile(args.inreg)

    if not args.doCompress and not args.doUncompress:
        logging.error("please, use either -c (compress) or -u (uncompress).")
        sys.exit()
    if args.doCompress and args.doUncompress:
        logging.error("please, use either -c (compress) or -u (uncompress).")
        sys.exit()
    args.threads = check_threads(args.threads)

    args.process_multiple_files = False
    if os.path.isdir(args.input):
        args.process_multiple_files = True
        assert not os.path.isfile(args.output), "in/output should be folders."
    else:
        assert not os.path.isdir(args.output), "in/output should be files."
        assert os.path.isfile(args.input), f"input not found: '{args.input}'"

    return args

def export_image(ipath: str, opath: str, compress: bool=None) -> str:
    idir = os.path.dirname(ipath)
    ipath = os.path.basename(ipath)
    odir = os.path.dirname(opath)
    opath = os.path.basename(opath)

    if compress is None: compress = False
    I = imt.ImageBase.from_tiff(os.path.join(idir, ipath))
    if opath is None: opath = ipath
    
    if not compress:
        I.to_tiff(os.path.join(odir, opath), compressed=False)
        label = "Uncompressed"
    else:
        I.to_tiff(os.path.join(odir, opath), compressed=True)
        label = "Compressed"

    logging.info(f"{label} '{os.path.join(idir, ipath)}'")
    return(os.path.join(odir, opath))

def run(args: argparse.Namespace) -> None:
    if args.process_multiple_files:
        if not os.path.isdir(args.input):
            logging.error(f"image folder not found: '{args.input}'")
            sys.exit()

        if not os.path.isdir(args.output):
            os.mkdir(args.output)

        imglist = [f for f in os.listdir(args.input) 
            if os.path.isfile(os.path.join(args.input, f))
            and not type(None) == type(re.match(args.inreg, f))]

        Parallel(n_jobs = args.threads)(
            delayed(export_image)(os.path.join(args.input, ipath),
                os.path.join(args.output, ipath),
                compress=args.doCompress) for ipath in imglist)
    else:
        export_image(args.input, args.output, args.doCompress)

def main() -> None:
    run(parse_arguments())
