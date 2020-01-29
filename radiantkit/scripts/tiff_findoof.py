'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import matplotlib
matplotlib.use('ps')

import argparse
from ggc.args import check_threads
from joblib import delayed, Parallel
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import radiantkit.image as imt
from radiantkit import plot, stat
import re
import sys
from tqdm import tqdm

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = '''
Calculate gradient magnitude over Z for every image in the input folder with a
filename matching the --pattern. Use --range to change the in-focus definition.
    ''', formatter_class = argparse.RawDescriptionHelpFormatter)

    parser.add_argument('imdir', type = str,
        help = 'Path to folder with tiff images.')
    parser.add_argument('output', type = str,
        help = 'Path to output tsv file.')

    parser.add_argument('-r', '--range', type = float, metavar = 'range',
        help = '''Specify %% of stack where the maximum of intensity
        distribution over Z is expected for an in-focus field of view.
        Default: 50%%''',
        default = .5)
    parser.add_argument('-p', '--pattern', type = str, metavar = 'regexp',
        help = '''Provide a regular expression pattern matching the images in
        the image folder that you want to check. Default: "^.*\.tif(f)$"''',
        default = "^.*\.tif(f)$")
    parser.add_argument('-t', '--threads', metavar = "nthreads", type = int,
        help = """Number of threads for parallelization. Default: 1""",
        default = 1)

    parser.add_argument('-P', '--plot', action = 'store_const',
        help = """Generate pdf plot of intensity sum per Z-slice.""",
        const = True, default = False)
    parser.add_argument('-S', '--intensity-sum', action = 'store_const',
        help = """Use intensity sum instead of gradient magnitude.""",
        const = True, default = False)
    parser.add_argument('-R', '--rename', action = 'store_const',
        help = """Rename out-of-focus images by adding the '.old' suffix.""",
        const = True, default = False)
    parser.add_argument('-s', '--silent', action = 'store_const',
        help = """Silent run.""",
        const = True, default = False)

    version = "0.3.1"
    parser.add_argument('--version', action = 'version',
        version = '%s %s' % (sys.argv[0], version,))

    args = parser.parse_args()
    args.threads = check_threads(args.threads)

    return args

def plot_profile(args: argparse.Namespace,
    series_data: pd.DataFrame, path: str) -> None:
    plt.figure(figsize = [12, 8])

    xmax = []
    ymax = []
    image_names = []
    for profile_data in series_data:
        image_names.append(os.path.basename(profile_data['path'].values[0]))
        xmax.append(max(profile_data['x']))
        ymax.append(max(profile_data['y']))
        plt.plot(profile_data['x'], profile_data['y'], linewidth=.5)
    xmax = max(xmax)
    ymax = max(ymax)

    plt.xlabel("Z-slice index")
    if args.intensity_sum: plt.ylabel("Intensity sum [a.u.]")
    else: plt.ylabel("Gradient magnitude [a.u.]")
    plt.title("Focus analysis")

    plt.legend(image_names,
        bbox_to_anchor = (1.04, 1), loc = "upper left",
        prop = {'size' : 6})
    plt.subplots_adjust(right = 0.75)

    plt.gca().axvline(x = xmax * args.range / 2, ymax = ymax,
        linestyle = "--", color = "k")
    plt.gca().axvline(x = xmax - xmax * args.range / 2, ymax = ymax,
        linestyle = "--", color = "k")

    plot.export(path)

def is_OOF(args: argparse.Namespace, ipath: str,
    logger: logging.RootLogger) -> pd.DataFrame:
    I = imt.Image.from_tiff(os.path.join(args.imdir, ipath)).pixels

    slice_descriptors = []
    for zi in range(I.shape[0]):
        if args.intensity_sum:
            slice_descriptors.append(I[zi].sum())
        else:
            dx = stat.gpartial(I[zi, :, :], 1, 1);
            dy = stat.gpartial(I[zi, :, :], 2, 1);
            slice_descriptors.append(np.mean(np.mean((dx**2 + dy**2) ** (1/2))))
    profile_data = pd.DataFrame.from_dict({'path':np.repeat(ipath, I.shape[0]),
        'x':np.array(range(I.shape[0]))+1, 'y':slice_descriptors})

    max_slice_id = slice_descriptors.index(max(slice_descriptors))
    halfrange = I.shape[0]*args.range/2.
    halfstack = I.shape[0]/2.

    response = "out-of-focus"
    if max_slice_id >= (halfstack-halfrange):
        if max_slice_id <= (halfstack+halfrange):
            response = "in-focus"
    logger.info(f"{ipath} is {response}.")
    profile_data['response'] = response

    if "out-of-focus" == response and args.rename:
        os.rename(os.path.join(args.imdir, ipath),
            os.path.join(args.imdir, ipath))

    return profile_data

def run(args: argparse.Namespace, logger: logging.RootLogger) -> None:
    if not "/" == args.imdir[-1]: args.imdir += "/"
    if not os.path.isdir(args.imdir):
        logger.error(f"image directory not found: '{args.imdir}'")
        sys.exit()

    flist = []
    for (dirpath, dirnames, filenames) in os.walk(args.imdir):
        flist.extend(filenames)
        break
    imlist = [f for f in flist if (
        not type(None) == type(re.match(args.pattern, f)))]

    if 1 == args.threads:
        if args.silent: t = imlist
        else: t = tqdm(imlist, desc = os.path.dirname(args.imdir))
        series_data = [is_OOF(args, impath, logger) for impath in t]
    else:
        verbosity = 11 if not args.silent else 0
        series_data = Parallel(n_jobs = args.threads, verbose = verbosity)(
            delayed(is_OOF)(args, impath, logger) for impath in imlist)

    pd.concat(series_data).to_csv(args.output, '\t', index=False)
    if args.plot: plot_profile(args, series_data,
        f"{os.path.splitext(args.output)[0]}.pdf")

def main():
    args = parse_arguments()

    logger = logging.getLogger()
    if 1 == args.threads:
        FH = logging.FileHandler(
            filename=f"{os.path.splitext(args.output)[0]}.log", mode="w+")
        FH.setLevel(logging.INFO)
        logger.addHandler(FH)

    run(args, logger)
