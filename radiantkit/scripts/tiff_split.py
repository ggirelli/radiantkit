'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import configparser as cp
from ggc.prompt import ask
import logging
import numpy as np
import os
import radiantkit.image as imt
from typing import List
import sys
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = '''
Split a TIFF image in smaller TIFF images of the specified side(s). If two
different sides are provided, the smaller images will be rectangular. The first
side corresponds to the X (columns) and the second to the Y (rows). By default,
only one side is required, which is used by the script for both X and Y sides.
In other words, square smaller images are produced.

If the original dimensions are not multiples of the specified side, a portion of
the image is lost, unless the --enlarge option is used. In that case, the
smaller images generated from the image border will contain empty pixels.

If the input image is a 3D stack, it will be split only on XY and the output
images will have the same number of slices. Using the --slice option, it is
possible to specify which slice to split (i.e., the output will be in 2D).
Defaults to first slice (--slice 0).

It is also possible to generate overlapping split images. This can be achieved
by using either the -S or -O options (which cannot be used together). With the
-S option, you can specify the step used when splitting the image, as a fraction
of its sides or as an absolute number of pixels. With the -O option, you can
specify the overlapping region between consecutive split images as a fraction of
their sides or as absolute pixels. In other words, the options -S 0.9 and -O 0.1
yield the same result. It is possible to provide two values to -S and -O, to
obtain different overlaps in
X and Y.

By default, split images are generated left-to-right, top-to-bottom, e.g.,
1 2 3
4 5 6
7 8 9

Use the option -I to generate them top-to-bottom, left-to-right, e.g.,
1 4 7
2 5 8
3 6 9

Examples:

- Square images of 100x100 px
tiff_split big_image.tif split_out_dir 100 -e

- Rectangular images of 125x100 px
tiff_split big_image.tif split_out_dir 100 125 -e

- Square images of 100x100 px, overlapping for 10 px in X and Y
tiff_split big_image.tif split_out_dir 100 -e -S 0.9
tiff_split big_image.tif split_out_dir 100 -e -S 90
tiff_split big_image.tif split_out_dir 100 -e -O 0.1
tiff_split big_image.tif split_out_dir 100 -e -O 10

- Square images of 100x100 px, overlapping for 10 px in X and 20 px in Y
tiff_split big_image.tif split_out_dir 100 -e -S 0.9 0.8
tiff_split big_image.tif split_out_dir 100 -e -S 90 80
tiff_split big_image.tif split_out_dir 100 -e -O 0.1 0.2
tiff_split big_image.tif split_out_dir 100 -e -O 10 20
    ''', formatter_class = argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input', type = str,
        help = '''Path to the TIFF image to split.''')
    parser.add_argument('outdir', type = str,
        help = '''Path to output TIFF folder, created if missing.''')
    parser.add_argument('side', type = int, nargs = '+',
        help = '''One or two (XY) sides,
        used to specify the smaller images dimensions.''')

    parser.add_argument('-S', '--step', metavar = "step", type = float,
        nargs = '+', help = """Step for splitting, defined as a fraction of the
        specified side(s).""")
    parser.add_argument('-O', '--overlap', metavar = "overlap", type = float,
        help = """Overlap fraction of splitted images, defined as a fraction of
        the specified side(s).""", nargs = '+')
    parser.add_argument('-s', '--slice', metavar = "slice", type = int,
        help = """ID of slice to be extracted from Z-stacks, 1-indexed.""")

    parser.add_argument('-e', '--enlarge',
        action = 'store_const', dest = 'enlarge',
        const = True, default = False,
        help = 'Expand to avoid pixel loss.')
    parser.add_argument('-I', '--invert',
        action = 'store_const', dest = 'inverted',
        const = True, default = False,
        help = '''Split top-to-bottom, left-to-right.''')
    parser.add_argument('-y', '--do-all', action = 'store_const',
        help = """Do not ask for settings confirmation and proceed.""",
        const = True, default = False)

    version = "0.0.1"
    parser.add_argument('--version', action = 'version',
        version = '%s %s' % (sys.argv[0], version,))

    args = parser.parse_args()
    args.version = version

    assert os.path.isfile(args.input), "input file not found: %s" % args.input
    assert not os.path.isfile(args.outdir
        ), "output directory cannot be a file: %s" % (args.outdir)

    if 1 == len(args.side): args.side = (args.side[0], args.side[0])
    else: args.side = args.side[:2]

    if args.slice is not None: assert args.slice > 0
    assert not (args.step is not None and args.overlap is not None
        ), "-S and -O are incompatible"

    relative_steps = True
    if args.step is not None:
        assert all([step > 0 for step in args.step])
        step_is_relative = all([step <= 1 for step in args.step])
        step_is_absolute = all([step > 1 for step in args.step])
        assert step_is_absolute or step_is_relative

        while len(args.step) < len(args.side): args.step.append(args.step[0])
        args.step = args.step[:len(args.side)]

        if step_is_absolute:
            relative_steps = False
            args.overlap = np.array([args.side[i] - args.step[i]
                for i in range(len(args.step))]).astype('int')
        elif step_is_relative:
            args.overlap = [np.round(1-s, 3) for s in args.step]

    if args.overlap is not None:
        assert all([overlap >= 0 for overlap in args.overlap])
        overlap_is_relative = all([overlap < 1 for overlap in args.overlap])
        overlap_is_absolute = all([overlap > 1 for overlap in args.overlap])
        assert overlap_is_absolute or overlap_is_relative

        while len(args.overlap) < len(args.side):
            args.overlap.append(args.overlap[0])
        args.overlap = args.overlap[:len(args.side)]

        if overlap_is_absolute:
            relative_steps = False
            args.step = np.array([args.side[i] - args.overlap[i]
                for i in range(len(args.overlap))]).astype('int')
        elif overlap_is_relative:
            args.overlap = [np.round(1-s, 3) for s in args.overlap]

    if (args.overlap is not None or args.step is not None) and relative_steps:
        args.step = np.array([np.round(args.side[i]*args.step[i])
            for i in range(len(args.step))]).astype('int')
        args.overlap = np.array([np.round(args.side[i]*args.overlap[i])
            for i in range(len(args.overlap))]).astype('int')

    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    return(args)

def enlarge_XY_tiff(I: np.ndarray, offset: List[int]) -> np.ndarray:
    new_shape = list(I.shape)
    new_shape[-1] += int(offset[0])
    new_shape[-2] += int(offset[1])
    new_image = I.copy()
    new_image = np.zeros(new_shape)
    new_image[np.ix_(*[range(I.shape[i]) for i in range(len(I.shape))])] = I
    return(new_image)

def get_pixel_loss(I: np.ndarray, side: List[int], step: float) -> int:
    N = len(I.shape)
    assert len(side) <= N

    if step is None:
        missed = [I.shape[-i-1]%side[i] for i in range(len(side))]
    else:
        assert len(side) == len(step)
        missed = [I.shape[-i-1]%side[i]%step[i] for i in range(len(side))]

    loss = []
    for i in range(len(side)):
        otherd = [I.shape[j] for j in range(N) if not N-i-1 == j]
        otherd.append(missed[-i-1])
        loss.append(np.prod(otherd))
    loss = int(np.sum(loss) - np.prod(I.shape[:-2]) * np.prod(missed))
    
    return (*missed, loss, loss/np.prod(I.shape)*100)

def tiff_split(I: np.ndarray, side: List[int], step: List[int],
    inverted: bool = False) -> np.ndarray:
    if step is None: step = side

    n = I.shape[-1]//side[0] * I.shape[-2]//side[1]
    logging.info(f"Output {n} images.")
    if 0 == n: return

    ys = [y for y in range(0,I.shape[-2],step[1])
        if y+side[1] <= I.shape[-2]]
    xs = [x for x in range(0,I.shape[-1],step[0])
        if x+side[0] <= I.shape[-1]]

    if inverted:
        logging.info("Image split top-to-bottom, left-to-right.")
        xy_gen = ((x, y) for x in xs for y in ys)
    else:
        logging.info("Image split left-to-right, top-to-bottom.")
        xy_gen = ((x, y) for y in ys for x in xs)

    assert len(I.shape) in [2, 3]
    if 3 == len(I.shape):
        tsplit = lambda i, x, y, s: i[:, y:(y+s[1]), x:(x+s[0])]
    elif 2 == len(I.shape):
        tsplit = lambda i, x, y, s: i[y:(y+s[1]), x:(x+s[0])]

    with tqdm(range(n)) as pbar:
        for (x_start, y_start) in xy_gen:
            yield tsplit(I, x_start, y_start, side)
            pbar.update(1)
    return

def print_settings(args: argparse.Namespace, clear: bool = True) -> None:
    settings_string = f""" # TIFF split v{args.version}
        Input file :  {args.input}
  Output directory :  {args.outdir}

            X side : {args.side[0]}
            Y side : {args.side[1]}

           Overlap : {args.overlap}
              Step : {args.step}

             Slice : {args.slice}
           Enlarge : {args.enlarge}
          Inverted : {args.inverted}
    """
    if clear: settings_string = f"\033[H\033[J{settings_string}"
    print(settings_string)

def save_settings(args: argparse.Namespace) -> None:
    config = cp.ConfigParser()
    config['MAIN'] = {'input' : args.input, 'outdir' : args.outdir}
    config['ADVANCED'] = {
        'x_side' : str(args.side[0]), 'y_side' : str(args.side[1]),
        'slice' : str(args.slice), 'enlarge' : str(args.enlarge),
        'inverted' : str(args.inverted)
    }
    if args.step is not None:
        config['ADVANCED'].update({'x_step' : str(args.step[0]),
            'y_step' : str(args.step[1])})

    with open(os.path.join(args.outdir, 'config.ini'), 'w') as CF:
        config.write(CF)

def run(args: argparse.Namespace) -> None:
    logging.info("Reading input image...")
    I = imt.Image.read_tiff(args.input).pixels

    if 3 == len(I.shape):
        logging.info(f"3D stack found: {I.shape}")
        if args.slice is not None:
            logging.info(f"Enforcing 2D split (slice #{args.slice} only).")
            umes = "pixel"
            assert args.slice <= I.shape[0]
            I = I[args.slice-1, :, :].copy()
        else:
            umes = "voxel"
    elif 2 == len(I.shape):
        logging.info(f"2D image found: {I.shape}")
        umes = "pixel"
    else:
        logging.error(f"cannot split a 1D image. File: {args.input}")
        sys.exit()

    x_loss, y_loss, loss, perc_loss = get_pixel_loss(I, args.side, args.step)
    if args.enlarge:
        I = enlarge_XY_tiff(I, np.array(args.side)-np.array((x_loss, y_loss)))
        logging.info(f"Image enlarged to {I.shape}")
    else:
        logging.info(f"{x_loss} {umes}s (X) and {y_loss} {umes}s (Y) are lost.")
        logging.info(f"In total, {loss} {umes}s lost ({perc_loss}%). "+
            "Use -e to avoid loss.")

    prefix = os.path.splitext(os.path.basename(args.input))[0]
    ext = os.path.splitext(os.path.basename(args.input))[1]

    image_counter = 1
    for sub_image in tiff_split(I, args.side, args.step, args.inverted):
        opath = os.path.join(args.outdir, f"{prefix}.sub{image_counter}{ext}")
        imt.Image.save_tiff(opath, sub_image, imt.get_dtype(sub_image.max()),
            compressed=False)
        image_counter += 1

def main() -> None:
    args = parse_arguments()
    print_settings(args)
    if not args.do_all: ask("Confirm settings and proceed?")
    save_settings(args)
    run(args)
