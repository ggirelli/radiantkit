"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
import configparser as cp
import logging
import numpy as np  # type: ignore
import os
from radiantkit.const import __version__
from radiantkit import image as imt
from rich.logging import RichHandler  # type: ignore
from rich.progress import track  # type: ignore
from rich.prompt import Confirm  # type: ignore
import sys
from typing import Iterable, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)],
)


def init_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1],
        description="""
Split a TIFF image in smaller TIFF images of the specified side(s).

If two different sides are provided, the smaller images will be rectangular.
The first side corresponds to the X (columns) and the second to the Y (rows).
By default, only one side is required, which is used by the script for both X
and Y sides. In other words, square smaller images are produced.

If the original dimensions are not multiples of the specified side, a portion
of the image is lost, unless the --enlarge option is used. In that case, the
smaller images generated from the image border will contain empty pixels.

If the input image is a 3D stack, it will be split only on XY and the output
images will have the same number of slices. Using the --slice option, it is
possible to specify which slice to split (i.e., the output will be in 2D).
Defaults to first slice (--slice 0).

It is also possible to generate overlapping split images. This can be achieved
by using either the -S or -O options (which cannot be used together). With the
-S option, you can specify the step used when splitting the image, as a
fraction of its sides or as an absolute number of pixels. With the -O option,
you can specify the overlapping region between consecutive split images as a
fraction of their sides or as absolute pixels. In other words, the options
-S 0.9 and -O 0.1 yield the same result. It is possible to provide two values
to -S and -O, to obtain different overlaps in X and Y.

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
tiff_split big_image.tif split_out_dir 100 -e -O 10 20""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Split a TIFF image in smaller images of the specified side(s).",
    )

    parser.add_argument("input", type=str, help="""Path to the TIFF image to split.""")
    parser.add_argument(
        "outdir", type=str, help="""Path to output TIFF folder, created if missing."""
    )
    parser.add_argument(
        "side",
        type=int,
        nargs="+",
        help="""One or two (XY) sides,
        used to specify the smaller images dimensions.""",
    )

    parser.add_argument(
        "--step",
        metavar="NUMBER",
        type=float,
        nargs="+",
        help="""Step for splitting, defined as a fraction of the
        specified side(s).""",
    )
    parser.add_argument(
        "--overlap",
        metavar="NUMBER",
        type=float,
        help="""Overlap fraction of splitted images, defined as a fraction of
        the specified side(s).""",
        nargs="+",
    )
    parser.add_argument(
        "--slice",
        metavar="NUMBER",
        type=int,
        help="""ID of slice to be extracted from Z-stacks, 1-indexed.""",
    )

    parser.add_argument(
        "--enlarge",
        action="store_const",
        dest="enlarge",
        const=True,
        default=False,
        help="Expand to avoid pixel loss.",
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
        "--invert",
        action="store_const",
        dest="inverted",
        const=True,
        default=False,
        help="""Split top-to-bottom, left-to-right.""",
    )
    advanced.add_argument(
        "-y",
        "--do-all",
        action="store_const",
        help="""Do not ask for settings confirmation and proceed.""",
        const=True,
        default=False,
    )

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


def update_args_from_step(
    args: argparse.Namespace, relative_steps: bool
) -> Tuple[argparse.Namespace, bool]:
    assert all([step > 0 for step in args.step])
    step_is_relative = all([step <= 1 for step in args.step])
    step_is_absolute = all([step > 1 for step in args.step])
    assert step_is_absolute or step_is_relative

    while len(args.step) < len(args.side):
        args.step.append(args.step[0])
    args.step = args.step[: len(args.side)]

    if step_is_absolute:
        relative_steps = False
        args.overlap = np.array(
            [args.side[i] - args.step[i] for i in range(len(args.step))]
        ).astype("int")
    elif step_is_relative:
        args.overlap = [np.round(1 - s, 3) for s in args.step]

    return args, relative_steps


def update_args_from_overlap(
    args: argparse.Namespace, relative_steps: bool
) -> Tuple[argparse.Namespace, bool]:
    assert all([overlap >= 0 for overlap in args.overlap])
    overlap_is_relative = all([overlap < 1 for overlap in args.overlap])
    overlap_is_absolute = all([overlap > 1 for overlap in args.overlap])
    assert overlap_is_absolute or overlap_is_relative

    while len(args.overlap) < len(args.side):
        args.overlap.append(args.overlap[0])
    args.overlap = args.overlap[: len(args.side)]

    if overlap_is_absolute:
        relative_steps = False
        args.step = np.array(
            [args.side[i] - args.overlap[i] for i in range(len(args.overlap))]
        ).astype("int")
    elif overlap_is_relative:
        args.overlap = [np.round(1 - s, 3) for s in args.overlap]

    return args, relative_steps


def check_step_and_overlap(args: argparse.Namespace) -> argparse.Namespace:
    relative_steps = True
    if args.step is not None:
        args, relative_steps = update_args_from_step(args, relative_steps)

    if args.overlap is not None:
        args, relative_steps = update_args_from_overlap(args, relative_steps)

    if (args.overlap is not None or args.step is not None) and relative_steps:
        args.step = np.array(
            [np.round(args.side[i] * args.step[i]) for i in range(len(args.step))]
        ).astype("int")
        args.overlap = np.array(
            [np.round(args.side[i] * args.overlap[i]) for i in range(len(args.overlap))]
        ).astype("int")

    return args


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    args.version = __version__

    assert os.path.isfile(args.input), "input file not found: %s" % args.input
    assert not os.path.isfile(
        args.outdir
    ), f"output directory cannot be a file: {args.outdir}"

    if 1 == len(args.side):
        args.side = (args.side[0], args.side[0])
    else:
        args.side = args.side[:2]

    if args.slice is not None:
        assert args.slice > 0
    assert not (
        args.step is not None and args.overlap is not None
    ), "-S and -O are incompatible"

    args = check_step_and_overlap(args)

    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    return args


def enlarge_XY_tiff(img: np.ndarray, offset: List[int]) -> np.ndarray:
    new_shape = list(img.shape)
    new_shape[-1] += int(offset[0])
    new_shape[-2] += int(offset[1])
    new_image = img.copy()
    new_image = np.zeros(new_shape)
    new_image[np.ix_(*[range(img.shape[i]) for i in range(len(img.shape))])] = img
    return new_image


def get_pixel_loss(
    img: np.ndarray, side: List[int], step: List[float]
) -> Tuple[int, ...]:
    N = len(img.shape)
    assert len(side) <= N

    if step is None:
        missed = [img.shape[-i - 1] % side[i] for i in range(len(side))]
    else:
        assert len(side) == len(step)
        missed = [img.shape[-i - 1] % side[i] % step[i] for i in range(len(side))]

    lost_parts = []
    for i in range(len(side)):
        otherd = [img.shape[j] for j in range(N) if not N - i - 1 == j]
        otherd.append(missed[-i - 1])
        lost_parts.append(np.prod(otherd))
    loss = int(np.sum(lost_parts) - np.prod(img.shape[:-2]) * np.prod(missed))

    return (*missed, loss, loss / np.prod(img.shape) * 100)


def init_xy(
    img: np.ndarray, step: List[int], side: List[int], inverted: bool = False
) -> Iterable[Tuple[int, int]]:
    ys = [y for y in range(0, img.shape[-2], step[1]) if y + side[1] <= img.shape[-2]]
    xs = [x for x in range(0, img.shape[-1], step[0]) if x + side[0] <= img.shape[-1]]

    if inverted:
        logging.info("Image split top-to-bottom, left-to-right.")
        xy_gen = ((x, y) for x in xs for y in ys)
    else:
        logging.info("Image split left-to-right, top-to-bottom.")
        xy_gen = ((x, y) for y in ys for x in xs)

    return xy_gen


def tsplit3d(img: np.ndarray, x: int, y: int, s: List[int]) -> np.ndarray:
    return img[:, y : (y + s[1]), x : (x + s[0])]


def tsplit2d(img: np.ndarray, x: int, y: int, s: List[int]) -> np.ndarray:
    return img[y : (y + s[1]), x : (x + s[0])]


tsplit_fun = {2: tsplit2d, 3: tsplit3d}


def tiff_split(
    img: np.ndarray, side: List[int], step: List[int], inverted: bool = False
) -> np.ndarray:
    n = (img.shape[-1] // side[0]) * (img.shape[-2] // side[1])
    logging.info(f"Output {n} images.")
    assert 0 != n

    xy_gen = init_xy(img, step, side, inverted)

    if not len(img.shape) in [2, 3]:
        logging.error("cannot split images with more than 3 dimensions.")
        raise ValueError

    for (x_start, y_start) in track(xy_gen):
        yield tsplit_fun[len(img.shape)](img, x_start, y_start, side)

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
    if clear:
        settings_string = f"\033[H\033[J{settings_string}"
    print(settings_string)


def save_settings(args: argparse.Namespace) -> None:
    config = cp.ConfigParser()
    config["MAIN"] = dict(input=args.input, outdir=args.outdir)
    config["ADVANCED"] = dict(
        x_side=str(args.side[0]),
        y_side=str(args.side[1]),
        slice=str(args.slice),
        enlarge=str(args.enlarge),
        inverted=str(args.inverted),
    )
    if args.step is not None:
        config["ADVANCED"].update(
            dict(x_step=str(args.step[0]), y_step=str(args.step[1]))
        )

    with open(os.path.join(args.outdir, "config.ini"), "w") as CF:
        config.write(CF)


def confirm_arguments(args: argparse.Namespace) -> None:
    print_settings(args)
    if not args.do_all:
        assert Confirm.ask("Confirm settings and proceed?")
    save_settings(args)


def enlarge_image(args: argparse.Namespace, img: np.ndarray, umes: str) -> np.ndarray:
    x_loss, y_loss, loss, perc_loss = get_pixel_loss(img, args.side, args.step)

    if args.enlarge:
        img = enlarge_XY_tiff(img, np.array(args.side) - np.array((x_loss, y_loss)))
        logging.info(f"Image enlarged to {img.shape}")
    else:
        logging.info(f"{x_loss} {umes}s (X) and {y_loss} {umes}s (Y) are lost.")
        logging.info(
            f"In total, {loss} {umes}s lost ({perc_loss}%). " + "Use -e to avoid loss."
        )

    return img


def run(args: argparse.Namespace) -> None:
    confirm_arguments(args)

    logging.info("Reading input image...")
    img = imt.Image.from_tiff(args.input).pixels

    if 3 == len(img.shape):
        logging.info(f"3D stack found: {img.shape}")
        if args.slice is not None:
            logging.info(f"Enforcing 2D split (slice #{args.slice} only).")
            umes = "pixel"
            assert args.slice <= img.shape[0]
            img = img[args.slice - 1, :, :].copy()
        else:
            umes = "voxel"
    elif 2 == len(img.shape):
        logging.info(f"2D image found: {img.shape}")
        umes = "pixel"
    else:
        logging.error(f"cannot split a 1D image. File: {args.input}")
        sys.exit()

    img = enlarge_image(args, img, umes)

    prefix = os.path.splitext(os.path.basename(args.input))[0]
    ext = os.path.splitext(os.path.basename(args.input))[1]

    image_counter = 1
    for sub_image in tiff_split(img, args.side, args.step, args.inverted):
        opath = os.path.join(args.outdir, f"{prefix}.sub{image_counter}{ext}")
        imt.save_tiff(
            opath, sub_image, compressed=False, dtype=imt.get_dtype(sub_image.max())
        )
        image_counter += 1
