'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import logging
import os
from radiantkit.conversion import CziFile2
import radiantkit.image as imt
from radiantkit.string import MultiRange
from string import Template
import sys
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = '''
Convert a czi file into single channel tiff images.

The output tiff file names follow the specified template (-T). A template is a
string including a series of "seeds" that are replaced by the corresponding
values when writing the output file. Available seeds are:
${channel_name} : the channel name, lower-cased.
${channel_id}   : the channel ID (number). Leading 0s added up to 3 digits.
${series_id}    : the series ID (number). Leading 0s added up to 3 digits.
${dimensions}   : the number of dimensions in the image, followed by a "D".
${axes_order}   : the order of the axes in the image (e.g., "TZYX").
The default template is "${channel_name}_${series_id}". Hence, when writing the
3rd series of the "a488" channel, the output file name would be:"a488_003.tiff".
Please, remember to escape the "$" when running from command line if using
double quotes, i.e., "\\$". Alternatively, use single quotes, i.e., '$'.
    ''', formatter_class = argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input', type = str,
        help = '''Path to the czi file to convert.''')

    parser.add_argument('-o', '--outdir', metavar = "outdir", type = str,
        help = """Path to output TIFF folder, created if missing. Default to a
        folder with the input file basename.""", default = None)

    parser.add_argument('-T', '--template', metavar = "template", type = str,
        help = """Template for output file name. See main description for more
        details. Default: '${channel_name}_${series_id}'""",
        default = "${channel_name}_${series_id}")
    parser.add_argument('-f', '--fields', metavar = "fields", type = str,
        help = """Extract only specified fields of view. Can be specified as
        when specifying which pages to print. E.g., '1-2,5,8-9'.""",
        default = None)
    parser.add_argument('-c', '--channels', metavar = "channels", type = str,
        help = """Extract only specified channels. Should be specified as a list
        of space-separated channel names. E.g., 'dapi cy5 a488'.""",
        default = None, nargs = "+")

    parser.add_argument('-C', '--compressed',
        action = 'store_const', dest = 'doCompress',
        const = True, default = False,
        help = 'Force compressed TIFF as output.')
    parser.add_argument('-n', '--dry-run',
        action = 'store_const', dest = 'dry',
        const = True, default = False,
        help = 'Describe input data and stop.')

    version = "0.0.1"
    parser.add_argument('--version', action = 'version',
        version = f'{sys.argv[0]} {version}')

    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = os.path.splitext(os.path.basename(args.input))[0]
        args.outdir = os.path.join(os.path.dirname(args.input), args.outdir)
        logging.info(f"Output directory: '{args.outdir}'")

    assert os.path.isfile(args.input), f"input file not found: {args.input}"
    assert not os.path.isfile(args.outdir
        ), f"output directory cannot be a file: {args.outdir}"
    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    if args.fields is not None:
        args.fields = MultiRange(args.fields)
        args.fields.zero_indexed = True

    if args.channels is not None:
        args.channels = [c.lower() for c in args.channels]

    return args

def get_output_path(args: argparse.Namespace, CZI: CziFile2,
    channel_id: int, field_id: int = 1) -> str:
    d = {
        'channel_name' : list(CZI.get_channel_names())[channel_id],
        'channel_id' : f"{(channel_id+1):03d}",
        'series_id' : f"{(field_id+1):03d}",
        'dimensions' : 3,
        'axes_order' : "ZYX"
    }
    t = Template(args.template)
    return f"{t.safe_substitute(d)}.tiff"

def run(args: argparse.Namespace) -> None:
    CZI = CziFile2(args.input)
    assert not CZI.isLive(), "time-course conversion images not implemented."
    logging.info(f"Found {CZI.field_count()} field(s) of view, " +
        f"with {CZI.channel_count()} channel(s).")
    logging.info(f"Channels: {list(CZI.get_channel_names())}.")

    resolution = CZI.get_resolution()
    x_size = CZI.pixels.shape[CZI.axes.index("X")]
    y_size = CZI.pixels.shape[CZI.axes.index("Y")]
    if CZI.is3D:
        z_size = CZI.pixels.shape[CZI.axes.index("Z")]
        logging.info(f"XYZ size: {x_size} x {y_size} x {z_size}")
    else: logging.info(f"XY size: {x_size} x {y_size}")

    if args.dry: sys.exit()

    if args.channels is not None:
        args.channels = [c for c in args.channels
            if c in list(CZI.get_channel_names())]
        if 0 == len(args.channels):
            logging.error("None of the specified channels was found.")
            sys.exit()
        logging.info(f"Converting only the following channels: {args.channels}")

    CZI.squeeze_axes("SCZYX")

    if 1 == CZI.field_count(): CZI.reorder_axes("CZYX")
    else: CZI.reorder_axes("SCZYX")

    if args.fields is not None:
        args.fields = list(args.fields)
        logging.info("Converting only the following fields: " +
            f"{[x+1 for x in args.fields]}")

    def field_generator(CZI: CziFile2, fields, channels):
        if fields is None: fields = range(CZI.field_count())
        if channels is None: channels = list(CZI.get_channel_names())
        for field_id in fields:
            if field_id >= CZI.field_count():
                logging.warning(f"Skipped field #{field_id+1}" +
                    "(from specified field range, not available in czi file).")
                continue
            for yieldedValue in CZI.get_channel_pixels(args, field_id):
                channel_pixels, channel_id = yieldedValue
                if not list(CZI.get_channel_names())[channel_id ] in channels:
                    continue
                yield (channel_pixels,
                    get_output_path(args, CZI, channel_id, field_id))

    export_total = min(CZI.field_count()*CZI.channel_count(),
        len(list(args.fields))*len(args.channels))
    for (OI, opath) in tqdm(field_generator(CZI, args.fields, args.channels),
        total=export_total):
        imt.save_tiff(os.path.join(args.outdir, opath),
            OI, imt.get_dtype(OI.max()), args.doCompress, bundled_axes = "ZYX",
            resolution = (1e-6/resolution["X"], 1e-6/resolution["Y"]),
            inMicrons = True, ResolutionZ = resolution["Z"]*1e6,
            forImageJ = True)

def main():
    run(parse_arguments())
