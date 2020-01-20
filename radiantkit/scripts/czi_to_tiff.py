'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S')

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
3rd series of the "a488" channel, the output file name would be: "a488_003.tiff"
    ''', formatter_class = argparse.RawDescriptionHelpFormatter)

    parser.add_argument('input', type = str,
        help = '''Path to the czi file to convert.''')

    parser.add_argument('-o', '--outdir', metavar = "outdir", type = str,
        help = """Path to output TIFF folder, created if missing. Default to a
        folder with the input file basename.""", default = None)

    parser.add_argument('-T', '--template', metavar = "template", type = str,
        help = """Template for output file name. See main description for more
        details. Default: '${channel_name}_${series_id}'""", default = None)

    parser.add_argument('--compressed',
        action = 'store_const', dest = 'doCompress',
        const = True, default = False,
        help = 'Force compressed TIFF as output.')

    version = "0.0.1"
    parser.add_argument('--version', action = 'version',
        version = f'{sys.argv[0]} {version}')

    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = os.path.splitext(os.path.basename(args.input))[0]
        args.outdir = os.path.join(os.path.dirname(args.input), args.outdir)
        print(f"Output directory: '{args.outdir}'")

    assert os.path.isfile(args.input), f"input file not found: {args.input}"
    assert not os.path.isfile(args.outdir
        ), f"output directory cannot be a file: {args.outdir}"
    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    return args

def run(args: argparse.Namespace) -> None:
    pass

def main():
    args = parse_arguments()
