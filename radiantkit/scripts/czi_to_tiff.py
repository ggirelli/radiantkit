'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import logging
import os
from radiantkit.const import __version__
from radiantkit.conversion import CziFile2
import radiantkit.image as imt
from radiantkit.string import MultiRange
from radiantkit.string import TIFFNameTemplateFields as TNTFields
from radiantkit.string import TIFFNameTemplate as TNTemplate
import sys
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s ' +
    '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

def init_parser(subparsers: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(__name__.split('.')[-1], description = f'''
Convert a czi file into single channel tiff images.

The output tiff file names follow the specified template (-T). A template is a
string including a series of "seeds" that are replaced by the corresponding
values when writing the output file. Available seeds are:
{TNTFields.CHANNEL_NAME} : channel name, lower-cased.
{TNTFields.CHANNEL_ID} : channel ID (number).
{TNTFields.SERIES_ID} : series ID (number).
{TNTFields.DIMENSIONS} : number of dimensions, followed by "D".
{TNTFields.AXES_ORDER} : axes order (e.g., "TZYX").
Leading 0s are added up to 3 digits to any ID seed.

The default template is "{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}".
Hence, when writing the 3rd series of the "a488" channel, the output file name
would be:"a488_003.tiff".

Please, remember to escape the "$" when running from command line if using
double quotes, i.e., "\\$". Alternatively, use single quotes, i.e., '$'.''',
        formatter_class = argparse.RawDescriptionHelpFormatter,
        help = f"{__name__.split('.')[-1]} -h")

    parser.add_argument('input', type = str,
        help = '''Path to the czi file to convert.''')

    parser.add_argument('-o', '--outdir', metavar = "outdir", type = str,
        help = """Path to output TIFF folder, created if missing. Default to a
        folder with the input file basename.""", default = None)

    parser.add_argument('-T', '--template', metavar = "template", type = str,
        help = """Template for output file name. See main description for more
        details. Default: '{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}'""",
        default = f"{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}")
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

    parser.add_argument('--version', action = 'version',
        version = f'{sys.argv[0]} {__version__}')

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser

def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    if args.outdir is None:
        args.outdir = os.path.splitext(os.path.basename(args.input))[0]
        args.outdir = os.path.join(os.path.dirname(args.input), args.outdir)

    assert os.path.isfile(args.input), f"input file not found: {args.input}"
    assert not os.path.isfile(args.outdir
        ), f"output directory cannot be a file: {args.outdir}"

    if args.fields is not None:
        args.fields = MultiRange(args.fields)
        args.fields.zero_indexed = True

    if args.channels is not None:
        args.channels = [c.lower() for c in args.channels]

    assert 0 != len(args.template)
    args.template = TNTemplate(args.template)

    return args

def run(args: argparse.Namespace) -> None:
    CZI = CziFile2(args.input)
    assert not CZI.isLive(), "time-course conversion images not implemented."
    CZI.log_details()
    if args.dry: sys.exit()

    if not args.template.can_export_fields(CZI.field_count(), args.fields):
        logging.critical("when exporting more than 1 field, the template " +
            f"must include the {TNTFields.SERIES_ID} seed. " +
            f"Got '{args.template.template}' instead.")
        sys.exit()

    logging.info(f"Output directory: '{args.outdir}'")
    if not os.path.isdir(args.outdir): os.mkdir(args.outdir)

    if args.channels is not None:
        args.channels = CZI.select_channels(args.channels)
        if 0 == len(args.channels):
            logging.error("None of the specified channels was found.")
            sys.exit()
        logging.info(f"Converting only the following channels: {args.channels}")

    if not args.template.can_export_channels(
        CZI.channel_count(), args.channels):
        logging.critical("when exporting more than 1 channel, the template " +
            f"must include either {TNTFields.CHANNEL_ID} or " +
            f"{TNTFields.CHANNEL_NAME} seeds. " +
            f"Got '{args.template.template}' instead.")
        sys.exit()

    CZI.squeeze_axes("SCZYX")

    if 1 == CZI.field_count(): CZI.reorder_axes("CZYX")
    else: CZI.reorder_axes("SCZYX")

    if args.fields is not None:
        args.fields = list(args.fields)
        logging.info("Converting only the following fields: " +
            f"{[x for x in args.fields]}")

    def field_generator(CZI: CziFile2, fields, channels):
        if fields is None: fields = range(CZI.field_count())
        if channels is None: channels = list(CZI.get_channel_names())
        for field_id in fields:
            if field_id-1 >= CZI.field_count():
                logging.warning(f"Skipped field #{field_id}" +
                    "(from specified field range, not available in czi file).")
                continue
            for yieldedValue in CZI.get_channel_pixels(args, field_id-1):
                channel_pixels, channel_id = yieldedValue
                if not list(CZI.get_channel_names())[channel_id ] in channels:
                    continue
                yield (channel_pixels,
                    CZI.get_tiff_path(args.template, channel_id, field_id-1))

    export_total = float('inf')
    if args.fields is not None and args.channels is not None:
        export_total = len(args.fields)*len(args.channels)
    elif args.fields is not None:
            export_total = len(args.fields)
    elif args.channels is not None:
            export_total = len(args.channels)
    export_total = min(CZI.field_count()*CZI.channel_count(), export_total)
    for (OI, opath) in tqdm(field_generator(CZI, args.fields, args.channels),
        total=export_total):
        imt.save_tiff(os.path.join(args.outdir, opath),
            OI, imt.get_dtype(OI.max()), args.doCompress, bundle_axes = "TZYX",
            resolution = (1e-6/CZI.get_axis_resolution("X"),
                1e-6/CZI.get_axis_resolution("Y")),
            inMicrons = True, ResolutionZ = CZI.get_axis_resolution("Z")*1e6)
