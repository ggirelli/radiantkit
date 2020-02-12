'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import logging
import os
import pims  # type: ignore
from radiantkit.const import __version__
from radiantkit.conversion import ND2Reader2
import radiantkit.image as imt
from radiantkit.string import MultiRange
from radiantkit.string import TIFFNameTemplateFields as TNTFields
from radiantkit.string import TIFFNameTemplate as TNTemplate
import sys
from tqdm import tqdm  # type: ignore
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s '
    + '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')


def init_parser(subparsers: argparse._SubParsersAction
                ) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1], description=f'''
Convert a nd2 file into single channel tiff images.

In the case of 3+D images, the script also checks for consistent deltaZ
distance across consecutive 2D slices (i.e., dZ). If the distance is consitent,
it is used to set the tiff image dZ metadata. Otherwise, the script stops. Use
the -Z argument to disable this check and provide a single dZ value to be used.

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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Convert a nd2 file into single channel tiff images.")

    parser.add_argument(
        'input', type=str, help='''Path to the nd2 file to convert.''')

    parser.add_argument(
        '--outdir', metavar="DIRPATH", type=str,
        help="""Path to output TIFF folder. Defaults to the input file
        basename.""", default=None)
    parser.add_argument(
        '--fields', metavar="STRING", type=str,
        help="""Extract only fields of view specified as when printing a set
        of pages. E.g., '1-2,5,8-9'.""", default=None)
    parser.add_argument(
        '--channels', metavar="STRING", type=str,
        help="""Extract only specified channels. Specified as space-separated
        channel names. E.g., 'dapi cy5 a488'.""", default=None, nargs="+")

    parser.add_argument(
        '--version', action='version', version=f'{sys.argv[0]} {__version__}')

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument(
        '--deltaZ', type=float, metavar='FLOAT',
        help="""If provided (in um), the script does not check delta Z
        consistency and instead uses the provided one.""", default=None)
    advanced.add_argument(
        '--template', metavar="STRING", type=str,
        help=f"""Template for output file name. See main description for more
        details. Default: '{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}'""",
        default=f"{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}")
    advanced.add_argument(
        '--compressed', action='store_const', dest='doCompress',
        const=True, default=False,
        help='Write compressed TIFF as output.')
    advanced.add_argument(
        '-n', '--dry-run', action='store_const', dest='dry',
        const=True, default=False,
        help='Describe input data and stop.')

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:

    if args.outdir is None:
        args.outdir = os.path.splitext(os.path.basename(args.input))[0]
        args.outdir = os.path.join(os.path.dirname(args.input), args.outdir)

    assert os.path.isfile(args.input), f"input file not found: {args.input}"
    assert not os.path.isfile(args.outdir), (
        f"output directory cannot be a file: {args.outdir}")

    if args.fields is not None:
        args.fields = MultiRange(args.fields)
        args.fields.zero_indexed = True

    if args.channels is not None:
        args.channels = [c.lower() for c in args.channels]

    assert 0 != len(args.template)
    args.template = TNTemplate(args.template)

    return args


def export_channel(
        args: argparse.Namespace, field_of_view: pims.frame.Frame,
        opath: str, metadata: dict, resolutionZ: float = None) -> None:
    resolutionXY = (1/metadata['pixel_microns'], 1/metadata['pixel_microns'])
    imt.save_tiff(
        os.path.join(args.outdir, opath), field_of_view,
        imt.get_dtype(field_of_view.max()), args.doCompress,
        resolution=resolutionXY, inMicrons=True, ResolutionZ=resolutionZ)


def export_field_3d(
        args: argparse.Namespace, nd2I: ND2Reader2,
        field_id: int, channels: Optional[List[str]] = None) -> None:
    if channels is None:
        channels = list(nd2I.get_channel_names())
    if args.deltaZ is not None:
        resolutionZ = args.deltaZ
    else:
        resolutionZ = ND2Reader2.get_resolutionZ(args.input, field_id)
        assert 1 == len(resolutionZ), (
            f"Z resolution is not constant: {resolutionZ}")
        resolutionZ = list(resolutionZ)[0]

    try:
        if not nd2I.hasMultiChannels():
            export_channel(args, nd2I[field_id],
                           nd2I.get_tiff_path(args.template, 0, field_id),
                           nd2I.metadata, resolutionZ)
        else:
            channels = nd2I.select_channels(channels)
            for channel_id in range(nd2I[field_id].shape[3]):
                channel_name = nd2I.metadata['channels'][channel_id].lower()
                if channel_name not in channels:
                    continue
                export_channel(
                    args, nd2I[field_id][:, :, :, channel_id],
                    nd2I.get_tiff_path(args.template, channel_id, field_id),
                    nd2I.metadata, resolutionZ)
    except ValueError as e:
        if "could not broadcast input array from shape" in e.args[0]:
            logging.error(f"corrupted file raised {type(e).__name__}. "
                          + "At least one frame has mismatching shape.")
            logging.critical(f"{e.args[0]}")
            sys.exit()
        raise e


def export_field_2d(args: argparse.Namespace, nd2I: ND2Reader2,
                    field_id: int, channels: Optional[List[str]] = None
                    ) -> None:
    if channels is None:
        channels = list(nd2I.get_channel_names())

    try:
        if not nd2I.hasMultiChannels():
            export_channel(args, nd2I[field_id],
                           nd2I.get_tiff_path(args.template, 0, field_id),
                           nd2I.metadata)
        else:
            channels = nd2I.select_channels(channels)
            for channel_id in range(nd2I[field_id].shape[3]):
                channel_name = nd2I.metadata['channels'][channel_id].lower()
                if channel_name not in channels:
                    continue
                export_channel(
                    args, nd2I[field_id][:, :, channel_id],
                    nd2I.get_tiff_path(args.template, channel_id, field_id),
                    nd2I.metadata)
    except ValueError as e:
        if "could not broadcast input array from shape" in e.args[0]:
            logging.error(f"corrupted file raised {type(e).__name__}. "
                          + "At least one frame has mismatching shape.")
            logging.critical(f"{e.args[0]}")
            sys.exit()
        raise e


def run(args: argparse.Namespace) -> None:
    if args.deltaZ is not None:
        logging.info(f"Enforcing a deltaZ of {args.deltaZ:.3f} um.")

    nd2I = ND2Reader2(args.input)
    assert not nd2I.isLive(), "time-course conversion images not implemented."
    nd2I.log_details()
    if args.dry:
        sys.exit()

    if not args.template.can_export_fields(nd2I.field_count(), args.fields):
        logging.critical("when exporting more than 1 field, the template "
                         + f"must include the {TNTFields.SERIES_ID} seed. "
                         + f"Got '{args.template.template}' instead.")
        sys.exit()

    logging.info(f"Output directory: '{args.outdir}'")
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    if args.channels is not None:
        args.channels = nd2I.select_channels(args.channels)
        if 0 == len(args.channels):
            logging.error("None of the specified channels was found.")
            sys.exit()
        logging.info(
            f"Converting only the following channels: {args.channels}")

    if not args.template.can_export_channels(
            nd2I.channel_count(), args.channels):
        logging.critical("when exporting more than 1 channel, the template "
                         + f"must include either {TNTFields.CHANNEL_ID} or "
                         + f"{TNTFields.CHANNEL_NAME} seeds. "
                         + f"Got '{args.template.template}' instead.")
        sys.exit()

    export_fn = export_field_3d if nd2I.is3D() else export_field_2d
    if 1 == nd2I.field_count():
        if args.fields is not None:
            if 1 not in list(args.fields):
                logging.warning("Skipped only available field "
                                + "(not included in specified field range.")
        nd2I.set_axes_for_bundling()
        export_fn(args, nd2I, 0, args.channels)
    else:
        nd2I.iter_axes = 'v'
        nd2I.set_axes_for_bundling()

        if args.fields is not None:
            args.fields = list(args.fields)
            logging.info("Converting only the following fields: "
                         + f"{[x for x in args.fields]}")
            field_generator = tqdm(args.fields)
        else:
            field_generator = tqdm(range(1, nd2I.sizes['v']+1))

        for field_id in field_generator:
            if field_id-1 >= nd2I.field_count():
                logging.warning(f"Skipped field #{field_id}(from specified "
                                + "field range, not available in nd2 file).")
            else:
                export_fn(args, nd2I, field_id-1, args.channels)
