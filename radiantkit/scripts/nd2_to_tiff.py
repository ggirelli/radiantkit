'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import logging
import os
import pims
from radiantkit.conversion import ND2Reader2
import radiantkit.image as imt
from radiantkit.string import MultiRange
from string import Template
import sys
from tqdm import tqdm
from typing import List

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = '''
Convert a nd2 file into single channel tiff images. In the case of 3+D images,
the script also checks for consistent deltaZ distance across consecutive 2D
slices (i.e., dZ). If the distance is consitent, it is used to set the tiff
image dZ metadata. Otherwise, the script stops. Use the -Z argument to disable
this check and provide a single dZ value to be used.

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
        help = '''Path to the nd2 file to convert.''')

    parser.add_argument('-o', '--outdir', metavar = "outdir", type = str,
        help = """Path to output TIFF folder, created if missing. Default to a
        folder with the input file basename.""", default = None)
    parser.add_argument('-Z', '--deltaZ', type = float, metavar = 'dZ',
        help = """If provided (in um), the script does not check delta Z
        consistency and instead uses the provided one.""", default = None)
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

def get_output_path(args: argparse.Namespace, bundle_axes: List[str],
    metadata: dict, channel_id: int, field_id: int) -> str:
    d = {
        'channel_name' : metadata['channels'][channel_id].lower(),
        'channel_id' : f"{(channel_id+1):03d}",
        'series_id' : f"{(field_id+1):03d}",
        'dimensions' : len(bundle_axes),
        'axes_order' : "".join(bundle_axes)
    }
    t = Template(args.template)
    return f"{t.safe_substitute(d)}.tiff"

def export_channel(args: argparse.Namespace, field_of_view: pims.frame.Frame,
    opath: str, metadata: dict, bundle_axes: List[str],
    resolutionZ: float = None) -> None:
    resolutionXY = (1/metadata['pixel_microns'], 1/metadata['pixel_microns'])
    imt.Image.save_tiff(os.path.join(args.outdir, opath), field_of_view,
        imt.get_dtype(field_of_view.max()), args.doCompress,
        bundled_axes = "".join(bundle_axes).upper(), resolution = resolutionXY,
        inMicrons = True, forImageJ = True, ResolutionZ = resolutionZ)

def export_field_3d(args: argparse.Namespace, field_of_view: pims.frame.Frame,
    metadata: dict, field_id: int, bundle_axes: List[str],
    channels: List[str] = None) -> None:
    if args.deltaZ is not None: resolutionZ = args.deltaZ
    else: resolutionZ = ND2Reader2.get_resolutionZ(args.input, field_id)

    if "c" not in bundle_axes:
        opath = get_output_path(args, bundle_axes, metadata, 0, field_id)
        export_channel(args, field_of_view, opath, metadata,
            bundle_axes, resolutionZ)
    else:
        if channels is None:
            channels = [c.lower() for c in metadata['channels']]
        for channel_id in range(field_of_view.shape[3]):
            if not metadata['channels'][channel_id].lower() in channels:
                continue
            opath = get_output_path(args, bundle_axes,
                metadata, channel_id, field_id)
            export_channel(args, field_of_view[:, :, :, channel_id], opath,
                metadata, bundle_axes, resolutionZ)

def export_field_2d(args: argparse.Namespace, field_of_view: pims.frame.Frame,
    metadata: dict, field_id: int, bundle_axes: List[str],
    channels: List[str] = None) -> None:
    if "c" not in bundle_axes:
        opath = get_output_path(args, bundle_axes, metadata, 0, field_id)
        export_channel(args, field_of_view, opath, metadata, bundle_axes)
    else:
        if channels is None:
            channels = [c.lower() for c in metadata['channels']]
        for channel_id in range(field_of_view.shape[3]):
            if not metadata['channels'][channel_id].lower() in channels:
                continue
            opath = get_output_path(args, metadata, channel_id, field_id)
            export_channel(args, field_of_view[:, :, channel_id],
                opath, metadata, bundle_axes)

def run(args: argparse.Namespace) -> None:
    if args.deltaZ is not None:
        logging.info(f"Enforcing a deltaZ of {args.deltaZ:.3f} um.")

    nd2I = ND2Reader2(args.input)
    assert not nd2I.isLive(), "time-course conversion images not implemented."
    logging.info(f"Found {nd2I.field_count()} field(s) of view, " +
        f"with {nd2I.channel_count()} channel(s).")
    logging.info(f"Channels: {list(nd2I.get_channel_names())}.")
    if nd2I.is3D: logging.info("XYZ size: " + 
        f"{nd2I.sizes['x']} x {nd2I.sizes['y']} x {nd2I.sizes['z']}")
    else: logging.info(f"XY size: {nd2I.sizes['x']} x {nd2I.sizes['y']}")
    if args.dry: sys.exit()

    if args.channels is not None:
        args.channels = [c for c in args.channels
            if c in list(nd2I.get_channel_names())]
        if 0 == len(args.channels):
            logging.error("None of the specified channels was found.")
            sys.exit()
        logging.info(f"Converting only the following channels: {args.channels}")

    export_fn = export_field_3d if nd2I.is3D() else export_field_2d
    if 1 == nd2I.field_count():
        if args.fields is not None:
            if not 1 in list(args.fields):
                logging.warning("Skipped only available field " +
                    "(not included in specified field range.")
        nd2I.set_axes_for_bundling()
        export_fn(args, nd2I[0], nd2I.metadata, 0, nd2I.bundle_axes,
            args.channels)
    else:
        nd2I.iter_axes = 'v'
        nd2I.set_axes_for_bundling()

        if args.fields is not None:
            args.fields = list(args.fields)
            logging.info("Converting only the following fields: " +
                f"{[x+1 for x in args.fields]}")
            field_generator = tqdm(args.fields)
        else: field_generator = tqdm(range(nd2I.sizes['v']))

        for field_id in field_generator:
            if field_id >= nd2I.field_count():
                logging.warning(f"Skipped field #{field_id+1}" +
                    "(from specified field range, not available in nd2 file).")
            else:
                export_fn(args, nd2I[field_id], nd2I.metadata,
                    field_id, nd2I.bundle_axes, args.channels)

def main():
    run(parse_arguments())
