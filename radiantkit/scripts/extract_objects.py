'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse as argp
import ggc
import joblib
import logging as log
import os
from radiantkit.const import __version__, default_inreg
from radiantkit import image
from radiantkit.series import Series, SeriesList
from radiantkit.particle import Nucleus
import re
import sys
from tqdm import tqdm

log.basicConfig(level=log.INFO, format='%(asctime)s ' +
    '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

def init_parser(subparsers: argp._SubParsersAction
    ) -> argp.ArgumentParser:
    parser = subparsers.add_parser(__name__.split(".")[-1], description = '''
Extract data of objects from masks.
''', formatter_class = argp.RawDescriptionHelpFormatter,
        help = "Extract data of objects from masks.")

    parser.add_argument('input', type=str,
        help='Path to folder containing deconvolved tiff images and masks.')
    parser.add_argument('ref_channel', type=str,
        help='Name of channel with masks to be used.')

    parser.add_argument('--version', action='version',
        version='%s %s' % (sys.argv[0], __version__,))

    critical = parser.add_argument_group("critical arguments")
    critical.add_argument('--aspect', type=float, nargs=3, help="""Physical size
        of Z, Y and X voxel sides in nm. Default: 300.0 216.6 216.6""",
        metavar=('Z','Y','X'), default=[300., 216.6, 216.6])
    critical.add_argument('--mask-prefix', type=str, metavar="TEXT",
        help="""Prefix for output binarized images name.
        Default: ''.""", default='')
    critical.add_argument('--mask-suffix', type=str, metavar="TEXT",
        help="""Suffix for output binarized images name.
        Default: 'mask'.""", default='mask')

    output = parser.add_argument_group("output mode")
    output.add_argument('--output', type=str,
        help='''Path to folder where output should be written to.
            Defaults to "objects" subfolder in the input directory.''')
    output.add_argument('--no-feature-export',
        action='store_const', dest='export_features',
        const=False, default=True,
        help='Skip calculation and export of table with object features.')
    output.add_argument('--no-tiff-export',
        action='store_const', dest='export_tiffs',
        const=False, default=True,
        help='Skip export of channel and mask tiffs.')

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument('--block-side', type=int, metavar="NUMBER",
        help="""Structural element side for dilation-based background/foreground
        measurement. Should be odd. Default: 11.""", default=11)
    advanced.add_argument('--use-labels',
        action='store_const', dest='labeled',
        const=True, default=False,
        help='Use labels from masks instead of relabeling.')
    advanced.add_argument('--no-rescaling',
        action='store_const', dest='do_rescaling',
        const=False, default=True,
        help='Do not rescale image even if deconvolved.')
    advanced.add_argument('--uncompressed',
        action='store_const', dest='compressed',
        const=False, default=True,
        help='Generate uncompressed TIFF binary masks.')
    advanced.add_argument('--inreg', type=str, metavar="REGEXP",
        help=f"""Regular expression to identify input TIFF images.
        Must contain 'channel_name' and 'series_id' fields.
        Default: '{default_inreg}'""", default=default_inreg)
    advanced.add_argument('-t', type=int, metavar="NUMBER", dest="threads",
        help="""Number of threads for parallelization. Default: 1""",
        default=1)
    advanced.add_argument('-y', '--do-all', action='store_const',
        help="""Do not ask for settings confirmation and proceed.""",
        const=True, default=False)

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser

def parse_arguments(args: argp.Namespace) -> argp.Namespace:
    args.version = __version__

    if args.output is None:
        args.output = os.path.join(args.input, 'objects')
    assert not os.path.isfile(args.output)
    if not os.path.isdir(args.output): os.mkdir(args.output)

    assert '(?P<channel_name>' in args.inreg
    assert '(?P<series_id>' in args.inreg
    args.inreg = re.compile(args.inreg)

    if 0 != len(args.mask_prefix):
        if '.' != args.mask_prefix[-1]:
            args.mask_prefix = f"{args.mask_prefix}."
    if 0 != len(args.mask_suffix):
        if '.' != args.mask_suffix[0]:
            args.mask_suffix = f".{args.mask_suffix}"

    if not 0 != args.block_side%2:
        log.warning("changed ground block side from " +
            f"{args.block_side} to {args.block_side+1}")
        args.block_side += 1

    args.threads = ggc.args.check_threads(args.threads)

    return args

def print_settings(args: argp.Namespace, clear: bool = True) -> str:
    s = f"""# Object extraction v{args.version}

    ---------- SETTING : VALUE ----------

       Input directory : '{args.input}'
      Output directory : '{args.output}'
Reference channel name : '{args.ref_channel}'
    Voxel aspect (ZYX) : {args.aspect}

           Mask prefix : '{args.mask_prefix}'
           Mask suffix : '{args.mask_suffix}'

  Export feature table : {args.export_features}
   Export object tiffs : {args.export_tiffs}

     Ground block side : {args.block_side}
            Use labels : {args.labeled}
               Rescale : {args.do_rescaling}
            Compressed : {args.compressed}

               Threads : {args.threads}
                Regexp : {args.inreg.pattern}
    """
    if clear: print("\033[H\033[J")
    print(s)
    return(s)

def confirm_arguments(args: argp.Namespace) -> None:
    settings_string = print_settings(args)
    if not args.do_all: ggc.prompt.ask("Confirm settings and proceed?")

    assert os.path.isdir(args.input), f"image folder not found: {args.input}"

    settings_path = os.path.join(args.output, "extract_objects.config.txt")
    with open(settings_path, "w+") as OH:
        ggc.args.export_settings(OH, settings_string)

def run(args: argp.Namespace) -> None:
    confirm_arguments(args)

    series_list = SeriesList.from_directory(args.input, args.inreg,
        args.ref_channel, (args.mask_prefix, args.mask_suffix), args.aspect)
    log.info(f"parsed {len(series_list)} series with " +
        f"{len(series_list.channel_names)} channels each" +
        f": {series_list.channel_names}")
    for series in series_list: series.labeled = args.labeled
    for series in series_list: series.ground_bloc_side = args.block_side

    if not args.export_tiffs and not args.export_features:
        log.info("Nothing to export when using both " +
            "--no-tiff-export and no-feature-export flags.")
        sys.exit()

    log.info(f"extracting nuclei")
    if 1 == args.threads:
        series_list = [Series.extract_particles(s, s.channel_names, Nucleus)
            for s in tqdm(series_list)]
    else:
        series_list = joblib.Parallel(n_jobs=args.threads, verbose=11)(
            joblib.delayed(Series.extract_particles
                )(s, s.channel_names, Nucleus) for s in series_list)

    if args.export_features:
        feat_path = os.path.join(args.output, "nuclear_features.tsv")
        log.info(f"exporting nuclear features to '{feat_path}'")
        for series in series_list:
            for nucleus in series.particles:
                print((nucleus.label, nucleus.mask, nucleus.region_of_interest,
                    nucleus.total_size, nucleus.surface,
                    [(cn, nucleus.get_intensity_sum(cn),
                        nucleus.get_intensity_mean(cn))
                        for cn in nucleus.channel_names]))

    if args.export_tiffs:
        tiff_path = os.path.join(args.output, "tiff")
        assert not os.path.isfile(tiff_path)
        if not os.path.isdir(tiff_path): os.mkdir(tiff_path)

        log.info(f"exporting nuclei images to '{tiff_path}'")
        if 1 == args.threads:
            for series in tqdm(series_list, desc="series"):
                series.export_particles(tiff_path, args.compressed)
        else:
            joblib.Parallel(n_jobs=args.threads, verbose=11)(
                joblib.delayed(Series.static_export_particles)(
                    s, tiff_path, args.compressed) for s in series_list)
