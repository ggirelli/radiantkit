'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
import radiantkit as ra
import sys

def default_parser(*args) -> None:
    print("radiant -h for usage details.")
    sys.exit()

def main():
    parser = argparse.ArgumentParser(description = f'''
Version:    {ra.const.__version__}
Author:     Gabriele Girelli
Docs:       http://ggirelli.github.io/radiantkit
Code:       http://github.com/ggirelli/radiantkit

Radial Image Analisys Toolkit (RadIAnTkit) is a Python3.6+ package containing
tools for radial analysis of microscopy image.
''',
        formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.set_defaults(parse=default_parser)
    parser.add_argument('--version', action = 'version',
        version = f'{sys.argv[0]} {ra.const.__version__}')

    subparsers = parser.add_subparsers(title="sub-commands",
        help='Access the help page for a sub-command with: sub-command -h')
    
    ra.scripts.czi_to_tiff.init_parser(subparsers)
    ra.scripts.nd2_to_tiff.init_parser(subparsers)

    ra.scripts.analyze_yfish.init_parser(subparsers)
    ra.scripts.select_nuclei.init_parser(subparsers)

    ra.scripts.tiff_findoof.init_parser(subparsers)
    ra.scripts.tiff_segment.init_parser(subparsers)
    ra.scripts.tiff_split.init_parser(subparsers)
    ra.scripts.tiffcu.init_parser(subparsers)

    args = parser.parse_args()
    args = args.parse(args)
    args.run(args)
