'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from radiantkit.const import __version__
from radiantkit import scripts
from sys import exit, argv as sys_argv

def default_parser(*args) -> None:
    print("radiant -h for usage details.")
    exit()

def main():
    parser = ArgumentParser(description = f'''
Version:    {__version__}
Author:     Gabriele Girelli
Docs:       http://ggirelli.github.io/radiantkit
Code:       http://github.com/ggirelli/radiantkit

Radial Image Analisys Toolkit (RadIAnTkit) is a Python3.6+ package containing
tools for radial analysis of microscopy image.
''',
        formatter_class = RawDescriptionHelpFormatter)
    parser.set_defaults(parse=default_parser)
    parser.add_argument('--version', action = 'version',
        version = f'{sys_argv[0]} {__version__}')

    subparsers = parser.add_subparsers(title="sub-commands",
        help='Access the help page for a sub-command with: sub-command -h')
    
    scripts.czi_to_tiff.init_parser(subparsers)
    scripts.nd2_to_tiff.init_parser(subparsers)

    scripts.analyze_yfish.init_parser(subparsers)
    scripts.select_nuclei.init_parser(subparsers)
    scripts.extract_objects.init_parser(subparsers)

    scripts.tiff_findoof.init_parser(subparsers)
    scripts.tiff_segment.init_parser(subparsers)
    scripts.tiff_split.init_parser(subparsers)
    scripts.tiffcu.init_parser(subparsers)

    args = parser.parse_args()
    args = args.parse(args)
    args.run(args)
