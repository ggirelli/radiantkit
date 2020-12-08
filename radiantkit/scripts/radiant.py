# PYTHON_ARGCOMPLETE_OK

"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

# import argcomplete  # type: ignore
import argparse
from radiantkit.const import __version__
from radiantkit import scripts
import sys


def default_parser(*args) -> None:
    print("radiant -h for usage details.")
    sys.exit()


def main():
    parser = argparse.ArgumentParser(
        description=f"""
Version:    {__version__}
Author:     Gabriele Girelli
Docs:       http://ggirelli.github.io/radiantkit
Code:       http://github.com/ggirelli/radiantkit

Radial Image Analisys Toolkit (RadIAnTkit) is a Python3.6+ package containing
tools for radial analysis of microscopy image.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(parse=default_parser)
    parser.add_argument(
        "--version", action="version", version=f"{sys.argv[0]} {__version__}"
    )

    subparsers = parser.add_subparsers(
        title="sub-commands",
        help="Access the help page for a sub-command with: sub-command -h",
    )

    # scripts.config.init_parser(subparsers)
    scripts.czi_to_tiff.init_parser(subparsers)
    scripts.nd2_to_tiff.init_parser(subparsers)

    scripts.select_nuclei.init_parser(subparsers)
    scripts.export_objects.init_parser(subparsers)
    scripts.measure_objects.init_parser(subparsers)

    scripts.radial_population.init_parser(subparsers)
    scripts.radial_object.init_parser(subparsers)
    scripts.radial_trajectory.init_parser(subparsers)
    scripts.radial_points.init_parser(subparsers)

    scripts.tiff_findoof.init_parser(subparsers)
    scripts.tiff_segment.init_parser(subparsers)
    scripts.tiff_desplit.init_parser(subparsers)
    scripts.tiff_split.init_parser(subparsers)
    scripts.tiffcu.init_parser(subparsers)

    scripts.pipeline.init_parser(subparsers)
    scripts.report.init_parser(subparsers)

    # argcomplete.autocomplete(parser)
    args = parser.parse_args()
    args = args.parse(args)
    args.run(args)
