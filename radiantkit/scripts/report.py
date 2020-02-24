'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
from enum import Enum
import logging
import os
from radiantkit import const, scripts
from radiantkit import path
import sys
from typing import Dict, List, Optional, Pattern

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s '
    + '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')


def init_parser(subparsers: argparse._SubParsersAction
                ) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split('.')[-1], description=f'''Long description''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Generate radiant report(s).")

    parser.add_argument(
        'input', type=str,
        help='''Path to folder with radiant output.''')

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument(
        '--subdir', type=str, metavar="STRING", default='objects',
        help=f"""Name of subfolder for nested search. Default: 'objects'""")
    advanced.add_argument(
        '--inreg', type=str, metavar="REGEXP",
        help=f"""Regular expression to identify input TIFF images.
        Must contain 'channel_name' and 'series_id' fields.
        Default: '{const.default_inreg}'""", default=const.default_inreg)

    parser.add_argument('--version', action='version',
                        version=f'{sys.argv[0]} {const.__version__}')

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    assert os.path.isdir(args.input)
    return args


class OutputType(Enum):
    SELECT_NUCLEI = "select_nuclei"
    MEASURE_OBJECTS = "measure_objects"
    RADIAL_POPULATION = "radial_population"


def output_in_folder(otype: OutputType, ipath: str) -> bool:
    if not os.path.isdir(ipath):
        return False
    flist = os.listdir(ipath)
    return all([oname in flist for oname in getattr(
            scripts, otype.value).__OUTPUT__])


def output_type_in_folder(
        otype: OutputType, ipath: str, subname: Optional[str] = None) -> bool:
    if output_in_folder(otype, ipath):
        return True

    if subname is not None:
        ipath = os.path.join(ipath, subname)
        if output_in_folder(otype, ipath):
            return True

    return False


def search_radiant_output_types(
        ipath: str, subname: str = "objects") -> List[OutputType]:
    return [otype for otype in OutputType
            if output_type_in_folder(otype, ipath, subname)]


def get_output_list(ipath: str) -> Optional[List[OutputType]]:
    output_list = search_radiant_output_types(ipath)
    if 0 == len(output_list):
        return None
    else:
        return output_list


def get_output_list_per_folder(
        ipath: str, inreg: Pattern) -> List[Dict[str, List[OutputType]]]:
    output_list = []
    if 0 == len(path.find_re(ipath, inreg)):
        subfolder_list = [f for f in os.scandir(ipath) if os.path.isdir(f)]
        for f in subfolder_list:
            fpath = f.path
            logging.info(f"looking into subfolder '{f.name}'")
            output = get_output_list(fpath)
            if output is not None:
                output_list.append({fpath: output})
    else:
        output = get_output_list(ipath)
        if output is not None:
            output_list.append({ipath: output})

    return output_list


def run(args: argparse.Namespace) -> None:
    logging.info(f"looking at '{args.input}'")
    output_list = get_output_list_per_folder(args.input, args.inreg)

    if len(OutputType) == len(output_list):
        pass
    else:
        pass

    raise NotImplementedError
