'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
from ggc.prompt import ask
import logging
import os
import re
import sys
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s ' +
    '[P%(process)s:%(module)s:%(funcName)s] %(levelname)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S')

def parse_arguments() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description='''
...
    ''', formatter_class=argparse.RawDescriptionHelpFormatter)

    version="0.0.1"
    parser.add_argument('--version', action='version',
        version='%s %s' % (sys.argv[0], version,))

    args = parser.parse_args()
    args.version = version

    return args

def print_settings(args: argparse.Namespace, clear: bool = True) -> str:
    s = f"""
# Nuclei selection v{args.version}

---------- SETTING :  VALUE ----------

    """
    if clear: print("\033[H\033[J")
    print(s)
    return(s)

def confirm_arguments(args: argparse.Namespace) -> None:
    settings_string = print_settings(args)
    if not args.do_all: ask("Confirm settings and proceed?")

def run(args: argparse.Namespace) -> None:
    pass

def main():
    args = parse_arguments()
    confirm_arguments(args)
    run(args)
