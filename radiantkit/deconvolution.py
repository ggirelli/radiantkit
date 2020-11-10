"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import logging
import numpy as np  # type: ignore
import os
import sys


def get_deconvolution_rescaling_factor(path: str, verbose: bool = False) -> float:
    huygens_factor = get_huygens_rescaling_factor(path)
    deconwolf_factor = get_deconwolf_rescaling_factor(path)

    if huygens_factor != 1:
        if deconwolf_factor != 1:
            logging.critical(
                f"found deconwolf and Huygens rescaling factor for file '{path}'"
            )
            sys.exit()
        else:
            logging.info(f"Huygens rescaling factor: {huygens_factor}")
            return huygens_factor
    else:
        if deconwolf_factor != 1:
            logging.info(f"Deconwolf rescaling factor: {deconwolf_factor}")
            return deconwolf_factor
        else:
            logging.info(f"no rescaling factor found for '{path}'.")
            return 1.0


def get_huygens_rescaling_factor(path: str) -> float:
    basename, ext = tuple(os.path.splitext(os.path.basename(path)))
    path = os.path.join(os.path.dirname(path), f"{basename}_history.txt")
    if not os.path.exists(path):
        return 1

    with open(path, "r") as log:
        factor = [x for x in log.readlines() if "Stretched to Integer type" in x]

    if 0 == len(factor):
        return 1
    elif 1 == len(factor):
        return float(factor[0].strip().split(" ")[-1])
    else:
        return np.prod([float(f.strip().split(" ")[-1]) for f in factor])


def get_deconwolf_rescaling_factor(path: str) -> float:
    basename, ext = tuple(os.path.splitext(os.path.basename(path)))
    path = os.path.join(os.path.dirname(path), f"{basename}.log.txt")
    if not os.path.exists(path):
        logging.debug(f"no deconwolf log found: '{path}'")
        return 1

    with open(path, "r") as log:
        factor = [x for x in log.readlines() if "scaling: " in x]

    if 0 == len(factor):
        return 1
    elif 1 == len(factor):
        return float(factor[0].strip().split(" ")[-1])
    else:
        return np.prod([float(f.strip().split(" ")[-1]) for f in factor])
