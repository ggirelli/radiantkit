"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import logging
import os
import sys

import numpy as np  # type: ignore


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
            logging.debug(f"Huygens rescaling factor: {huygens_factor}")
            return huygens_factor
    elif deconwolf_factor != 1:
        logging.debug(f"Deconwolf rescaling factor: {deconwolf_factor}")
        return deconwolf_factor
    else:
        logging.debug(f"no rescaling factor found for '{path}'.")
        return 1.0


def get_huygens_rescaling_factor(path: str) -> float:
    basename, ext = tuple(os.path.splitext(os.path.basename(path)))
    path = os.path.join(os.path.dirname(path), f"{basename}_history.txt")
    if not os.path.exists(path):
        logging.debug(f"no Huygens log found: '{path}'")
        return 1

    with open(path, "r") as log:
        factor = [x for x in log.readlines() if "Stretched to Integer type" in x]

    if not factor:
        return 1
    elif len(factor) == 1:
        return float(factor[0].strip().split(" ")[-1])
    else:
        return float(np.prod([float(f.strip().split(" ")[-1]) for f in factor]))


def get_deconwolf_rescaling_factor(path: str) -> float:
    path = os.path.join(os.path.dirname(path), f"{os.path.basename(path)}.log.txt")
    if not os.path.exists(path):
        logging.debug(f"no deconwolf log found: '{path}'")
        return 1

    with open(path, "r") as log:
        factor = [x for x in log.readlines() if "scaling: " in x]

    if not factor:
        return 1
    elif len(factor) == 1:
        return float(factor[0].strip().split(" ")[-1])
    else:
        return float(np.prod([float(f.strip().split(" ")[-1]) for f in factor]))
