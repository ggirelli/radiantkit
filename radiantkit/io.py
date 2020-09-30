# PYTHON_ARGCOMPLETE_OK

"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import sys


def ask(q: str, elseAbort: bool = True) -> bool:
    answer = ""
    while not answer.lower() in ["y", "n"]:
        print(f"{q} (y/n)")
        answer = sys.stdin.readline().strip()
        if "n" == answer.lower():
            if elseAbort:
                sys.exit("Aborted.\n")
            else:
                return False
        elif "y" != answer.lower():
            print("Please, answer 'y' or 'n'.\n")
    return True
