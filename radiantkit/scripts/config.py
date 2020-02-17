'''
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
'''

import argparse
from ggc.prompt import ask  # type: ignore
import os
from radiantkit.const import __version__
import sys


AVAILABLE_SHELLS = ("bash", "zsh", "tcsh", "fish")


def init_parser(subparsers: argparse._SubParsersAction
                ) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split('.')[-1], description=f'''''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Set config settings for RadIAnTkit.")
    parser.add_argument(
        '--version', action='version', version=f'{sys.argv[0]} {__version__}')

    nested = parser.add_subparsers(title="Config fields", help='')

    autocomplete = nested.add_parser(
        "autocomplete", help="turn on autocompletion for radiant scripts")
    required = autocomplete.add_argument_group("required arguments")
    required.add_argument(
        '--shell-type', metavar="STRING", type=str, help=f'''which shell to
        turn on autocompletion for. Available shells: {AVAILABLE_SHELLS}''',
        choices=AVAILABLE_SHELLS, required=True)
    autocomplete.add_argument(
        '--version', action='version', version=f'{sys.argv[0]} {__version__}')
    autocomplete.set_defaults(parse=parse_arguments, run=run_autocomplete)

    autocomplete_global = nested.add_parser(
        "autocomplete-global",
        help="turn on global autocompletion for radiant scripts")
    required = autocomplete_global.add_argument_group("required arguments")
    autocomplete_global.add_argument(
        '--version', action='version', version=f'{sys.argv[0]} {__version__}')
    autocomplete_global.set_defaults(
        parse=parse_arguments, run=run_autocomplete_global)

    return parser


def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    return args


def print_argcomplete_disclaimer() -> None:
    print("""NOTE: autocompeltion requires the 'argcomplete' package to be
          installed for the current user to properly work.""")


def run_autocomplete(args: argparse.Namespace) -> None:
    if args.shell_type == "bash":
        CMD = 'eval "$(register-python-argcomplete radiant)"'
        CONFIG = "~/.bashrc"
    elif args.shell_type == "zsh":
        CMD = 'autoload -U bashcompinit;bashcompinit;'
        CMD += 'eval "$(register-python-argcomplete radiant)"'
        CONFIG = "~/.zshrc"
    elif args.shell_type == "tcsh":
        CMD = "eval `register-python-argcomplete --shell tcsh radiant`"
        CONFIG = "~/.tcshrc"
    elif args.shell_type == "fish":
        CMD = "register-python-argcomplete --shell fish radiant | ."
        CONFIG = "~/.config/fish/config.fish"
    CONFIG = os.path.expanduser(CONFIG)

    print(f"Command: {CMD}")
    ask("Do you want to automatically add the aforementioned command "
        + f"to your '{CONFIG}' file?")
    assert os.path.isfile(CONFIG), f"'{CONFIG}' file not found"

    with open(CONFIG, "a+") as CH:
        CH.write("\n# >>> RadIAnTkit autocompletion >>>")
        CH.write(f"\n{CMD}")
        CH.write("\n# <<< RadIAnTkit autocompletion <<<")

    print(f"Restart your '{args.shell_type}' shell to use autocompletion.")

    print_argcomplete_disclaimer()


def run_autocomplete_global(args: argparse.Namespace) -> None:
    ask("This will turn on autocompletion for the curren use. Confirm?")
    CMD = "activate-global-python-argcomplete --user"
    print(f"Executing: {CMD}")
    os.system(CMD)

    print_argcomplete_disclaimer()
