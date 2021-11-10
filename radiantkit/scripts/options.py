"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click  # type: ignore


def reference():
    return click.option(
        "--reference", "-r", type=click.STRING, help="Name of reference channel."
    )


def threads():
    return click.option(
        "--threads",
        "-T",
        type=click.INT,
        help="Number of threads for parallelization.",
        default=1,
    )


def input_regexp(default: str):
    return click.option(
        "--input-regexp",
        "-R",
        type=click.STRING,
        metavar="RE",
        help=f"""
    Regexp used to identify input ND2 files.
    Default: {default}""",
        default=default,
    )


def mask_prefix(default: str = ""):
    return click.option(
        "--mask-prefix",
        type=click.STRING,
        help="Segmented file prefix",
        default=default,
    )


def mask_suffix(default: str = ""):
    return click.option(
        "--mask-suffix",
        type=click.STRING,
        help="Segmented file suffix",
        default=default,
    )


def is_pre_labeled():
    return click.option(
        "--labeled",
        is_flag=True,
        help="Masks are pre-labeled",
    )


def do_not_rescale():
    return click.option(
        "--no-rescale",
        is_flag=True,
        help="Skip rescaling. Useful only for debugging purposes.",
    )


def compress_output(extra_text: str = ""):
    return click.option(
        "--compress",
        is_flag=True,
        help=f"Compress output files. {extra_text}",
    )


def agree_to_all():
    return click.option(
        "--agree-to-all",
        "-y",
        is_flag=True,
        help="Do not ask for confirmation and proceed",
    )


def filename_template(default: str):
    return click.option(
        "--template",
        "-T",
        type=click.STRING,
        help=f"""\b
    Output file name template. See --long-help for more details.
    Default: '{default}'""",
        default=default,
    )
