"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import argparse
from collections import defaultdict
from joblib import delayed, Parallel  # type: ignore
import logging
import os
import pandas as pd  # type: ignore
from plotly import graph_objects as go, express as px  # type: ignore
from radiantkit import const, exception, image, io, path, report
from radiantkit.scripts.common import argtools
from typing import DefaultDict, Dict


@exception.enable_rich_exceptions
def init_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        __name__.split(".")[-1],
        description="""
Calculate gradient magnitude over Z for every image in the input folder with a
filename matching the --inreg. Use --range to change the in-focus
definition.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help="Find out of focus fields of view.",
    )

    parser.add_argument("input", type=str, help="Path to folder with tiff images.")

    parser.add_argument(
        "--output",
        type=str,
        help="Path to output tsv file. Default: oof.tsv in input folder.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        help="Fraction of stack (middle-centered) for in-focus fields. Default: .5",
        default=0.5,
    )
    parser = argtools.add_version_argument(parser)

    advanced = parser.add_argument_group("advanced arguments")
    advanced.add_argument(
        "--inreg",
        type=str,
        metavar="REGEXP",
        help=f"""Regular expression to identify input TIFF images.
        Default: '{const.default_inreg}'""",
        default=const.default_inreg,
    )
    advanced = argtools.add_threads_argument(advanced)
    advanced.add_argument(
        "--intensity-sum",
        action="store_const",
        const=True,
        default=False,
        help="""Use intensity sum instead of gradient magnitude.""",
    )
    advanced.add_argument(
        "--rename",
        action="store_const",
        const=True,
        default=False,
        help="""Rename out-of-focus images by adding the '.old' suffix.""",
    )

    parser.set_defaults(parse=parse_arguments, run=run)

    return parser


@exception.enable_rich_exceptions
def parse_arguments(args: argparse.Namespace) -> argparse.Namespace:
    if args.output is None:
        args.output = os.path.join(args.input, "oof.tsv")
    args.threads = argtools.check_threads(args.threads)
    args.descriptor_mode = (
        image.SliceDescriptorMode.INTENSITY_SUM
        if args.intensity_sum
        else image.SliceDescriptorMode.GRADIENT_OF_MAGNITUDE
    )
    return args


def check_focus(args: argparse.Namespace, ipath: str) -> pd.DataFrame:
    img = image.ImageGrayScale.from_tiff(os.path.join(args.input, ipath))
    response, profile_data = img.is_in_focus(args.descriptor_mode, args.fraction)
    profile_data["path"] = ipath
    profile_data["response"] = "in-focus" if response else "out-of-focus"
    if "out-of-focus" == response and args.rename:
        os.rename(os.path.join(args.input, ipath), os.path.join(args.input, ipath))
    return profile_data


@exception.enable_rich_exceptions
def run(args: argparse.Namespace) -> None:
    assert os.path.isdir(args.input), f"image directory not found: '{args.input}'"
    argtools.dump_args(args, "oof.args.pkl")
    io.add_log_file_handler(os.path.join(args.input, "oof.log.txt"))

    logging.info(f"Input:\t\t{args.input}")
    logging.info(f"Output:\t\t{args.output}")
    logging.info(f"Fraction:\t{args.fraction}")
    logging.info(f"Rename:\t\t{args.rename}")
    logging.info(f"Mode:\t\t{args.descriptor_mode.value}")
    logging.info(f"Regexp:\t\t{args.inreg}")
    logging.info(f"Threads:\t{args.threads}")

    series_data = Parallel(n_jobs=args.threads, verbose=11)(
        delayed(check_focus)(args, impath)
        for impath in path.find_re(args.input, args.inreg)
    )

    pd.concat(series_data).to_csv(args.output, "\t", index=False)

    logging.info("Done. :thumbs_up: :smiley:")


class Report(report.ReportBase):
    def __init__(self, *args, **kwargs):
        super(Report, self).__init__(*args, **kwargs)
        self._idx = 0.0
        self._stub = "tiff_findoof"
        self._title = "Focus analysis"
        self._files = {"focus_data": ("oof.tsv", True, [])}
        self._log = {"log": ("oof.log.txt", False, [])}
        self._args = {"args": ("oof.args.pkl", True, [])}

    def _plot(
        self, data: DefaultDict[str, Dict[str, pd.DataFrame]]
    ) -> DefaultDict[str, Dict[str, go.Figure]]:
        logging.info(f"plotting '{self._stub}'.")
        fig_data: DefaultDict[str, Dict[str, go.Figure]] = defaultdict(lambda: {})
        assert "focus_data" in data

        for dirpath, dirdata in data["focus_data"].items():
            assert isinstance(dirdata, pd.DataFrame)
            dirdata.sort_values(["path", "Z-slice index"], inplace=True)
            fig = px.line(
                dirdata,
                x="Z-slice index",
                y=dirdata.columns[1],
                color="path",
                line_dash="response",
                labels={"path": "Image", "response": "Result"},
            )
            fig.update_layout(
                title=f"""Focus analysis<br>
<sub>Condition: {os.path.basename(dirpath)}</sub>""",
                autosize=False,
                width=1000,
                height=800,
            )
            fig_data[self._stub][dirpath] = fig

        return fig_data
