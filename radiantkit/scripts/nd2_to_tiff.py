"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

import click
import logging
import numpy as np
from os import listdir, mkdir
from os.path import basename, dirname, isdir, isfile, join as join_paths, splitext
import radiantkit as ra
from radiantkit.string import TIFFNameTemplate as TNTemplate
from radiantkit.string import TIFFNameTemplateFields as TNTFields
import re
from rich.progress import track
import sys
from typing import Any, Dict, List, Optional, Set, Union


@click.command(
    name="nd2_to_tiff",
    context_settings=ra.const.CONTEXT_SETTINGS,
    help=f"""
Convert ND2 file(s) into TIFF.

To convert a single file, provide its path as INPUT. To convert all nd2 files in a
folder, instead, specify the folder path as INPUT. To convert specific files, specify
them one after the other as INPUT.
""",
)
@click.argument("input", nargs=-1, type=click.Path(exists=True))
@click.option("--info", is_flag=True, help="Show INPUT details and stop.")
@click.option("--list", is_flag=True, help="List INPUT files and stop.")
@click.option("--long-help", is_flag=True, help="Show long help page and stop.")
@click.option(
    "--output",
    "-o",
    type=click.Path(exists=False),
    help="Output folder path. Defaults to INPUT without extension.",
)
@click.option(
    "--fields",
    "-F",
    type=click.STRING,
    help="""\b
    Limit conversion to the specified fields of view.
    E.g., '1-3,5' converts fields: 1,2,3,5.""",
)
@click.option(
    "--channels",
    "-C",
    type=click.STRING,
    help="""\b
    Limit conversion to the specified channels.
    Separate multiple channels with a comma, e.g., 'dapi,a647'.
    """,
)
@click.option(
    "--dz",
    "-Z",
    type=click.FLOAT,
    help="Delta Z in um. Use when the script fails to recognize the correct value.",
)
@click.option(
    "--input-re",
    "-R",
    type=click.STRING,
    metavar="RE",
    help=f"""
    Regexp used to identify input ND2 files.
    Default: {ra.const.DEFAULT_INPUT_RE['nd2']}""",
    default=ra.const.DEFAULT_INPUT_RE["nd2"],
)
@click.option(
    "--template",
    "-T",
    type=click.STRING,
    help=f"""\b
    Output file name template. See --long-help for more details.
    Default: '{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}'""",
    default=f"{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}",
)
@click.option(
    "--compress/--no-compress",
    help="Compress output TIFF files. Useful with low bit-depth output.",
    default=False,
)
def run(**args: Dict[str, Any]):
    if args["long_help"]:
        print_long_help()
        return

    settings = Settings(**args)
    logging.info(f"Input: {settings.input_paths}")
    for path in settings.input_paths:
        if isdir(path):
            logging.info(f"Looking into folder: {path}")
            convert_folder(settings, path)
        else:
            convert_file(settings, path, settings.output_dirpath)
        logging.info("Done. :thumbs_up: :smiley:")


class Settings(object):
    """docstring for ND2toTIFFsettings"""

    _input_paths: Set[str] = set()
    _output_dirpath: Optional[str] = None
    _fields: Optional[ra.string.MultiRange] = None
    _channels: Optional[Set[str]] = None
    dz: Optional[float] = None
    input_re: str = ra.const.DEFAULT_INPUT_RE["nd2"]
    template: TNTemplate
    compress: bool = False
    just_info: bool = False
    just_list: bool = False

    def __init__(self, **args):
        super(Settings, self).__init__()
        for field in [
            "input",
            "output",
            "fields",
            "channels",
            "dz",
            "input_re",
            "template",
            "compress",
            "info",
            "list",
        ]:
            assert field in args, f"missing field '{field}'"
        self.input_paths = args["input"]
        self.output_dirpath = args["output"]
        self.fields = args["fields"]
        self.channels = args["channels"]
        self.dz = args["dz"]
        self.input_re = args["input_re"]
        self.template = TNTemplate(args["template"])
        self.compress = args["compress"]
        self.just_info = args["info"]
        self.just_list = args["list"]

    @property
    def input_paths(self) -> Set[str]:
        return self._input_paths

    @input_paths.setter
    def input_paths(self, path_list: List[str]) -> None:
        file_count = 0
        dir_count = 0
        for path in set(path_list):
            if isfile(path):
                file_count += 1
            elif isdir(path):
                dir_count += 1
            else:
                assert False, f"input path not found: {path}"
        if 0 < dir_count:
            assert 1 == dir_count, "only one directory is allowed per run"
            assert (
                0 == file_count
            ), "please provide either files or a directory, not both."
        self._input_paths = set(path_list)

    @property
    def output_dirpath(self) -> Optional[str]:
        return self._output_dirpath

    @output_dirpath.setter
    def output_dirpath(self, path: Optional[str]) -> None:
        if path is not None:
            assert not isfile(path), f"output path exists as a file: {path}"
            assert not isdir(path), f"output path exists as a folder: {path}"
        self._output_dirpath = path

    @property
    def fields(self) -> Optional[ra.string.MultiRange]:
        return self._fields

    @fields.setter
    def fields(self, fields_str: Optional[str]) -> None:
        if fields_str is not None:
            self._fields = ra.string.MultiRange(fields_str)
        else:
            self._fields = None

    @property
    def channels(self) -> Optional[Set[str]]:
        return self._channels

    @channels.setter
    def channels(self, channels_str: Optional[str]) -> None:
        if channels_str is not None:
            self._channels = set(channels_str.split(","))
        else:
            self._channels = None

    def check_fields(self, nd2_image: ra.conversion.ND2Reader2) -> bool:
        assert self.template.can_export_fields(
            nd2_image.field_count(), self.fields
        ), "".join(
            [
                "when exporting more than 1 field, the template ",
                f"must include the {TNTFields.SERIES_ID} seed. ",
                f"Got '{self.template.template}' instead.",
            ]
        )
        if self.fields is not None:
            if np.array(list(self.fields)).min() > nd2_image.field_count():
                logging.warning(
                    "".join(
                        [
                            "Skipped all available fields ",
                            "(not included in specified field range.",
                        ]
                    )
                )
        return True

    def check_channels(self, nd2_image: ra.conversion.ND2Reader2) -> bool:
        if self.channels is not None:
            assert 0 != len(
                nd2_image.select_channels(self.channels)
            ), "none of the specified channels was found."

        assert self.template.can_export_channels(
            nd2_image.channel_count(), self.channels
        ), "".join(
            [
                "when exporting more than 1 channel, the template ",
                f"must include either {TNTFields.CHANNEL_ID} or ",
                f"{TNTFields.CHANNEL_NAME} seeds. ",
                f"Got '{self.template.template}' instead.",
            ]
        )

        return True

    def check_dz(self, nd2_image: ra.conversion.ND2Reader2) -> bool:
        if self.dz is not None:
            logging.info(f"Enforcing a deltaZ of {self.dz:.3f} um.")
        elif 1 < len(nd2_image.z_resolution):
            logging.warning(
                " ".join(
                    [
                        "Z resolution is not constant across fields.",
                        "It will be automagically identified, field-by-field.",
                        "If the automatic Z resolution reported in the log is wrong,",
                        "please enforce the correct one using the --dz option.",
                    ]
                )
            )
            logging.debug(f"Z steps histogram: {nd2_image.z_resolution}.")
        return True

    def is_nd2_compatible(self, nd2_image: ra.conversion.ND2Reader2) -> bool:
        assert not nd2_image.isLive(), "time-course conversion images not implemented."
        self.check_fields(nd2_image)
        self.check_channels(nd2_image)
        self.check_dz(nd2_image)
        return True

    def get_output_dirpath_for_single_file(
        self, file_path: str, output_dirpath: Optional[str] = None
    ) -> str:
        if output_dirpath is None:
            output_dirpath = join_paths(
                dirname(file_path), splitext(basename(file_path))[0]
            )
        assert not isfile(output_dirpath), f"output path is a file: {output_dirpath}"
        return output_dirpath

    def select_field_list(
        self, nd2_image: ra.conversion.ND2Reader2, verbose: bool = False
    ) -> List[int]:
        field_list: List[int]
        if self.fields is not None:
            logging.info(
                "Converting only the following fields: " + f"{[x for x in self.fields]}"
            )
            return [x - 1 for x in self.fields if x <= nd2_image.field_count()]
        else:
            return list(range(nd2_image.field_count()))


def print_long_help() -> None:
    print(
        f"""
# Converting 3+D ND2 files to TIFF

In the case of 3+D images, radiant checks for a consistent dZ across consecutive 2D
slices and saves it in the output TIFF metadata. In case of inconsistent dZ, the script
tries to guess the correct value, then report it and proceed. If the reported dZ is
wrong, please enforce the correct one using the -Z option.

If a dZ cannot be automatically guessed, the affected field is skipped and the user is
warned. Use the -F and -Z options to convert the skipped field(s).

# Output file name template

The output tiff file name follows the specified template (-T option). A template is a
string which includes a series of "seeds", which radiant replaces with the corresponding
values when writing the output. Available seeds are:
- '{TNTFields.CHANNEL_NAME}'\t: channel name, lower-cased.
- '{TNTFields.CHANNEL_ID}'\t: channel ID (number).
- '{TNTFields.SERIES_ID}'\t: series ID (number).
- '{TNTFields.DIMENSIONS}'\t: number of dimensions, followed by "D".
- '{TNTFields.AXES_ORDER}'\t: axes order (e.g., "TZYX").

Note: Leading 0s are added up to 3 digits to all ID seed.

The default template is "{TNTFields.CHANNEL_NAME}_{TNTFields.SERIES_ID}". Hence, when
writing the 3rd series of the "a488" channel, the output file name would be:
"a488_003.tiff".

IMPORTANT: please, remember to escape the "$" when running from command line if using
double quotes, i.e., "\\$". Alternatively, use single quotes, i.e., '$'.
"""
    )


def convert_folder(args: Settings, path: str) -> None:
    assert isdir(path)
    for file_path in sorted(listdir(path)):
        if re.match(args.input_re, file_path) is not None:
            convert_file(args, join_paths(path, file_path))


def convert_file(
    args: Settings, path: str, output_dirpath: Optional[str] = None
) -> None:
    logging.info(f"Working on file '{path}'.")
    assert isfile(path), f"input file not found: {path}"
    if args.just_list:
        return

    nd2_image = ra.conversion.ND2Reader2(path)
    if args.just_info:
        nd2_image.log_details()
        logging.info("")
        return

    output_dirpath = args.get_output_dirpath_for_single_file(path, output_dirpath)
    if not isdir(output_dirpath):
        mkdir(output_dirpath)
    ra.io.add_log_file_handler(join_paths(output_dirpath, "nd2_to_tiff.log.txt"))

    nd2_image.log_details()
    args.is_nd2_compatible(nd2_image)
    logging.info(f"Output directory: '{output_dirpath}'")

    nd2_image.set_iter_axes("v")
    nd2_image.set_axes_for_bundling()

    for field_id in track(
        args.select_field_list(nd2_image, True), description="Converting field"
    ):
        export_field(nd2_image, field_id, output_dirpath, args)


def export_field(
    nd2_image: ra.conversion.ND2Reader2,
    field_id: int,
    output_dirpath: str,
    args: Settings,
) -> None:
    dz = nd2_image.get_dz(field_id, args.dz)
    if np.isnan(dz):
        return

    try:
        export_channels(nd2_image, field_id, output_dirpath, args, dz)
    except ValueError as e:
        if "could not broadcast input array from shape" in e.args[0]:
            logging.error(
                " ".join(
                    [
                        f"corrupted file raised {type(e).__name__}.",
                        "At least one frame has mismatching shape.",
                    ]
                )
            )
            logging.critical(f"{e.args[0]}")
            sys.exit()
        raise e


def export_channels(
    nd2_image: ra.conversion.ND2Reader2,
    field_id: int,
    outdir: str,
    args,
    z_resolution: float = 0.0,
) -> None:
    channels = nd2_image.select_channels(
        list(nd2_image.get_channel_names()) if args.channels is None else args.channels
    )

    bundle_axes = nd2_image.bundle_axes.copy()
    if nd2_image.has_multi_channels():
        bundle_axes.pop(bundle_axes.index("c"))
    bundle_axes = "".join(bundle_axes).upper()

    for channel_id in range(nd2_image.channel_count()):
        channel_name = nd2_image.metadata["channels"][channel_id].lower()
        if channel_name in channels:
            ra.image.save_tiff(
                join_paths(
                    outdir,
                    nd2_image.get_tiff_path(args.template, channel_id, field_id),
                ),
                get_field(nd2_image, field_id, channel_id),
                args.compress,
                bundle_axes=bundle_axes,
                inMicrons=True,
                z_resolution=z_resolution,
                resolution=(
                    0 if 0 == nd2_image.xy_resolution else 1 / nd2_image.xy_resolution,
                    0 if 0 == nd2_image.xy_resolution else 1 / nd2_image.xy_resolution,
                    None,
                ),
            )


def get_field(
    nd2_image: ra.conversion.ND2Reader2, field_id: int, channel_id: int
) -> np.ndarray:
    slicing: List[Union[slice, int]] = []
    field_of_view = nd2_image[field_id]
    for a in nd2_image.bundle_axes:
        axis_size = field_of_view.shape[nd2_image.bundle_axes.index(a)]
        if "c" == a:
            assert channel_id < axis_size
            slicing.append(channel_id)
        else:
            slicing.append(slice(0, axis_size))
    return field_of_view[tuple(slicing)]
