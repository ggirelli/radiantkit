"""
@author: Gabriele Girelli
@contact: gigi.ga90@gmail.com
"""

from radiantkit.scripts.conversion.settings import ConversionSettings
from os import listdir
from os.path import isdir, join as join_paths
from radiantkit.string import TIFFNameTemplateFields as TNTFields
import re
from typing import Callable

CONVERSION_TEMPLATE_LONG_HELP_STRING = f"""
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


def convert_folder(
    args: ConversionSettings,
    path: str,
    convert_file: Callable[[ConversionSettings, str], None],
) -> None:
    assert isdir(path)
    for file_path in sorted(listdir(path)):
        if re.match(args.input_re, file_path) is not None:
            convert_file(args, join_paths(path, file_path))
