---
title: nd2_to_tiff
---

<!-- MarkdownTOC -->

- [In short](#in-short)
- [Introduction](#introduction)
- [File naming](#file-naming)
- [More arguments](#more-arguments)

<!-- /MarkdownTOC -->

### In short

Run `radiant nd2_to_tiff -h` to access the command's help page.

```bash
usage: radiant nd2_to_tiff [-h] [--outdir DIRPATH] [--fields STRING]
    [--channels STRING [STRING ...]] [--version] [--deltaZ FLOAT]
    [--inreg REGEXP] [--template STRING] [--compressed] [-i] [-l] input [input ...]
```

### Introduction

Use `radiant nd2_to_tiff` to convert one or more nd2 files into single channel tiff images. You can specify multiple nd2 files, or multiple folders containing nd2 files, by separating them with a space. When a folder is specified as input, all files matching the "inreg" regular expression are converted. You can change the regular expression to convert a specific files subset.

In the case of 3+D images, the script also checks for consistent deltaZ distance across consecutive 2D slices (i.e., dZ). If the distance is consitent, it is used to set the tiff image dZ metadata. Otherwise, the script tries to guess the correct dZ and reports it in the log. If the reported dZ is wrong, please enforce the correct one using the -Z option. If a correct dZ cannot be automatically guessed, the field of view is skipped and a warning is issued to the user. Use the --fields and -Z options to convert the skipped field(s).

### File naming

The output tiff file names follow the specified template (-T). A template is a string including a series of "seeds" that are replaced by the corresponding values when writing the output file. Available seeds are:
- `${channel_name}` : channel name, lower-cased.
- `${channel_id}` : channel ID (number).
- `${series_id}` : series ID (number).
- `${dimensions}` : number of dimensions, followed by "D".
- `${axes_order}` : axes order (e.g., "TZYX").

Leading 0s are added up to 3 digits to any ID seed.

The default template is "`${channel_name}_${series_id}`". Hence, when writing the 3rd series of the "a488" channel, the output file name would be: "a488_003.tiff".

Please, remember to escape the "$" when running from command line if using double quotes, i.e., "\$". Alternatively, use single quotes, i.e., '$'.

### More arguments

```bash
positional arguments:
  input                 Path an nd2 file to convert, or to a folder containing nd2 files. To specify
                        multiple inputs, separate them with a space.

optional arguments:
  -h, --help            show this help message and exit
  --outdir DIRPATH      Path to output TIFF folder. Defaults to the input file basename. This is
                        ignored when input is a folder.
  --fields STRING       Convert only fields of view specified as when printing a set of pages. Omit if
                        all fields should be converted. E.g., '1-2,5,8-9'.
  --channels STRING [STRING ...]
                        Convert only specified channels. Specified as space-separated channel names.
                        Omit if all channels should be converted. E.g., 'dapi cy5 a488'.
  --version             show program's version number and exit

advanced arguments:
  --deltaZ FLOAT, -Z FLOAT
                        If provided (in um), the script does not check delta Z consistency and instead
                        uses the provided one.
  --inreg REGEXP        Regular expression to identify input ND2 images. Default: '^.*\.nd2$'
  --template STRING, -T STRING
                        Template for output file name. See main description for more details. Default:
                        '${channel_name}_${series_id}'
  --compressed          Write compressed TIFF as output. Useful especially for binary or low-depth
                        (e.g. labeled) images.
  -i, --info            Show details of input nd2 files and stop (nothing is converted).
  -l, --list            List input nd2 files and stop (nothing is converted).
```
