import os
import re
from typing import Iterator, Pattern

ROOT = config['root']
CONDITIONS = config['conditions']['labels']
CONDITION_FOLDERS = [os.path.join(ROOT, condition) for condition in CONDITIONS]

REFERENCE = config['images']['reference']
BG_RADIUS = config['images']['background_radius']

MASK_PREFIX = config['segmentation']['prefix']
MASK_SUFFIX = config['segmentation']['suffix']

IMAGE_PATTERN = config['images']['pattern']
SEGMENTATION_PATTERN = IMAGE_PATTERN.replace(
    "{channel_name}", REFERENCE)
SEGMENTATION_RE = re.compile(SEGMENTATION_PATTERN.replace(
    "{series_id}", "(?P<series_id>[0-9]+)"))
MASK_PATTERN = MASK_PREFIX+os.path.splitext(
    IMAGE_PATTERN)[0]+MASK_SUFFIX+".tif"


def max_series_id(condition: str, regexp: Pattern) -> int:
    maxid = 0
    for f in os.listdir(os.path.join(ROOT, condition)):
        if re.search(regexp, f) is not None:
            maxid = max(maxid, int(re.match(
                regexp, f).groupdict()["series_id"]))
    return maxid


def list_series_id(condition: str, regexp: Pattern) -> Iterator[int]:
    for f in os.listdir(os.path.join(ROOT, condition)):
        if re.search(regexp, f) is not None:
            yield int(re.match(regexp, f).groupdict()["series_id"])


def add_leading_dot(suffix: str) -> str:
    if 0 != len(suffix):
        if not suffix.startswith("."):
            suffix = f".{suffix}"
    return suffix


def add_suffix(path: str, suffix: str) -> str:
    fname, fext = os.path.splitext(path)
    if not fname.endswith(suffix):
        fname += add_leading_dot(suffix)
    return f"{fname}{fext}"


final_output = [
    [os.path.join(cf, "objects", "nuclear_features.tsv")
     for cf in CONDITION_FOLDERS],
    [os.path.join(cf, "objects", "radial_population.profile.poly_fit.pkl")
     for cf in CONDITION_FOLDERS],
    [os.path.join(cf, "objects", "radial_population.profile.raw_data.tsv")
     for cf in CONDITION_FOLDERS]]

if config['measure_objects']['measure_single_voxel']:
    final_output.append([os.path.join(
        cf, "objects", "single_pixel_features.tsv")
        for cf in CONDITION_FOLDERS])

if config['export_objects']:
    final_output.append([directory(os.path.join(
        cf, "objects", "tiff")) for cf in CONDITION_FOLDERS])

rule all:
    input: final_output

rule segmentation:
    input:
        os.path.join(ROOT, "{condition}", IMAGE_PATTERN)
    output:
        os.path.join(ROOT, "{condition}", MASK_PATTERN)
    log:
        os.path.join(ROOT, "{condition}",
                     "segmentation.{channel_name}.{series_id}.log")
    params:
        root = ROOT,
        condition = "{condition}",
        prefix = f"{MASK_PREFIX}",
        suffix = MASK_SUFFIX,
        pattern = SEGMENTATION_RE.pattern
    threads: 1
    shell: """
        radiant tiff_segment --inreg '{params.pattern}' -y {input} &> {log}
        """

rule nuclear_selection:
    input:
        lambda wildcards: expand(os.path.join(
            ROOT, "{condition}", MASK_PATTERN),
            condition=wildcards.condition, channel_name=REFERENCE,
            series_id=[f"{sid:03d}" for sid in list_series_id(
                wildcards.condition, SEGMENTATION_RE)])
    output:
        os.path.join(ROOT, "{condition}", "select_nuclei.data.tsv")
    log:
        os.path.join(ROOT, "{condition}", "select_nuclei.data.log")
    params:
        root = ROOT,
        condition = "{condition}",
        prefix = f"{MASK_PREFIX}",
        suffix = MASK_SUFFIX,
        reference = REFERENCE,
        sigma = config['nuclear_selection']['k_sigma'],
        bg_radius = BG_RADIUS
    threads: config['nuclear_selection']['threads']
    shell: """
        radiant select_nuclei\
        {params.root}/{params.condition} {params.reference}\
        --mask-prefix "{params.prefix}" --mask-suffix "{params.suffix}"\
        --k-sigma {params.sigma} --block-side {params.bg_radius}\
        --export-instance --threads {threads} -y &> {log}
        """

measure_objects_output = [
    os.path.join(ROOT, "{condition}", "objects", "nuclear_features.tsv")]
if config['measure_objects']['measure_single_voxel']:
    measure_objects_output.append(os.path.join(
        ROOT, "{condition}", "objects", "single_pixel_features.tsv"))

rule measure_objects:
    input:
        os.path.join(ROOT, "{condition}",
                     "select_nuclei.data.tsv")
    output: measure_objects_output
    log:
        os.path.join(ROOT, "{condition}", "objects",
                     "measure_objects.{condition}.log")
    threads: config['measure_objects']['threads']
    params:
        root = ROOT,
        condition = "{condition}",
        prefix = f"{MASK_PREFIX}",
        suffix = f"{MASK_SUFFIX}.selected",
        reference = REFERENCE,
        aspect = " ".join([str(a) for a in config['images']['aspect']]),
        bg_radius = BG_RADIUS,
        single_voxel = " --export-single-voxel" if config[
            'measure_objects']['measure_single_voxel'] else ''
    shell: """
        radiant measure_objects\
        {params.root}/{params.condition} {params.reference}\
        --aspect {params.aspect} --block-side {params.bg_radius}\
        --mask-prefix "{params.prefix}" --mask-suffix "{params.suffix}"\
        --import-instance -y {params.single_voxel}&> {log}
        """

QUANTILE = ""
if 'quantile' in config['radiality']:
    QUANTILE = f"--quantile {config['radiality']['quantile']}"

rule radial_population:
    input:
        os.path.join(ROOT, "{condition}",
                     "select_nuclei.data.tsv")
    output:
        os.path.join(ROOT, "{condition}", "objects",
                     "radial_population.profile.poly_fit.pkl"),
        os.path.join(ROOT, "{condition}", "objects",
                     "radial_population.profile.raw_data.tsv")
    log:
        os.path.join(ROOT, "{condition}", "objects",
                     "radial_population.{condition}.log")
    threads: config['radiality']['threads']
    params:
        root = ROOT,
        condition = "{condition}",
        prefix = f"{MASK_PREFIX}",
        suffix = f"{MASK_SUFFIX}.selected",
        reference = REFERENCE,
        aspect = " ".join([str(a) for a in config['images']['aspect']]),
        bg_radius = BG_RADIUS,
        center_type = config['radiality']['center_type'],
        quantile = QUANTILE,
        bins = config['radiality']['bins'],
        degree = config['radiality']['degree']
    shell: """
        radiant radial_population\
        {params.root}/{params.condition} {params.reference}\
        --aspect {params.aspect} --block-side {params.bg_radius}\
        --mask-prefix "{params.prefix}" --mask-suffix "{params.suffix}"\
        --center-type {params.center_type} {params.quantile}\
        --bins {params.bins} --degree {params.degree}\
        --import-instance --export-instance -y &> {log}
        """

rule export_objects:
    input:
        os.path.join(ROOT, "{condition}",
                     "select_nuclei.data.tsv")
    output:
        directory(os.path.join(ROOT, "{condition}", "objects", "tiff"))
    log:
        os.path.join(ROOT, "{condition}", "objects",
                     "export_objects.{condition}.log")
    threads: 1
    params:
        root = ROOT,
        condition = "{condition}",
        prefix = f"{MASK_PREFIX}",
        suffix = f"{MASK_SUFFIX}selected.",
        reference = REFERENCE
    shell: """
        radiant export_objects\
        {params.root}/{params.condition} {params.reference}\
        --mask-prefix "{params.prefix}" --mask-suffix "{params.suffix}"\
        --import-instance -y &> {log}
        """
