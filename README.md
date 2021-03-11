# radiantkit

![](https://img.shields.io/librariesio/github/ggirelli/radiantkit.svg?style=flat) ![](https://img.shields.io/github/license/ggirelli/radiantkit.svg?style=flat)  
![](https://github.com/ggirelli/radiantkit/workflows/Python%20package/badge.svg?branch=main&event=push) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/radiantkit) ![PyPI - Format](https://img.shields.io/pypi/format/radiantkit) ![PyPI - Status](https://img.shields.io/pypi/status/radiantkit)  
![](https://img.shields.io/github/release/ggirelli/radiantkit.svg?style=flat) ![](https://img.shields.io/github/release-date/ggirelli/radiantkit.svg?style=flat) ![](https://img.shields.io/github/languages/code-size/ggirelli/radiantkit.svg?style=flat)  
![](https://img.shields.io/github/watchers/ggirelli/radiantkit.svg?label=Watch&style=social) ![](https://img.shields.io/github/stars/ggirelli/radiantkit.svg?style=social)

[PyPi](https://pypi.org/project/radiantkit/) | [docs](https://ggirelli.github.io/radiantkit/)

**Rad**ial **I**mage **An**alysis **T**ool**kit** (RadIAnTkit)j is a Python3.7+ package containing tools for full-stack image analysis - from proprietary format conversion to tiff to cellular nuclei segmentation, from the selection of G1 nuclei to the measurement of radial patterns.

## Features (in short)

* **Convert** proprietary microscope formats CZI (Zeiss) and ND2 (Nikon) to open-source TIFF format.
* **Segment** cellular nuclei or other objects, in 2D or 3D, in an unsupervised manner.  
Then use the automatic segmentation to **estimate background** and foreground intensity.
* **Select** cellular nuclei, in G1-phase of the cell cycle, based on DNA staining and nuclear volume.
* **Extract** segmented objects and **measure** their features (e.g., volume, integral of intensity, shape descriptors).
* Measure **radial patterns** as radial profiles (with different center definitions),  
and characterize them (e.g., peaks, inflection points, contrast).
* Generate **snakemake-based workflows** for seamless integration into fully reproducible streamlined analysis pipelines.

For more available features, check out our [docs](https://ggirelli.github.io/radiantkit/)!

## Requirements

`radiantkit` has been tested with Python 3.7, 3.8, and 3.9. We recommend installing it using `pipx` (see [below](https://github.com/ggirelli/radiantkit#install)) to avoid dependency conflicts with other packages. The packages it depends on are listed in our [dependency graph](https://github.com/ggirelli/radiantkit/network/dependencies). We use [`poetry`](https://github.com/python-poetry/poetry) to handle our dependencies.

## Install

We recommend installing `radiantkit` using [`pipx`](https://github.com/pipxproject/pipx). Check how to install `pipx` [here](https://github.com/pipxproject/pipx#install-pipx) if you don't have it yet!

Once you have `pipx` ready on your system, install the latest stable release of `radiantkit` by running:
```
pipx install radiantkit
```
If you see the stars (✨ 🌟 ✨), then the installation went well!

Alternatively, you can use `pipx` (v0.15.5+) to install directly from git, with the command:
```
pipx install git+https://github.com/ggirelli/radiantkit.git --force
```

## Usage

Run: `radiant -h`.

All RadIAnTkit tools are accessible from the terminal using the `radiant` keyword.  

```bash
usage: radiant [-h] [--version] sub_command ...
```

## Contributing

We welcome any contributions to `radiantkit`. In short, we use [`black`](https://github.com/psf/black) to standardize code format. Any code change also needs to pass `mypy` checks. For more details, please refer to our [contribution guidelines](https://github.com/ggirelli/radiantkit/blob/main/CONTRIBUTING.md) if this is your first time contributing! Also, check out our [code of conduct](https://github.com/ggirelli/radiantkit/blob/main/CODE_OF_CONDUCT.md).

## License

`MIT License - Copyright (c) 2020 Gabriele Girelli`
