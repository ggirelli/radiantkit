---
title: radiantkit
---

![](https://img.shields.io/librariesio/github/ggirelli/radiantkit.svg?style=flat) ![](https://img.shields.io/github/license/ggirelli/radiantkit.svg?style=flat)  
![](https://github.com/ggirelli/radiantkit/workflows/Python%20package/badge.svg?branch=main&event=push) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/radiantkit) ![PyPI - Format](https://img.shields.io/pypi/format/radiantkit) ![PyPI - Status](https://img.shields.io/pypi/status/radiantkit)  
![](https://img.shields.io/github/release/ggirelli/radiantkit.svg?style=flat) ![](https://img.shields.io/github/release-date/ggirelli/radiantkit.svg?style=flat) ![](https://img.shields.io/github/languages/code-size/ggirelli/radiantkit.svg?style=flat)  
![](https://img.shields.io/github/watchers/ggirelli/radiantkit.svg?label=Watch&style=social) ![](https://img.shields.io/github/stars/ggirelli/radiantkit.svg?style=social)

**Rad**ial **I**mage **An**alysis **T**ool**kit** (RadIAnTkit)j is a Python3.7+ package containing tools for full-stack image analysis - from proprietary format conversion to tiff to cellular nuclei segmentation, from the selection of G1 nuclei to the measurement of radial patterns.

<!-- MarkdownTOC -->

- [Features \(in short\)](#features-in-short)
- [Contributing](#contributing)
- [License](#license)

<!-- /MarkdownTOC -->

## Features (in short)

* **Convert** proprietary microscope formats CZI (Zeiss) and ND2 (Nikon) to open-source TIFF format.
* **Segment** cellular nuclei or other objects, in 2D or 3D, in an unsupervised manner. Then use the automatic segmentation to **estimate background** and foreground intensity.
* **Select** cellular nuclei, in G1-phase of the cell cycle, based on DNA staining and nuclear volume.
* **Extract** segmented objects and **measure** their features (e.g., volume, integral of intensity, shape descriptors).
* * Measure **radial patterns** as radial profiles (with different center definitions), and characterize them (e.g., peaks, inflection points, contrast).
* Generate **snakemake-based workflows** for seamless integration into fully reproducible streamlined analysis pipelines.

## Contributing

We welcome any contributions to `radiantkit`. In short, we use [`black`](https://github.com/psf/black) to standardize code format. Any code change also needs to pass `mypy` checks. For more details, please refer to our [contribution guidelines](https://github.com/ggirelli/radiantkit/blob/main/CONTRIBUTING.md) if this is your first time contributing! Also, check out our [code of conduct](https://github.com/ggirelli/radiantkit/blob/main/CODE_OF_CONDUCT.md).

## License

`MIT License - Copyright (c) 2020 Gabriele Girelli`
