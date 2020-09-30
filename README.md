# radiantkit

![Python package](https://github.com/ggirelli/radiantkit/workflows/Python%20package/badge.svg?branch=master)

**Rad**ial **I**mage **An**alysis **T**ool**kit** (RadIAnTkit)j is a Python3.7/3.8 package containing tools for full-stack image analysis - from proprietary format conversion to tiff to cellular nuclei segmentation, from the selection of G1 nuclei to the measurement of radial patterns.

## Features

* **Convert** proprietary microscope formats CZI (Zeiss) and ND2 (Nikon) to open-source TIFF format.
* **Segment** cellular nuclei or other objects, in 2D or 3D, in an unsupervised manner.  
Then use the automatic segmentation to **estimate background** and foreground intensity.
* **Select** cellular nuclei, in G1-phase of the cell cycle, based on DNA staining and nuclear volume.
* **Extract** segmented objects and **measure** their features (e.g., volume, integral of intensity, shape descriptors).
* Measure **radial patterns** as radial profiles (with different center definitions),  
and characterize them (e.g., peaks, inflection points, contrast).
* Generate **snakemake-based workflows** for seamless integration into fully reproducible streamlined analysis pipelines.

## Requirements

RadIAnTkit requires Python 3.7 or 3.8. In the [setup.py](https://github.com/ggirelli/radiantkit/blob/master/setup.py) file, you can find a list with all the Python package dependencies.

(*optional*) The pipeline features require the snakemake package. As these are optional features, the automatic installation process does not install the package. To manually install it on your system, follow [this guide](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html).

## Installation

We recommend using a python environment manager (our favorite one being conda) to install the package. Why is that? For each dependency, RadIAnTkit requires a specific version to avoid breaks due to non-retro compatible dependency upgrades. Requiring specific dependency versions can cause conflicts with other packages installed in the same environment. The easiest way to avoid such issues is to have a separate environment for the RadIAnTkit package, which can be easily achieved by following the steps below.

### Setup using conda

First, install conda if it is not already available on your system.  
A full installation guide is available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/#regular-installation), follow the instructions to install miniconda.

Then, the setup requires only three steps:

1) create an environment with Python3.6+,
2) install pip in the environment,
3) install the package with pip.

```bash
conda create --name radiantkit
conda install pip
pip install radiantkit
```

## Usage

All RadIAnTkit tools are accessible from the terminal using the radiant keyword.  
We are planning to add a pyQT5-based GUI.

```bash
usage: radiant [-h] [--version] sub_command ...
```

## Contributing

We welcome any contributions to RadIAnTkit.  
Please, refer to our [contribution guidelines](https://github.com/ggirelli/radiantkit/blob/master/CONTRIBUTING.md) if this is your first time contributing!  
Also, check out our [code of conduct](https://github.com/ggirelli/radiantkit/blob/master/CODE_OF_CONDUCT.md).

## License

MIT License
Copyright (c) 2020 Gabriele Girelli
