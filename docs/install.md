---
title: Install
---

<!-- MarkdownTOC -->

- [Requirements](#requirements)
- [Install](#install)
    - [Install with `pip`](#install-with-pip)
    - [Install with `pipx`](#install-with-pipx)
- [Check your installation](#check-your-installation)

<!-- /MarkdownTOC -->

## Requirements

`radiantkit` has been tested with Python 3.7, 3.8, and 3.9. The packages it depends on are listed in our [dependency graph](https://github.com/ggirelli/radiantkit/network/dependencies). We use [`poetry`](https://github.com/python-poetry/poetry) to handle our dependencies.

## Install

#### Install with `pip`

```bash
git clone https://github.com/ggirelli/radiantkit.git
cd radiantkit
pip install --user .
```

#### Install with `pipx`

If you want access only to `radiantkit` scripts (i.e., not the API), we recommend installing using [`pipx`](https://github.com/pipxproject/pipx). Check how to install `pipx` [here](https://github.com/pipxproject/pipx#install-pipx) if you don't have it yet!

Once you have `pipx` (v0.15.5+) ready on your system, install the latest stable release of `radiantkit` by running:
```bash
pipx install git+https://github.com/ggirelli/radiantkit.git --force
```
If you see the stars (✨ 🌟 ✨), then the installation went well!

## Check your installation

To check your installation, simply run:
```bash
radiant --version
```

If you see the version of `radiantkit` that you installed, everything went well! If you see an error or `command not found`, try again or [get in touch](https://github.com/ggirelli/radiantkit/issues)!
