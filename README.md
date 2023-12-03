# IBM Quantum Summit practitioners forum PEA utility module

This is a small library of helper functions used during the IBM Quantum Summit practitioners forum.

## Documentation

The  API documentation can be built using tox. To build it run

```bash
> cd pea-summit
> pip install tox
> tox -e docs-clean; tox -e docs
```

## Installation

This module contains a small python package `pea_summit` which includes helper functions
and all needed dependencies to run the tutorial notebooks.

If you use conda it can be installed in a fresh environment using the `environment.yml`

```bash
> cd pea-summit
> conda env create -f environment.yml
> conda activate pea-summit
> juputer lab
```

To install via pip in a location of your choice run
```bash
> cd pea-summit
> pip install .
```
