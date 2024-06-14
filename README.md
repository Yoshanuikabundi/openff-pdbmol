OpenFF PDBScan
==============================
[//]: # (Badges)

| **Latest release** | [![Last release tag](https://img.shields.io/github/release-pre/yoshanuikabundi/openff-pdbscan.svg)](https://github.com/yoshanuikabundi/openff-pdbscan/releases) ![GitHub commits since latest release (by date) for a branch](https://img.shields.io/github/commits-since/yoshanuikabundi/openff-pdbscan/latest)  [![Documentation Status](https://readthedocs.org/projects/openff-pdbscan/badge/?version=latest)](https://openff-pdbscan.readthedocs.io/en/latest/?badge=latest)                                                                                                                                                                                                                        |
| :----------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Status**         | [![GH Actions Status](https://github.com/yoshanuikabundi/openff-pdbscan/actions/workflows/gh-ci.yaml/badge.svg)](https://github.com/yoshanuikabundi/openff-pdbscan/actions?query=branch%3Amain+workflow%3Agh-ci) [![codecov](https://codecov.io/gh/yoshanuikabundi/openff-pdbscan/branch/main/graph/badge.svg)](https://codecov.io/gh/yoshanuikabundi/openff-pdbscan/branch/main) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/yoshanuikabundi/openff-pdbscan.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/yoshanuikabundi/openff-pdbscan/context:python) |

Stats collection from the PDB databank.

OpenFF PDBScan is bound by a [Code of Conduct](https://github.com/yoshanuikabundi/openff-pdbscan/blob/main/CODE_OF_CONDUCT.md).

### Installation

To build OpenFF PDBScan from source,
we highly recommend using virtual environments.
If possible, we strongly recommend that you use
[Anaconda](https://docs.conda.io/en/latest/) as your package manager.
Below we provide instructions both for `conda` and
for `pip`.

#### From Conda Forge

```
conda install -c conda-forge openff-pdbscan
```

#### With conda

Ensure that you have [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

Create a virtual environment and activate it:

```
conda create --name pdbscan
conda activate pdbscan
```

Install the development and documentation dependencies:

```
conda env update --name pdbscan --file devtools/conda-envs/test_env.yaml
conda env update --name pdbscan --file docs/requirements.yaml
```

Build this package from source:

```
pip install -e .
```

If you want to update your dependencies (which can be risky!), run:

```
conda update --all
```

And when you are finished, you can exit the virtual environment with:

```
conda deactivate
```

#### With pip

To build the package from source, run:

```
pip install -e .
```

If you want to create a development environment, install
the dependencies required for tests and docs with:

```
pip install -e ".[test,doc]"
```

### Copyright

The OpenFF PDBScan source code is hosted at https://github.com/yoshanuikabundi/openff-pdbscan
and is available under the GNU General Public License, version 3 (see the file [LICENSE](https://github.com/yoshanuikabundi/openff-pdbscan/blob/main/LICENSE)).

Copyright (c) 2024, Josh Mitchell


#### Acknowledgements
 
Project based on the 
[OpenFF Cookiecutter](https://github.com/lilyminium/cookiecutter-openff) version 0.1.
