[![Unit Tests](https://github.com/spang-lab/adadmire/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/spang-lab/adadmire/actions/workflows/unit-tests.yml)
[![Coverage Badge](https://img.shields.io/codecov/c/github/spang-lab/adadmire?label=Code%20Coverage)](https://app.codecov.io/gh/spang-lab/adadmire?branch=main)
[![Download Badge](https://img.shields.io/pypi/dm/adadmire.svg?label=PyPI%20Downloads)](
https://pypi.org/project/adadmire/)

# adadmire

<!-- ATTENTION: this file will be displayed not only on Github, but also on PyPI, so NO relative relative to files in the repo must be used -->

Functions for detecting anomalies in molecular data sets using Mixed Graphical Models.

## Installation

Enter the following commands in a shell like *bash*, *zsh* or *powershell*:

```bash
pip install -U adadmire
```

## Usage

The usage example in this section requires that you first download the data files from the [data](https://github.com/spang-lab/adadmire/tree/main/data) folder. For a description of the contents of this folder, see section [Data](https://spang-lab.github.io/adadmire/usage.html#data) of the adadmire documentation site.

```python
from adadmire import admire, penalty
import numpy as np

# Load example data
X = np.load('data/Feist_et_al/scaled_data_raw.npy') # continuous data
D = np.load('data/Feist_et_al/pheno.npy') # discrete data
levels = np.load('data/Feist_et_al/levels.npy') # levels of discrete variables

# Define lambda sequence of penalty values
lam = penalty(X, D, min= -2.25, max = -1.5, step =0.25)

# Get anomalies in continuous and discrete data
X_cor, n_cont, position_cont, D_cor, n_disc, position_disc = admire(X, D, levels, lam)
print(X_cor) # corrected X
print(n_cont) # number of continuous anomalies
print(position_cont) # position in X
print(D_cor) # corrected D
print(n_disc) # number of discrete anomalies
print(position_disc) # position in D
```

You can find more usage examples in the [Usage](https://spang-lab.github.io/adadmire/usage.html) section of adadmire's documentation site.

## Documentation

You can find the full documentation for adadmire at [spang-lab.github.io/adadmire](https://spang-lab.github.io/adadmire/). Amongst others, it includes chapters about:

- [Installation](https://spang-lab.github.io/adadmire/installation.html)
- [Usage](https://spang-lab.github.io/adadmire/usage.html)
- [Modules](https://spang-lab.github.io/adadmire/modules.html)
- [Contributing](https://spang-lab.github.io/adadmire/contributing.html)
- [Testing](https://spang-lab.github.io/adadmire/testing.html)
- [Documentation](https://spang-lab.github.io/adadmire/documentation.html)
