# Usage

The usage example in this section requires you to download the data files from folder [data](https://github.com/spang-lab/adadmire/tree/main/data) or the urnc repository first. For a description of the contents of this folder, see section [Data](#data).

```python
from adadmire import admire, penalty
import numpy as np

# Load Feist et al example data into python
X = np.load('data/Feist_et_al/scaled_data_raw.npy') # continuous data
D = np.load('data/Feist_et_al/pheno.npy') # discrete data
levels = np.load('data/Feist_et_al/levels.npy') # levels of discrete variables

# Define lambda sequence of penalty values
lam = penalty(X, D, min= -2.25, max = -1.5, step =0.25)
print(lam)

# Get anomalies in continuous and discrete data
X_cor, n_cont, position_cont, D_cor, n_disc, position_disc = admire(X, D, levels, lam)
print(X_cor) # Corrected X
print(n_cont) # Number of continuous anomalies (46)
print(position_cont) # Position in X
print(D_cor) # Corrected D
print(n_disc) # Number of discrete anomalies (0)
print(position_disc) # Position in D
```

## Documentation

You can find the full documentation for adadmire at [spang-lab.github.io/adadmire](https://spang-lab.github.io/adadmire/). Amongst others, it includes chapters about:

- [Installation](https://spang-lab.github.io/adadmire/installation.html)
- [Usage](https://spang-lab.github.io/adadmire/usage.html)
- [Modules](https://spang-lab.github.io/adadmire/modules.html)
- [Contributing](https://spang-lab.github.io/adadmire/contributing.html)
- [Testing](https://spang-lab.github.io/adadmire/testing.html)
- [Documentation](https://spang-lab.github.io/adadmire/documentation.html)
