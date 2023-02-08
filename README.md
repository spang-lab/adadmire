# adadmire

<!-- ATTENTION: this file will be displayed not only on Github, but also on PyPI, so NO relative relative to files in the repo must be used -->

Functions for detecting anomalies in tabular datasets using Mixed Graphical Models.

## Installation

Enter the following commands in a shell like *bash*, *zsh* or *powershell*:

```bash
pip install -U adadmire
```


## Data

In the directory **data** you can find two sub directories: 
* **Feist_et_al**: contains data set as discribed in (cite1, cite2). 
    * **data_raw.xlsx**: raw, unscaled data, contains measurements of 100 samples and 49 metabolites
    *  **scaled_data_raw.npy**: numpy file containing scaled version of **data_raw.xlsx**
    *  **pheno_with_simulations.xlsx**: pheno data corresponding to **data_raw.xlsx**, also contains cell stimulations
    *  **pheno.npy**: numpy file corresponding to **pheno_with_simulations.xlsx** (only contains variables batch and myc)
    *  **levels.npy**: numpy file containing the levels of the discrete variables in **pheno.npy**
* **Higuera_et_al**: contains data set as discribed in (cite1, cite3).
    * **data_raw.xlsx**: raw, unscaled data, contains measurements of 400 samples and 68 proteins (downsamples from cite3)
    *  **scaled_data_raw.npy**: numpy file containing scaled version of **data_raw.xlsx**
    *  **pheno_.xlsx**: pheno data corresponding to **data_raw.xlsx**
    *  **pheno.npy**: numpy file corresponding to **pheno.xlsx**
    *  **levels.npy**: numpy file containing the levels of the discrete variables in **pheno.npy**
## Usage

⚠️Attention: this section is currently in draft mode, i.e. the listed examples are **not yet** working and must be updated first.
<!-- TODO -->

### Example 1

```python
from adadmire import loo_cv_cor, get_threshold_continuous, get_threshold_discrete
import numpy as np

X = np.load('C:/Users/l_buc/paper_mgm/val_wolfram/data/scaled.npy')
D = np.load('C:/Users/l_buc/paper_mgm/val_wolfram/data/bm.npy')
levels = np.load('C:/Users/l_buc/paper_mgm/val_wolfram/data/levels_bm.npy')
lam_zero = np.sqrt(np.log(X.shape[1] + D.shape[1]/2)/X.shape[0])
lam_seq = np.array([-2.0,-2.25])
lam = [pow(2, x) for x in lam_seq]
lam = np.array(lam)
lam = lam_zero * lam
prob_hat, B_m, lam_opt,  x_hat_cor_xp, d_hat_cor_xp = loo_cv_cor(X,D,levels,lam)
X_cor, threshold, n_ano,  ano_index = get_threshold_continuous(X, x_hat_cor_xp, B_m)
n_ano, threshold, pos = get_threshold_discrete(D, levels, d_hat_cor_xp)
```

### Example 2

```python
import adadmire
import sklearn.datasets

# load diabetes dataset from sklearn
diab = sklearn.datasets.load_diabetes()
# 442x4 matrix with scaled features: age, sex, bmi, blood pressure
X = diab.data[:, 1:4]
y = diab.target
# Lets introduce some faulty entries
X[100, 1] = 0.8
X[200, 2] = 0.7
X[300, 3] = 0.1
X[400, 4] = 0.2
# Lets detect (and correct) them using adadmire
ca = adadmire.detect_anomalies(X)
print(ca)
X_corrected = adadmire.correct_anomalies(X)
```

## Contribute

In case you have **questions**, **feature requests** or find any **bugs** in adadmire, please create a corresponding issue at [gitlab.spang-lab.de/bul38390/admire/issues](https://github.com/spang-lab/adadmire/issues).

In case you want to **write code** for this package, see [Contribute](https://github.com/spang-lab/adadmire/blob/main/doc/contribute.md) for details.