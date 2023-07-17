# adadmire

<!-- ATTENTION: this file will be displayed not only on Github, but also on PyPI, so NO relative relative to files in the repo must be used -->

Functions for detecting anomalies in molecular data sets using Mixed Graphical Models.

## Installation

Enter the following commands in a shell like *bash*, *zsh* or *powershell*:

```bash
pip install -U adadmire
```

## Usage

The usage example in this section require you to download the data files from folder [data](https://github.com/spang-lab/adadmire/tree/main/data) first. For a description of the contents of this folder, see section [Data](#data).

### Example 1

```python
from adadmire import loo_cv_cor, get_threshold_continuous, get_threshold_discrete
import numpy as np

# download data/Feist_et_al
# and load data
X = np.load('data/Feist_et_al/scaled_data_raw.npy') # continuous data
D = np.load('data/Feist_et_al/pheno.npy') # discrete data
levels = np.load('data/Feist_et_al/levels.npy') # levels of discrete variables
# define lambda sequence
lam_zero = np.sqrt(np.log(X.shape[1] + D.shape[1]/2)/X.shape[0])
lam_seq = np.array([-1.75,-2.0,-2.25])
lam = [pow(2, x) for x in lam_seq]
lam = np.array(lam)
lam = lam_zero * lam
# perform cross validation
prob_hat, B_m, lam_opt,  x_hat, d_hat = loo_cv_cor(X,D,levels,lam)
# determine continuous threshold
X_cor, threshold_cont, n_ano_cont,  position_cont = get_threshold_continuous(X, x_hat, B_m)
# returns: X corrected for detected anomalies, threshold, number of detected anomalies (n_ano_cont) and position
print(n_ano_cont) # 46 detected continuous anomalies
n_ano_disc, threshold_cont, position_disc = get_threshold_discrete(D, levels, d_hat)
# returns:  number of detected anomalies (n_ano_disc), threshold and position
print(n_ano_disc)
# 0 detected discrete anomalies
```

### Example 2

```python
from adadmire import loo_cv_cor, get_threshold_continuous, place_anomalies_continuous
import numpy as np
X = np.load('data/Higuera_et_al/scaled_data_raw.npy') # continuous data
D = np.load('data/Higuera_et_al/pheno.npy') # discrete data
levels = np.load('data/Higuera_et_al/levels.npy') # levels of discrete variables

# use originial data set and create simulations by introducing artificial anomalies with various strengths
X_ano, pos = place_anomalies_continuous( X, n_ano = 1360, epsilon = np.array([0.6, 0.8, 1.0, 1.2, 1.4]))
# n_ano: how many anomalies should be introduced?
# epsilon defines "strength" of introduced anomalies

# now detect anomalies using ADMIRE
lam_zero = np.sqrt(np.log(X.shape[1] + D.shape[1]/2)/X.shape[0])
lam_seq = np.array([-1.75,-2.0,-2.25])
lam = [pow(2, x) for x in lam_seq]
lam = np.array(lam)
lam = lam_zero * lam
prob_hat, B_m, lam_opt,  x_hat, d_hat = loo_cv_cor(X_ano[0],D,levels,lam) # perform cv and estimation 
X_cor, threshold_cont, n_ano_cont,  position_cont = get_threshold_continuous(X_ano, x_hat, B_m) # determine threshold
```

### Example 3

```python
from adadmire import impute
import numpy as np

# load data containing missing values in continuous
X = np.load('data/Higuera_et_al/data_na_scaled.npy')
# as well as in discrete features
D = np.load('data/Higuera_et_al/pheno_na.npy')

print(np.sum(np.isnan(X))) # 1360
print(np.sum(np.isnan(D))) # 120

levels = np.load('data/Higuera_et_al/levels.npy') # levels of discrete variables

# define Lambda sequence
lam_zero = np.sqrt(np.log(X.shape[1] + D.shape[1]/2)/X.shape[0])
lam_seq = np.array([-1.75,-2.0,-2.25])
lam = [pow(2, x) for x in lam_seq]
lam = np.array(lam)
lam = lam_zero * lam

# now impute with ADMIRE
X_imp, D_imp,lam_o = impute(X,D,levels,lam)

print(np.sum(np.isnan(X_imp))) # 0
print(np.sum(np.isnan(D_imp))) # 0
```

### Data

In the directory **data** you can find two sub directories:
* **Feist_et_al**: contains data set as discribed in [Feist et al, 2018](#feist-et-al-2018) and [Buck et al, 2023](#buck-et-al-2023).
    * **data_raw.xlsx**: raw, unscaled data, contains measurements of 100 samples and 49 metabolites
    *  **scaled_data_raw.npy**: numpy file containing scaled version of **data_raw.xlsx**
    *  **pheno_with_simulations.xlsx**: pheno data corresponding to **data_raw.xlsx**, also contains cell stimulations
    *  **pheno.npy**: numpy file corresponding to **pheno_with_simulations.xlsx** (only contains variables batch and myc)
    *  **levels.npy**: numpy file containing the levels of the discrete variables in **pheno.npy**
* **Higuera_et_al**: contains down sampled data set from [Higuera et al, 2015](#higuera-et-al-2015) as described in [Buck et al, 2023](#buck-et-al-2023).
    * **data_raw.xlsx**: raw, unscaled data, contains measurements of 400 samples and 68 proteins (down sampled from [Higuera et al, 2015](#higuera-et-al-2015))
    *  **scaled_data_raw.npy**: numpy file containing scaled version of **data_raw.xlsx**
    *  **pheno_.xlsx**: pheno data corresponding to **data_raw.xlsx**
    *  **pheno.npy**: numpy file corresponding to **pheno.xlsx**
    *  **levels.npy**: numpy file containing the levels of the discrete variables in **pheno.npy**
    *  **data_na_scaled.npy**: numpy file containing scaled version of **data_raw.xlsx** where 5% of the values are missing
    *  **pheno_na.npy**: numpy file corresponding to **pheno.xlsx** with 5% of missing values included

## Contribute

In case you have **questions**, **feature requests** or find any **bugs** in adadmire, please create a corresponding issue at [gitlab.spang-lab.de/bul38390/admire/issues](https://github.com/spang-lab/adadmire/issues).

In case you want to **write code** for this package, see [Contribute](https://github.com/spang-lab/adadmire/blob/main/doc/contribute.md) for details.

## References

##### Feist et al, 2018

Feist, Maren, et al. "Cooperative stat/nf-kb signaling regulates lymphoma metabolic reprogramming and aberrant got2 expression." Nature Communications, 2018

##### Higuera et al, 2015

Higuera, Clara et al, "Self-organizing feature maps identify proteins critical to learning in a mouse model of down syndrome." PLOS ONE, 2015

##### Buck et al, 2023

Buck, Lena et al. "Anomaly detection in mixed high dimensional molecular data"
