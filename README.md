# adadmire

Functions for detecting anomalies in tabular datasets using Mixed Graphical Models.

## Installation

Enter the following commands in a shell like *bash*, *zsh* or *powershell*:

```bash
pip install -U adadmire
pip install git+https://github.com/yhat/ggplot.git
```

## Usage

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

## Questions, Feature-Requests, Bugs

In case you have questions, feature requests or find any bugs in adadmire, please create a corresponding issue at [gitlab.spang-lab.de/bul38390/admire/issues](https://gitlab.spang-lab.de/bul38390/admire/issues).

## Contribute code

In case you want to write code for this package, please create an [Issue](https://gitlab.spang-lab.de/bul38390/admire/issues) first, in which you describe your planned code contribution. After acceptance of your proposal by an active maintainer of the repository you will get permissions to create branches for this repository. After this, please follow the steps outlined in the following subsections to create new versions of adadmire.

🗒️Note: all following subsections are based on <https://packaging.python.org/en/latest/tutorials/packaging-projects/>.

⚠️Important: all commands mentioned in the following subsections should be run from the root folder of this repository.


### Build the package:

* Run command `python -m pip install --upgrade build twine` to install modules [build](https://pypi.org/project/build/) and [twine](https://pypi.org/project/twine/) (required for building and uploading of packages) 
* Run command `python -m build` to generate pre-built version of your package (`dist/adadmire*.whl`) as well as a source version of your package (`dist/adadmire*.tar.gz`)

### Upload to PyPI Test Server

This step is optional and only required if you want to try out the upload and following installation.

* Run command `python3 -m twine upload --repository testpypi dist/*` to upload your package to [test.pypi.org](https://test.pypi.org/). **Important**: for this to work, you need to create an account at [test.pypi.org](https://test.pypi.org/) first! Please also note, that you need two separate accounts for [test.pypi.org](https://test.pypi.org/) and [pypi.org](https://pypi.org/).
* Check that your new version is listed at [test.pypi.org/project/adadmire](https://test.pypi.org/project/adadmire)
* Run command `py -m pip install --upgrade --index-url https://test.pypi.org/simple/ --no-deps adadmire` to check that your new version can be installed via pip

### Upload to PyPI Production Server

* Run command `python3 -m twine upload dist/*` to upload your package to [test.pypi.org](https://test.pypi.org/). **Important**: for this to work, you need to create an account at [pypi.org](https://pypi.org/) first! Please also note, that you need two separate accounts for [pypi.org](https://pypi.org/) and [test.pypi.org](https://test.pypi.org/).
* Check that your new version is listed at [pypi.org/project/adadmire](https://pypi.org/project/adadmire)
* Run command `py -m pip install --upgrade adadmire` to check that your new version can be installed via pip
