# Contributing

In case you have **questions**, **feature requests** or find any **bugs** in adadmire, please create a corresponding issue at [gitlab.spang-lab.de/bul38390/admire/issues](https://github.com/spang-lab/adadmire/issues).

In case you want to **write code** for this package, please also create an [Issue](https://github.com/spang-lab/adadmire/issues) first, in which you describe your planned code contribution. After acceptance of your proposal by an active maintainer of the repository you will get permissions to create branches for this repository. After this, please follow the steps outlined in the following to create new versions of adadmire.

🗒️Note: all following steps are based on <https://packaging.python.org/en/latest/tutorials/packaging-projects/>.

⚠️Important: all commands mentioned in the following should be run from the root folder of this repository.

1. **Make your code changes**
2. **Increase the version** in [pyproject.toml](../pyproject.toml)
3. **Build the package**:
   * Run command `python -m pip install --upgrade build twine` to install modules [build](https://pypi.org/project/build/) and [twine](https://pypi.org/project/twine/) (required for building and uploading of packages)
   * Run command `python -m build` to generate pre-built version of your package (`dist/adadmire*.whl`) as well as a source version of your package (`dist/adadmire*.tar.gz`)
4. **Upload to PyPI Test Server**: this step is optional and only required if you want to try out the upload and following installation.
   * Run command `python -m twine upload --repository testpypi dist/*` to upload your package to [test.pypi.org](https://test.pypi.org/). **Important**: for this to work, you need to create an account at [test.pypi.org](https://test.pypi.org/) first! Please also note, that you need two separate accounts for [test.pypi.org](https://test.pypi.org/) and [pypi.org](https://pypi.org/).
   * Check that your new version is listed at [test.pypi.org/project/adadmire](https://test.pypi.org/project/adadmire)
   * Run command `python -m pip install --upgrade --index-url https://test.pypi.org/simple/ --no-deps adadmire` to check that your new version can be installed via pip
5. **Upload to PyPI Production Server**
   * Run command `python -m twine upload dist/*` to upload your package to [test.pypi.org](https://test.pypi.org/). **Important**: for this to work, you need to create an account at [pypi.org](https://pypi.org/) first! Please also note, that you need two separate accounts for [pypi.org](https://pypi.org/) and [test.pypi.org](https://test.pypi.org/).
   * Check that your new version is listed at [pypi.org/project/adadmire](https://pypi.org/project/adadmire)
   * Run command `python -m pip install --upgrade adadmire` to check that your new version can be installed via pip

All above commands in short

```bash
python -m pip install --upgrade build twine wheel # Update packages
rm dist/* # Clean dist folder
python -m build # Build package. IMPORTANT: version updated in pyproject.toml?
# IF POWERSHELL
$whl=$(ls ./dist/adadmire-*-py3-none-any.whl).FullName
python -m pip install $whl # Test installation locally
# ELSE
python -m pip install ./dist/adadmire-*-py3-none-any.whl
# END IF
python -m twine upload --repository testpypi dist/* # Upload to test server
python -m pip install --upgrade --index-url https://test.pypi.org/simple/ --no-deps adadmire
python -m twine upload dist/* # Upload to production server
python -m pip install --upgrade adadmire # Test installation from server
# IMPORTANT: do not forget to push your changes to Github as well ;)
```

## Tips and Tricks

### Create a .pypirc file

Create a [.pypirc](https://packaging.python.org/en/latest/specifications/pypirc/) file in your local home directory. This way you don't have to type your PyPI username and password each time. Mine looks as follows:

```
[testpypi]
repository = https://test.pypi.org/legacy/
username = <my_test_pypi_username>
password = <my_test_pypi_password>

[pypi]
repository = https://upload.pypi.org/legacy/
username = <my_pypi_username>
password = <my_pypi_password>
```
