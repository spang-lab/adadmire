# Testing

Adadmire uses [pytest](https://docs.pytest.org/en/latest/), [pytest-cov](https://pypi.org/project/pytest-cov/) and [pytest-xdist](https://pypi.org/project/pytest-xdist/) for testing. These are declared as the `test` extra, so `pip install -e ".[test]"` installs adadmire together with all testing dependencies. To run the full test suite, call `pytest --runslow` from the root of the repository. Note that the slow tests take roughly 10 to 20 minutes; passing `-n auto` distributes them over all available cores. The tests are located in the [tests](https://github.com/spang-lab/adadmire/tree/main/tests) folder. To skip the slow tests, omit the `--runslow` flag. To run a specific test, use the `-k` flag, e.g. `pytest -k test_transform_data`. To avoid having to install the package repeatedly after each change, we recommend performing an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) using `pip install -e .`. This allows you to modify the source code and run the tests for the modified version without needing to reinstall the package. Note, however, that it is still required to reinstall the package whenever you modify the [pyproject.toml](https://github.com/spang-lab/adadmire/tree/main/pyproject.toml) file.

```bash
# All commands for copy pasting
git clone https://github.com/spang-lab/adadmire.git # clone adadmire
cd adadmire
pip install -e ".[test]" # install adadmire (editable) plus testing dependencies
pytest --tb=short # run all fast tests and show short traceback
pytest --runslow -n auto # run all tests/test*py files, in parallel
pytest -k test_transform_data # run tests/test_transform_data.py only
pytest -s # show stdout of commands during tests (useful for debugging)
```
