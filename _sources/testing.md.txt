# Testing

Adadmire uses [pytest](https://docs.pytest.org/en/latest/), [freezegun](https://github.com/spulec/freezegun), and [pytest-cov](https://pypi.org/project/pytest-cov/) for testing. To run the full test suite, install the mentioned packages and adadmire and then call `pytest --runslow` from the root of the repository. The tests are located in the [tests](https://github.com/spang-lab/adadmire/tree/main/tests) folder. To skip the slow tests, omit the `--runslow` flag. To run a specific test, use the `-k` flag, e.g. `pytest -k test_transform_data`. To avoid having to install the package repeatedly after each change, we recommend performing an [editable install](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) using `pip install -e .`. This allows you to modify the source code and run the tests for the modified version without needing to reinstall the package. Note, however, that it is still required to reinstall the package whenever you modify the [pyproject.toml](https://github.com/spang-lab/adadmire/tree/main/pyproject.toml) file.

```bash
# All commands for copy pasting
git clone https://github.com/spang-lab/adadmire.git # clone adadmire
cd adadmire
pip install -e . # install adadmire in editable mode
pip install pytest pytest-cov pytest-xdist freezegun # install testing dependencies
pytest --tb=short # run all fast tests and show short traceback
pytest --runslow  # run all tests/test*py files
pytest -k test_transform_data # run tests/test_transform_data.py only
pytest -s # show stdout of commands during tests (useful for debugging)
```
