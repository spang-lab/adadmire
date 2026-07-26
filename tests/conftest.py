# Pytest imports this file before running any tests. Its purpose is to add a
# command-line option `--runslow`. If this option is provided when running
# pytest, all tests will run. If it's not provided, tests marked as slow will be
# skipped. This is implemented as described here:
# https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option.

import pytest


def pytest_addoption(parser):
    """
    Add the --runslow option to the pytest command-line parser.
    The option is stored as a boolean value, with a default of False.
    """
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    """
    Add a new marker named slow to the pytest configuration. This marker can be
    used to mark tests as slow.
    """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """
    Called after pytest has collected all the tests but before it starts running
    them. If the --runslow option is not provided, it marks all tests that have
    the slow keyword with the skip marker, which causes pytest to skip these
    tests.
    """
    if config.getoption("--runslow"):
        return
    else:
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
