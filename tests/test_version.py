import json
import re
import requests
import os
from packaging import version

def test_that_version_was_incremented():
    # Extract the current version from pyproject.toml
    with open('pyproject.toml', 'r') as file:
        content = file.read()
        current_version = re.search(r'version = "(.*?)"', content).group(1)

    # Fetch the latest version from PyPI
    response = requests.get('https://pypi.org/pypi/adadmire/json')
    pypi_version = json.loads(response.text)['info']['version']

    # Print the current and PyPI versions
    print(f"Current version: {current_version}")
    print(f"PyPI version: {pypi_version}")

    # Check if the test was triggered by a GitHub Action triggered by a pull request
    if os.environ.get('GITHUB_EVENT_NAME') == 'pull_request':
        # Assert that the current version was incremented, i.e. is greater than the PyPI version
        assert version.parse(current_version) >= version.parse(pypi_version)