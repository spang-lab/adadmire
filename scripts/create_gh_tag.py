"""
Purpose:
    This script is used to create a tag in a GitHub repository.

Usage:
    python create_gh_tag.py

Required environment variables:
    GITHUB_TOKEN: The GitHub token used to authenticate with the GitHub API.
    GITHUB_REPOSITORY: The repository to create the tag in, in the format 'owner/repo'.
    VERSION: The version to create a tag for.

Details:
    It uses the PyGithub library to interact with the GitHub API. The script retrieves the GitHub token, repository, and version from environment variables. It then attempts to get a reference to a tag with the name 'v' + version. If the tag does not exist, it creates a new tag with that name, pointing to the latest commit in the repository.
"""

from github import Github
import os

g = Github(os.getenv("GITHUB_TOKEN"))
repo = g.get_repo(os.getenv("GITHUB_REPOSITORY"))
version = os.getenv("VERSION")

try:
    ref = repo.get_git_ref('tags/v' + version)
except:
    ref = repo.create_git_ref('refs/tags/v' + version, repo.get_commits()[0].sha)
