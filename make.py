import git
import toml
import github
import argparse
import sys
import os
import subprocess

prog = "make.py"
usage = "python make.py (gh_release|pypi_release) [--force|--dry-run] [--help]"
description = "Adadmire Make Script"
epilog = f"For further details see {__file__}"
details = """
When called with argument `gh_release`, this script creates a tag and release in adadmire's GitHub repository with the version taken from `pyproject.toml` if the following conditions are fulfilled:

1. The local git client is able to authenticate with the GitHub API. This can be achieved by setting up SSH keys or by configuring environment variable `GITHUB_TOKEN`.
2. Tag `vX.Y.Z` does not exist yet where `X.Y.Z` is the version taken from `pyproject.toml`.
3. The current commit hash matches the latest commit hash on the main branch.
4. The script is run by a GitHub Action triggered by a push to the main branch.

When called with argument `pypi_release`, this script creates a release on PyPI with the version taken from `pyproject.toml` if the following conditions are fulfilled:

1. The local twine installation is able to authenticate with the PyPI API. This can be achieved by setting up a local `.pypirc` file or by configuring environment variables `TWINE_USERNAME` and `TWINE_PASSWORD`.
1. Builds the package by running `python -m build`. This creates a dist directory containing the built package.
2. Publishes the package to PyPI by running `twine upload dist/*`. This uploads all files in the dist directory to PyPI.
"""

# Constants
blue = "\033[94m"
norm = "\033[0m"
def title(text): return print(f"{blue}{text}{norm}")


# Parse commandline arguments
parser = argparse.ArgumentParser(prog, usage, description, epilog, formatter_class=argparse.RawTextHelpFormatter, )
parser.add_argument('target', choices=['gh_release', 'pypi_release'], help='Target to make', )
parser.add_argument('--force', action='store_true',
                    help='Force making of target even when run outside a github action?')
parser.add_argument('--dry-run', action='store_true', help='Skip making of target even when conditions are fulfilled?')
args = parser.parse_args()
target = args.target
force = args.force
dry_run = args.dry_run
title(f"Commandline arguments:")
print(f"target = {target}")
print(f"force = {force}")
print(f"dry_run = {dry_run}")

# Collect info about local repo
repo = git.Repo()
url = repo.remotes.origin.url
branch = repo.active_branch
commit_hash = repo.head.object.hexsha
commit_datetime = repo.head.object.committed_datetime
commit_tags = repo.git.tag('--points-at', 'HEAD')
tags = [str(tag) for tag in repo.tags]
title(f"\nLocal Repo Details:")
print(f"Current repo url: {url}")
print(f"Current repo branch: {branch}")
print(f"Current commit hash: {commit_hash}")
print(f"Current commit datetime: {commit_datetime}")
print(f"Current commit tags: {commit_tags}")
print(f"Existing tags: {', '.join(tags)}")

# Read version from pyproject.toml
version = toml.load('pyproject.toml')['project']['version']
tag = 'v' + version
title(f"\nPyproject Details:")
print(f"Current version: {version}")

# Collect info about remote Github repo
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
gh_repo = github.Github(GITHUB_TOKEN).get_repo("spang-lab/adadmire")
latest_commit_hash_main = gh_repo.get_branch("main").commit.sha
title(f"\nGithub Repo Details:")
print(f"Latest commit hash on main branch: {latest_commit_hash_main}")

# Collect info about Github Action triggering this script
GITHUB_REF = os.getenv("GITHUB_REF")
GITHUB_EVENT_NAME = os.getenv("GITHUB_EVENT_NAME")
triggered_by_push_to_main = GITHUB_REF == "refs/heads/main" and GITHUB_EVENT_NAME == "push"
title(f"\nGithub Action Details:")
print(f"Triggered by Github Event: {GITHUB_EVENT_NAME}")
print(f"Triggered by Github Ref: {GITHUB_REF}")
print(f"Triggered by push to main branch: {triggered_by_push_to_main}")

# Actually make target
title(f"\nMaking Target: {target}")

if target == "gh_release":
    assert tag not in tags, f"Tag {tag} already exists. Skipping creation."
    assert commit_hash == latest_commit_hash_main, f"Current commit_hash does not match latest commit hash on main branch. Skipping creation of tag {tag}."
    assert triggered_by_push_to_main or force == True, "This script should be run by a GitHub Action triggered by a push to the main branch."
    if dry_run:
        print(f"Skipping creation of tag {tag} because dry_run is True")
    else:
        print(f"Creating tag {tag} and pushing to GitHub.")
        gh_repo.create_git_tag_and_release(tag=tag, tag_message=tag, release_name=tag, release_message=tag, object=commit_hash, type="commit")

if target == "pypi_release":
    assert tag not in tags, f"Tag {tag} doesn't exist on Github. Run `python make gh_release` first."
    assert commit_hash == latest_commit_hash_main or force == True, f"Current commit_hash {commit_hash} does not match latest commit hash on main branch {latest_commit_hash_main}. Skipping creation of PyPI release."
    assert triggered_by_push_to_main or force == True, f"This script should be run by a GitHub Action triggered by a push to the main branch. Skipping creation of PyPI release."
    if dry_run:
        print(f"Skipping creation of PyPI release because dry_run is True")
    else:
        print(f"Building and publishing package to PyPI.")
        subprocess.run([sys.executable, "-m", "build"])
        subprocess.run(["twine", "upload", "dist/*"])
