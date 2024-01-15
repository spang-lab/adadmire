import argparse
import json
import os
import subprocess
import sys

import git
import github
import packaging.version
import requests
import toml

prog = "make.py"
description = "Adadmire Make Script"
usage = "python make.py TARGET [--force|--dry-run] [--help]"
epilog = f"For further details see {__file__ if '__file__' in globals() else 'make.py'}"


def parse_args(argv=sys.argv[1:]):
    targets = ['gh_release', 'pypi_release', 'version_check', 'docs', 'docs_release']
    parser = argparse.ArgumentParser(prog, usage, description, epilog, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('target', choices=targets, metavar="TARGET", help=f'Target to make. Valid targets are:\n{", ".join(targets)}')
    parser.add_argument('--force', action='store_true', help='Force making of target despite failed checks?')
    parser.add_argument('--dry-run', action='store_true', help='Skip making of target?')
    args = parser.parse_args(args=argv)
    h1(f"Commandline arguments:")
    print(f"target = {args.target}")
    print(f"force = {args.force}")
    print(f"dry_run = {args.dry_run}")
    return args


cyan = "\033[36m"
blue = "\033[94m"
red = "\033[91m"
green = "\033[92m"
norm = "\033[0m"


def h1(text, *args, **kwargs):
    print(f"{blue}# {text}{norm}", *args, **kwargs)


def h2(text):
    print(f"{blue}## {text}{norm}")


def print_test_results(test_descriptions, tests_passed):
    for desc, passed in zip(test_descriptions, tests_passed):
        print(f"{desc}: {f'{green}ok{norm}' if passed else f'{red}failed{norm}'}")


def make_gh_release(args):

    h1("Collecting info about local repo, github repo, github action and pyproject.toml")
    local_repo = LocalRepo()
    github_repo = GithubRepo()
    github_action = GithubAction()
    pyproject = Pyproject()
    tag = pyproject.tag

    h1("Testing whether conditions for making a Github release are fulfilled")
    test_descriptions = ["Tag does not exist yet",
                         "Current commit hash matches latest commit hash on main branch",
                         "This script is run by a GitHub Action triggered by a push to main"]
    tests_passed = [pyproject.tag not in github_repo.tags,
                    local_repo.commit_hash == github_repo.latest_commit_hash_main,
                    github_action.triggered_by_push_to_main]
    print_test_results(test_descriptions, tests_passed)

    if (all(tests_passed) or args.force) and (not args.dry_run):
        h1(f"Tagging commit {local_repo.commit_hash} as {tag}")
        github_repo.repo.create_git_tag(tag=tag, message=tag, object=local_repo.commit_hash, type="commit")
        h1(f"Creating release {tag} from tag {tag}")
        try:
            github_repo.repo.create_git_release(tag=tag, name=tag, message=tag)
            h1(f"Finished creation of tag and release")
        except github.GithubException as e:
            print(f"Failed to create release. Error details:")
            print(e)
            sys.exit(1)
    sys.exit(0 if all(tests_passed) or args.force or args.dry_run else 1)


def make_pypi_release(args):

    h1("Collecting info about local repo, github repo, pyproject.toml, github action and PyPI package")
    local_repo = LocalRepo()
    github_repo = GithubRepo()
    github_action = GithubAction()
    pyproject = Pyproject()
    pypi = PyPi()

    h1("Testing whether conditions for making a PyPI release are fulfilled")
    test_descriptions = ["Version as specified in .pyproject.toml does not exist on PyPI yet",
                         "Current commit hash matches latest commit hash on main branch",
                         "This script is run by a GitHub Action triggered by a push to main"]
    tests_passed = [pyproject.version not in pypi.versions,
                    local_repo.commit_hash == github_repo.latest_commit_hash_main,
                    github_action.triggered_by_push_to_main]
    print_test_results(test_descriptions, tests_passed)

    if (all(tests_passed) or args.force) and (not args.dry_run):
        h1(f"Building adadmire...", end=" ")
        subprocess.run([sys.executable, "-m", "build"], check=True)
        h1(f"Uploading adadmire to PyPI")
        cmd = ["twine", "upload", "dist/*", "--non-interactive", "--verbose"]
        proc = subprocess.run(cmd, check=True, stderr=subprocess.STDOUT)
    sys.exit(0 if all(tests_passed) or args.force or args.dry_run else 1)


def make_version_check(args):

    h1("Collecting info about pyproject.toml and PyPI package")
    pyproject = Pyproject()
    pypi = PyPi()
    vloc = packaging.version.parse(pyproject.version)
    vpip = packaging.version.parse(pypi.latest_version)

    h1("Testing whether local version has been updated compared to latest PyPI version")
    if vloc > vpip:
        print(f"Test result: {green}ok{norm}")
        sys.exit(0)
    else:
        print(f"Test result: {red}failed{norm}")
        sys.exit(1)


def make_docs(args):
    if not args.dry_run:
        h1("Building docs")
        subprocess.run(["make", "html"], cwd="docs")


def make_docs_release(args):
    """Upload documentation to gh-pages branch using [ghg-import](https://github.com/c-w/ghp-import)

    Usage: ghp-import [OPTIONS] DIRECTORY

    Options:
        -n         Include a .nojekyll file in the branch.
        -c CNAME   Write a CNAME file with the given CNAME.
        -m MESG    The commit message to use on the target branch.
        -p         Push the branch to origin/{branch} after committing.
        -x PREFIX  The prefix to add to each file that gets pushed to the remote.
        -f         Force the push to the repository.
        -o         Force new commit without parent history.
        -r REMOTE  The name of the remote to push to. [origin]
        -b BRANCH  Name of the branch to write to. [gh-pages]
        -s         Use the shell when invoking Git. [False]
        -l         Follow symlinks when adding files. [False]
        -h         show this help message and exit
    """
    if not args.dry_run:
        h1("Building docs")
        subprocess.run(["make", "html"], cwd="docs")
        h1("Uploading docs to GitHub Pages")
        subprocess.run(["ghp-import", "-npfo", "docs/build/html"])


class LocalRepo():
    def __init__(self):
        h2(f"Local Repo Details:")
        self.repo = repo = git.Repo()
        self.url = repo.remotes.origin.url
        self.branch = repo.active_branch
        self.commit_hash = repo.head.object.hexsha
        self.commit_datetime = repo.head.object.committed_datetime
        self.commit_tags = repo.git.tag('--points-at', 'HEAD')
        self.tags = [str(tag) for tag in repo.tags]
        print(f"Current repo url: {self.url}")
        print(f"Current repo branch: {self.branch}")
        print(f"Current commit hash: {self.commit_hash}")
        print(f"Current commit datetime: {self.commit_datetime}")
        print(f"Current commit tags: {self.commit_tags}")
        print(f"Existing tags: {', '.join(self.tags)}")


class GithubRepo():
    def __init__(self):
        h2(f"Github Repo Details:")
        self.auth = github.Auth.Token(os.getenv("GITHUB_TOKEN"))
        self.conn = github.Github(auth=self.auth)
        self.repo = self.conn.get_repo("spang-lab/adadmire")
        try:
            self.user = self.conn.get_user().login
        except github.GithubException as e:
            print(f"Failed to get user. Setting user to UNKNOWN. Error details:")
            print(e)
            self.user = "UNKNOWN"
        self.tags = [tag.name for tag in self.repo.get_tags()]
        self.latest_commit_hash_main = self.repo.get_branch("main").commit.sha
        print(f"Latest commit hash on main branch: {self.latest_commit_hash_main}")
        print(f"Logged in as: {self.user}")


class Pyproject():
    def __init__(self):
        self.toml = toml.load('pyproject.toml')
        self.version = self.toml['project']['version']
        self.tag = 'v' + self.version
        h2(f"Pyproject Toml Details:")
        print(f"Version: {self.version}")
        print(f"Tag: {self.tag}")


class GithubAction():
    def __init__(self):
        GITHUB_REF = os.getenv("GITHUB_REF")
        GITHUB_EVENT_NAME = os.getenv("GITHUB_EVENT_NAME")
        self.triggered_by_push_to_main = GITHUB_REF == "refs/heads/main" and GITHUB_EVENT_NAME == "push"
        h2(f"Github Action Details:")
        print(f"Triggered by Github Event: {GITHUB_EVENT_NAME}")
        print(f"Triggered by Github Ref: {GITHUB_REF}")
        print(f"Triggered by push to main branch: {self.triggered_by_push_to_main}")


class PyPi():
    def __init__(self):
        response = requests.get('https://pypi.org/pypi/adadmire/json')
        self.json = json.loads(response.text)
        self.latest_version = self.json['info']['version']
        self.versions = list(self.json['releases'].keys())

        h2(f"PyPI Details:")
        print(f"PyPI versions: {self.versions}")
        print(f"Latest PyPI version: {self.latest_version}")
        print(f"len(TWINE_USERNAME) : {len(os.getenv('TWINE_USERNAME', ''))}")
        print(f"len(TWINE_PASSWORD): {len(os.getenv('TWINE_PASSWORD', ''))}")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    target_funcs = {
        "gh_release": make_gh_release,
        "pypi_release": make_pypi_release,
        "version_check": make_version_check,
        "docs": make_docs,
        "docs_release": make_docs_release
    }
    target_func = target_funcs[args.target]
    target_func(args)
