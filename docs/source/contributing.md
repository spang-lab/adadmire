# Contributing

In case you have **questions**, **feature requests** or find any **bugs** in adadmire, please create a corresponding issue at [gitlab.spang-lab.de/bul38390/admire/issues](https://github.com/spang-lab/adadmire/issues).

In case you want to **write code** for this package, please also create an [Issue](https://github.com/spang-lab/adadmire/issues) first, in which you describe your planned code contribution. After acceptance of your proposal by an active maintainer of the repository you will get permissions to create branches for this repository. After this, please follow the steps outlined in the following to create a new version of adadmire.

1. Clone the [adadmire repository](https://github.com/spang-lab/adadmire)
2. Create a new branch for your changes
3. Make your code changes by updating the files in folder [src/adadmire](https://github.com/spang-lab/adadmire/tree/main/src/adadmire)
4. Test your changes by following the steps outlined in [Testing](https://spang-lab.github.io/adadmire/testing.html)
5. Update the tests in folder [tests](https://github.com/spang-lab/adadmire/tree/main/tests) is necessary
6. Increase the version in [pyproject.toml](https://github.com/spang-lab/adadmire/blob/main/pyproject.toml)
7. Push your changes to the repository
8. Create a pull request for your changes
9. Wait for the automated tests to finish.
10. In case the automated tests fail, fix the problems and push the changes to the repository again.
11. In case the automated tests pass, wait for a maintainer to review your changes.
12. After your changes have been accepted, a new github release and pypi release will be created automatically.
