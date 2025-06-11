# Salient Software Development Kit

This document is intended for Salient developers who are contributing to the package.
For user-facing documentation (including a full API reference), see [docs/index](./docs/index.md).

## Installation

First, you'll need to set up ssh so you can git clone. Then:

```bash
cd ~
mkdir salientsdk
cd salientsdk
git clone git@github.com:Salient-Predictions/salientsdk.git
```

## Environment Setup

(1) If you use vscode or cursor, `./vscode/extensions.json` will suggest extensions and `./vscode/settings.json` will set you up automatically.

(2) We use `poetry` to manage dependencies. Install it and execute

```bash
# Get and install poetry, if you don't have it already
curl -sSL https://install.python-poetry.org | python3 -
# This will install all the dependency packages (including dev dependencies)
poetry install

# Activate your poetry environment
source $(poetry env info --path)/bin/activate

```

(3) Running `poetry install` should grab `pre-commit`. Activate it, and you'll get a suite of automated quality checks as specified in `.pre-commit-config.yaml`.

```bash
pre-commit install
# or, if you haven't activated a poetry virtual environment
poetry run pre-commit install
```

To make use of Salient's testing credentials, you'll need to authorize with the gcs secrets manager. If you have already installed the `salient` development environment, you have probably already done this.

```
gcloud init
gcloud auth application-default login
```

## Contributing

- We use google-style docstrings for all user-facing functions, which `ruff` will enforce rigidly.
- Export user-facing functions via `salientsdk/__init__.py/__all__`
- To make a change, branch `develop` and submit a pull request.

## Pull Requests

By default, pull requests will go to the `develop` branch. Github actions will run the `pre-commit` workflow, `pytest` everything in `/tests`, and execute all noteboox in `/examples`. To preempt this, include `skip-ci` in your commit message or PR label.

## Release

A pull request will be automatically created every other Monday that attempts to merge `develop` with `main`. All merges to `main` will trigger:

- Build doc and deploy to [github-pages](https://salient-predictions.github.io/salientsdk/), which redirects [sdk.salientpredictions.com](https://sdk.salientpredictions.com/)
- Increment the "patch" version number
- Deploy to [PyPI](https://pypi.org/project/salientsdk/).

To skip the deployment process on `main`, add `skip-deploy" in your PR label or commit message.

## Checklist for Adding a New API Endpoint

- Add a file in `/salientsdk/<endpoint>_api.py`
- Add a function in that file called `<endpoint>`
- Export the function in `/salientsdk/__init.py`
- Add command line support in `/salientsdk/__main__.py`. This can be a lightweight way to test the function while it's under development.
- Write a test in `/tests/test_<endpoint>_api.py`. Don't forget to test the command line interface in `main`.
- Reference the exported function in `/docs/api.md`
