# How to Create Requirements File

The requirements file should include all needed python packages.

- Create a clean venv and install all needed packages
- Use pip freeze to save the requirements into `.aitk/requirements/XXX/YYY_py3.12.9.txt`
    - Python version is also flexible as long as it is in uv supported list [Python versions | uv](https://docs.astral.sh/uv/concepts/python-versions/#viewing-available-python-versions)
    - Choose a python version that most packages have already built against it. For example, onnx 1.17.0 build against 312 but [not latest 313](https://pypi.org/pypi/onnx/1.17.0/json)

## Create venv patch file

If recipes need a venv that is only a little different from another one, we could use patch to avoid creating a new venv.

It will be in the format of .aitk/requirements/XXX/YYY_py3.12.9-ZZZ.txt. ZZZ is the patch name.

In demo pr, we use [olive-recipes/.aitk/requirements/Intel/Test_py3.12.9-Transformers4.49.txt at main · microsoft/olive-recipes](https://github.com/microsoft/olive-recipes/blob/main/.aitk/requirements/Intel/Test_py3.12.9-Transformers4.49.txt) to bump version of transformers lib.

## Venv file special commands (WIP)

We support special commands to enable better venv setup. See [Requirement Special Commands](../guide/ReqCommands.md).
