# Contributing

We're always looking for your help to improve the product (bug fixes, new features, documentation, etc).

## Repository Organization

Repository is organized by "model's name" as root folder, with each folder containing recipes specific to it.
Each sub-folder within the root folder of the model might be organized by whatever criteria (based on devices, ExecutionProvider, maintainer, or something entirely different) best suits the recipes maintainer.

Each folder (and subsequent sub-folders), however, **are required to have a file named `info.yml`** that lists details about what that specific folder contains. This information file is not strictly structured expect that it be legal yaml file and contain a few required fields. An automation job scans these information file to collect and populate a table (viewable on repository's home page) for ease of navigation (and visibility) for users.

## Core requirements for `info.yml`

Each `info.yml` is unique to the folder it contains and, at the minimum, should contain the following information:

NOTE: All information is case-sensitive.

* arch [str]: Architecture of the model
* recipes [Recipe[]]: A list of recipes (file names) in the parent folder. For each entry,
  * name [str]: Name of the file
  * eps: [str | str[]]: One or list of supported EPs
  * devices: [str | str[]]: One or list of supported devices

Beyond the required fields, the file can include any information relevant to the recipes maintainer.
Here's an example of `info.yml` file for a bert model.
```yaml
arch: bert
recipes:
  - name: qdq.json
    devices: cpu
    eps: CPUExecutionProvider

  - name: trtrtx.json
    devices: gpu
    eps: NvTensorRTRTXExecutionProvider

  - name: vitis_ai.json
    devices: cpu
    eps: VitisAIExecutionProvider
```

## Coding conventions and standards

### Testing and Code Coverage
In the test folder, run `pip install -r requirements-test.txt` to install test dependencies.

#### Unit testing
There should be unit tests that cover the core functionality of the product, expected edge cases, and expected errors.
Code coverage from these tests should aim at maintaining over 80% coverage.

All changes should be covered by new or existing unit tests.

#### Style

Test the *behavior*, instead of the *implementation*. To make what a test is testing clear, the test methods should be named following the pattern `test_<method or function name>_<expected behavior>_[when_<condition>]`.

e.g. `test_method_x_raises_error_when_dims_is_not_a_sequence`

### Linting
Ensure that the correct develop packages are installed by `pip install -r requirements-dev.txt`.

This project uses [lintrunner](https://github.com/suo/lintrunner) for linting. It provides a consistent linting experience locally and in CI. You can initialize with

```sh
lintrunner init
```

This will install lintrunner on your system and download all the necessary
dependencies to run linters locally.
If you want to see what lintrunner init will install, run
`lintrunner init --dry-run`.

To lint local changes:

```bash
lintrunner
```

To format files and apply suggestions:

```bash
lintrunner -a
```

To lint all files:

```bash
lintrunner --all-files
```

To show help text:

```bash
lintrunner -h
```

To read more about lintrunner, see [wiki](https://github.com/pytorch/pytorch/wiki/lintrunner).
To update an existing linting rule or create a new one, modify `.lintrunner.toml` or create a
new adapter following examples in https://github.com/justinchuby/lintrunner-adapters.

### Python Code Style

Follow the [Black formatter](https://black.readthedocs.io)'s coding style when possible. A maximum line length of 120 characters is allowed.

Please adhere to the [PEP8 Style Guide](https://www.python.org/dev/peps/pep-0008/). We use [Google's python style guide](https://google.github.io/styleguide/pyguide.html) as the style guide which is an extension to PEP8.

Auto-formatting is done with `black` and `isort`. The tools are configured in `pyproject.toml`. From the root of the repository, you can run

```sh
lintrunner f --all-files
```

to format Python files.

## Licensing guidelines

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

## Code of conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Report a security issue

Security issues and bugs should be reported privately, via email, to the Microsoft Security
Response Center (MSRC) at [secure@microsoft.com](mailto:secure@microsoft.com). You should
receive a response within 24 hours. If for some reason you do not, please follow up via
email to ensure we received your original message. Further information, including the
[MSRC PGP](https://technet.microsoft.com/en-us/security/dn606155) key, can be found in
the [Security TechCenter](https://technet.microsoft.com/en-us/security/default).
