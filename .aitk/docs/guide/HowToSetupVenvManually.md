# How to setup venv manually

Sometimes, you may want to setup venv and convert model via olive recipe yourselves without AITK. Here is how to do that.

## Venv Creation

We recommend using [uv](https://docs.astral.sh/uv/reference/cli/#uv-venv) to manage venv and now our venvs are using python 3.12.9.

## Requirements installation

When AITK setups a venv, we usually install 3 kinds of requirements in order.

- Base requirements: the fundamental requirements including all packages
- Patch (feature) requirements: the additional requirements for the recipe that are installed after base
- Project requirements: the requirements.txt file inside project to allow user for customization

To install a requirements file, one should first check [Special commands](./ReqCommands.md) and then install it normally like via `uv pip install -r xxx.txt`.

### Base requirements

Base requirements is determined by two ways:
- Runtime Overwrite: it is setup in json.config.
    - for example, in [DeepSeek QNN](../../../deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_qnn_config.json.config), `runtimeOverwrite.executeEp` is `CUDAExecutionProvider`, so it will use [requirements-NvidiaGPU](../../requirements/requirements-NvidiaGPU.txt)
    - for example, in [Qwen OV](../../../Qwen-Qwen2.5-0.5B/aitk/qwen2_5_ov_config.json.config), `runtimeOverwrite.executeRequirement` is `Intel/Test_py3.12.9`, so it will use [Intel/Test](../../requirements/Intel/Test_py3.12.9.txt)
- Runtime Map: it is determined in `get_execute_runtime` in [utils.py](../../scripts/sanitize/utils.py).
    - for example, for intel recipe, it will use [requirements-IntelNPU](../../requirements/requirements-IntelNPU.txt)
    - for example, for dml recipe, it will use [requirements-WCR](../../requirements/requirements-WCR.txt)

### Patch (feature) requirements

This is determined by `executeRuntimeFeatures` in json.config. One could find it by append the name in base requirements.
- for example, in [DeepSeek QNN](../../../deepseek-ai-DeepSeek-R1-Distill-Qwen-1.5B/aitk/deepseek_qnn_config.json.config), `executeRuntimeFeatures` is `AutoGptq`, so it will use [requirements-NvidiaGPU-AutoGptq](../../requirements/requirements-NvidiaGPU-AutoGptq.txt)
- for example, in [Qwen OV](../../../Qwen-Qwen2.5-0.5B/aitk/qwen2_5_ov_config.json.config), `executeRuntimeFeatures` is `Transformers4.49`, so it will use [Intel/Test-Transformers4.49](../../requirements/Intel/Test_py3.12.9-Transformers4.49.txt)

### Project requirements

The requirements.txt file inside project.
