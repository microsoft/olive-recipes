from pathlib import Path

from .generator_amd import generate_quantization_config
from .generator_common import create_model_parameter
from .model_info import ModelList
from .utils import isLLM_by_id


def generator_qnn(id: str, recipe, folder: Path, modelList: ModelList):
    aitk = recipe.get("aitk", {})
    auto = aitk.get("auto", True)
    isLLM = isLLM_by_id(id)
    if not auto or not isLLM:
        return
    runtime_values: list[str] = recipe.get("devices", [recipe.get("device")])
    name = f"Convert to Qualcomm {"/".join([runtime.upper() for runtime in runtime_values])}"

    file = recipe.get("file")
    configFile = folder / file

    parameter = create_model_parameter(aitk, name, configFile)
    parameter.isQNNLLM = True

    quantize = generate_quantization_config(configFile, modelList)
    if quantize:
        parameter.sections.append(quantize)

    parameter.writeIfChanged()
    print(f"\tGenerated QNN configuration for {file}")
