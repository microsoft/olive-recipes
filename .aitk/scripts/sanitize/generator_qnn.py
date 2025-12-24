from pathlib import Path

from .generator_amd import generate_quantization_config
from .generator_common import create_model_parameter, set_optimization_path
from .model_info import ModelList
from .model_parameter import ModelParameter
from .utils import isLLM_by_id


def generator_qnn(id: str, recipe, folder: Path, modelList: ModelList):
    aitk = recipe.get("aitk", {})
    auto = aitk.get("auto", True)
    if not auto:
        return

    isLLM = isLLM_by_id(id)
    file = recipe.get("file")
    configFile = folder / file

    if not isLLM:
        modelParameter = ModelParameter.Read(str(configFile) + ".config")
        set_optimization_path(modelParameter, str(configFile))
        modelParameter.writeIfChanged()
        return

    runtime_values: list[str] = recipe.get("devices", [recipe.get("device")])
    name = f"Convert to Qualcomm {"/".join([runtime.upper() for runtime in runtime_values])}"

    parameter = create_model_parameter(aitk, name, configFile)
    parameter.isQNNLLM = True

    quantize = generate_quantization_config(configFile, modelList, parameter)
    if quantize:
        parameter.sections.append(quantize)

    parameter.writeIfChanged()
    print(f"\tGenerated QNN configuration for {file}")
