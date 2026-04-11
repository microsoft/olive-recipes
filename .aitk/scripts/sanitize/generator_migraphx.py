from pathlib import Path

from .generator_common import create_model_parameter, set_optimization_path
from .generator_dml import generate_quantization_config
from .model_info import ModelList
from .model_parameter import ModelParameter
from .utils import isLLM_by_id


def generator_migraphx(id: str, recipe, folder: Path, modelList: ModelList):
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

    name = "Convert to AMD GPU"

    parameter = create_model_parameter(aitk, name, configFile)
    parameter.isLLM = isLLM

    quantize = generate_quantization_config(configFile, parameter)
    if quantize:
        parameter.sections.append(quantize)

    parameter.writeIfChanged()
    print(f"\tGenerated MIGraphX configuration for {file}")
