from pathlib import Path

from .generator_common import create_model_parameter
from .generator_dml import generate_quantization_config
from .model_info import ModelList
from .utils import isLLM_by_id


def generator_migraphx(id: str, recipe, folder: Path, modelList: ModelList):
    aitk = recipe.get("aitk", {})
    auto = aitk.get("auto", True)
    if not auto:
        return

    isLLM = isLLM_by_id(id)
    if not isLLM:
        return

    file = recipe.get("file")
    configFile = folder / file

    name = "Convert to AMD GPU"

    parameter = create_model_parameter(aitk, name, configFile)
    parameter.isLLM = isLLM

    quantize = generate_quantization_config(configFile, parameter)
    if quantize:
        parameter.sections.append(quantize)

    parameter.writeIfChanged()
    print(f"\tGenerated MIGraphX configuration for {file}")
    
