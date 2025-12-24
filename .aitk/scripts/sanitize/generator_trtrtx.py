import json
from pathlib import Path

from .constants import OlivePassNames, OlivePropertyNames
from .generator_common import create_model_parameter, set_optimization_path
from .generator_dml import generate_quantization_config
from .model_info import ModelList
from .model_parameter import ModelParameter
from .utils import isLLM_by_id, open_ex


def generate_additional_config(configFile: Path, parameter: ModelParameter):
    with open_ex(configFile, "r") as f:
        content = json.load(f)
    for k, v in content[OlivePropertyNames.Passes].items():
        if v[OlivePropertyNames.Type].lower() == OlivePassNames.NVModelOptQuantization:
            if not parameter.executeRuntimeFeatures:
                parameter.executeRuntimeFeatures = []
            parameter.executeRuntimeFeatures.append("NVModelOptQuantization")


def generator_trtrtx(id: str, recipe, folder: Path, modelList: ModelList):
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

    name = "Convert to NVIDIA TRT for RTX"

    parameter = create_model_parameter(aitk, name, configFile)
    parameter.isLLM = isLLM

    generate_additional_config(configFile, parameter)

    quantize = generate_quantization_config(configFile, parameter)
    if quantize:
        parameter.sections.append(quantize)

    parameter.writeIfChanged()
    print(f"\tGenerated NVIDIA TRT configuration for {file}")
