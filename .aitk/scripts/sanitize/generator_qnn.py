from pathlib import Path
import json

from .constants import OlivePassNames, OlivePropertyNames, PhaseTypeEnum
from .generator_amd import generate_quantization_config
from .generator_common import create_model_parameter, set_optimization_path
from .model_info import ModelList
from .model_parameter import ModelParameter
from .utils import isLLM_by_id, open_ex


def setup_features(content: dict, parameter: ModelParameter):
    def add(feature: str):
        if parameter.executeRuntimeFeatures is None:
            parameter.executeRuntimeFeatures = []
        if feature not in parameter.executeRuntimeFeatures:
            parameter.executeRuntimeFeatures.append(feature)

    for k, v in content[OlivePropertyNames.Passes].items():
        if v[OlivePropertyNames.Type].lower() == OlivePassNames.GptqQuantizer:
            add("AutoGptq")
        elif v[OlivePropertyNames.Type].lower() == OlivePassNames.GptqModel:
            add("GptqModel")


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
    if "npu" in runtime_values:
        parameter.isQNNLLM = True

    with open_ex(configFile, "r") as f:
        content = json.load(f)
    quantize = generate_quantization_config(content, modelList, parameter)
    if quantize:
        parameter.sections.append(quantize)

    setup_features(content, parameter)

    parameter.writeIfChanged()
    print(f"\tGenerated QNN configuration for {file}")
