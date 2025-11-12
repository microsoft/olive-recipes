from pathlib import Path
from .model_info import ModelList
from .utils import isLLM_by_id, open_ex
from .generator_common import create_model_parameter

def generator_trtrtx(id: str, recipe, folder: Path, modelList: ModelList):
    aitk = recipe.get("aitk", {})
    auto = aitk.get("auto", True)
    isLLM = isLLM_by_id(id)
    if not auto or not isLLM:
        return
    name = f"Convert to NVIDIA TRT for RTX"

    file = recipe.get("file")
    configFile = folder / file

    parameter = create_model_parameter(aitk, name, configFile)
    parameter.addCpu = False
    parameter.isLLM = isLLM

    parameter.writeIfChanged()
    print(f"\tGenerated NVIDIA TRT configuration for {file}")
