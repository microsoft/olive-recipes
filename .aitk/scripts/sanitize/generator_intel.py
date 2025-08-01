from pathlib import Path

from .constants import OliveDeviceTypes
from .model_parameter import ModelParameter


def generator_intel(recipe, folder: Path):
    intel_runtime_values: list[str] = recipe.get("devices", [recipe.get("device")])
    aitk = recipe.get("aitk", {})
    file = recipe.get("file")

    name = f"Convert to Intel {"/".join([runtime.upper() for runtime in intel_runtime_values])}"
    isLLM = aitk.get("isLLM", False)
    addCpu = False
    oliveFile = aitk.get("oliveFile")

    sections = []
    
    parameter = ModelParameter(
        name=name,
        isLLM=isLLM,
        isIntel=True,
        intelRuntimeValues=[OliveDeviceTypes(runtime) for runtime in intel_runtime_values],
        addCpu=addCpu,
        oliveFile=oliveFile,
    )
    parameter._file = str(folder / (file + ".config"))
    parameter.writeIfChanged()
