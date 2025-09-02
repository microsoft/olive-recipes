"""
File validation functions
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from .constants import (
    EPNames,
    OlivePassNames,
    OlivePropertyNames,
    importOnnxgenairuntime,
    importOnnxruntime,
    outputModelIntelNPURelativePath,
    outputModelModelBuilderPath,
    outputModelRelativePath,
)
from .model_info import ModelInfo
from .model_parameter import ModelParameter
from .utils import GlobalVars, open_ex, printError, printProcess, printWarning


def check_case(path: Path) -> bool:
    path = Path(path)
    try:
        abs_path = path.resolve(strict=False)
    except Exception:
        return False

    if str(path) != str(abs_path):
        printError(f"Path case mismatch: {path} vs {abs_path}")
        return False
    return True


def process_gitignore(modelVerDir: str, configDir: str):
    gitignoreFile = os.path.join(modelVerDir, ".gitignore")
    GlobalVars.gitignoreCheck.append(gitignoreFile)
    templateFile = os.path.join(configDir, "gitignore.md")
    if not os.path.exists(gitignoreFile):
        printWarning(f"{gitignoreFile} not exists. Copy the template one")
        shutil.copy(templateFile, gitignoreFile)
    else:
        # Ensure each non-empty line in template is present in the .gitignore file (exact match)
        with open_ex(gitignoreFile, "r") as file:
            gitignoreLines = [line.strip() for line in file if line.strip()]
        with open_ex(templateFile, "r") as file:
            templateLines = [line.strip() for line in file if line.strip()]
        missing = [line for line in templateLines if line not in gitignoreLines]
        for line in missing:
            printError(f"{gitignoreFile} does not have line '{line}'")


def checkSystem(oliveJsonFile: str, system):
    accelerators = system[OlivePropertyNames.Accelerators]
    if len(accelerators) != 1:
        printError(f"{oliveJsonFile} should have only one accelerator")
        return False
    eps = accelerators[0][OlivePropertyNames.ExecutionProviders]
    if len(eps) != 1:
        printError(f"{oliveJsonFile} should have only one execution provider")
        return False
    if eps[0] not in EPNames:
        printError(f"{oliveJsonFile} has wrong execution provider {eps[0]}")
        return False
    return True


def readCheckOliveConfig(oliveJsonFile: str, model: ModelInfo):
    """
    This will set phases to modelParameter
    """
    GlobalVars.oliveJsonCheck.append(oliveJsonFile)

    printProcess(oliveJsonFile)
    with open_ex(oliveJsonFile, "r") as file:
        oliveJson = json.load(file)
    # check if engine is in oliveJson
    if OlivePropertyNames.Engine in oliveJson:
        printError(f"{oliveJsonFile} has engine. Should place in the root instead")
        return
    if OlivePropertyNames.Evaluator in oliveJson and not isinstance(oliveJson[OlivePropertyNames.Evaluator], str):
        printError(f"{oliveJsonFile} evaluator property should be str")
        return

    jsonUpdated = False

    # check target
    if OlivePropertyNames.Target not in oliveJson:
        printError(f"{oliveJsonFile} should have target")
        return
    target = oliveJson[OlivePropertyNames.Target]
    if OlivePropertyNames.Systems not in oliveJson or target not in oliveJson[OlivePropertyNames.Systems]:
        printError(f"{oliveJsonFile} should have {target} system")
        return
    if not checkSystem(oliveJsonFile, oliveJson[OlivePropertyNames.Systems][target]):
        return

    # check host
    if OlivePropertyNames.Host in oliveJson:
        if oliveJson[OlivePropertyNames.Host] == target:
            printError(
                f"{oliveJsonFile} should not use same host as target. You should either remove it or setup a separate one"
            )

    # cache / output / evaluate_input_model
    if OlivePropertyNames.CleanCache in oliveJson and oliveJson[OlivePropertyNames.CleanCache]:
        oliveJson.pop(OlivePropertyNames.CleanCache)
        jsonUpdated = True

    if OlivePropertyNames.CacheDir not in oliveJson or oliveJson[OlivePropertyNames.CacheDir] != "cache":
        oliveJson[OlivePropertyNames.CacheDir] = "cache"
        jsonUpdated = True

    if OlivePropertyNames.OutputDir not in oliveJson or not str(oliveJson[OlivePropertyNames.OutputDir]).startswith(
        "model/"
    ):
        printError(f"{oliveJsonFile} should have use model/XXX as {OlivePropertyNames.OutputDir}")

    if OlivePropertyNames.EvaluateInputModel not in oliveJson or oliveJson[OlivePropertyNames.EvaluateInputModel]:
        oliveJson[OlivePropertyNames.EvaluateInputModel] = False
        jsonUpdated = True

    # update save_as_external_data
    supportedPasses = [
        v
        for k, v in oliveJson[OlivePropertyNames.Passes].items()
        if v[OlivePropertyNames.Type].lower()
        in [
            OlivePassNames.OnnxConversion,
            OlivePassNames.OnnxQuantization,
            OlivePassNames.OnnxStaticQuantization,
            OlivePassNames.OnnxDynamicQuantization,
            OlivePassNames.OrtTransformersOptimization,
        ]
    ]
    for conversionPass in supportedPasses:
        if OlivePropertyNames.ExternalData not in conversionPass or not conversionPass[OlivePropertyNames.ExternalData]:
            conversionPass[OlivePropertyNames.ExternalData] = True
            jsonUpdated = True

    if jsonUpdated:
        with open_ex(oliveJsonFile, "w") as file:
            json.dump(oliveJson, file, indent=4)
            file.write("\n")
    return oliveJson


def readCheckIpynb(ipynbFile: str, modelItems: dict[str, ModelParameter]):
    """
    Note this return exists or not, not valid or not
    """
    if os.path.exists(ipynbFile):
        GlobalVars.ipynbCheck.append(ipynbFile)

        with open_ex(ipynbFile, "r") as file:
            ipynbContent: str = file.read()
        allRuntimes: list[str] = []
        for name, modelParameter in modelItems.items():
            if modelParameter.runtime == None:
                printError(f"When checking {ipynbFile}, modelItem {name} does not have runtime!")
                continue
            testPath = outputModelRelativePath
            importStr = importOnnxruntime
            if modelParameter.isLLM:
                testPath = outputModelModelBuilderPath
                importStr = importOnnxgenairuntime
            elif modelParameter.runtime.values and modelParameter.isIntel:
                testPath = outputModelIntelNPURelativePath
            for item in [testPath, importStr]:
                if not re.search(item, ipynbContent):
                    printError(f"{ipynbFile} does not have '{item}' for {name}, please use it as input")
            if modelParameter.evalRuntime:
                runtime = GlobalVars.RuntimeToEPName[modelParameter.evalRuntime]
                if runtime not in allRuntimes:
                    allRuntimes.append(runtime.value)
            else:
                if modelParameter.isIntel:
                    allRuntimes.append(EPNames.OpenVINOExecutionProvider.value)
                elif modelParameter.runtime.values:
                    for runtime in modelParameter.runtime.values:
                        if runtime not in allRuntimes:
                            allRuntimes.append(runtime)

        targetEP = None
        if len(allRuntimes) == 2 and EPNames.CPUExecutionProvider.value in allRuntimes:
            allRuntimes.remove(EPNames.CPUExecutionProvider.value)
            targetEP = allRuntimes[0]
        elif len(allRuntimes) == 1:
            targetEP = allRuntimes[0]
        elif len(allRuntimes) > 1:
            # TODO we use QNN as default because currently we only replace this
            if EPNames.QNNExecutionProvider.value in allRuntimes:
                targetEP = EPNames.QNNExecutionProvider.value
            elif EPNames.CPUExecutionProvider.value in allRuntimes:
                targetEP = EPNames.CPUExecutionProvider.value
            else:
                targetEP = allRuntimes[0]
        if targetEP:
            targetStr = f'ExecutionProvider=\\"{targetEP}\\"'
            if ipynbContent.count(targetStr) != 1:
                printError(f"{ipynbFile} should have 1 {targetStr}")
        else:
            printError(f"{ipynbFile} has no runtime for it!")
        return True
    return False


def readCheckRequirements(requirementsFile: str):
    """
    Check requirements.txt file
    """
    if not os.path.exists(requirementsFile):
        printWarning(f"{requirementsFile} not exists.")
        return

    GlobalVars.requirementsCheck.append(requirementsFile)
    with open_ex(requirementsFile, "r") as file:
        requirementsContent: str = file.read()
    requirementsLines = requirementsContent.splitlines()

    oldContents = []
    newContents = [
        "# This file will be installed together with AITK runtime requirements",
        "# For the full requirements, see AITK",
    ]
    ready = False
    if len(requirementsLines) >= len(newContents):
        ready = True
        for i in range(len(newContents)):
            if requirementsLines[i] != newContents[i]:
                ready = False
                break
    if not ready:
        requirementsLines = [line for line in requirementsLines if line not in oldContents]
        requirementsLines = newContents + requirementsLines
    newContent = "\n".join(requirementsLines) + "\n"
    if requirementsContent != newContent:
        with open_ex(requirementsFile, "w") as file:
            file.write(newContent)
