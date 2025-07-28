"""
Main sanitize script
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
from pathlib import Path

from .constants import EPNames, ModelStatusEnum
from .copy_config import CopyConfig
from .file_validation import check_case, process_gitignore, readCheckIpynb, readCheckOliveConfig
from .model_info import ModelInfo, ModelList
from .model_parameter import ModelParameter
from .parameters import readCheckParameterTemplate
from .project_config import ModelInfoProject, ModelProjectConfig
from .utils import GlobalVars, open_ex, printError, printWarning


def shouldCheckModel(configDir: str, model: ModelInfo) -> str | None:
    modelDir = os.path.join(configDir, model.id)
    # If we have folder, we also check it
    if model.status == ModelStatusEnum.Ready or os.path.exists(modelDir):
        return modelDir
    return None


def main():
    argparser = argparse.ArgumentParser(description="Check model lab configs")
    argparser.add_argument("-v", "--verbose", action="store_true", help="Verbose mode")
    argparser.add_argument(
        "-o",
        "--olive",
        default="",
        type=str,
        help="Path to olive repo to check json files",
    )
    args = argparser.parse_args()
    GlobalVars.verbose = args.verbose
    GlobalVars.olivePath = args.olive

    # need to resolve due to d:\ vs D:\
    # Now main.py is in sanitize/ folder, so we need to go up three levels:
    # sanitize/main.py -> scripts/ -> model_lab_configs/
    configDir = str(Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))).resolve(strict=False))

    # get model list
    modelList = ModelList.Read(configDir)
    # check parameter template
    parameterTemplate = readCheckParameterTemplate(os.path.join(configDir, "parameter_template.json"))

    # check each model
    for model in modelList.allModels():
        modelDir = shouldCheckModel(configDir, model)
        if modelDir:
            if not check_case(Path(modelDir)):
                printError(
                    f"Model folder does not exist, or check if case matches between model.id {model.id} and model folder."
                )

            # get all versions
            allVersions = [int(name) for name in os.listdir(modelDir) if os.path.isdir(os.path.join(modelDir, name))]
            allVersions.sort()
            model.version = allVersions[-1]
            # check if version is continuous
            if allVersions[0] != 1 or allVersions[-1] != len(allVersions):
                printError(f"{modelDir} has wrong versions {allVersions}")

            # process each version
            for version in allVersions:
                # deep copy model for version usage
                modelInVersion = copy.deepcopy(model)
                modelInVersion.version = version
                modelVerDir = os.path.join(modelDir, str(version))

                # process copy
                copyConfigFile = os.path.join(modelVerDir, "_copy.json.config")
                if os.path.exists(copyConfigFile):
                    with open_ex(copyConfigFile, "r") as file:
                        copyConfigContent = file.read()
                    copyConfig = CopyConfig.model_validate_json(copyConfigContent, strict=True)
                    copyConfig.process(modelVerDir)

                # get model space config
                modelSpaceConfig = ModelProjectConfig.Read(os.path.join(modelVerDir, "model_project.config"))
                modelSpaceConfig.modelInfo.version = int(os.path.basename(modelVerDir))

                # check md
                mdFile = os.path.join(modelVerDir, "README.md")
                if not os.path.exists(mdFile):
                    printError(f"{mdFile} not exists")

                # check requirement.txt
                if not model.extension:
                    requirementFile = os.path.join(modelVerDir, "requirements.txt")
                    if not os.path.exists(requirementFile):
                        printWarning(f"{requirementFile} not exists.")

                # copy .gitignore
                if not model.extension:
                    process_gitignore(modelVerDir, configDir)

                # check ipynb & parameter
                sharedIpynbFile = os.path.join(modelVerDir, "inference_sample.ipynb")
                hasSharedIpynb = os.path.exists(sharedIpynbFile)
                workflowsAgainstShared: dict[str, ModelParameter] = {}

                if modelSpaceConfig.modelInfo:
                    modelSpaceConfig.modelInfo.id = modelInVersion.id
                else:
                    modelSpaceConfig.modelInfo = ModelInfoProject(id=modelInVersion.id)

                hasLLM = False
                for _, modelItem in enumerate(modelSpaceConfig.workflows):
                    # set template
                    fileName = os.path.basename(modelItem.file)[:-5]
                    modelItem.templateName = fileName

                    # read parameter
                    modelParameter = ModelParameter.Read(os.path.join(modelVerDir, f"{modelItem.file}.config"))

                    # check olive json
                    oliveJsonFile = os.path.join(modelVerDir, modelItem.file)
                    oliveJson = readCheckOliveConfig(oliveJsonFile)
                    if not oliveJson:
                        printError(f"{oliveJsonFile} not exists or is not a valid olive json file")
                        continue

                    # check parameter
                    modelParameter.Check(parameterTemplate, oliveJson, modelList)
                    if modelParameter.isIntel:
                        tmpDevices = modelParameter.getIntelDevices()
                        # Remove items containing "intel" (case-insensitive) from runtime values
                        filteredValues = [v for v in model.runtimes if "intel" not in v.lower()]
                        # Add Intel runtime values
                        intelRuntimes = [
                            GlobalVars.GetRuntimeRPC(EPNames.OpenVINOExecutionProvider, device) for device in tmpDevices
                        ]
                        filteredValues.extend([runtime.value for runtime in intelRuntimes])
                        model.runtimes = filteredValues

                    hasLLM = hasLLM or modelParameter.isLLM

                    # check ipynb
                    if not model.extension:
                        # although filename and templateName are same here, use fileName to align with Skylight implementation
                        ipynbFile = os.path.join(modelVerDir, f"{fileName}_inference_sample.ipynb")
                        hasSpecialIpynb = readCheckIpynb(ipynbFile, {modelItem.file: modelParameter})
                        if not hasSpecialIpynb:
                            if not hasSharedIpynb:
                                printError(f"{ipynbFile} nor {sharedIpynbFile} not exists.")
                            else:
                                workflowsAgainstShared[modelItem.file] = modelParameter

                if not model.extension:
                    readCheckIpynb(sharedIpynbFile, workflowsAgainstShared)

                if model.extension:
                    GlobalVars.extensionCheck += 1

                modelSpaceConfig.Check(modelInVersion)

                if hasLLM:
                    # check inference_model.json
                    inferenceModelFile = os.path.join(modelVerDir, "inference_model.json")
                    GlobalVars.inferenceModelCheck.append(inferenceModelFile)
                    if not os.path.exists(inferenceModelFile):
                        printWarning(f"{inferenceModelFile} not exists.")

    modelList.Check()

    if GlobalVars.olivePath:
        printWarning(f"Total {GlobalVars.oliveCheck} config files checked against olive json files")

    GlobalVars.Check(configDir)

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=configDir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if len(GlobalVars.errorList) == 0:
        # If the output is not empty, there are uncommitted changes
        if bool(result.stdout.strip()):
            printError("Please commit changes!")

    for filename, lineno, msg in GlobalVars.errorList:
        # Red text, with file and line number, clickable in terminal
        print(f"\033[31mERROR: {filename}:{lineno}: {msg}\033[0m")


if __name__ == "__main__":
    main()
