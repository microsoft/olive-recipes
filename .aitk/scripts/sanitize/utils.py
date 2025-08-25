"""
Utility functions for the sanitize module
"""

import inspect
import json
import os
from contextlib import contextmanager
from typing import Any

import pydash
from model_lab import RuntimeEnum

from .constants import EPNames, OliveDeviceTypes, OlivePropertyNames


class GlobalVars:
    errorList = []
    verbose = False
    # Initialize checks
    pathCheck = 0
    configCheck = []
    oliveJsonCheck = []
    ipynbCheck = []
    gitignoreCheck = []
    modelProjectCheck = []
    extensionCheck = 0
    # Should align with number of LLM models
    inferenceModelCheck = []
    requirementsCheck = []
    venvRequirementsCheck = set()

    olivePath = None
    oliveCheck = 0
    RuntimeToEPName = {
        RuntimeEnum.CPU: EPNames.CPUExecutionProvider,
        RuntimeEnum.QNN: EPNames.QNNExecutionProvider,
        RuntimeEnum.IntelAny: EPNames.OpenVINOExecutionProvider,
        RuntimeEnum.IntelCPU: EPNames.OpenVINOExecutionProvider,
        RuntimeEnum.IntelNPU: EPNames.OpenVINOExecutionProvider,
        RuntimeEnum.IntelGPU: EPNames.OpenVINOExecutionProvider,
        RuntimeEnum.AMDNPU: EPNames.VitisAIExecutionProvider,
        RuntimeEnum.NvidiaGPU: EPNames.CUDAExecutionProvider,
        RuntimeEnum.NvidiaTRTRTX: EPNames.NvTensorRTRTXExecutionProvider,
        RuntimeEnum.DML: EPNames.DmlExecutionProvider,
    }
    RuntimeToOliveDeviceType = {
        RuntimeEnum.CPU: OliveDeviceTypes.CPU,
        RuntimeEnum.QNN: OliveDeviceTypes.NPU,
        RuntimeEnum.IntelAny: OliveDeviceTypes.Any,
        RuntimeEnum.IntelCPU: OliveDeviceTypes.CPU,
        RuntimeEnum.IntelNPU: OliveDeviceTypes.NPU,
        RuntimeEnum.IntelGPU: OliveDeviceTypes.GPU,
        RuntimeEnum.AMDNPU: OliveDeviceTypes.NPU,
        RuntimeEnum.NvidiaGPU: OliveDeviceTypes.GPU,
        RuntimeEnum.DML: OliveDeviceTypes.GPU,
    }
    RuntimeToDisplayName = {
        RuntimeEnum.CPU: "CPU",
        RuntimeEnum.QNN: "Qualcomm NPU",
        RuntimeEnum.IntelAny: "Intel Any",
        RuntimeEnum.IntelCPU: "Intel CPU",
        RuntimeEnum.IntelNPU: "Intel NPU",
        RuntimeEnum.IntelGPU: "Intel GPU",
        RuntimeEnum.AMDNPU: "AMD NPU",
        RuntimeEnum.NvidiaGPU: "NVIDIA GPU",
        RuntimeEnum.NvidiaTRTRTX: "NVIDIA TensorRT for RTX",
        RuntimeEnum.DML: "DirectML",
    }

    @classmethod
    def Check(cls, configDir: str):
        if len(cls.configCheck) != len(cls.oliveJsonCheck):
            printError(f"Config check {len(cls.configCheck)} does not match olive json check {len(cls.oliveJsonCheck)}")
        if len(cls.gitignoreCheck) != len(cls.modelProjectCheck) - cls.extensionCheck:
            printError(
                f"Gitignore check {len(cls.gitignoreCheck)} does not match model project check {len(cls.modelProjectCheck)} - {cls.extensionCheck}"
            )
        # We add this test to make sure the sanity check is working: i.e. paths are checked and files are checked
        with open_ex(os.path.join(configDir, "checks.json"), "w") as file:
            # get class properties and dump all ends with Check
            properties = [attr for attr in dir(cls) if attr.endswith("Check") and attr != "Check"]

            # save len if list else save the value
            def getlen(value):
                return len(value) if isinstance(value, list) or isinstance(value, set) else value

            properties = {prop: getlen(getattr(cls, prop)) for prop in properties}
            json.dump(properties, file, indent=4)
            file.write("\n")

    @classmethod
    def GetRuntimeRPC(cls, epName: EPNames, oliveDeviceType: OliveDeviceTypes) -> RuntimeEnum:
        # Accept epName as either Enum or string, convert to Enum if needed
        if not isinstance(epName, EPNames):
            epName = EPNames(epName)
        # Accept oliveDeviceType as either Enum or string, convert to Enum if needed
        if not isinstance(oliveDeviceType, OliveDeviceTypes):
            oliveDeviceType = OliveDeviceTypes(oliveDeviceType)

        matching_runtimes = [runtime for runtime, ep in cls.RuntimeToEPName.items() if ep == epName]
        if not matching_runtimes:
            raise ValueError(f"No runtime found for EPName: {epName}")
        if len(matching_runtimes) == 1:
            return matching_runtimes[0]
        # If multiple runtimes match, filter by oliveDeviceType
        for runtime in matching_runtimes:
            if cls.RuntimeToOliveDeviceType[runtime] == oliveDeviceType:
                return runtime
        raise ValueError(f"No matching runtime found for EPName: {epName} and OliveDeviceType: {oliveDeviceType}")


def printProcess(msg: str):
    if GlobalVars.verbose:
        print(f"Process {msg}")


def printInfo(msg: str):
    if GlobalVars.verbose:
        print(msg)


def printTip(msg: str):
    """Print important information with special color formatting (cyan)"""
    frame = inspect.currentframe()
    if frame and frame.f_back:
        frame = frame.f_back
        filename = os.path.relpath(frame.f_code.co_filename)
        lineno = frame.f_lineno
    else:
        filename = "unknown"
        lineno = 0
    # Cyan text with file and line number, clickable in terminal
    print(f"\033[36mTip: {filename}:{lineno}: {msg}\033[0m")


def printError(msg: str):
    frame = inspect.currentframe()
    if frame and frame.f_back:
        frame = frame.f_back
        filename = os.path.relpath(frame.f_code.co_filename)
        lineno = frame.f_lineno
    else:
        filename = "unknown"
        lineno = 0
    # print all errors in the end
    GlobalVars.errorList.append((filename, lineno, msg))


def printWarning(msg: str):
    frame = inspect.currentframe()
    if frame and frame.f_back:
        frame = frame.f_back
        filename = os.path.relpath(frame.f_code.co_filename)
        lineno = frame.f_lineno
    else:
        filename = "unknown"
        lineno = 0
    # Yellow text, with file and line number, clickable in terminal
    print(f"\033[33mWARNING: {filename}:{lineno}: {msg}\033[0m")


@contextmanager
def open_ex(file_path, mode):
    # Note: The `newline` parameter has no effect when reading a file.
    file = open(file_path, mode, encoding="utf-8", newline="\n")
    try:
        yield file
    finally:
        file.close()


def get_target_system(oliveJson: Any):
    syskey = oliveJson[OlivePropertyNames.Target]
    sysValue = oliveJson[OlivePropertyNames.Systems][syskey]
    return syskey, sysValue


def checkPath(path: str, oliveJson: Any, printOnNotExist: bool = True):
    printInfo(path)
    GlobalVars.pathCheck += 1
    if pydash.get(oliveJson, path) is None:
        syskey, system = get_target_system(oliveJson)
        currentEp = system[OlivePropertyNames.Accelerators][0][OlivePropertyNames.ExecutionProviders][0]
        # TODO some ov recipes do not have device but we set it in config
        if path == f"systems.{syskey}.accelerators.0.device" and currentEp == EPNames.OpenVINOExecutionProvider.value:
            printWarning(f"Not in olive json: {path}")
            return True
        if printOnNotExist:
            printError(f"Not in olive json: {path}")
        return False
    return True


def isLLM_by_id(id: str) -> bool:
    check_list = ["deepseek-ai/DeepSeek", "meta-llama/Llama", "microsoft/Phi", "mistralai/Mistral", "Qwen/Qwen"]
    return any(check in id for check in check_list)


# TODO align with Skylight\vscode\ai-mlstudio\src\model-lab\utilities\runtimeUtils.ts
def get_execute_runtime(runtime: RuntimeEnum) -> RuntimeEnum:
    if runtime in [RuntimeEnum.IntelAny, RuntimeEnum.IntelCPU, RuntimeEnum.IntelGPU, RuntimeEnum.IntelNPU]:
        return RuntimeEnum.IntelNPU
    if runtime == RuntimeEnum.NvidiaGPU:
        return RuntimeEnum.NvidiaGPU
    if runtime == RuntimeEnum.NvidiaTRTRTX:
        return RuntimeEnum.WCR_CUDA
    return RuntimeEnum.WCR


def get_eval_runtime(runtime: RuntimeEnum, isLLM: bool) -> RuntimeEnum:
    if runtime == RuntimeEnum.QNN and isLLM:
        return RuntimeEnum.QNN_LLLM
    if runtime == RuntimeEnum.NvidiaTRTRTX:
        return RuntimeEnum.WCR_CUDA
    return RuntimeEnum.WCR


def get_eval_in_execute_runtime(runtime: RuntimeEnum) -> RuntimeEnum:
    if runtime == RuntimeEnum.QNN:
        return RuntimeEnum.QNN
    raise ValueError(f"Unsupported runtime for eval in execute: {runtime}")
