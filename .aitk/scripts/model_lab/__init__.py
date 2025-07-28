from enum import Enum


class RuntimeEnum(Enum):
    CPU = "CPU"
    QNN = "QNN"
    IntelAny = "IntelAny"
    IntelCPU = "IntelCPU"
    IntelNPU = "IntelNPU"
    IntelGPU = "IntelGPU"
    AMDNPU = "AMDNPU"
    NvidiaGPU = "NvidiaGPU"
    NvidiaTRTRTX = "NvidiaTRTRTX"
    DML = "DML"
    WCR = "WCR"
    WCR_CUDA = "WCR_CUDA"
    # Inference
    QNN_LLLM = "QNN_LLM"


class RuntimeFeatureEnum(Enum):
    AutoGptq = "AutoGptq"
    AutoAwq = "AutoAwq"
    Nightly = "Nightly"
    Genai = "Genai"
