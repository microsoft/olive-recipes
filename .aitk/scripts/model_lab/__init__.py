from enum import Enum


class RuntimeEnum(Enum):
    CPU = "CPU"
    QNN = "QNN"
    AMDNPU = "AMDNPU"
    NvidiaTRTRTX = "NvidiaTRTRTX"
    IntelAny = "IntelAny"
    IntelCPU = "IntelCPU"
    IntelGPU = "IntelGPU"
    IntelNPU = "IntelNPU"
    DML = "DML"
    NvidiaGPU = "NvidiaGPU"
    WCR = "WCR"
    WCR_CUDA = "WCR_CUDA"
    # Inference
    QNN_LLLM = "QNN_LLM"
