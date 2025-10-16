from enum import Enum


class RuntimeEnum(Enum):
    CPU = "CPU"
    QNN = "QNN"
    AMDNPU = "AMDNPU"
    AMDGPU = "AMDGPU"
    NvidiaTRTRTX = "NvidiaTRTRTX"
    IntelAny = "IntelAny"
    IntelCPU = "IntelCPU"
    IntelGPU = "IntelGPU"
    IntelNPU = "IntelNPU"
    DML = "DML"
    WebGPU = "WebGPU"
    NvidiaGPU = "NvidiaGPU"
    WCR = "WCR"
    WCR_CUDA = "WCR_CUDA"
    WCR_INIT = "WCR_INIT"
    # Inference
    QNN_LLLM = "QNN_LLM"
