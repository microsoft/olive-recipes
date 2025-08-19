"""
Constants and Enums for the sanitize module
"""

from enum import Enum


class IconEnum(Enum):
    Intel = "intel"
    Gemini = "gemini"
    OpenAI = "OpenAI"
    Microsoft = "ms"
    Meta = "meta"
    CompVis = "compvis"
    BAAI = "baai"
    tiiuae = "tiiuae"
    EleutherAI = "eleutherai"
    openlm = "openlm"
    DeepSeek = "DeepSeek"
    laion = "laion"
    qwen = "qwen"
    mistralai = "mistralai"
    HuggingFace = "HuggingFace"


class ArchitectureEnum(Enum):
    Transformer = "Transformer"
    CNN = "CNN"
    Diffusion = "Diffusion"
    Others = "Others"


class ModelStatusEnum(Enum):
    Ready = "Ready"
    Coming = "Coming"
    Hide = "Hide"


class ParameterTypeEnum(Enum):
    Enum = "enum"
    Int = "int"
    Bool = "bool"
    String = "str"


class ParameterDisplayTypeEnum(Enum):
    Dropdown = "Dropdown"
    RadioGroup = "RadioGroup"


class ParameterCheckTypeEnum(Enum):
    Exist = "exist"
    NotExist = "notExist"


class ParameterActionTypeEnum(Enum):
    # Update and Insert are both upsert in runtime. Separate them for validation
    Update = "update"
    Insert = "insert"
    Delete = "delete"


class ParameterTagEnum(Enum):
    QuantizationDataset = "QuantizationDataset"
    QuantizationDatasetSubset = "QuantizationDatasetSubset"
    QuantizationDatasetSplit = "QuantizationDatasetSplit"
    EvaluationDataset = "EvaluationDataset"
    EvaluationDatasetSubset = "EvaluationDatasetSubset"
    EvaluationDatasetSplit = "EvaluationDatasetSplit"
    DependsOnDataset = "DependsOnDataset"
    ActivationType = "ActivationType"
    WeightType = "WeightType"


class PhaseTypeEnum(Enum):
    Conversion = "Conversion"
    Quantization = "Quantization"
    Evaluation = "Evaluation"


class ReplaceTypeEnum(Enum):
    String = "string"
    Path = "path"
    PathAdd = "pathAdd"


class EPNames(Enum):
    CPUExecutionProvider = "CPUExecutionProvider"
    CUDAExecutionProvider = "CUDAExecutionProvider"
    QNNExecutionProvider = "QNNExecutionProvider"
    OpenVINOExecutionProvider = "OpenVINOExecutionProvider"
    VitisAIExecutionProvider = "VitisAIExecutionProvider"
    NvTensorRTRTXExecutionProvider = "NvTensorRTRTXExecutionProvider"
    DmlExecutionProvider = "DmlExecutionProvider"
    WebGpuExecutionProvider = "WebGpuExecutionProvider"


class OliveDeviceTypes(Enum):
    Any = "any"
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"


# Pass name is case insensitive, so we use lower case for all pass names
class OlivePassNames:
    OnnxConversion = "onnxconversion"
    OnnxQuantization = "onnxquantization"
    OnnxStaticQuantization = "onnxstaticquantization"
    OnnxDynamicQuantization = "onnxdynamicquantization"
    ModelBuilder = "modelbuilder"
    OpenVINOConversion = "openvinoconversion"
    OpenVINOOptimumConversion = "openvinooptimumconversion"
    OpenVINOQuantization = "openvinoquantization"
    OpenVINOEncapsulation = "openvinoencapsulation"
    OrtTransformersOptimization = "orttransformersoptimization"


class OlivePropertyNames:
    Engine = "engine"
    Passes = "passes"
    Evaluator = "evaluator"
    Evaluators = "evaluators"
    Type = "type"
    ExternalData = "save_as_external_data"
    Systems = "systems"
    Accelerators = "accelerators"
    Device = "device"
    TargetDevice = "target_device"
    ExecutionProviders = "execution_providers"
    DataConfigs = "data_configs"
    Target = "target"
    CacheDir = "cache_dir"
    OutputDir = "output_dir"
    PythonEnvironmentPath = "python_environment_path"
    EvaluateInputModel = "evaluate_input_model"
    Metrics = "metrics"
    UserConfig = "user_config"
    CleanCache = "clean_cache"
    ExtraArgs = "extra_args"


# Path constants
outputModelRelativePath = r"\\\"./model/model.onnx\\\""
outputModelIntelNPURelativePath = (
    r"\\\"./model/(ov_model_st_quant|openvino_model_quant_st|openvino_model_st_quant).onnx\\\""
)
outputModelModelBuilderPath = r"\\\"./model\\\""

# Import constants
importOnnxruntime = r"import onnxruntime as ort"
importOnnxgenairuntime = r"import onnxruntime_genai as og"
