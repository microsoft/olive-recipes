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


class OliveDeviceTypes(Enum):
    Any = "any"
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"


# Pass name is case insensitive, so we use lower case for all pass names
# Should sort by value
class OlivePassNames:
    OnnxFloatToFloat16 = "onnxfloattofloat16"
    ModelBuilder = "modelbuilder"
    OnnxConversion = "onnxconversion"
    OnnxDynamicQuantization = "onnxdynamicquantization"
    OnnxQuantization = "onnxquantization"
    OnnxStaticQuantization = "onnxstaticquantization"
    OpenVINOConversion = "openvinoconversion"
    OpenVINOEncapsulation = "openvinoencapsulation"
    OpenVINOOptimumConversion = "openvinooptimumconversion"
    OpenVINOQuantization = "openvinoquantization"
    OrtTransformersOptimization = "orttransformersoptimization"
    QuarkQuantization = "quarkquantization"



# Should sort by value
class OlivePropertyNames:
    Accelerators = "accelerators"
    ActivationType = "activation_type"
    CacheDir = "cache_dir"
    CleanCache = "clean_cache"
    DataConfig = "data_config"
    DataConfigs = "data_configs"
    DataName = "data_name"
    Dataset = "dataset"
    Device = "device"
    Engine = "engine"
    EvaluateInputModel = "evaluate_input_model"
    Evaluator = "evaluator"
    Evaluators = "evaluators"
    ExecutionProviders = "execution_providers"
    ExtraArgs = "extra_args"
    Host = "host"
    LoadDatasetConfig = "load_dataset_config"
    MaxSamples = "max_samples"
    Metrics = "metrics"
    Name = "name"
    NumCalibData = "num_calib_data"
    OutputDir = "output_dir"
    Passes = "passes"
    Precision = "precision"
    PreProcessDataConfig = "pre_process_data_config"
    PythonEnvironmentPath = "python_environment_path"
    ExternalData = "save_as_external_data"
    Split = "split"
    Subset = "subset"
    Systems = "systems"
    Target = "target"
    TargetDevice = "target_device"
    Type = "type"
    UserConfig = "user_config"


# Path constants
outputModelRelativePath = r"\\\"./model/model.onnx\\\""
outputModelIntelNPURelativePath = (
    r"\\\"./model/(ov_model_st_quant|openvino_model_quant_st|openvino_model_st_quant).onnx\\\""
)
outputModelModelBuilderPath = r"\\\"./model\\\""

# Import constants
importOnnxruntime = r"import onnxruntime as ort"
importOnnxgenairuntime = r"import onnxruntime_genai as og"
