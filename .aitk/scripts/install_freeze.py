import argparse
import os
import subprocess
from os import path

from model_lab import RuntimeEnum

# This script is used to generate the requirements-*.txt
# Usage: uv run .\install_freeze.py --python PATH_TO_RUNTIME
# They also have special comments:
# - `# pip:`: anything after it will be sent to pip command like `# pip:--no-build-isolation`
# - `# copy:`: copy from cache to folder in runtime like `# copy:a/*.dll;b;pre`, `# copy:a/*.dll;b;post`
# - `# download:`: download from release and save it to cache folder like `# download:onnxruntime-genai-cuda-0.7.0-cp39-cp39-win_amd64.whl`
uvpipInstallPrefix = "# uvpip:install"
cudaExtraUrl = "--extra-index-url https://download.pytorch.org/whl/cu128"
torchCudaVersion = "torch==2.7.0+cu128"
onnxruntimeWinmlVersion = f"{uvpipInstallPrefix} onnxruntime-winml==1.22.0.post1 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple --no-deps;post"
onnxruntimeGenaiWinmlVersion = f"{uvpipInstallPrefix} onnxruntime-genai-winml==0.8.3 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple --no-deps;post"
evaluateVersion = "evaluate==0.4.3"
scikitLearnVersion = "scikit-learn==1.6.1"
optimumVersion = "optimum==1.26.0"
# if from git: "git+https://github.com/microsoft/Olive.git@COMMIT_ID#egg=olive_ai
oliveAi = "olive-ai@git+https://github.com/microsoft/Olive.git@8365802b68c32725418ae2c8999b9a90af0d41e0#egg=olive-ai"
torchVision = "torchvision==0.22.0"
amdQuark = "AMD__Quark_py3.10.17"

def get_requires(name: str, args):
    # TODO for this case, need to install via Model Lab first
    viaModelLab = False
    if name.startswith(uvpipInstallPrefix):
        name = name.split(" ")[2].strip()
        viaModelLab = True

    if "#egg=" in name:
        package_name = name.split("#egg=")[1]
    elif name.startswith("./"):
        package_name = name[2:].split("-")[0].replace("_", "-")
    else:
        package_name = name.split("==")[0]  # Remove version if present
    if "[" in package_name:
        package_name = package_name.split("[")[0]
    requires = []
    try:
        output = subprocess.check_output(["uv", "pip", "show", package_name, "-p", args.python]).decode("utf-8")
        for line in output.splitlines():
            if line.startswith("Requires"):
                requires = line.split(":")[1].strip().split(", ")
                break
    except subprocess.CalledProcessError:
        pass
    return [req for req in requires if req], package_name, viaModelLab


def get_name_outputFile(python: str, configs_dir: str):
    pythonSegs = python.split("-")
    if "__" in python:
        folder_name = pythonSegs[-4].split("__")
        folder = folder_name[0]
        name = f"{folder_name[1]}_py{pythonSegs[-1]}"
        runtime = f"{folder}__{name}"
        outputFile = path.join(configs_dir, "requirements", folder, f"{name}.txt")
    else:
        runtime = pythonSegs[-4]
        outputFile = path.join(configs_dir, "requirements", f"requirements-{runtime}.txt")
        runtime = RuntimeEnum(runtime)
    return runtime, outputFile

def main():
    pre = {
        RuntimeEnum.NvidiaGPU: [
            cudaExtraUrl,
            torchCudaVersion,
        ],
        RuntimeEnum.WCR_CUDA: [
            cudaExtraUrl,
            torchCudaVersion,
        ],
        RuntimeEnum.IntelNPU: [
            "torch==2.6.0",
        ],
        amdQuark: [
            "transformers==4.50.0",
            "amd-quark==0.9",
            "--extra-index-url=https://pypi.amd.com/simple",
            "model-generate==1.5.1",
            # olive.passes.quark_quantizer.torch.language_modeling.llm_utils.model_preparation
            "psutil==7.0.0",
            # ValueError: Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)` requires `accelerate`. You can install it with `pip install accelerate`
            "accelerate==1.10.1",
        ]
    }
    shared_conversion = [
        "huggingface-hub[hf_xet]==0.34.4",
        # sticking to ONNX IR version 10 which can still be consumed by ORT v1.22.0
        "onnx==1.17.0",
        oliveAi,
        "tabulate==0.9.0",
        "datasets==3.5.0",
    ]
    shared_ipynb = [
        "ipykernel==6.29.5",
        "ipywidgets==8.1.5",
    ]
    shared_both = shared_conversion + shared_ipynb
    shared = {
        RuntimeEnum.QNN: shared_conversion,
        RuntimeEnum.IntelNPU: shared_conversion,
        RuntimeEnum.NvidiaGPU: shared_conversion,
        RuntimeEnum.WCR: shared_both,
        RuntimeEnum.WCR_CUDA: shared_both,
        RuntimeEnum.QNN_LLLM: shared_ipynb,
        amdQuark: shared_conversion,
    }
    # torchvision, onnxruntime and genai go here. others should go feature
    post = {
        RuntimeEnum.QNN: [
            torchVision,
            "onnxruntime-qnn==1.21.1",
            "# uvpip:install onnxruntime-genai==0.7.0 --no-deps;post",
        ],
        # now optimum-intel does not depend on onnxruntime, but we use a separate venv to simplify management
        RuntimeEnum.IntelNPU: [
            # nncf needs torch 2.6 so torchvision is downgraded
            "torchvision==0.21.0",
            "onnxruntime==1.21.0",
            # from olive[openvino]
            "openvino==2025.1.0",
            "nncf==2.16.0",
            "numpy==1.26.4",
            "optimum[openvino]==1.24.0",
            # TODO for model builder
            "onnxruntime-genai==0.7.0",
        ],
        # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
        RuntimeEnum.NvidiaGPU: [
            "torchvision==0.22.0+cu128",
            "onnxruntime-gpu==1.21.0",
            # 0.8.X is not working for DML LLM because
            # File "onnxruntime_genai\models\builder.py", line 571, in make_tensor_proto_from_tensor
            #    data_type=self.to_onnx_dtype[tensor.dtype],
            #KeyError: torch.uint8
            "onnxruntime-genai-cuda==0.7.0",
            optimumVersion,
        ],
        RuntimeEnum.WCR: [
            torchVision,
            onnxruntimeWinmlVersion,
            onnxruntimeGenaiWinmlVersion,
            evaluateVersion,
            scikitLearnVersion,
            optimumVersion,
        ],
        RuntimeEnum.WCR_CUDA: [
            "torchvision==0.22.0+cu128",
            onnxruntimeWinmlVersion,
            onnxruntimeGenaiWinmlVersion,
            evaluateVersion,
            scikitLearnVersion,
            optimumVersion,
        ],
        RuntimeEnum.QNN_LLLM: [
            # for onnxruntime-winml
            "numpy==2.2.4",
            onnxruntimeGenaiWinmlVersion,
        ],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--python", "-p", required=True, type=str, help="python path")
    args = parser.parse_args()

    configs_dir = path.dirname(path.dirname(__file__))
    runtime, outputFile = get_name_outputFile(args.python, configs_dir)

    # prepare file
    configs_dir = path.dirname(path.dirname(__file__))
    temp_dir = path.join(configs_dir, "scripts", "model_lab", "__pycache__")
    os.makedirs(temp_dir, exist_ok=True)
    temp_req = path.join(temp_dir, "temp_req.txt")
    all: list[str] = []
    with open(temp_req, "w") as f:
        if runtime in pre:
            for line in pre[runtime]:
                f.write(line + "\n")
                all.append(line)
        for line in shared[runtime]:
            f.write(line + "\n")
            all.append(line)
        if runtime in post:
            for line in post[runtime]:
                f.write(line + "\n")
                all.append(line)

    # Install
    print(f"Installing dependencies: {temp_req}")
    result = subprocess.run(["uv", "pip", "install", "-r", temp_req, "-p", args.python], text=True)

    # Get freeze
    pip_freeze = subprocess.check_output(["uv", "pip", "freeze", "-p", args.python]).decode("utf-8").splitlines()
    freeze_dict = {}
    for line in pip_freeze:
        if "==" in line:
            name, version = line.split("==")
            # requires outputs lower case names
            freeze_dict[name.lower()] = version
    print(f"Installed dependencies: {freeze_dict}")
    freeze_dict_used = set()

    # write result
    with open(outputFile, "w", newline="\n") as f:
        def get_write_require(req: str):
            if req in freeze_dict:
                if req not in freeze_dict_used:
                    f.write(f"{req}=={freeze_dict[req]}\n")
                    freeze_dict_used.add(req)
                    write_requires_recursively(req)
                return True
            return False

        def write_requires_recursively(name: str):
            requires, package_name, viaModelLab = get_requires(name, args)
            print(f"Requires for {name} by {package_name}: {requires}")
            freeze_dict_used.add(package_name)
            
            for req in requires:
                if get_write_require(req):
                    continue
                newReq = req.replace("-", "_")
                if get_write_require(newReq):
                    continue
                # in QNN for onnxruntime-genai
                if req == "onnxruntime":
                    if get_write_require("onnxruntime-qnn"):
                        continue
                raise Exception(f"Cannot find {req} in pip freeze")

        for name in all:
            if (
                name.startswith("#") and not name.startswith(uvpipInstallPrefix)
            ) or name.startswith("--"):
                f.write(name + "\n")
                continue
            if not name.startswith("#"):
                f.write("# " + name + "\n")
            f.write(name + "\n")
            write_requires_recursively(name)
        f.write("# not in requires\n")
        for k in freeze_dict:
            if k not in freeze_dict_used:
                f.write(f"{k}=={freeze_dict[k]}\n")

    # remove duplicate lines from output file
    with open(outputFile, "r") as f:
        lines = f.readlines()
    unique_lines = list(dict.fromkeys(lines))  # Preserve order and remove duplicates
    assert len(lines) == len(unique_lines), "Duplicate lines found."


if __name__ == "__main__":
    main()
