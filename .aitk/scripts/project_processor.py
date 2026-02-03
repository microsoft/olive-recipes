import os
from pathlib import Path
from typing import Dict

import yaml
from model_lab import RuntimeEnum
from sanitize.constants import ArchitectureEnum, EPNames, IconEnum, ModelStatusEnum
from sanitize.copy_config import CopyConfig
from sanitize.generator_amd import generator_amd
from sanitize.generator_dml import generator_dml
from sanitize.generator_intel import generator_intel
from sanitize.generator_qnn import generator_qnn
from sanitize.generator_trtrtx import generator_trtrtx
from sanitize.model_info import ModelInfo, ModelList
from sanitize.project_config import ModelInfoProject, ModelProjectConfig, WorkflowItem
from sanitize.utils import GlobalVars, isLLM_by_id, open_ex

org_to_icon = {
    "Intel": IconEnum.Intel,
    "google-bert": IconEnum.Gemini,
    "openai": IconEnum.OpenAI,
    "laion": IconEnum.laion,
    "microsoft": IconEnum.Microsoft,
    "google": IconEnum.Gemini,
    "deepseek-ai": IconEnum.DeepSeek,
    "Qwen": IconEnum.qwen,
    "meta-llama": IconEnum.Meta,
    "mistralai": IconEnum.mistralai,
    # TODO add
    "OFA-Sys": IconEnum.HuggingFace,
}


class ModelSummary:
    def __init__(self, modelInfo: ModelInfo):
        self.modelInfo = modelInfo
        self.modelName = modelInfo.displayName.split("/")[1].replace("-", " ").title()
        self.recipes = dict[RuntimeEnum, list[str]]()


class AllModelSummary:
    def __init__(self):
        self.llmModels = list[ModelSummary]()
        self.nonLlmModels = list[ModelSummary]()

    def write(self, root_dir: Path):
        md = root_dir / ".aitk" / "docs" / "guide" / "ModelList.md"
        with open_ex(md, "w") as f:
            f.write("# Model List\n\n")
            self.write_list(f, "LLM Models", self.llmModels, GlobalVars.RuntimeToDisplayName, root_dir, md)
            self.write_list(f, "Non-LLM Models", self.nonLlmModels, GlobalVars.RuntimeToDisplayName, root_dir, md)

    def write_list(
        self,
        f,
        title: str,
        modelList: list[ModelSummary],
        runtimeToDisplayName: Dict[RuntimeEnum, str],
        root_dir: Path,
        md_path: Path,
    ):
        modelList.sort(key=lambda x: (x.modelName))
        f.write(f"## {title}\n\n")
        f.write("| Model Name | Supported Runtimes |\n")
        f.write("|------------|--------------------|\n")
        for model in modelList:

            def get_runtime_str(runtime: RuntimeEnum, recipes: list[str]) -> str:
                name = runtimeToDisplayName.get(runtime)
                # TODO only show first one
                recipe_path = root_dir / str(model.modelInfo.relativePath) / recipes[0]
                recipe_path = os.path.relpath(recipe_path, md_path.parent).replace("\\", "/")
                return f"[{name}]({recipe_path})"

            runtimes = ", ".join([get_runtime_str(r, model.recipes[r]) for r in RuntimeEnum if r in model.recipes])
            f.write(f"| [{model.modelName}]({model.modelInfo.modelLink}) | {runtimes} |\n")


def get_runtime(recipe: dict):
    eps = recipe.get("eps", [recipe.get("ep")])
    devices = recipe.get("devices", [recipe.get("device")])
    for ep in eps:
        for device in devices:
            yield GlobalVars.GetRuntimeRPC(ep, device)


def convert_yaml_to_model_info(root_dir: Path, yml_file: Path, yaml_object: dict) -> ModelInfo:
    """
    Convert a YAML object to a ModelInfo instance.
    """
    aitk = yaml_object.get("aitk", {})
    modelInfo = aitk.get("modelInfo", {})
    id = modelInfo.get("id")
    version = modelInfo.get("version", 1)
    if not id:
        raise ValueError(f"Model ID is required in {yml_file}")
    if not isinstance(version, int) or version <= 0:
        raise ValueError(f"Model version must be a positive integer in {yml_file}")
    id_segs = id.split("/")

    display_name = modelInfo.get("displayName", "/".join(id_segs[1:]))
    icon = IconEnum(modelInfo.get("icon", org_to_icon.get(id_segs[1])))
    model_link = modelInfo.get("modelLink", "/".join(["https://huggingface.co"] + id_segs[1:]))
    architecture = ArchitectureEnum(modelInfo.get("architecture", ArchitectureEnum.Transformer))
    status = ModelStatusEnum(modelInfo.get("status", ModelStatusEnum.Ready))
    recipes = yaml_object.get("recipes", [])
    runtimes = set()
    for recipe in recipes:
        runtimes.update(get_runtime(recipe))
    runtimes = [r for r in RuntimeEnum if r in runtimes]
    relative_path = str(yml_file.parent.relative_to(root_dir)).replace("\\", "/")
    groupId = modelInfo.get("groupId")
    groupItemName = modelInfo.get("groupItemName")
    p0 = modelInfo.get("p0")
    model_info = ModelInfo(
        displayName=display_name,
        icon=icon,
        modelLink=model_link,
        id=id,
        runtimes=runtimes,
        architecture=architecture,
        status=status,
        version=version,
        relativePath=relative_path,
        groupId=groupId,
        groupItemName=groupItemName,
        p0=p0,
    )
    return model_info


def convert_yaml_to_project_config(
    yml_file: Path, yaml_object: dict, modelList: ModelList, modelSummary: ModelSummary
) -> ModelProjectConfig:
    aitk = yaml_object.get("aitk", {})
    modelInfo = aitk.get("modelInfo", {})
    id = modelInfo.get("id")
    recipes = yaml_object.get("recipes", [])
    items = []
    for recipe in recipes:
        file = recipe.get("file")
        items.append(
            WorkflowItem(
                file=file,
                templateName=file[:-5] if file and file.endswith(".json") else file,
            )
        )
        if recipe.get("ep") == EPNames.OpenVINOExecutionProvider.value:
            generator_intel(id, recipe, yml_file.parent)
        elif recipe.get("ep") == EPNames.VitisAIExecutionProvider.value:
            generator_amd(id, recipe, yml_file.parent, modelList)
        elif recipe.get("ep") == EPNames.QNNExecutionProvider.value:
            generator_qnn(id, recipe, yml_file.parent, modelList)
        elif recipe.get("ep") == EPNames.NvTensorRTRTXExecutionProvider.value:
            generator_trtrtx(id, recipe, yml_file.parent, modelList)
        elif recipe.get("ep") == EPNames.DmlExecutionProvider.value:
            generator_dml(id, recipe, yml_file.parent, modelList)
        runtimes = get_runtime(recipe)
        for runtime in runtimes:
            modelSummary.recipes.setdefault(runtime, []).append(file)

    version = modelInfo.get("version", 1)
    result = ModelProjectConfig(
        workflows=items,
        modelInfo=ModelInfoProject(
            id=id,
            version=version,
        ),
    )
    result._file = str(yml_file.parent / "model_project.config")
    result.writeIfChanged()
    return result


def project_processor():
    root_dir = Path(__file__).parent.parent.parent

    modelList = ModelList.Read(str(root_dir / ".aitk" / "configs"))
    modelList.models.clear()

    all_ids = set()
    all_summary = AllModelSummary()
    for yml_file in root_dir.rglob("info.yml"):
        # if "DEBUG_ID" in str(yml_file):
        #    pass
        # read yml file as yaml object
        with yml_file.open("r", encoding="utf-8") as file:
            try:
                yaml_content = file.read()
                yaml_object = yaml.safe_load(yaml_content)
            except yaml.YAMLError as e:
                print(f"Error reading {yml_file}: {e}")
                continue
            aitk = yaml_object.get("aitk", [])
            if not aitk:
                if yml_file.parent.name == "aitk":
                    raise KeyError(f"aitk not found in {yml_file}")
                continue
        print(f"Process aitk for {yml_file}")
        # model info
        modelInfo = convert_yaml_to_model_info(root_dir, yml_file, yaml_object)
        if modelInfo.id.lower() in all_ids:
            raise KeyError(f"same id found in {yml_file}")
        all_ids.add(modelInfo.id.lower())
        modelList.models.append(modelInfo)
        # copy pre
        copyConfigFile = yml_file.parent / "_copy.json.config"
        if copyConfigFile.exists():
            copyConfig = CopyConfig.Read(copyConfigFile.as_posix())
            copyConfig.process(yml_file.parent.as_posix(), pre=True)
            copyConfig.writeIfChanged()
        # model summary
        model_summary = ModelSummary(modelInfo)
        if modelInfo.status == ModelStatusEnum.Ready:
            if isLLM_by_id(modelInfo.id):
                all_summary.llmModels.append(model_summary)
            else:
                all_summary.nonLlmModels.append(model_summary)
        # project config and json configs
        convert_yaml_to_project_config(yml_file, yaml_object, modelList, model_summary)

    modelList.models.sort(key=lambda x: (x.GetSortKey()))
    modelList.writeIfChanged()
    all_summary.write(root_dir)


if __name__ == "__main__":
    project_processor()
