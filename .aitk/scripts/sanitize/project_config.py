"""
Model project configuration classes
"""

from typing import List, Optional

from pydantic import BaseModel

from .base import BaseModelClass
from .constants import IconEnum, ArchitectureEnum
from .model_info import ModelInfo
from .utils import GlobalVars, open_ex, printError, printProcess


class WorkflowItem(BaseModel):
    displayName: Optional[str] = None
    file: str
    templateName: str
    # DO NOT ADD ANYTHING ELSE HERE
    # We should add it to the *.json.config

    def Check(self):
        if not self.file:
            return False
        if "\\" in self.file:
            printError("Please use / instead of \\")
            return False
        if not self.templateName:
            return False
        return True


class ModelInfoProject(BaseModel):
    id: str
    version: int = -1
    displayName: Optional[str] = None
    icon: Optional[IconEnum] = None
    modelLink: Optional[str] = None
    architecture: Optional[ArchitectureEnum] = None

    def Check(self, modelInfo: ModelInfo):
        if not self.id:
            return False
        if self.displayName and self.displayName != modelInfo.displayName:
            return False
        if self.icon and self.icon != modelInfo.icon:
            return False
        if self.modelLink and self.modelLink != modelInfo.modelLink:
            return False
        if self.architecture and self.architecture != modelInfo.architecture:
            return False
        return True


class ModelProjectConfig(BaseModelClass):
    workflows: List[WorkflowItem]
    modelInfo: ModelInfoProject

    @staticmethod
    def Read(modelSpaceConfigFile: str):
        printProcess(modelSpaceConfigFile)
        with open_ex(modelSpaceConfigFile, "r") as file:
            modelSpaceConfigContent = file.read()
        modelSpaceConfig = ModelProjectConfig.model_validate_json(modelSpaceConfigContent, strict=True)
        modelSpaceConfig._file = modelSpaceConfigFile
        modelSpaceConfig._fileContent = modelSpaceConfigContent
        return modelSpaceConfig

    # after template is set
    def Check(self, modelInfo: ModelInfo):
        GlobalVars.modelProjectCheck.append(self._file)

        for i, model in enumerate(self.workflows):
            if not model.Check():
                printError(f"{self._file} model {i} has error")

        if not self.modelInfo.Check(modelInfo):
            printError(f"{self._file} modelInfo has error")

        self.writeIfChanged()
