"""
Model information and model list classes
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

from model_lab import RuntimeEnum
from pydantic import BaseModel

from .base import BaseModelClass
from .constants import ArchitectureEnum, IconEnum, ModelStatusEnum
from .utils import GlobalVars, open_ex, printError, printProcess

# This file is import by others
# To avoid circular import issues, we should carefully manage imports


class ModelInfo(BaseModel):
    displayName: str
    discription: Optional[str] = None
    icon: IconEnum
    modelLink: str
    id: str
    groupId: Optional[str] = None
    groupItemName: Optional[str] = None
    runtimes: List[RuntimeEnum]
    architecture: ArchitectureEnum
    status: ModelStatusEnum = ModelStatusEnum.Hide
    relativePath: Optional[str] = None
    version: int = -1
    extension: Optional[bool] = None
    p0: Optional[bool] = None

    def Check(self):
        if self.status == ModelStatusEnum.Hide:
            return True
        if not self.displayName:
            return False
        if not self.modelLink:
            return False
        if not self.id and self.status == ModelStatusEnum.Ready:
            return False
        if not self.runtimes:
            return False
        if self.version <= 0 and self.status == ModelStatusEnum.Ready:
            return False
        return True

    def GetSortKey(self):
        lowerName = self.displayName.lower()
        match = re.search(r"-(\d+(?:\.\d+)?)b", lowerName)
        if match:
            return (lowerName.replace(match.group(0), "-0b", 1), float(match.group(1)))
        else:
            return (lowerName, 0)


class ModelList(BaseModelClass):
    models: List[ModelInfo]
    template_models: List[ModelInfo]
    HFDatasets: Dict[str, str]
    LoginRequiredDatasets: List[str]
    LoginRequiredModelIds: List[str]
    # If exist in the dict, we will use the one from dict
    # If not exist in the dict, we will use the config from json
    # - if only one value, don't need to add
    # - custom config could provide a combined list for new datasets
    DatasetSplit: Dict[str, List[str]]
    DatasetSubset: Dict[str, List[str]]
    DisplayNameToRuntimeRPC: Dict[str, RuntimeEnum] = {}
    RuntimeToDisplayName: Dict[RuntimeEnum, str] = {}

    @staticmethod
    def Read(scriptFolder: str):
        modelListFile = os.path.join(scriptFolder, "model_list.json")
        printProcess(modelListFile)
        with open_ex(modelListFile, "r") as file:
            modelListContent = file.read()
        modelList = ModelList.model_validate_json(modelListContent, strict=True)
        modelList._file = modelListFile
        modelList._fileContent = modelListContent
        return modelList

    def allModels(self):
        return self.models + self.template_models

    # Check after set version
    def Check(self):
        self.models.sort(key=lambda x: x.GetSortKey())
        # TODO template models order needs manually set
        # self.template_models.sort(key=lambda x: x.displayName.lower())
        for i, model in enumerate(self.allModels()):
            if not model.Check():
                printError(f"{self._file} model {i} has error")
        self.SetupConstants()
        self.writeIfChanged()

        self.CheckDataset(self.LoginRequiredDatasets, "LoginRequiredDatasets")
        self.CheckDataset(self.DatasetSplit.keys(), "DatasetSplit")
        self.CheckDataset(self.DatasetSubset.keys(), "DatasetSubset")
        self.CheckModel(self.LoginRequiredModelIds, "LoginRequiredModelIds")

    def SetupConstants(self):
        self.RuntimeToDisplayName = GlobalVars.RuntimeToDisplayName
        self.DisplayNameToRuntimeRPC = {v: k for k, v in self.RuntimeToDisplayName.items()}

    def CheckDataset(self, datasetKeys, name: str):
        for key in datasetKeys:
            if key not in self.HFDatasets:
                printError(f"{self._file} {name} {key} not in HFDatasets")

    def CheckModel(self, modelIds, name: str):
        tmpAllModelIds = {model.id for model in self.models}
        for key in modelIds:
            if key not in tmpAllModelIds:
                printError(f"{self._file} {name} {key} not in ModelInfos")
