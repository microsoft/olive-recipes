"""
Copy configuration classes
"""

import json
import os
import shutil
from typing import Any, List, Optional, Union

import pydash
from pydantic import BaseModel

from .base import BaseModelClass
from .constants import ReplaceTypeEnum
from .utils import GlobalVars, open_ex, printError, printInfo, printProcess


class Replacement(BaseModel):
    find: str
    replace: Union[str, Any]
    type: Optional[ReplaceTypeEnum] = None


class Copy(BaseModel):
    src: str
    dst: str
    replacements: Optional[List[Replacement]] = None


class CopyConfig(BaseModelClass):
    copies: List[Copy] = []

    @staticmethod
    def Read(copyConfigFile: str):
        printProcess(copyConfigFile)
        with open_ex(copyConfigFile, "r") as file:
            copyConfigContent = file.read()
        copyConfig = CopyConfig.model_validate_json(copyConfigContent, strict=True)
        copyConfig._file = copyConfigFile
        copyConfig._fileContent = copyConfigContent
        return copyConfig

    def process(self, modelVerDir: str, pre: bool = True):
        if not self.copies:
            return
        for copy in self.copies:
            src = os.path.join(modelVerDir, copy.src)
            # validation
            if src.endswith("model_project.config"):
                printError("Should not copy model_project.config, it will be generated")
                continue
            # separate pre and post
            if src.endswith(".json.config"):
                if pre:
                    continue
            else:
                if not pre:
                    continue

            dst = os.path.join(modelVerDir, copy.dst)
            if not os.path.exists(src):
                printError(f"{src} does not exist")
                continue
            shutil.copy(src, dst)
            GlobalVars.copyCheck += 1
            if copy.replacements:
                stringReplacements = [
                    repl for repl in copy.replacements if repl.type == None or repl.type == ReplaceTypeEnum.String
                ]
                if stringReplacements:
                    with open_ex(dst, "r") as file:
                        content = file.read()
                    for replacement in stringReplacements:
                        printInfo(replacement.find)
                        if replacement.find not in content:
                            printError(f"Not in dst file {dst}: {replacement.find}")
                            continue
                        content = content.replace(replacement.find, replacement.replace)
                    with open_ex(dst, "w") as file:
                        file.write(content)
                pathReplacements = [
                    repl
                    for repl in copy.replacements
                    if repl.type == ReplaceTypeEnum.Path or repl.type == ReplaceTypeEnum.PathAdd
                ]
                if pathReplacements:
                    with open_ex(dst, "r") as file:
                        jsonObj = json.load(file)
                    for replacement in pathReplacements:
                        printInfo(replacement.find)
                        target = pydash.get(jsonObj, replacement.find)
                        if (
                            replacement.type == ReplaceTypeEnum.Path
                            and target is None
                            or replacement.type == ReplaceTypeEnum.PathAdd
                            and target
                        ):
                            printError(f"Not match type in dst json {dst}: {replacement.find}")
                            continue
                        pydash.set_(jsonObj, replacement.find, replacement.replace)
                    with open_ex(dst, "w") as file:
                        json.dump(jsonObj, file, indent=4)
                        file.write("\n")
