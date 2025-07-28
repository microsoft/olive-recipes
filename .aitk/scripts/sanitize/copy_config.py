"""
Copy configuration classes
"""

import json
import os
import shutil
from typing import Any, List, Union

import pydash
from pydantic import BaseModel

from .constants import ReplaceTypeEnum
from .utils import open_ex, printError, printInfo


class Replacement(BaseModel):
    find: str
    replace: Union[str, Any]
    type: ReplaceTypeEnum = ReplaceTypeEnum.String


class Copy(BaseModel):
    src: str
    dst: str
    replacements: List[Replacement] = []


class CopyConfig(BaseModel):
    copies: List[Copy] = []

    def process(self, modelVerDir: str):
        if not self.copies:
            return
        for copy in self.copies:
            src = os.path.join(modelVerDir, copy.src)
            dst = os.path.join(modelVerDir, copy.dst)
            if not os.path.exists(src):
                printError(f"{src} does not exist")
                continue
            shutil.copy(src, dst)
            if copy.replacements:
                stringReplacements = [repl for repl in copy.replacements if repl.type == ReplaceTypeEnum.String]
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
