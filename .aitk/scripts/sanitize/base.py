"""
Base model classes
"""

from typing import Optional

from pydantic import BaseModel

from .utils import open_ex


class BaseModelClass(BaseModel):
    """Base class for all model classes with file I/O capabilities"""

    _file: Optional[str] = None
    _fileContent: Optional[str] = None

    def writeIfChanged(self):
        newContent = self.model_dump_json(indent=4, exclude_none=True)
        if self._file:
            BaseModelClass.writeJsonIfChanged(newContent, self._file, self._fileContent)

    @classmethod
    def writeJsonIfChanged(cls, newContent: str, filePath: str, fileContent: str | None):
        newContent += "\n"
        if newContent != fileContent:
            with open_ex(filePath, "w") as file:
                file.write(newContent)

    class Config:
        arbitrary_types_allowed = True
