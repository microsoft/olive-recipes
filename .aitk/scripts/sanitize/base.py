"""
Base model classes
"""

import json
from typing import Optional

from pydantic import BaseModel

from .utils import open_ex


def add_schema_to_config(content: str, schema_url: str) -> str:
    """Inject a '$schema' key as the first entry of a JSON object string."""
    data = json.loads(content)
    # Ensure any existing "$schema" in the input does not override the injected schema URL
    if isinstance(data, dict):
        data.pop("$schema", None)
    with_schema = {"$schema": schema_url}
    with_schema.update(data)
    return json.dumps(with_schema, indent=4)


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
