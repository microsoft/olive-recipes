# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import shutil
from pathlib import Path
from typing import Optional, Union

from diffusers import OnnxRuntimeModel
from diffusers.utils import ONNX_EXTERNAL_WEIGHTS_NAME, ONNX_WEIGHTS_NAME

import numpy as np
import onnxruntime as ort


class PatchedOnnxRuntimeModel(OnnxRuntimeModel):
    def __init__(self, model=None, **kwargs):
        self.model = model
        self.model_save_dir = kwargs.get("model_save_dir", None)

    def __call__(self, **kwargs):
        inputs = {k: np.array(v) for k, v in kwargs.items()}
        return self.model.run(None, inputs)

    @staticmethod
    def load_model(path: Union[str, Path], provider=None, sess_options=None, provider_options=None):
        return ort.InferenceSession(path, sess_options=sess_options)

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
        model_file_name = ONNX_WEIGHTS_NAME

        src_path = self.model_save_dir.joinpath(model_file_name)
        dst_path = Path(save_directory).joinpath(model_file_name)

        try:
            shutil.copyfile(src_path, dst_path)
        except shutil.SameFileError:
            pass

        # copy external weights (for models >2GB)
        src_path = self.model_save_dir.joinpath(ONNX_EXTERNAL_WEIGHTS_NAME)
        if src_path.exists():
            dst_path = Path(save_directory).joinpath(ONNX_EXTERNAL_WEIGHTS_NAME)
            try:
                shutil.copyfile(src_path, dst_path)
            except shutil.SameFileError:
                pass

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        **kwargs,
    ):
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        self._save_pretrained(save_directory, **kwargs)


    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        provider: Optional[str] = None,
        sess_options: Optional["ort.SessionOptions"] = None,
        **kwargs,
    ):
        model_file_name = ONNX_WEIGHTS_NAME
        model_path = Path(model_id, model_file_name).as_posix()

        model = PatchedOnnxRuntimeModel.load_model(
            model_path,
            provider=provider,
            sess_options=sess_options,
        )

        kwargs["model_save_dir"] = Path(model_id)

        return cls(model=model, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_id: Union[str, Path],
        **model_kwargs,
    ):
        return cls._from_pretrained(model_id=model_id, **model_kwargs)
