import glob
import os

import numpy as np
from qai_hub_models.utils.input_spec import make_torch_inputs

from olive.data.registry import Registry


def model_loader(model_name):
    if model_name == "openai/whisper-large-v3-turbo":
        from qai_hub_models.models.whisper_large_v3_turbo import Model

        model = Model.from_pretrained()
        component = model.components["HfWhisperDecoder"]
        return component
    else:
        raise ValueError(f"Invalid model id provided: {model_name}")


def generate_dummy_inputs(model=None):
    from qai_hub_models.models.whisper_large_v3_turbo import Model

    model = Model.from_pretrained()
    component = model.components["HfWhisperDecoder"]
    input_spec = component.get_input_spec()
    return tuple(make_torch_inputs(input_spec))


class DecoderBaseDataLoader:
    def __init__(self, data_path):
        self.data_files = glob.glob(os.path.join(data_path, "**", "*_decoder_input.npy"), recursive=True)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        return np.load(self.data_files[idx], allow_pickle=True).item()


@Registry.register_dataloader()
def decoder_data_loader(dataset, data_path):
    return DecoderBaseDataLoader(data_path)
