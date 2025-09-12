# ResNet optimization with Nvidia TensorRT-RTX execution provider

This example performs ResNet optimization with Nvidia TensorRT-RTX execution provider. It performs the optimization pipeline:

- *ONNX Model -> fp16 Onnx Model*

Config file: [resnet_trtrtx.json](resnet_trtrtx.json)

## How to run

### Pip requirements

Install the necessary python packages:

```bash
python -m pip install -r requirements.txt
```

### Prepare data and model

```bash
python prepare_model_data.py
```

### Run sample using config

The optimization techniques to run are specified in the relevant config json file.

First, install required packages according to passes.

```bash
olive run --config resnet_trtrtx.json --setup
```

Then, optimize the model

```bash
olive run --config resnet_trtrtx.json
```

After running the above command, the final model will be saved in the *output_dir* specified in the config file.
