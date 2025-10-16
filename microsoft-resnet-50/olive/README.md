# ResNet optimization with PTQ on CPU

This workflow performs ResNet optimization on CPU with ONNX Runtime PTQ. It performs the optimization pipeline:

- *PyTorch Model -> Onnx Model -> Quantized Onnx Model -> ONNX Runtime performance tuning*

Config file: [resnet_ptq_cpu.json](resnet_ptq_cpu.json)

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
olive run --config resnet_ptq_cpu.json --setup
```

Then, optimize the model

```bash
olive run --config resnet_ptq_cpu.json
```

After running the above command, the final model will be saved in the *output_dir* specified in the config file.
