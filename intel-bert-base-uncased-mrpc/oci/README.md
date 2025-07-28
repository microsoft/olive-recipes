# BERT Optimization
This folder contains recipe for BERT optimization using different workflows.

- CPU: PTQ Optimization using CPUExecutionProvider
- GPU: PTQ Optimization using CUDAExeuctionProvider

Go to [How to run](#how-to-run)

### BERT PTQ optimization 

This workflow performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Quantized Onnx Model -> ONNX Runtime performance tuning*

## How to run
### Pip requirements
Install the necessary python packages:
```sh
# [CPU]
pip install git+https://github.com/microsoft/Olive#egg=olive-ai[cpu]
# [GPU]
pip install git+https://github.com/microsoft/Olive#egg=olive-ai[gpu]
```

### Install other dependencies
python -m pip install -r requirements.txt

### Optimize the model
```sh
olive run --config <config_file>.json
```

After running the above command, the final model will be saved in the *output_dir* specified in the config file.
