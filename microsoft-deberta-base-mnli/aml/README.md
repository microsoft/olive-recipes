# Deberta Optimization
This folder contains a sample use case of Olive to optimize a [microsoft-deberta-base-mnli](https://ml.azure.com/models/microsoft-deberta-base-mnli/version/5/catalog/registry/HuggingFace) model from Azure ML Model catalog.

## Optimization Workflows
This workflow performs Deberta optimization on CPU with ONNX Runtime. It performs the optimization pipeline:
- *PyTorch Model -> Onnx Model -> Transformers Optimized Onnx Model -> Quantized Onnx Model

Config file: [deberta.json](deberta.json)

## How to run
### Install Olive
Install Olive from pip:
```
pip install olive-ai[azureml]
```

### Log into the az account
Run `az login` to login your Azure account.

### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

### Run sample using config

```
olive run --config deberta.json
```

or run simply with python code:
```python
from olive.workflows import run as olive_run
olive_run("deberta.json")
```

After running the above command, the model candidates and corresponding config will be saved in the output directory.
You can then select the best model and config from the candidates and run the model with the selected config.
