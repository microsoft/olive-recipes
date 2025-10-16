# Falcon Optimization
This folder contains a sample use case of Olive to optimize a [opt-125m](https://huggingface.co/facebook/opt-125m) model.

### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

### Optimize the model

```
olive optimize --precision int4 --provider CUDAExecutionProvider -m facebook/opt-125m -o opt_125_out
```
After running the above command, the model candidates and corresponding config will be saved in the output directory.
