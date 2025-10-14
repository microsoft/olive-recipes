# Falcon Optimization
This folder contains a sample use case of Olive to optimize a [falcon-7b](https://huggingface.co/tiiuae/falcon-7b) model.

## How to run
### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

### Optimize the model

```
olive optimize --precision fp16 --provider CUDAExecutionProvider -m tiiuae/falcon-7b -o falcon_out
```
After running the above command, the model candidates and corresponding config will be saved in the output directory.
