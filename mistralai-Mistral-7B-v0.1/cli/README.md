# Falcon Optimization
This folder contains a sample use case of Olive to optimize a [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) model.

### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

### Optimize the model

```
olive optimize --precision fp16 --provider CUDAExecutionProvider -m mistralai/Mistral-7B-v0.1 -o mistral_fp16_out
```
After running the above command, the model candidates and corresponding config will be saved in the output directory.
