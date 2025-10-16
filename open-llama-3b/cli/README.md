# Falcon Optimization
This folder contains a sample use case of Olive to optimize a [Open-llama-3B](https://huggingface.co/openlm-research/open_llama_3b) model.

### Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

### Optimize the model

```
olive optimize --provider CUDAExecutionProvider -m openlm-research/open_llama_3b -o open_llama_out
```
After running the above command, the model candidates and corresponding config will be saved in the output directory.
