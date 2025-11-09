# Bert-base-multilingual-cased Model Optimization

### QNN-GPU:

Running QNN-GPU configs requires features and fixes that are not available in the released Olive version 0.9.3.
To ensure compatibility, please install Olive directly from the source at the required commit:

```bash
pip install git+https://github.com/microsoft/Olive.git@da24463e14ed976503dc5871572b285bc5ddc4b2
```

If you previously installed Olive via PyPI or pinned it to version 0.9.3, please uninstall it first and then use the above
commit to install:

```bash
pip uninstall olive-ai
```

To run the config:

```bash
olive run --config config_qnn_gpu.json
```

✅ Optimized model saved in: `output/`
