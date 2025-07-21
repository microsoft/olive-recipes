# Introduction

For each folder, a info.yml should be created. This file is divided into three parts.

## General information

```YAML
keywords:
    aitk
ep:
    OpenVINOExecutionProvider
    VitisAIExecutionProvider
    QNNExecutionProvider
    NvTensorRTRTXExecutionProvider
device:
    cpu
    npu
    gpu
arch: bert
```

The first part contains general information for all recipes, for example architecture, ep, and device.
If not specified, we could gather them from the second part.

## Per file information

```YAML
files:
    - file: "bert_qdq_qnn.json"
      device: npu
      ep: QNNExecutionProvider
    - file: "bert_qdq_amd.json"
      device: npu
      ep: VitisAIExecutionProvider
    - file: "bert_ov.json"
      ep: VitisAIExecutionProvider
    - file: "bert_trtrtx.json"
      device: gpu
      ep: NvTensorRTRTXExecutionProvider
```

The second part contains specific information for each file.
If not speficied, the property will inherit from general information.
For example, bert_ov.json will target cpu, npu and gpu devices.

## Special information

```YAML
aitk:
    modelInfo:
        id: "huggingface/Intel/bert-base-uncased-mrpc"
        version: 1
    workflows:
    - file: "bert_qdq_qnn.json"
    - file: "bert_qdq_amd.json"
    - file: "bert_ov.json"
    - file: "bert_trtrtx.json"
```

The last part is optional and it contains special information for a special target like aitk, fl.
