# The encoding validation tool

- [ev - Encoding validation](./ev.py)
----

## Overview
The motivation of the encoding validation is to detect the potential encoding exception that is generated in Notebook 1 and 2 for diverse machine learning models. It includes single encoding validation for non-lora models, and lora encoding validation for lora models, the mha2sha encoding validation for G2G and the safetensor validation.
### Target users
The encoding validation script is intended for the following users:
- Developers working on Notebook 1, responsible for exporting ONNX graphs and encoding artifacts.
- Developers involved in Notebook 2, focused on on-target deployment and exporting either net.json or dlc files.
- QNN developers who need to verify the correctness of dlc or net.json files.
## AIMET Encoding format
The current script supports AIMET encoding versions 0.6 and 1.0. The primary difference between these two encoding formats lies in the data structure used to store the encodings:

- Version 0.6: The activation_encodings and param_encodings sections are dictionaries. Each key is a string representing the tensor name, and the corresponding value is a list of one or more dictionaries for per-tensor, per-channel, or block encoding. Each encoding includes a min and max range. The format example of encoding version 0.6 is shown below:
```
"name": [
          {
            "bitwidth": ,
            "dtype": "",
            "is_symmetric": "",
            "max": ,
            "min": ,
            "offset": ,
            "scale":
            },
          {...
          }
          ...
        ],
```
- Version 1.0: The activation_encodings and param_encodings sections are lists containing multiple dictionaries, each representing the encoding of a specific tensor. The offset and scale are list structures containing one or more values for per-tensor, per-channel, or block encoding. This version does not include min and max ranges. If AIMET adds min and max values, they should also be in list structures to support per-channel and block encoding. Additionally, each encoding in version 1.0 includes an encoding type, which is not present in version 0.6.
The format example of encoding version 1.0 is shown as below
```
{
  "bw": "",
  "dtype": "",
  "enc_type": "",
  "is_sym": ,
  "name": "",
  "offset": [...],
  "scale": [...]
},
```
## Encoding Validation
### Single encoding validation
Single encoding validation includes aimet encoding validation, qnn encoding validation and dlc encoding validation.
#### AIMET encoding validation
The AIMET encoding validation uses the onnx and qimet encodings file that are exported from Notebook 1 as the inputs. AIMET encoding validation checks AIMET encodings to detect the encodings that vialate the validation rules.
#### QNN encoding validation
The QNN encoding validation uses the net.json that is generated from qnn-onnx-converter in Notebook 2, and the AIMET encodings file from Notebook 1 as the inputs. The QNN encoding validation first identify the mismatches between QNN encodings and AIMET encodings, then it checks QNN encodings to detect the encodings that vialate the validation rules.
#### DLC encoding validation
The DLC encoding validation uses the .dlc file that is generated from qairt-converter and qairt-quantizer in Notebook 2, and the qimet encodings file from Notebook 1 as the inputs. The DLC encoding validation first identify the mismatches between the DLC encodings and AIMET encodings, then it checks QNN encodings to detect the encodings that vialate the validation rules.
#### Validation rules
The validation rules are listed below:

1. Rules by Op type:
- Gather: input encoding equals to output encoding
- Concat: input encoding equals to output encoding
- Transpose: input encoding equals to output encoding
- Reshape: input encoding equals to output encoding
- StrideSlice: input encoding equals to output encoding
- MatMul: Second input is symmetric; data type and bitwidth are float and 16 for BQ, otherwise int and 8; Output scale should not be too large to make integer softmax implementation
- Conv: Weight parameter is symmetric. Bitwidth listed in the table below:

    | Model type | parameter type | bitwidth |
    |------------|----------------|----------|
    | LLM | "lm_head" base weight  | 8 |
    | LLM | other base weights  | 4 |
    | LVM | all base weights  | 8 |
    | LVM+LoRA | lora weights  | 16 |
    | LLM+LoRA | lora weights  | 16 |
    | LLM-BQ | all base weights  | 4 |
    | LLM-LPBQ | all base weights | 8 |

- Sigmoid: min=0, max=1
- Sofmax: min=0, max=1

2. Rules by tensor:
- key cache (with "past_key" keyword): is symmetric, bitwidth is 16 and dtype is float for bq, otherwise bitwidth is 8 and dtype is int
- value cache (with "past_value" keyword): is symmetric, bitwidth is 16 and dtype is float for bq, otherwise bitwidth is 8 and dtype is int

3. Rules for all encodings:
- If the encoding is symmetric, the offset should be -2^(bitwidth-1)
- The scale of all the encoding should be less than 1e+10 and larger than 1e-10

4. Rules for LoRA:
- LoRA Alpha encoding exists inside MHA or SHA aimet encodings
### LoRA encoding validation:
LoRA encoding validation gets onnx graph, net.json graph or dlc graph as the first input, the corresponding AIMET encoding file as the second input and the second AIMET encoding of different adapter as the third input. The validation first do single encoding validation for the first AIMET encoding file, and compares the items between two AIMET encoding files according to the following rules:

1. The name of each activation encoding between two encoding files should be exactly the same. If there is any missing encoding, show the missing name.
2. The name and value of each base weight encoding between two encoding files should be exactly the same.
3. LoRA weight is 16 bit per tensor in both files
4. The name of every LoRA weight encoding between two encoding files should be exactly the same, encoding should not be all the same.

### MHA2SHA encoding validation:
The MHA2SHA encoding validation gets MHA onnx graph, net.json graph or dlc graph as the first input, the corresponding MHA AIMET encoding file as the second input, the SHA AIMET encoding from G2G as the third input and the MHA to SHA map json as the fourth input. The MHA2SHA validation first do single encoding validation for the MHA AIMET encoding file, then it compares the items between MHA and SHA AIMET encoding files according to the following rules:

1. For one-to-one MHA2SHA encoding mapping, encoding should be exactly the same.
2. For one-to-multiple encoding mapping, if MHA is per-tensor encoding, all the SHA encoding shoule be equal to their corresponding MHA encoding; if MHA is per-channel encoding, the SHA encodings should be equal to the corresponding channels of their corresponding MHA encoding.
## Options overview
```shell
$ python ev.py -h
usage: ev.py [-h] [-e ENCODINGS] [-c] [-s] [-t {lora,llm-bq,llm-lpbq,lvm,llm}]
          [-m MAP_PATH] [-2e SEC_ENCODINGS]
          inputfile [outputfile]

Encoding Validation

positional arguments:
  inputfile             onnx file name
  outputfile            optional output file name

options:
  -h, --help            show this help message and exit
  -e ENCODINGS, --encodings ENCODINGS
                        AIMET encoding file name
  -t {lora,llm-bq,llm-lpbq,lvm,llm}, --model-type {lora,llm-bq,llm-lpbq,lvm,llm}
                        Sepcify the model type
  -m MAP_PATH, --map_path MAP_PATH
                        MHA2SHA encoding map json path
  -2e SEC_ENCODINGS, --sec-encodings SEC_ENCODINGS
                        The second AIMET encoding file name
```
## Command examples
### Single encoding validation using different graphs:
- The single encoding validation using onnx graph:
  ```shell
  python ev.py path/to/model.onnx -e path/to/model.encodings -t <model_type>
  ```
- The single encoding validation using dlc graph:
  ```shell
  python ev.py path/to/model.dlc -e path/to/model.encodings -t <model_type>
  ```
- The single encoding validation using net.json graph:
  ```shell
  python ev.py path/to/model_net.json -e path/to/model.encodings -t <model_type>
  ```
### LoRA adapter encoding validation:
  ```shell
  python ev.py path/to/model.onnx -e path/to/model_base.encodings -2e path/to/model_stickers.encodings -t <model_type>
  or python ev.py path/to/model.dlc -e path/to/model_base.encodings -2e path/to/model_stickers.encodings -t <model_type>
  or python ev.py path/to/model_net.json -e path/to/model_base.encodings -2e path/to/model_stickers.encodings -t <model_type>
  ```
### LoRA mha2sha encoding validation for single SHA graph
  ```shell
  python ev.py path/to/model.onnx -e path/to/whole_mha.encodings -2e path/to/model_stickers.encodings -m mha_to_sha_map.json -t <model_type>
  or python ev.py path/to/model.dlc -e path/to/whole_mha.encodings -2e path/to/model_stickers.encodings -m mha_to_sha_map.json -t <model_type>
  or python ev.py path/to/model_net.json -e path/to/whole_mha.encodings -2e path/to/model_stickers.encodings -m mha_to_sha_map.json -t <model_type>
  ```
### LoRA mha2sha encoding validation for split SHA graphs (assume 2 splits)
  ```shell
  python ev.py path/to/model.onnx -e path/to/whole_mha.encodings -2e path/to/split_1.encodings,path/to/split_2.encodings -m path/to/split_1_mha_to_sha_map.json,path/to/split_2_mha_to_sha_map.json -t <model_type>
  or python ev.py path/to/model_net.json -e path/to/whole_mha.encodings -2e path/to/split_1.encodings,path/to/split_2.encodings -m path/to/split_1_mha_to_sha_map.json,path/to/split_2_mha_to_sha_map.json -t <model_type>
  or python ev.py path/to/model.dlc -e path/to/whole_mha.encodings -2e path/to/split_1.encodings,path/to/split_2.encodings -m path/to/split_1_mha_to_sha_map.json,path/to/split_2_mha_to_sha_map.json -t <model_type>
  ```
### Supported mode types:

  | Models | <model_type> | command example |
  |--------|--------------|-----------------|
  | SD1, SD2 true base model (Text encoder, UNET, Vae decoder), VEG | lvm | -t lvm |
  | llama1, llama2 and llama3 true base model, JAIS | llm | -t llm |
  | llama1 and llama2 LoRA model | llm,lora | -t llm,lora |
  | SD1, SD2 LoRA model (UNET with LoRA) | lvm,lora | -t lvm,lora |
  | llama BQ | llm-bq | -t llm-bq |
  | llama LPBQ | llm-lpbq | -t llm-lpbq |
  | SS Combo | llm-lpbq,lora  | -t llm-lpbq,lora |
