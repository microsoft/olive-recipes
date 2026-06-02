# SAM2.1 Hiera Small Quantization

This folder contains a sample use case of Olive to optimize the [facebook/sam2.1-hiera-small](https://huggingface.co/facebook/sam2.1-hiera-small) model using Intel® OpenVINO tools.

- SAM2.1 Hiera Small Vision Encoder: [sam2.1-hiera-small vision encoder](#sam-21-hiera-small-vision-encoder)
- SAM2.1 Hiera Small Mask Decoder: [sam2.1-hiera-small mask decoder](#sam-21-hiera-small-mask-decoder)

## Quantization Workflows

For both the SAM 2.1 Vision Encoder and SAM 2.1 Mask Decoder, this workflow performs the optimization pipeline:

- *PyTorch Model -> OpenVINO model -> Quantized OpenVINO model -> Quantized encapsulated ONNX OpenVINO IR model*

## How to run

### Model Quantization Environment Setup

Create a python virtual environment for model conversion and install the necessary python packages:

```bash
python -m venv sam2_ov_quant_env
sam2_ov_quant_env\Scripts\activate
python -m pip install -r ..\..\.aitk\requirements\requirements-IntelNPU.txt
python -m pip install -r ..\..\.aitk\requirements\requirements-IntelNPU-SAM.txt
```

### Run the full pipeline

The full pipeline can be run with a single script that executes all steps from downloading and generating the calibration dataset, to running the quantization workflow for both the SAM 2.1 Hiera Small vision encoder and mask decoder models using Intel® OpenVINO NNCF tools.

After following the virtual environment creation, activation and package installation steps in [Model Quantization Environment Setup](#model-quantization-environment-setup), run the full pipeline in the created virtual environment with:

```bash
python sam2_ov_workflow.py
```

Once complete, the steps to run the inference are listed in [Run Inference Sample IPython Notebook](#run-inference-sample-ipython-notebook) to execute the optimized models.

---

### (Optional) Run each step manually

The steps can also be run separately as outlined below.

#### Generate Calibration Dataset

Download and prepare the calibration dataset for both the SAM 2.1 Vision Encoder and Mask Decoder models.

The script downloads the COCO128 dataset, extracts all images, and runs the FP32 Vision Encoder model forward pass after image pre-processing to save the image_embeddings, high_res_feats_256 and high_res_feats_128 which are the Mask Decoder model inputs.

After following the virtual environment creation, activation and package installation steps in [Model Quantization Environment Setup](#model-quantization-environment-setup), run the data preparation script in the created virtual environment with:

```bash
python prep_ov_quant_data.py
```

#### SAM 2.1 Hiera Small Vision Encoder

The workflow in Config file: [sam21_vision_encoder_ov.json](sam21_vision_encoder_ov.json) executes the above workflow producing a static shape quantized ONNX Encapsulated OpenVINO IR SAM 2.1 Hiera Small Vision Encoder model.

#### SAM 2.1 Hiera Small Mask Decoder

The workflow in Config file: [sam21_mask_decoder_ov.json](sam21_mask_decoder_ov.json) executes the above workflow producing a static shape quantized ONNX Encapsulated OpenVINO IR SAM 2.1 Hiera Small Mask Decoder model.

#### Run Olive config

The optimization techniques to run are specified in the relevant config json file.

After following the virtual environment creation, activation and package installation steps in [Model Quantization Environment Setup](#model-quantization-environment-setup), run the olive configs in the created virtual environment with:

##### CLI

```bash
olive run --config sam21_vision_encoder_ov.json
olive run --config sam21_mask_decoder_ov.json
```

— **or** —

##### Python

```python
import olive.workflows
workflow_output = olive.workflows.run("sam21_vision_encoder_ov.json")
workflow_output = olive.workflows.run("sam21_mask_decoder_ov.json")
```

After running the above command, the models will be saved in the output folder specified in the olive recipe JSON file.

### Run Inference Sample IPython Notebook

#### Inference Sample Environment Setup

Create a python virtual environment for inference and install the necessary python packages to run ONNX OpenVINO IR Encapsulated models with ONNXRuntime OpenVINO EP support via WindowsML:

```bash
python -m venv sam2_ov_inference_env
sam2_ov_inference_env\Scripts\activate
python -m pip install -r ..\..\.aitk\requirements\requirements-WCR.txt
python -m pip install -r ..\..\.aitk\requirements\requirements-WCR-SAM.txt
python -m pip install wasdk-Microsoft.Windows.AI.MachineLearning==2.0.1 --no-deps
python -m pip install notebook
```

#### Inference execution

After following the virtual environment creation, activation and package installation steps in [Inference Sample Environment Setup](#inference-sample-environment-setup), run the notebook in the created virtual environment with:

```bash
jupyter notebook inference_sample.ipynb
```

This will start the Jupyter Notebook server and open a browser window at `http://localhost:8888`. Run the [SAM2.1 Hiera Small Inference Sample Notebook](inference_sample.ipynb) with the Olive generated ONNX OpenVINO IR Encapsulated models.
