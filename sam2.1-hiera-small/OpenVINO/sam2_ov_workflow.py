import subprocess
import sys

import olive.workflows


def main():
    # define script and config file paths
    data_prep_script = "prep_ov_quant_data.py"
    vision_encoder_config = "sam21_vision_encoder_ov.json"
    mask_decoder_config = "sam21_mask_decoder_ov.json"

    # prepare the calibration data in blocking mode
    # data is required for running the olive workflows for
    # both SAM2.1 Vision Encoder and Mask Decoder models.
    subprocess.run([sys.executable, data_prep_script], check=True)

    # run SAM 2.1 Vision Encoder workflow to generate
    # Intel® OpenVINO Execution Provider ONNX Encapsulated OVIR model
    output_ve = olive.workflows.run(vision_encoder_config)
    if output_ve is None or not output_ve.has_output_model():
        error = f"Execution of {vision_encoder_config} was unsuccessful. SAM 2.1 ONNX Intel® OpenVINO IR Encapsulated Vision Encoder model file was not generated."
        raise RuntimeError(error)

    # run SAM 2.1 Mask Decoder workflow to generate
    # Intel® OpenVINO Execution Provider ONNX Encapsulated OVIR model
    output_md = olive.workflows.run(mask_decoder_config)
    if output_md is None or not output_md.has_output_model():
        error = f"Execution of {mask_decoder_config} was unsuccessful. SAM 2.1 ONNX Intel® OpenVINO IR Encapsulated Mask Decoder model file was not generated."
        raise RuntimeError(error)

if __name__ == "__main__":
    main()
