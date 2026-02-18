# -------------------------------------------------------------------------
# Copyright (c) Intel® Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import sys
import shutil
import argparse
import json

from transformers import WhisperConfig


def run_olive_workflow(config_json: dict):
    """
    Run Olive workflow given a config JSON object.

    @param config_json: JSON dictionary representing Olive workflow configuration.
    """
    from olive.workflows import run
    run(config_json)


def str2bool(v: str) -> bool:
    """
    Convert string to boolean.

    @param v: Input string.
    @return: Boolean value.
    @raises argparse.ArgumentTypeError: If the input string is not a valid boolean representation.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", 't', 'y', '1'):
        return True
    elif v.lower() in ("no", "false", 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def handle_arguments() -> argparse.Namespace:
    """
    Handle command line arguments.

    @return: Parsed arguments.
    """
    # setup argparse
    desc_txt = "Whisper model conversion to ONNX OpenVINO IR encapsulated model using Olive"
    parser = argparse.ArgumentParser(description=desc_txt)
    parser.add_argument(
        '-m',
        "--model",
        type=str,
        default="openai/whisper-large-v3-turbo",
        help="Huggingface ID for whisper model to convert. Default: openai/whisper-large-v3-turbo"
    )
    parser.add_argument(
        '-w',
        "--weight-format",
        type=str,
        default="fp16",
        choices=["fp16", "int8", "int4"],
        help="The weight format used to compress the model. Default: fp16. Other options: int8, int4"
    )
    parser.add_argument(
        "--enable_npu_ws",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="Enable use of NPUW weight sharing between prefill & generate compiled models. "
             "Can be used as a flag (--enable_npu_ws) or with a value (--enable_npu_ws True). Default: False"
    )
    parser.add_argument(
        "--reshape",
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help="Reshape encoder & decoder models at conversion time, instead of adding reshape_input "
             "provider options to genai_config.json. "
             "Can be used as a flag (--reshape) or with a value (--reshape True). Default: False"
    )

    # parse args
    args = parser.parse_args()
    return args


def load_templates() -> list[dict]:
    """
    Load default whisper conversion, encapsulation and audio configs

    @return: list of JSON dictionaries
        1st dict: default whisper conversion config
        2nd dict: default whisper encapsulation config
        3rd dict: default whisper audio processor config
    """
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # load default config json
    default_config_path = os.path.join(script_dir, "whisper_large_v3_turbo_default_ov_npu.json")
    with open(default_config_path, 'r') as f:
        default_json = json.load(f)

    # load the default encapsulation json
    default_encapsulation_path = os.path.join(script_dir, "whisper_large_v3_turbo_encapsulate.json")
    with open(default_encapsulation_path, 'r') as f:
        default_encapsulation_json = json.load(f)

    # load the default audio preprocessor config json
    audio_processor_config_path = os.path.join(script_dir, "audio_processor_config_default.json")
    with open(audio_processor_config_path, 'r') as f:
        audio_processor_config_json = json.load(f)

    return [default_json, default_encapsulation_json, audio_processor_config_json]


def reshape_and_save_to_tmp(input_ov_path: str, reshape_dict: dict, tmp_ov_path: str):
    """
    Reshape OpenVINO model and save to temporary path.

    @param input_ov_path: Path to input OpenVINO model.
    @param reshape_dict: Dictionary of input names to new shapes.
    @param tmp_ov_path: Path to save the reshaped OpenVINO model.
    """
    from openvino.runtime import Core, serialize
    core = Core()
    ov_model = core.read_model(input_ov_path)
    ov_model.reshape(reshape_dict)
    serialize(ov_model, tmp_ov_path)


def update_genai_overrides(encapsulate_json, w_config: WhisperConfig, enable_npu_ws: bool, reshape: bool, output_directory: str):
    """
    Update genai_config overrides based on NPU_WS and reshape flags.

    @param encapsulate_json: JSON dictionary representing Olive workflow configuration.
    @param w_config: Whisper configuration object.
    @param enable_npu_ws: Boolean flag to enable NPU weight sharing.
    @param reshape: Boolean flag to enable reshaping of models.
    @param output_directory: Directory where converted models are stored.
    @return: encoder_provider_options for post-processing
    """
    # for whisper, typically a prefill consists of 4 or fewer tokens
    max_prompt_len = 4

    # For Whisper models, the effective maximum decoder sequence length is given by
    # 'max_target_positions'. In some configs (e.g., openai/whisper-large-v3-turbo),
    # 'max_length' is set to a much smaller value (e.g., 20 instead of 448), while
    # 'max_target_positions' is consistently set to the correct value (448 in that case).
    # Therefore we rely on 'max_target_positions' here instead of 'max_length' for all
    # Whisper models.
    max_length = w_config.max_target_positions

    if enable_npu_ws:
        print("  [OK] Enabling NPU Weight Sharing between prefill & generate models")
        decoder_load_config = {
            "NPU": {
                "MAX_PROMPT_LEN": max_prompt_len,
                "MIN_RESPONSE_LEN": max_length - max_prompt_len,
                "NPUW_FUNCALL_FOR_ALL": "YES",
                "NPUW_FOLD": "YES",
                "NPUW_WHISPER": "YES",
                "NPUW_WEIGHTS_BANK": "whisper-shared",
                "NPUW_LLM_PREFILL_HINT": "STATIC",
                "NPUW_ONLINE_PIPELINE": "NONE"
            }
        }
    else:
        decoder_load_config = {
            "NPU": {
                "MAX_PROMPT_LEN": max_prompt_len,
                "MIN_RESPONSE_LEN": max_length - max_prompt_len,
                "NPUW_FUNCALL_FOR_ALL": "NO",
                "NPUW_FOLD": "NO",
                "NPUW_WHISPER": "YES",
                "NPUW_LLM_PREFILL_HINT": "STATIC",
                "NPUW_ONLINE_PIPELINE": "NONE"
            }
        }

    # Only stringify the load_config value, not the whole provider_options
    decoder_load_config_str = json.dumps(decoder_load_config, separators=(',', ':'))

    provider_options_encoder = {
        "OpenVINO": {
            "device_type": "NPU"
        }
    }

    # This 3000 magic number is common across all official whisper models.
    # It could potentially be calculated by loading preprocessor_config.json and performing the calculation. But just fix it at 3000 for now.
    # But in general, it's derived from:
    # sampling_rate = 16000
    # hop_length = 160
    # chunk_length = 30
    # sequence_length = chunk_length * sampling_rate / hop_length
    # 3000 = 30 * 16000 / 160
    sequence_length = 3000
    input_features_shape = [1, w_config.num_mel_bins, sequence_length]
    encoder_hidden_states_shape = [1, w_config.max_source_positions, w_config.d_model]
    
    encoder_hidden_states_shape_str = None
    input_features_shape_str = None

    if reshape:
        print("  [OK] Reshaping decoder model now...")
        reshape_and_save_to_tmp(os.path.join(output_directory, "openvino_decoder_model.xml"),
                                {"encoder_hidden_states": encoder_hidden_states_shape}, "reshaped_tmp.xml")

        os.replace("reshaped_tmp.xml", os.path.join(output_directory, "openvino_decoder_model.xml"))
        os.replace("reshaped_tmp.bin", os.path.join(output_directory, "openvino_decoder_model.bin"))

        print("  [OK] Reshaping encoder model now...")
        reshape_and_save_to_tmp(os.path.join(output_directory, "openvino_encoder_model.xml"),
                            {"input_features": input_features_shape}, "reshaped_tmp.xml")

        os.replace("reshaped_tmp.xml", os.path.join(output_directory, "openvino_encoder_model.xml"))
        os.replace("reshaped_tmp.bin", os.path.join(output_directory, "openvino_encoder_model.bin"))
    else:
        encoder_hidden_states_shape_str = "encoder_hidden_states" + str(encoder_hidden_states_shape).replace(" ", "")
        input_features_shape_str = "input_features" + str(input_features_shape).replace(" ", "")
        provider_options_encoder["OpenVINO"]["reshape_input"] = input_features_shape_str

    # Build decoder provider_options as proper JSON object (only load_config is a string)
    provider_options_decoder = {
        "OpenVINO": {
            "device_type": "NPU",
            "enable_causallm": "True",
            "load_config": decoder_load_config_str
        }
    }

    if encoder_hidden_states_shape_str:
        provider_options_decoder["OpenVINO"]["reshape_input"] = encoder_hidden_states_shape_str

    # Update the encapsulate_json with decoder provider options
    model_config = encapsulate_json["passes"]["encapsulation"]["genai_config_override"]["model"]
    
    # Update decoder options - keep as dict, not string
    if "decoder" in model_config and "session_options" in model_config["decoder"]:
        model_config["decoder"]["session_options"]["provider_options"][0] = provider_options_decoder

    # Return encoder info for post-processing genai_config.json after encapsulation
    return provider_options_encoder


def post_process_genai_config(genai_config_path: str, provider_options_encoder: dict, w_config: WhisperConfig):
    """
    Post-process genai_config.json to add encoder section and fix decoder section.
    
    The genai_config.json is generated by Olive's encapsulation pass during the first model
    (encoder) processing. This means the decoder section may have incorrect values copied
    from the encoder. This function fixes those values and adds the encoder section.
    
    @param genai_config_path: Path to the genai_config.json file
    @param provider_options_encoder: Encoder provider options dict
    @param w_config: Whisper configuration object
    """
    with open(genai_config_path, 'r') as f:
        genai_config = json.load(f)


    max_length = w_config.max_target_positions
    
    # Add encoder section with all required fields
    genai_config["model"]["encoder"] = {
        "session_options": {
            "log_id": "onnxruntime-genai",
            "provider_options": [provider_options_encoder]
        },
        "filename": "openvino_encoder_model.onnx",
        "head_size": w_config.d_model // w_config.encoder_attention_heads,
        "hidden_size": w_config.d_model,
        "inputs": {
            "audio_features": "input_features"
        },
        "outputs": {
            "encoder_hidden_states": "last_hidden_state"
        },
        "num_attention_heads": w_config.encoder_attention_heads,
        "num_hidden_layers": w_config.encoder_layers,
        "num_key_value_heads": w_config.encoder_attention_heads
    }
    
    # Fix decoder section - the genai_config may have been generated from encoder encapsulation
    # so decoder values may be incorrect and need to be overwritten
    if "decoder" in genai_config["model"]:
        decoder = genai_config["model"]["decoder"]

        # Fix filename (may have been set to encoder filename)
        decoder["filename"] = "openvino_decoder_model.onnx"

        # Fix head_size and hidden_size from config
        decoder["head_size"] = w_config.d_model // w_config.decoder_attention_heads
        decoder["hidden_size"] = w_config.d_model
        decoder["num_attention_heads"] = w_config.decoder_attention_heads
        decoder["num_hidden_layers"] = w_config.decoder_layers
        decoder["num_key_value_heads"] = w_config.decoder_attention_heads

        # Fix inputs (decoder takes input_ids and encoder_hidden_states)
        decoder["inputs"] = {
            "input_ids": "input_ids",
            "encoder_hidden_states": "encoder_hidden_states"
        }

        # Fix outputs (decoder outputs logits)
        decoder["outputs"] = {
            "logits": "logits"
        }

        # Remove graph_optimization_level if present (not needed)
        if "session_options" in decoder and "graph_optimization_level" in decoder["session_options"]:
            del decoder["session_options"]["graph_optimization_level"]
    
    # Fix pad_token_id - Olive may set it incorrectly from tokenizer/config
    # The correct value comes from WhisperConfig.pad_token_id
    genai_config["model"]["pad_token_id"] = w_config.pad_token_id
    
    # Update context_length to match max_length
    genai_config["model"]["context_length"] = max_length
    genai_config["search"]["max_length"] = max_length
    
    with open(genai_config_path, 'w') as f:
        json.dump(genai_config, f, indent=4)


def run_encapsulation(encapsulation_config_json, output_directory: str, w_config: WhisperConfig, enable_npu_ws: bool, reshape: bool, cache_dir: str = "cache"):
    """
    Run Olive encapsulation workflow for Whisper models.

    @param encapsulation_config_json: JSON dictionary representing Olive workflow configuration.
    @param output_directory: Directory where converted models are stored.
    @param w_config: Whisper configuration object.
    @param enable_npu_ws: Boolean flag to enable NPU weight sharing.
    @param reshape: Boolean flag to enable reshaping of models.
    @param cache_dir: Directory where Olive cache is stored.
    """
    # Models to encapsulate (encoder and decoder)
    models_to_encapsulate = ["openvino_encoder_model", "openvino_decoder_model"]
    cache_models_dir = os.path.join(cache_dir, "default_workflow", "runs")
    
    if os.path.exists(cache_models_dir):
        for run_hash in os.listdir(cache_models_dir):
            models_subdir = os.path.join(cache_models_dir, run_hash, "models")
            if os.path.exists(models_subdir):
                for item in os.listdir(models_subdir):
                    src_path = os.path.join(models_subdir, item)
                    if os.path.isfile(src_path):
                        dst_path = os.path.join(output_directory, item)
                        if not os.path.exists(dst_path):
                            shutil.copy(src_path, dst_path)
                    elif os.path.isdir(src_path):
                        if any(item.startswith(model) for model in models_to_encapsulate):
                            continue
                        for subitem in os.listdir(src_path):
                            if subitem.endswith((".xml", ".bin")):
                                sub_src = os.path.join(src_path, subitem)
                                sub_dst = os.path.join(output_directory, subitem)
                                if not os.path.exists(sub_dst):
                                    shutil.copy(sub_src, sub_dst)
    
    # Find and rename model files from subdirectories
    os.chdir(output_directory)
    subdirs_to_delete = []
    
    for item in os.listdir("."):
        if os.path.isdir(item):
            for subitem in os.listdir(item):
                # Look for encoder/decoder model files with _st/_dy suffix
                for model_name in models_to_encapsulate:
                    if subitem.startswith(f"{model_name}_") and subitem.endswith(".xml"):
                        # Move and rename XML file
                        src_xml = os.path.join(item, subitem)
                        dst_xml = f"{model_name}.xml"
                        if os.path.exists(dst_xml):
                            os.remove(dst_xml)
                        shutil.move(src_xml, dst_xml)
                        
                        # Move and rename BIN file
                        src_bin = src_xml.replace(".xml", ".bin")
                        dst_bin = f"{model_name}.bin"
                        if os.path.exists(dst_bin):
                            os.remove(dst_bin)
                        shutil.move(src_bin, dst_bin)
                        
                        if item not in subdirs_to_delete:
                            subdirs_to_delete.append(item)
                
                # Move any JSON files from subdirectories to root
                if subitem.endswith(".json"):
                    src_json = os.path.join(item, subitem)
                    if os.path.exists(subitem):
                        os.remove(subitem)
                    shutil.move(src_json, subitem)
    
    # Delete emptied subdirectories
    for subdir in subdirs_to_delete:
        if os.path.exists(subdir) and os.path.isdir(subdir):
            shutil.rmtree(subdir)
    
    # Verify both models exist
    for model_name in models_to_encapsulate:
        if not os.path.exists(f"{model_name}.xml"):
            raise FileNotFoundError(f"Could not find {model_name}.xml in output directory")
    
    # Update genai overrides based on NPU_WS and reshape flags
    provider_options_encoder = update_genai_overrides(encapsulation_config_json, w_config, enable_npu_ws, reshape, ".")
    
    # Store absolute path to output directory for genai_config post-processing
    output_dir_abs = os.path.abspath(".")
    
    # Get list of all files in output directory
    all_files = [f for f in os.listdir(".") if os.path.isfile(f)]
    
    # Separate model files (to be encapsulated) from other files
    model_files = {}
    for model_name in models_to_encapsulate:
        model_files[model_name] = [f"{model_name}.xml", f"{model_name}.bin"]
    
    # All other files (not ANY XML/BIN files) will be copied to tmp_dir once
    # We exclude ALL XML/BIN files because the encapsulation pass requires exactly 1 XML and 1 BIN
    other_files = [f for f in all_files if not f.endswith((".xml", ".bin"))]
    
    # Run encapsulation for each model
    # The encapsulation pass requires exactly 1 XML and 1 BIN file in the directory
    tmp_dir = "olive_wrap_tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    
    # Copy all non-XML/BIN files to tmp_dir once
    for f in other_files:
        shutil.copy(f, os.path.join(tmp_dir, f))
    
    genai_config_copied = False
    for model_name in models_to_encapsulate:
        model_xml = f"{model_name}.xml"
        model_bin = f"{model_name}.bin"
        
        # Copy this model's XML/BIN files to tmp_dir
        shutil.copy(model_xml, os.path.join(tmp_dir, model_xml))
        shutil.copy(model_bin, os.path.join(tmp_dir, model_bin))
        
        os.chdir(tmp_dir)
        
        # Update input_model path and run encapsulation
        encapsulation_config_json["input_model"]["model_path"] = "."
        encapsulation_config_json["output_dir"] = "model"
        
        run_olive_workflow(encapsulation_config_json)
        
        # Move ONNX file back to output directory (parent of tmp_dir)
        model_onnx = f"{model_name}.onnx"
        shutil.move(os.path.join("model", model_onnx), os.path.join("..", model_onnx))
        
        # Copy genai_config.json if generated (only once, from first model)
        genai_config_path = os.path.join("model", "genai_config.json")
        if os.path.exists(genai_config_path) and not genai_config_copied:
            shutil.copy(genai_config_path, os.path.join("..", "genai_config.json"))
            genai_config_copied = True
        
        os.chdir("..")
        
        # Remove this model's XML/BIN from tmp_dir and clean up model output
        os.remove(os.path.join(tmp_dir, model_xml))
        os.remove(os.path.join(tmp_dir, model_bin))
        shutil.rmtree(os.path.join(tmp_dir, "model"))

        # Also clean up cache created by encapsulation
        cache_in_tmp = os.path.join(tmp_dir, "cache")
        if os.path.exists(cache_in_tmp):
            shutil.rmtree(cache_in_tmp)

        # Remove any ONNX files that might have been created in tmp_dir root
        for f in os.listdir(tmp_dir):
            if f.endswith(".onnx"):
                os.remove(os.path.join(tmp_dir, f))
    
    # Clean up tmp_dir
    shutil.rmtree(tmp_dir)
    
    # Post-process genai_config.json to add encoder section and fix formatting
    genai_config_dest = os.path.join(output_dir_abs, "genai_config.json")
    if genai_config_copied and os.path.exists(genai_config_dest):
        post_process_genai_config(genai_config_dest, provider_options_encoder, w_config)
    
    os.chdir("..")


def main():
    # parse arguments
    args = handle_arguments()

    # capture all args locally
    hf_model_id = args.model
    weight_format = args.weight_format
    enable_npu_ws = args.enable_npu_ws
    reshape = args.reshape
    print(f"Converting model: {hf_model_id} with weight format: {weight_format}, enable_npu_ws: {enable_npu_ws}, reshape: {reshape}")

    # load default JSONs
    [default_json, default_encapsulation_json, audio_processor_config_json] = load_templates()

    # load whisper config from transformers for given hf_model_id
    w_config = WhisperConfig.from_pretrained(hf_model_id)

    # override the model ID
    default_json["input_model"]["model_path"] = hf_model_id

    # get just the model name without the org
    model_name = hf_model_id.split("/")[-1]
    model_path = f"model/{model_name}-{weight_format}-ov"

    # override the input model path
    default_encapsulation_json["input_model"]["model_path"] = model_path
    
    # override the output dir
    default_json["output_dir"] = model_path
    default_encapsulation_json["output_dir"] = model_path

    # override the weight format in optimum convert pass ov_quant_config
    default_json["passes"]["optimum_convert"]["ov_quant_config"]["weight_format"] = weight_format

    # override the mel spectrogram config in audio preprocessor config
    audio_processor_config_json["feature_extraction"]["sequence"][-1]["operation"]["attrs"]["n_mel"] = w_config.num_mel_bins

    # Get absolute path for the model output directory before running workflows
    model_path_abs = os.path.abspath(model_path)

    # run the conversion workflow
    print("\n[1/2] Running Whisper conversion workflow...")
    run_olive_workflow(default_json)

    # run the encapsulation workflow
    print("\n[2/2] Running Whisper encapsulation workflow...")
    run_encapsulation(default_encapsulation_json, model_path, w_config, enable_npu_ws, reshape)

    # JSON dump the audio preprocessor config into the output directory
    audio_processor_config_json = json.dumps(audio_processor_config_json, indent=4)
    with open(os.path.join(model_path_abs, "audio_processor_config.json"), "w") as f:
        f.write(audio_processor_config_json)

if __name__ == "__main__":
    main()
