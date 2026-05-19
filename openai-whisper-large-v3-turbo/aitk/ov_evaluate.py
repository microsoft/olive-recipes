from pathlib import Path
import numpy as np
import argparse
import librosa
import subprocess
import json
import sys
import os
import time
import onnxruntime_genai as og

def register_execution_providers():
    import ctypes
    import importlib.util
    from pathlib import Path

    # Locate onnxruntime package path without importing it first
    ort_spec = importlib.util.find_spec("onnxruntime")
    assert ort_spec is not None and ort_spec.origin is not None
    ort_package_path = Path(ort_spec.origin).parent
    ort_capi_dir = ort_package_path / "capi"
    ort_dll_path = ort_capi_dir / "onnxruntime.dll"

    # Load the onnxruntime DLL because "C:\Windows\System32\onnxruntime.dll" may be exist and loaded first
    ctypes.WinDLL(str(ort_dll_path))

    worker_script = os.path.abspath('winml.py')
    result = subprocess.check_output([sys.executable, worker_script], text=True)
    paths = json.loads(result)
    for item in paths.items():
        try:
            og.register_execution_provider_library(item[0], item[1])  # pyright: ignore[reportAttributeAccessIssue]
            print(f"Successfully registered execution provider {item[0]} from {item[1]}")
        except Exception as e:
            print(f"Failed to register execution provider {item[0]} from {item[1]}: {e}")


def test_transcript(model_path, audio_path, num_beams=0, execution_provider="OpenVINO", device_type="NPU"):
    print("Loading audio...")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    raw_speech, samplerate = librosa.load(audio_path, sr=16000)
    input_speech = raw_speech.tolist()

    print("Loading model...")
    print(f"Model path: {model_path}")

    config = og.Config(model_path)
    config.set_provider_option(execution_provider, "device_type", device_type)
    model = og.Model(config)
    processor = model.create_multimodal_processor()    

    print(f"Processing audio: {audio_path}")
    batch_size = 1
    decoder_prompt_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    prompts = ["".join(decoder_prompt_tokens)]

    # librosa usually returns a 1-D array of np.float32 values by default on load() with normalized values [-1.0,1.0]
    # input_speech is a list[float] now
    samples = np.array(input_speech, dtype=np.float32)
    # convert to 16-bit un-normalized PCM [-32768,32767]
    samples = (samples * np.iinfo(np.int16).max).astype(np.int16)
    import io
    import wave
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1) # 1 = mono, change to 2 for stereo
        wf.setsampwidth(2) # 2 bytes per sample for int16
        wf.setframerate(16000) # sample rate at which we loaded using librosa
        wf.writeframes(samples.tobytes()) # write to buffer
    buff_val = buffer.getvalue()
    audios = og.Audios.open_bytes(buff_val)

    inputs = processor(prompts, audios=audios)

    print(f"Processing:")
    params = og.GeneratorParams(model)
    params.set_search_options(
        do_sample=False,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        max_length=448,
        batch_size=batch_size,
    )

    latencies = []

    for _ in range(20):
        generator = og.Generator(model, params)
        generator.set_inputs(inputs)

        while not generator.is_done():
            start_time = time.perf_counter()
            generator.generate_next_token()
            latencies.append(time.perf_counter() - start_time)

    return latencies

def main():
    parser = argparse.ArgumentParser(description="Test Whisper ONNX GenAI models.")
    parser.add_argument("--execution_provider", type=str, default="CPUExecutionProvider", help="OG Execution provider")
    parser.add_argument("--device_str", type=str, default="cpu")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_path", required=True, help="Path to Whisper ONNX GenAI model")
    args = parser.parse_args()

    # model path
    model_path = args.model_path
    test_audio_url = "https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/librispeech_s5/how_are_you_doing_today.wav"
    test_audio_name = "how_are_you_doing_today.wav"

    import requests
    r = requests.get(test_audio_url)
    open(test_audio_name, "wb").write(r.content)

    register_execution_providers()

    num_beams = 1
    latencies = test_transcript(model_path, test_audio_name, num_beams, args.execution_provider, args.device_str.upper())

    latency_avg = round(sum(latencies) / len(latencies) * 1000, 5)
    throughput_avg = round(1 / latency_avg * 1000, 5)

    metrics = {
        "latency-avg": latency_avg,
        "throughput-avg": throughput_avg,
    }
    resultStr = json.dumps(metrics, indent=4)
    with open(args.output_file, 'w') as file:
        file.write(resultStr)
    print("Model lab succeeded for evaluation.\n%s", resultStr)


if __name__ == "__main__":
    main()
