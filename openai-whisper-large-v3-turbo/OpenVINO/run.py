from pathlib import Path
import numpy as np
import librosa
import onnxruntime_genai as og
dllname = "onnxruntime_providers_openvino_plugin.dll"
og.register_execution_provider_library("OpenVINOExecutionProvider", dllname)

def generate_transcript(model_path, audio_path, num_beams=0):
    """Generate transcript using onnxruntime-genai.

    Args:
        model_path: Path to the genai model directory
        audio_path: Path to audio file
        num_beams: Number of beams for beam search
        execution_provider: Execution provider (cpu, cuda, or follow_config)

    Returns:
        Transcription text
    """
    print("Loading audio...")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    # install librosa with python -m pip install librosa
    raw_speech, samplerate = librosa.load(audio_path, sr=16000)
    input_speech = raw_speech.tolist()
    
    print("Loading model...")
    print(f"Model path: {model_path}")

    config = og.Config(model_path)
    config.set_provider_option("OpenVINO", "device_type", "NPU")
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

    generator = og.Generator(model, params)
    generator.set_inputs(inputs)

    while not generator.is_done():
        generator.generate_next_token()

    print("Finish generation. Decoding outputs...")
    transcriptions = []
    for i in range(batch_size * num_beams):
        tokens = generator.get_sequence(i)
        transcription = processor.decode(tokens)
        transcriptions.append(transcription.strip())

    return transcriptions[0]

def main():
    # model path
    model_path = f".\\model\\whisper-large-v3-turbo-int4-ov\\"
    audio_path = f".\\how_are_you_doing_today.wav"
    num_beams = 1
    transcript = generate_transcript(model_path, audio_path, num_beams)
    print("Transcript: ", transcript)


if __name__ == "__main__":
    main()

 