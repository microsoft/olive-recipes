# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from urllib import request
import onnxruntime_genai as og

def generate_transcript(model_path, audio_path, num_beams=0, execution_provider="cuda"):
    """Generate transcript using onnxruntime-genai.

    Args:
        model_path: Path to the genai model directory
        audio_path: Path to audio file
        num_beams: Number of beams for beam search
        execution_provider: Execution provider (cpu, cuda, or follow_config)

    Returns:
        Transcription text
    """
    print("Loading model...")
    print(f"Model path: {model_path}")
    config = og.Config(model_path)
    if execution_provider != "follow_config":
        config.clear_providers()
        if execution_provider != "cpu":
            print(f"Setting model to {execution_provider}")
            config.append_provider(execution_provider)
    model = og.Model(config)
    processor = model.create_multimodal_processor()

    print("Loading audio...")
    if not Path(audio_path).exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    audios = og.Audios.open(audio_path)

    print(f"Processing audio: {audio_path}")
    batch_size = 1
    decoder_prompt_tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    prompts = ["".join(decoder_prompt_tokens)]
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


def download_audio_test_data():
    cur_dir = Path(__file__).parent
    data_dir = cur_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    test_audio_name = "1272-141231-0002.mp3"
    test_audio_url = (
        "https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/test/data/"
        + test_audio_name
    )
    test_audio_path = data_dir / test_audio_name
    if not test_audio_path.exists():
        request.urlretrieve(test_audio_url, test_audio_path)

    return test_audio_path.relative_to(cur_dir)


def main():
    base = Path(__file__).parent
    audio_path = download_audio_test_data()

    models_dir = base / "models"

    # Generate transcription using genai
    text = generate_transcript(
        str(models_dir),
        str(audio_path),
        num_beams=1,
        execution_provider="cuda"
    )

    print("\nTranscription:", text)


if __name__ == "__main__":
    main()
