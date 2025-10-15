# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
from pathlib import Path
from urllib import request
import numpy as np
import onnxruntime as ort
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_id = "openai/whisper-large-v3"


def _load_audio(audio_path: Path, target_sampling_rate: int = 16000) -> np.ndarray:
    audio, sr = librosa.load(audio_path.as_posix(), sr=target_sampling_rate, mono=True)
    return audio.astype(np.float32)


def _run_encoder(
    session: ort.InferenceSession, input_features: np.ndarray
) -> np.ndarray:
    feeds = {session.get_inputs()[0].name: np.ascontiguousarray(input_features)}
    last_hidden_state = session.run(None, feeds)[0]
    return last_hidden_state


def _prepare_decoder_prompt(processor: WhisperProcessor) -> np.ndarray:
    """Prepare decoder input IDs with proper prompt.

    The prompt should be: [<|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>]
    = [50258, 50259, 50360, 50364]
    """
    bos_token_id = 50258  # <|startoftranscript|>
    forced_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    # forced_ids are (position, token_id) pairs, starting from position 1
    # We need to prepend the BOS token at position 0
    prompt = [bos_token_id] + [token_id for _, token_id in forced_ids]
    return np.array([prompt], dtype=np.int64)


def _flatten_present_to_past(
    output_names, present_tensors, previous=None, skip_encoder=False
):
    """Convert present KV outputs to past KV inputs for next iteration.

    IMPORTANT: For merged decoder:
    - First step (use_cache=False): outputs include valid encoder present KVs (129 outputs)
    - Subsequent steps (use_cache=True): outputs include encoder KVs but with wrong shapes (129 outputs)
    - We must preserve encoder past KVs from first step and skip encoder updates in later steps
    """
    past = {} if previous is None else dict(previous)

    # Process all tensors
    for name, tensor in zip(output_names[: len(present_tensors)], present_tensors):
        if name.startswith("present"):
            # Skip encoder outputs in subsequent steps (use_cache_branch=True)
            # They have invalid shapes (batch_size=0)
            if skip_encoder and "encoder" in name:
                continue

            past_name = name.replace("present", "past_key_values")
            past[past_name] = np.ascontiguousarray(tensor)

    return past


def _suppress_logits(logits: np.ndarray, token_ids):
    """Suppress specific token IDs by setting their logits to -inf.

    Args:
        logits: Shape (batch, seq_len, vocab_size) or (batch, vocab_size)
        token_ids: List or set of token IDs to suppress
    """
    for token_id in token_ids:
        if 0 <= token_id < logits.shape[-1]:
            # Handle both 2D and 3D logits
            if len(logits.shape) == 3:
                logits[0, -1, token_id] = -np.inf  # Only suppress at last position
            else:
                logits[0, token_id] = -np.inf


def _build_decoder_merged_feeds(
    session, input_ids, encoder_hidden_states, past_map=None, use_cache=True
):
    """Build feeds for decoder_merged model.

    The merged decoder has a use_cache_branch input:
    - False (0): First decoding step (no past KV cache, uses encoder_hidden_states)
    - True (1): Subsequent steps (with past KV cache including encoder cache)
    """
    feeds = {}
    batch_size = input_ids.shape[0]

    for inp in session.get_inputs():
        name = inp.name
        if name == "input_ids":
            feeds[name] = np.ascontiguousarray(input_ids)
        elif name == "encoder_hidden_states":
            feeds[name] = np.ascontiguousarray(encoder_hidden_states)
        elif name == "use_cache_branch":
            # False for first step, True for subsequent steps
            feeds[name] = np.array([use_cache], dtype=np.bool_)
        elif name.startswith("past_key_values"):
            if past_map and name in past_map:
                # Use actual past KV values
                feeds[name] = np.ascontiguousarray(past_map[name])
            else:
                # First step: provide dummy tensors (will be ignored when use_cache_branch=False)
                # The shape is [batch, num_heads, seq_len, head_dim]
                if "encoder" in name:
                    # For encoder KV: shape should match encoder_hidden_states seq length
                    encoder_seq_len = encoder_hidden_states.shape[1]
                    feeds[name] = np.zeros(
                        (batch_size, 20, encoder_seq_len, 64), dtype=np.float32
                    )
                else:
                    # For decoder KV: use seq_len=0 for first step
                    feeds[name] = np.zeros((batch_size, 20, 0, 64), dtype=np.float32)
    return feeds


def generate_transcript_merged_decoder(
    decoder_merged_session: ort.InferenceSession,
    processor: WhisperProcessor,
    encoder_hidden_states: np.ndarray,
    max_new_tokens: int = 64,
):
    """Generate transcript using the merged decoder model."""
    from transformers import WhisperForConditionalGeneration

    # Load generation config to get proper suppression tokens
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    gen_config = model.generation_config

    suppress_tokens = (
        set(gen_config.suppress_tokens) if gen_config.suppress_tokens else set()
    )
    begin_suppress_tokens = (
        set(gen_config.begin_suppress_tokens)
        if gen_config.begin_suppress_tokens
        else set()
    )
    timestamp_begin = 50365  # Timestamp tokens start here

    # First decoding step: use_cache_branch = False
    decoder_input_ids = _prepare_decoder_prompt(processor)

    decoder_outputs = decoder_merged_session.run(
        None,
        _build_decoder_merged_feeds(
            decoder_merged_session,
            decoder_input_ids,
            encoder_hidden_states,
            use_cache=False,
        ),
    )

    logits = decoder_outputs[0]

    # Suppress tokens in first step
    _suppress_logits(logits, suppress_tokens | begin_suppress_tokens)
    _suppress_logits(logits, set(range(timestamp_begin, logits.shape[-1])))

    present_names = [output.name for output in decoder_merged_session.get_outputs()][1:]
    past_map = _flatten_present_to_past(
        present_names, decoder_outputs[1:], skip_encoder=False
    )

    generated = []
    last_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
    generated.append(last_token)

    eos_token_id = processor.tokenizer.eos_token_id
    pad_token_id = processor.tokenizer.pad_token_id

    # Subsequent steps: use_cache_branch = True
    for step in range(max_new_tokens - 1):
        if last_token in {eos_token_id, pad_token_id}:
            break

        next_input_ids = np.array([[last_token]], dtype=np.int64)

        decoder_outputs = decoder_merged_session.run(
            None,
            _build_decoder_merged_feeds(
                decoder_merged_session,
                next_input_ids,
                encoder_hidden_states,
                past_map,
                use_cache=True,
            ),
        )

        logits = decoder_outputs[0]
        # Suppress suppress_tokens and timestamps in subsequent steps
        _suppress_logits(logits, suppress_tokens)
        _suppress_logits(logits, set(range(timestamp_begin, logits.shape[-1])))

        updated_kv = decoder_outputs[1:]

        # Skip encoder outputs in subsequent steps (they have invalid shapes from merged decoder)
        past_map = _flatten_present_to_past(
            present_names, updated_kv, previous=past_map, skip_encoder=True
        )

        last_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        generated.append(last_token)

    full_sequence = decoder_input_ids[0].tolist() + generated
    transcription = processor.tokenizer.decode(full_sequence, skip_special_tokens=True)
    return transcription


def generate_transcript(
    decoder_session: ort.InferenceSession,
    decoder_with_past_session: ort.InferenceSession,
    processor: WhisperProcessor,
    encoder_hidden_states: np.ndarray,
    max_new_tokens: int = 256,
):
    """Generate transcript using models WITHOUT post-processing.

    Key differences from post-processed version:
    - decoder_with_past outputs include encoder present KVs (not just decoder)
    - Need to handle all outputs properly
    """
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    gen_config = model.generation_config

    suppress_tokens = (
        set(gen_config.suppress_tokens) if gen_config.suppress_tokens else set()
    )
    begin_suppress_tokens = (
        set(gen_config.begin_suppress_tokens)
        if gen_config.begin_suppress_tokens
        else set()
    )
    timestamp_begin = 50365

    # First decoding step with decoder (no past)
    decoder_input_ids = _prepare_decoder_prompt(processor)

    decoder_outputs = decoder_session.run(
        None,
        {
            "input_ids": np.ascontiguousarray(decoder_input_ids),
            "encoder_hidden_states": np.ascontiguousarray(encoder_hidden_states),
        },
    )

    logits = decoder_outputs[0]

    # Suppress tokens in first step
    _suppress_logits(logits, suppress_tokens | begin_suppress_tokens)
    _suppress_logits(logits, set(range(timestamp_begin, logits.shape[-1])))

    # Extract past KV from decoder outputs
    # Without post-processing, decoder outputs ALL present KVs (decoder + encoder)
    output_names = [out.name for out in decoder_session.get_outputs()]

    past_kvs = {}
    for name, tensor in zip(output_names[1:], decoder_outputs[1:]):
        past_name = name.replace("present", "past_key_values")
        past_kvs[past_name] = np.ascontiguousarray(tensor)

    generated = []
    last_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
    generated.append(last_token)

    eos_token_id = processor.tokenizer.eos_token_id
    pad_token_id = processor.tokenizer.pad_token_id

    # Subsequent steps with decoder_with_past
    decoder_with_past_input_names = [
        inp.name for inp in decoder_with_past_session.get_inputs()
    ]
    decoder_with_past_output_names = [
        out.name for out in decoder_with_past_session.get_outputs()
    ]

    for step in range(max_new_tokens - 1):
        if last_token in {eos_token_id, pad_token_id}:
            break

        next_input_ids = np.array([[last_token]], dtype=np.int64)

        # Build feeds for decoder_with_past
        feeds = {"input_ids": np.ascontiguousarray(next_input_ids)}
        for inp_name in decoder_with_past_input_names[1:]:
            if inp_name in past_kvs:
                feeds[inp_name] = past_kvs[inp_name]
            else:
                print(f"[WARNING] Missing input: {inp_name}")

        decoder_outputs = decoder_with_past_session.run(None, feeds)

        logits = decoder_outputs[0]

        # Suppress tokens
        _suppress_logits(logits, suppress_tokens)
        _suppress_logits(logits, set(range(timestamp_begin, logits.shape[-1])))

        # Update past KV
        # Without post-processing, decoder_with_past may output different KVs
        # We need to update only the changed ones (decoder KVs)
        for name, tensor in zip(
            decoder_with_past_output_names[1:], decoder_outputs[1:]
        ):
            past_name = name.replace("present", "past_key_values")
            past_kvs[past_name] = np.ascontiguousarray(tensor)

        last_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
        generated.append(last_token)

    full_sequence = decoder_input_ids[0].tolist() + generated
    transcription = processor.tokenizer.decode(full_sequence, skip_special_tokens=True)
    return transcription


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
    parser = argparse.ArgumentParser("Inference arguments")
    parser.add_argument(
        "--merge_decoder", action="store_true", help="Merge decoder models"
    )
    args = parser.parse_args()

    base = Path(__file__).parent
    audio_path = download_audio_test_data()

    models_dir = base / "models"
    encoder_path = models_dir / "encoder" / "model.onnx"
    decoder_path = models_dir / "decoder" / "model.onnx"
    decoder_with_past_path = models_dir / "decoder_with_past" / "model.onnx"
    decoder_merged_path = models_dir / "decoder_merged" / "model.onnx"

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

    audio = _load_audio(audio_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="np")

    providers = ["CPUExecutionProvider"]
    text = ""

    if args.merge_decoder:
        print("Using merged decoder model...")
        encoder_session = ort.InferenceSession(
            encoder_path.as_posix(), providers=providers
        )
        decoder_merged_session = ort.InferenceSession(
            decoder_merged_path.as_posix(), providers=providers
        )

        encoder_hidden_states = _run_encoder(encoder_session, inputs.input_features)

        text = generate_transcript_merged_decoder(
            decoder_merged_session,
            processor,
            encoder_hidden_states,
            max_new_tokens=256,
        )

    else:
        encoder_session = ort.InferenceSession(
            encoder_path.as_posix(), providers=providers
        )
        decoder_session = ort.InferenceSession(
            decoder_path.as_posix(), providers=providers
        )
        decoder_with_past_session = ort.InferenceSession(
            decoder_with_past_path.as_posix(), providers=providers
        )

        encoder_hidden_states = _run_encoder(encoder_session, inputs.input_features)

        text = generate_transcript(
            decoder_session,
            decoder_with_past_session,
            processor,
            encoder_hidden_states,
            max_new_tokens=256,
        )

    print("\nTranscription:", text)


if __name__ == "__main__":
    main()
